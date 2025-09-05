from pydantic import BaseModel
from data_structures.prompt import Prompt
from data_structures.generation_model import GenerationModel
from data_structures.prompt_result import PromptResultLite, ProcessedDocument
from data_structures.data_handling import OutputData, InputData
from typing import List, Callable, Tuple, Optional, Any
import concurrent.futures
from logging import Logger
import logging


class Evaluator(BaseModel):
    model_name_inference: str
    response_schema: Any
    logger: Logger = None

    def __init__(self, **data):
        super().__init__(**data)

        def addLoggingLevel(levelName, levelNum, methodName=None):
            if not methodName:
                methodName = levelName.lower()

            if hasattr(logging, levelName):
                raise AttributeError(
                    "{} already defined in logging module".format(levelName)
                )
            if hasattr(logging, methodName):
                raise AttributeError(
                    "{} already defined in logging module".format(methodName)
                )
            if hasattr(logging.getLoggerClass(), methodName):
                raise AttributeError(
                    "{} already defined in logger class".format(methodName)
                )

            def logForLevel(self, message, *args, **kwargs):
                if self.isEnabledFor(levelNum):
                    self._log(levelNum, message, args, **kwargs)

            def logToRoot(message, *args, **kwargs):
                logging.log(levelNum, message, *args, **kwargs)

            logging.addLevelName(levelNum, levelName)
            setattr(logging, levelName, levelNum)
            setattr(logging.getLoggerClass(), methodName, logForLevel)
            setattr(logging, methodName, logToRoot)

        try:
            addLoggingLevel("CUSTOM", 25)
        except AttributeError:
            print("Logging defined already, falling back on it.")
        # Create and configure logger explicitly
        self.logger = logging.getLogger("EvaluatorLogger")

        # Remove all existing handlers (including Jupyter's StreamHandler)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add only FileHandler
        file_handler = logging.FileHandler("eval_info.log", mode="w")
        formatter = logging.Formatter("%(asctime)s %(message)s")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.CUSTOM)
        self.logger.propagate = False

    class Config:
        arbitrary_types_allowed = True

    # wrapper for the evaluate prompt function to access from protegi and other algo.
    def evaluate_prompt(
        self,
        current_prompt: Prompt,
        dataset_batch: List[InputData],
        metric_fn: Callable,
        num_threads: Optional[int] = 1,
    ) -> Tuple[float, List[OutputData], List[OutputData], "Evaluator.ApiStatistics"]:
        return self.__evaluate_prompt(
            current_prompt=current_prompt,
            dataset_batch=dataset_batch,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )

    def __evaluate_prompt(
        self,
        current_prompt: Prompt,
        dataset_batch: List[InputData],
        metric_fn: Callable,
        num_threads: Optional[int] = 1,
    ) -> Tuple[float, List[OutputData], List[OutputData], "Evaluator.ApiStatistics"]:
        """
        Evaluate a prompt on a batch of data using a given metric function.
        The metric function should measure how well the LLM output
        matches the ground truth in dataset_batch.

        :param current_prompt: The prompt we are evaluating.
        :param dataset_batch: A batch of examples (input + ground truth).
        :param metric_fn: A function that takes in (predicted_text, ground_truth)
                        and returns a performance score.
        :num_threads: How many threads to use.
        :return: Average performance score (float), incorrectly predicted examples, correctly predicted examples.
        """
        scores = []
        incorrect_samples = []
        correct_examples = []
        # generate new connectors to avoid race conditions in a multi-threaded environment. One connector = one datapoint,
        # is always used for that datapoint in this function and only for that data point.
        models = [
            GenerationModel(model_name=self.model_name_inference) for _ in dataset_batch
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_evaluation = {
                executor.submit(
                    self.__evaluate_one_prompt,
                    current_prompt,
                    datapoint,
                    metric_fn,
                    models[i],
                ): datapoint
                for i, datapoint in enumerate(dataset_batch)
            }
            for future in concurrent.futures.as_completed(future_to_evaluation):
                score, output_object = future.result()
                scores.append(score)
                if score == 0.0:
                    incorrect_samples.append(output_object)
                elif score == 1.0:
                    correct_examples.append(output_object)
                else:
                    raise ValueError(
                        "Score evaluated is not 0 or 1, but something else."
                    )
        # store token usages
        api_calls = 0
        total_tokens = 0
        total_time = 0
        for model in models:
            api_calls += model.api_calls
            total_tokens += model.total_tokens
            total_time += model.total_time
        api_statistics = Evaluator.ApiStatistics(
            api_calls=api_calls, total_tokens=total_tokens, total_time=total_time
        )
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score, incorrect_samples, correct_examples, api_statistics

    class ApiStatistics(BaseModel):
        api_calls: int
        total_tokens: int
        total_time: float

    def __evaluate_one_prompt(
        self,
        current_prompt: Prompt,
        datapoint: InputData,
        metric_fn: Callable,
        model: GenerationModel,
    ) -> Tuple[float, OutputData]:
        # we use a schema, so no need for a meta prompt.
        meta_prompt = current_prompt.text

        predicted = model.query_llm(
            prompt_text=meta_prompt,
            doc=datapoint.doc,
            schema=self.response_schema,
        )
        # Compare with ground truth using the provided metric function
        score = metric_fn(predicted, datapoint.ground_truth)
        output_object = OutputData(**datapoint.model_dump(), predicted=predicted)
        self.logger.custom(
            f"""Running inference: \n 
                    Prompt: {current_prompt.text}. \n 
                    doc_id: {output_object.doc_id}, gt: {output_object.ground_truth}, predicted: {output_object.predicted}."""
        )
        return score, output_object

    def __metric_fn(
        self,
        predicted: Any,
        ground_truth: Any,
    ) -> float:
        assert type(predicted) == type(ground_truth)
        assert type(predicted) == self.response_schema
        msg = f"predicted: {predicted}, gt: {ground_truth}"
        if predicted == ground_truth:
            msg += ". Evaluated as same."
            self.logger.custom(msg)
            return 1.0
        else:
            msg += ". Evaluated as different."
            self.logger.custom(msg)
            return 0.0

    def evaluate(
        self,
        prompt: Prompt,
        training_data: InputData,
        test_data: InputData,
        num_threads: int,
        metric_fn: Callable = None,
    ) -> PromptResultLite:
        """
        Function to evaluate a prompt on a subset of data.

        :param prompt: Starting prompt we wish to improve.
        :param training_data: Full training data used to refine the prompt before.
        :param test_data: Data to test the prompt.
        :param num_threads: Number of threads to use for evaluation.
        :return: Evaluation result with training derived parameters.
        """
        if metric_fn is None:
            metric_fn = self.__metric_fn
        (
            training_score,
            training_incorrect_examples,
            training_correct_examples,
            test_score,
            test_incorrect_examples,
            test_correct_examples,
        ) = self.__evaluate_both_test_train(
            prompt=prompt,
            training_data=training_data,
            test_data=test_data,
            num_threads=num_threads,
            metric_fn=metric_fn,
        )
        result = PromptResultLite(
            prompt=prompt,
            training_evaluation_correct=[
                ProcessedDocument.create_processed_document(x)
                for x in training_correct_examples
            ],
            training_evaluation_incorrect=[
                ProcessedDocument.create_processed_document(x)
                for x in training_incorrect_examples
            ],
            test_evaluation_correct=[
                ProcessedDocument.create_processed_document(x)
                for x in test_correct_examples
            ],
            test_evaluation_incorrect=[
                ProcessedDocument.create_processed_document(x)
                for x in test_incorrect_examples
            ],
            training_accuracy=training_score,
            test_accuracy=test_score,
        )
        return result

    def __evaluate_both_test_train(
        self,
        prompt: Prompt,
        training_data: InputData,
        test_data: InputData,
        num_threads: int,
        metric_fn: Callable,
    ) -> Tuple[
        float,
        List[OutputData],
        List[OutputData],
        float,
        List[OutputData],
        List[OutputData],
    ]:
        (
            training_score,
            training_incorrect_examples,
            training_correct_examples,
            _,
        ) = self.__evaluate_prompt(
            current_prompt=prompt,
            dataset_batch=training_data,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        test_score, test_incorrect_examples, test_correct_examples, _ = (
            self.__evaluate_prompt(
                current_prompt=prompt,
                dataset_batch=test_data,
                metric_fn=metric_fn,
                num_threads=num_threads,
            )
        )
        return (
            training_score,
            training_incorrect_examples,
            training_correct_examples,
            test_score,
            test_incorrect_examples,
            test_correct_examples,
        )
