import os
from pydantic import BaseModel, create_model, Field
from data_structures.prompt import Prompt
from data_structures.generation_model import GenerationModel
from data_structures.data_handling import OutputData, InputData
import re
from typing import List, Callable, Tuple, Any
import concurrent.futures
import logging
import traceback
from logging import Logger
from data_structures.prompt_result import TrainingStatistics
from evaluator import Evaluator
import pandas as pd
from itertools import combinations
import pulp
from collections import defaultdict, OrderedDict

from data_structures.smart_prompt_config import (
    GROUP_PROMPTS_LLM,
    IDENTIFY_CONTRADICTIONS_PROMPT,
    ASSESS_SIMILARITY_PROMPT,
    GRADIENT_DESCENT_PROMPT_SCHEMA,
    COMBINE_SIMILAR_INSTRUCTIONS_PROMT,
)


class SmartOptimizer(BaseModel):
    model_name_feedback: str
    model_name_inference: str
    response_schema: Any
    results_path: str
    doc_skip_threshold: int = 5
    majority_number_tries: int = 2
    majority_score: float = 1
    error_restarts: int = 5
    shared_model_feedback: "GenerationModel" = None
    shared_model_inference: "GenerationModel" = None
    evaluator: "Evaluator" = None
    response_extract_pattern: str = r"<START>(.*?)<END>"
    logger: Logger = None

    def __init__(self, **data):
        super().__init__(**data)
        self.shared_model_feedback = GenerationModel.create_model(
            model_name=self.model_name_feedback
        )
        self.shared_model_inference = GenerationModel.create_model(
            model_name=self.model_name_inference
        )
        self.evaluator = Evaluator(
            model_name_inference=self.model_name_inference,
            response_schema=self.response_schema,
        )

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
        logging.basicConfig(
            filename="smart_optimizer.log",
            format="%(asctime)s %(message)s",
            filemode="w",
            level=logging.CUSTOM,
        )
        self.logger = logging.getLogger()
        self.logger.setLevel("CUSTOM")

    class Config:
        arbitrary_types_allowed = True

    class ApiStatistics(BaseModel):
        api_calls: int
        total_tokens: int
        total_time: float

    def __extract_answer(self, response: str) -> str:
        # relaxing regex requirement to deal with flash shenanigans.
        patterns = [
            self.response_extract_pattern,
            r"<START>(.*?)</END>",
            r"<START>(.*?)<\\END>",
            r"<START>(.*?)<START>",
        ]
        for pattern in patterns:
            extracted_answer = re.findall(pattern, response, flags=re.DOTALL)
            if extracted_answer:
                break
        if not extracted_answer:
            raise ValueError(f"Response couldn't be extracted. Response: {response}")
        return extracted_answer

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

    def __generate_dynamic_response_schema_for_gradients(self, num_gradients: int):
        grads = {f"fix_{i}": (str, ...) for i in range(1, num_gradients + 1)}
        GradientsResponseSchema = create_model("Gradients", **grads)
        self.logger.custom(
            f"Generated gradient response schema for {num_gradients} gradients: {GradientsResponseSchema}"
        )
        return GradientsResponseSchema

    def __generate_gradients(
        self,
        current_prompt: Prompt,
        error_examples: List[OutputData],
        num_gradients: int,
        model: GenerationModel = None,
    ) -> List[str]:
        """
        Generate gradients that describe how to fix
        current_prompt, based on errors observed. Explanation to behaviour: all of the error examples passed
        will be used in a single prompt to generate gradients. Hence, for a bigger sample problems might occur.

        :param current_prompt: The existing prompt that made mistakes.
        :param error_examples: A small batch of examples the prompt got wrong.
        :param num_gradients: How many different gradient strings to produce.
        :return: List of textual gradients describing potential improvements.
        """
        if model is None:
            model = self.shared_model_feedback
        error_str, docs = self.__make_error_string_to_work_with_pdfs(
            error_examples=error_examples
        )
        schema = self.__generate_dynamic_response_schema_for_gradients(
            num_gradients=num_gradients
        )
        meta_prompt = GRADIENT_DESCENT_PROMPT_SCHEMA.format(
            prompt_text=current_prompt.text,
            num_gradients=num_gradients,
            error_str=error_str,
        )
        response = model.query_llm(
            prompt_text=meta_prompt,
            doc=docs,
            config=self.shared_model_feedback.default_diversity_config,
            schema=schema,
        )
        gradients = list(response.model_dump().values())

        # gradients = self.__extract_answer(response=gradient_text)
        # logging block start
        self.logger.custom("-----GENERATING GRADIENTS-----")
        log_list = []
        for elem in error_examples:
            log_list.append({
                "doc_id": elem.doc_id,
                "gt": elem.ground_truth,
                "predicted": elem.predicted,
            })
        logger_message = (
            f"Generated gradients: \nPrompt: {current_prompt.text}. \nGradients: "
        )
        for i, gradient in enumerate(gradients):
            logger_message += f"\nGradient {i + 1}: {gradient}"
        logger_message += "\nGenerated from incorrect examples: "
        for elem in log_list:
            logger_message += f"\n{elem}."
        self.logger.custom(logger_message)
        # logging block end
        return gradients

    def __make_error_string_to_work_with_pdfs(
        self, error_examples: List[OutputData]
    ) -> Tuple[str, List[Any]]:
        error_str = ""
        docs = []
        for i, example in enumerate(error_examples):
            docs.append(example.doc)
            error_str = (
                error_str
                + f"Example {i}: extracted: {example.predicted}, ground truth: {example.ground_truth}. \n"
            )
        return error_str, docs

    def __get_input_data_for_output_data(
        self, output_data: List[OutputData], all_input_data: List[InputData]
    ) -> List[InputData]:
        input_docs = []
        for doc in output_data:
            input_docs.append(
                [elem for elem in all_input_data if elem.doc_id == doc.doc_id][0]
            )
        return input_docs

    def __classify_training_set(
        self,
        prompt: Prompt,
        training_data: List[InputData],
        metric_fn,
        num_threads: int,
    ) -> Tuple[List[OutputData], List[OutputData]]:
        _, incorrect, correct, api_statistics = self.evaluator.evaluate_prompt(
            current_prompt=prompt,
            dataset_batch=training_data,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        # a temporary solution to do 2 majority voting. no time for normal solutions.
        # get only docs which are correct:
        """if len(correct) == 0:
            return incorrect, correct

        correct_input_docs = self.__get_input_data_for_output_data(
            correct, training_data
        )
        _, incorrect1, correct1, api_statistics1 = self.evaluator.evaluate_prompt(
            current_prompt=prompt,
            dataset_batch=correct_input_docs,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        incorrect_final = incorrect + incorrect1
        self.__update_api_statistics_inference(api_statistics)
        self.__update_api_statistics_inference(api_statistics1)
        return incorrect_final, correct1
        """
        self.__update_api_statistics_inference(api_statistics)
        return incorrect, correct

    def __get_gradients_with_restarts(
        self,
        current_prompt: Prompt,
        error_examples: List[OutputData],
        num_gradients: int,
        model: GenerationModel,
    ) -> List[str]:
        for _ in range(self.error_restarts):
            try:
                gradients = self.__generate_gradients(
                    current_prompt=current_prompt,
                    error_examples=error_examples,
                    num_gradients=num_gradients,
                    model=model,
                )
                return gradients
            except Exception:
                self.logger.custom(
                    f"Gradient generation exception: {traceback.format_exc()}."
                )
                continue
        else:
            raise ValueError("Gradients couldn't be generated in 5 tries.")

    class PromptResultWrapper(BaseModel):
        avg_score: float
        incorrect_examples: List[OutputData]
        correct_examples: List[OutputData]
        prompt: "SmartOptimizer.PromptWrapper"

    def evaluate_multiple_prompts(
        self,
        prompts: List[Prompt],
        dataset_batch: List[InputData],
        gradients: List[str] | None,
        metric_fn: Callable | None = None,
        num_threads: int = 1,
    ) -> Tuple[List["SmartOptimizer.PromptResultWrapper"], ApiStatistics]:
        if len(prompts) == 0 or len(gradients) == 0:
            raise ValueError("prompts and gradients cannot be empty.")
        if metric_fn is None:
            metric_fn = self.__metric_fn
        prompt_wrappers = []
        for prompt, gradient in zip(prompts, gradients):
            prompt_wrappers.append(
                SmartOptimizer.PromptWrapper(prompt=prompt, gradient=gradient)
            )
        res_wrappers, api_statistics = self.__evaluate_multiple_prompts(
            prompts=prompt_wrappers,
            dataset_batch=dataset_batch,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        return res_wrappers, api_statistics

    def __evaluate_multiple_prompts(
        self,
        prompts: List["SmartOptimizer.PromptWrapper"],
        dataset_batch: List[InputData],
        metric_fn: Callable,
        num_threads: int,
    ) -> Tuple[List["SmartOptimizer.PromptResultWrapper"], ApiStatistics]:
        # handle the case where more prompts than threads.
        num_threads_dedicated_to_prompts = min(len(prompts), num_threads)
        num_document_threads = num_threads // num_threads_dedicated_to_prompts
        api_statistics_collection = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads_dedicated_to_prompts
        ) as executor:
            future_to_evaluation = {
                executor.submit(
                    self.evaluator.evaluate_prompt,
                    current_prompt.prompt,
                    dataset_batch,
                    metric_fn,
                    num_threads=num_document_threads,
                ): current_prompt
                for current_prompt in prompts
            }
            result = []
            for future in concurrent.futures.as_completed(future_to_evaluation):
                avg_score, incorrect_examles, correct_examples, api_statistics = (
                    future.result()
                )
                prompt = future_to_evaluation[future]
                result.append(
                    SmartOptimizer.PromptResultWrapper(
                        avg_score=avg_score,
                        incorrect_examples=incorrect_examles,
                        correct_examples=correct_examples,
                        prompt=prompt,
                    )
                )
                api_statistics_collection.append(api_statistics)
        total_api_statistics_inference = self.gather_api_stats(
            api_statistics_collection=api_statistics_collection
        )
        return result, total_api_statistics_inference

    class PromptWrapper(BaseModel):
        prompt: Prompt
        gradient: str

    def attempt_getting_gradient(
        self,
        doc: InputData,
        prompt: Prompt,
        num_gradients: int,
        failed_prediction: OutputData,
        metric_fn: Callable,
        model_feedback: GenerationModel,
        num_threads: int = 1,
    ) -> Tuple[str, ApiStatistics]:
        api_statistics_inference_collection = []
        for _ in range(self.doc_skip_threshold):
            gradients = self.__get_gradients_with_restarts(
                current_prompt=prompt,
                error_examples=[failed_prediction],
                num_gradients=num_gradients,
                model=model_feedback,
            )
            prompts = []
            for gradient in gradients:
                # fix = f"Here is the reason you could fail when performing this task. Take it into account. Potential failure reason: \n{gradient}."
                fix = f"{gradient}"
                prompt_candidate_text = "\n".join([prompt.text, fix])
                prompt_candidate = Prompt(text=prompt_candidate_text)
                prompt_candidate = SmartOptimizer.PromptWrapper(
                    prompt=prompt_candidate, gradient=gradient
                )
                prompts.append(prompt_candidate)
            results, api_statistics_inference = self.__evaluate_multiple_prompts(
                prompts=prompts,
                dataset_batch=[doc] * self.majority_number_tries,
                metric_fn=metric_fn,
                num_threads=num_threads,
            )
            api_statistics_inference_collection.append(api_statistics_inference)
            for result in results:
                if result.avg_score >= self.majority_score:
                    return result.prompt.gradient, self.gather_api_stats(
                        api_statistics_inference_collection
                    )
        # return empty if couldn't be found
        return "", self.gather_api_stats(api_statistics_inference_collection)

    def gather_api_stats(
        self, api_statistics_collection: List["SmartOptimizer.ApiStatistics"]
    ) -> ApiStatistics:
        total_api_statistics = SmartOptimizer.ApiStatistics(
            api_calls=sum([_.api_calls for _ in api_statistics_collection]),
            total_tokens=sum([_.total_tokens for _ in api_statistics_collection]),
            total_time=sum([_.total_time for _ in api_statistics_collection]),
        )
        return total_api_statistics

    def __calc_threads(
        self, num_threads: int, incorrect_docs: List[Any]
    ) -> Tuple[int, int]:
        num_threads_dedicated_to_grads = min(len(incorrect_docs), num_threads)
        num_document_threads = num_threads // num_threads_dedicated_to_grads
        if num_document_threads < 1:
            num_document_threads = 1
        return num_threads_dedicated_to_grads, num_document_threads

    def optimize_prompt_with_all_shenanigans(
        self,
        prompt: Prompt,
        training_data: List[InputData],
        num_gradients: int,
        num_threads: int,
        metric_fn: Callable = None,
    ) -> Tuple[Prompt, ApiStatistics]:
        res_df, bad_docs, _ = self.optimize_prompt(
            prompt=prompt,
            training_data=training_data,
            num_gradients=num_gradients,
            num_threads=num_threads,
            metric_fn=metric_fn,
        )
        # replace doc texts with ids to not break prints and saves down the line.
        res_df["docs"] = res_df["docs"].apply(lambda x: x.doc_id)
        # save all confirmed grads and bad docs.
        res_df.to_csv(os.path.join(self.results_path, "grads.csv"), index=False)
        bad_df = pd.DataFrame({"bad_docs": bad_docs})
        bad_df.to_csv(os.path.join(self.results_path, "bad_docs.csv"), index=False)
        # filter gradients by pairwise combinations of similar using llms
        grads = res_df["gradients"].values.tolist()
        filtered_grads = self.filter_and_combine_using_llm_pairwise(
            instructions=grads, starting_prompt=prompt
        )
        # save filtered grads
        pd.DataFrame({"gradients": filtered_grads}).to_csv(
            os.path.join(self.results_path, "filtered_grads.csv")
        )
        # make prompts from grads for scoring. Each prompt = starting prompt + one grad.
        prompts = [f"{prompt.text}\n{fix}" for fix in filtered_grads]
        # include grads for later retrieval of prompt results based on grads as keys.
        gradients = ["starting prompt"] + [fix for fix in filtered_grads]
        # convert all into prompt objects
        prompts = [prompt] + [Prompt(text=prompt) for prompt in prompts]
        # evaluate all produced prompts
        evals, api_stats = self.evaluate_multiple_prompts(
            prompts=prompts,
            gradients=gradients,
            dataset_batch=training_data,
            num_threads=20,
        )
        self.__update_api_statistics_inference(api_statistics=api_stats)
        # save the results
        evals_dict = {
            "gradient": [eval.prompt.gradient for eval in evals],
            "score": [eval.avg_score for eval in evals],
        }
        df = pd.DataFrame(evals_dict)
        df.to_csv(
            os.path.join(self.results_path, "filtered_grads_eval.csv"), index=False
        )
        # remove baseline (starting prompt), convert to grad lists and scores
        col = df.loc[df.gradient == "starting prompt"]
        grad_df = df[df.gradient != "starting prompt"]
        baseline_score = col["score"].iloc[0]
        instructions = grad_df["gradient"].values.tolist()
        scores = grad_df["score"].values.tolist()
        # drop those that make the performance worse than the baseline
        cut_grad_df = grad_df[grad_df.score >= baseline_score]
        instructions = cut_grad_df["gradient"].values.tolist()
        scores = cut_grad_df["score"].values.tolist()
        clean_grads, _ = self.filter_contradictions_using_llm(
            instructions=instructions, instruction_scores=scores
        )
        # save final results
        final_instructions_and_scores = pd.DataFrame({"instructions": clean_grads})
        final_instructions_and_scores = final_instructions_and_scores.merge(
            grad_df, left_on="instructions", right_on="gradient", how="left"
        )
        final_instructions_and_scores = final_instructions_and_scores[
            ["instructions", "score"]
        ]
        final_instructions_and_scores.to_csv(
            os.path.join(self.results_path, "grad_experiment_final_instructions.csv"),
            index=False,
        )

        # make final prompt
        clean_grads = final_instructions_and_scores["instructions"].values.tolist()
        new_prompt = "\n".join([prompt.text] + clean_grads)
        print(new_prompt)
        new_prompt = Prompt(text=new_prompt)
        training_statistics = TrainingStatistics.factory(
            inference_model=self.shared_model_inference,
            feedback_model=self.shared_model_feedback,
            final_beam_prompts=[],
            final_beam_scores=[],
            beams=[[]],
        )
        training_statistics.save_to_json(
            os.path.join(self.results_path, "training_statistics.csv")
        )
        return new_prompt, training_statistics

    def optimize_prompt(
        self,
        prompt: Prompt,
        training_data: List[InputData],
        num_gradients: int,
        num_threads: int,
        metric_fn: Callable = None,
    ) -> Tuple[pd.DataFrame, List[str], TrainingStatistics]:
        self.logger.custom("----------------------STARTING SMART----------------------")
        # if no metric function, fallback on the default.
        if metric_fn is None:
            metric_fn = self.__metric_fn
        # first classification of the dataset, finding incorrect documents.
        incorrect_docs, _ = self.__classify_training_set(
            prompt=prompt,
            training_data=training_data,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        incorrect_doc_inputs = self.__get_input_data_for_output_data(
            output_data=incorrect_docs, all_input_data=training_data
        )
        # dedicate most threads to generating gradients, less to document evaluations.
        num_grad_threads, num_doc_threads = self.__calc_threads(
            num_threads, incorrect_docs
        )
        # set a pool of feedback models to not lose api statistics.
        feedback_models_pool = [
            GenerationModel(model_name=self.model_name_feedback)
            for _ in incorrect_doc_inputs
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_grad_threads
        ) as executor:
            future_to_evaluation = {
                executor.submit(
                    self.attempt_getting_gradient,
                    doc=doc,
                    prompt=prompt,
                    num_gradients=num_gradients,
                    failed_prediction=[
                        d for d in incorrect_docs if d.doc_id == doc.doc_id
                    ][0],
                    metric_fn=metric_fn,
                    num_threads=num_doc_threads,
                    model_feedback=feedback_models_pool[
                        incorrect_doc_inputs.index(doc)
                    ],
                ): doc
                for doc in incorrect_doc_inputs
            }
            gradients = []
            fixed_docs = []
            bad_docs = []
            for future in concurrent.futures.as_completed(future_to_evaluation):
                gradient, api_statistics_inference = future.result()
                self.__update_api_statistics_inference(api_statistics_inference)
                doc = future_to_evaluation[future]
                if not gradient:
                    bad_docs.append(doc)
                # self.__update_api_statistics_inference(api_statistics=api_statistics)
                else:
                    gradients.append(gradient)
                    fixed_docs.append(doc)
            df = pd.DataFrame({"docs": fixed_docs, "gradients": gradients})
        for model in feedback_models_pool:
            self.__update_api_statistics_feedback(
                SmartOptimizer.ApiStatistics(
                    api_calls=model.api_calls,
                    total_tokens=model.total_tokens,
                    total_time=model.total_time,
                )
            )

        training_statistics = TrainingStatistics.factory(
            inference_model=self.shared_model_inference,
            feedback_model=self.shared_model_feedback,
            final_beam_prompts=[],
            final_beam_scores=[],
            beams=[[]],
        )
        self.logger.custom("----------------------SMART COMPLETE----------------------")
        self.logger.custom(f"Training statistics: {training_statistics}")
        return df, [doc.doc_id for doc in bad_docs], training_statistics

    def filter_and_combine_using_llm(self, instructions: List[str]) -> List[str]:
        if len(instructions) <= 1:
            return instructions
        for _ in range(self.error_restarts):
            try:
                meta_prompt = GROUP_PROMPTS_LLM
                formated_instructions = ""
                for i, instruction in enumerate(instructions):
                    formated_instructions = (
                        f"{formated_instructions} \nInstruction {i}: {instruction}"
                    )
                meta_prompt = f"{meta_prompt}. Instructions: {formated_instructions}"
                self.logger.custom(f"Grouping query: {meta_prompt}")
                response_raw = self.shared_model_feedback.query_llm(
                    prompt_text=meta_prompt
                )
                self.logger.custom(f"Grouping query response: {response_raw}.")
                refined_instructions = self.__extract_answer(response=response_raw)
                return refined_instructions
            except Exception("Something broke"):
                continue
        else:
            raise Exception("Everything broke.")

    def filter_and_combine_using_llm_pairwise(
        self,
        instructions: List[str],
        starting_prompt: Prompt,
    ) -> List[str]:
        if len(instructions) <= 1:
            return instructions
        instruction_pairs = list(combinations(instructions, 2))
        similar_instructions = []

        class SimilarityResponseSchema(BaseModel):
            are_instructions_similar: bool

        for instruction_pair in instruction_pairs:
            for _ in range(self.error_restarts):
                try:
                    meta_prompt = ASSESS_SIMILARITY_PROMPT.format(
                        instruction_1=instruction_pair[0],
                        instruction_2=instruction_pair[1],
                        starting_prompt=starting_prompt,
                    )
                    are_similar = self.shared_model_feedback.query_llm(
                        prompt_text=meta_prompt,
                        schema=SimilarityResponseSchema,
                    ).are_instructions_similar
                    if are_similar:
                        similar_instructions.append(instruction_pair)
                    break
                except Exception("Something broke"):
                    self.logger.custom(
                        f"ERROR: LLM pairwise similarity filtering exception: {traceback.format_exc()}."
                    )
                    continue
            else:
                self.logger.custom(
                    "ERROR: LLM pairwise similarity filtering critical exception."
                )
                raise Exception("Everything broke.")
        # contains only vars with similars
        debug_msg = "Pairs of instructions classified as similar"
        for _ in similar_instructions:
            debug_msg = f"{debug_msg}. \n{_}"
        self.logger.custom(debug_msg)
        flattened_similarity_array = [
            instruction for t in similar_instructions for instruction in t
        ]
        similarities = {}
        unique_instructions = [
            instruction
            for instruction in instructions
            if instruction not in flattened_similarity_array
        ]
        similar_instructions_set = set(flattened_similarity_array)
        for ins in similar_instructions_set:
            temp_similarities = []
            for pair in similar_instructions:
                if ins in pair:
                    temp_similarities += list(pair)
            similarities[ins] = list(set(temp_similarities))
        colapsed_similarities = self.__colapse_similar_instructions(similarities)

        class FinalInstruction(BaseModel):
            final_instruction: str = Field(description="Text of the final instruction.")

        combined = []
        for val in colapsed_similarities.values():
            debug_msg = "Combining similar instructions: "
            meta_prompt = COMBINE_SIMILAR_INSTRUCTIONS_PROMT
            for i, ins in enumerate(val):
                debug_msg = f"{debug_msg}\n{ins}"
                meta_prompt = f"{meta_prompt}. \nInstruction {i + 1}: {ins}"
            comb = self.shared_model_feedback.query_llm(
                prompt_text=meta_prompt, schema=FinalInstruction
            )
            comb = comb.final_instruction
            debug_msg = f"{debug_msg}\n COMBINED INTO:  {comb}"
            self.logger.custom(debug_msg)
            combined.append(comb)
        return unique_instructions + combined

    def __colapse_similar_instructions(
        self, similarities, order=None, include_key=True
    ):
        self.logger.custom(
            f"Similarities in the collapse graph traversal aglorithm: {similarities}"
        )
        adj = defaultdict(set)
        seen = []
        seen_set = set()
        for ins, similars in similarities.items():
            if ins not in seen_set:
                seen.append(ins)
                seen_set.add(ins)
            for similar in similars:
                if similar not in seen_set:
                    seen.append(similar)
                    seen_set.add(similar)
                adj[ins].add(similar)
                adj[similar].add(ins)
        if order is None:
            order = seen
        pos = {v: i for i, v in enumerate(order)}
        out = OrderedDict()
        visited = set()
        for start in order:
            if start in visited or start not in adj:
                continue
            stack = [start]
            visited.add(start)
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            rep = min(comp, key=lambda x: pos.get(x, float("inf")))
            members = sorted(comp, key=lambda x: pos.get(x, float("inf")))
            out[rep] = members if include_key else [m for m in members if m != rep]
        self.logger.custom(
            f"Similarities out of the collapse graph traversal algorithm: {dict(out)}"
        )
        return out

    def __identify_contradictions_using_llm(self, instructions: List[str]) -> Any:
        instruction_pairs = list(combinations(instructions, 2))
        contradicting = {}

        class ContradictionResponseSchema(BaseModel):
            are_instructions_contradicting: bool

        for instruction_pair in instruction_pairs:
            for _ in range(self.error_restarts):
                try:
                    meta_prompt = IDENTIFY_CONTRADICTIONS_PROMPT.format(
                        instruction_pair=instruction_pair
                    )
                    self.logger.custom(f"Contradiction filter query: {meta_prompt}")
                    are_contradicting = self.shared_model_feedback.query_llm(
                        prompt_text=meta_prompt, schema=ContradictionResponseSchema
                    ).are_instructions_contradicting
                    contradicting[instruction_pair] = are_contradicting
                    break
                except Exception("Something broke in contradictions."):
                    self.logger.custom(
                        f"ERROR: LLM contradiction filtering exception: {traceback.format_exc()}."
                    )
                    continue
            else:
                self.logger.custom("FATAL: LLM contradiction filtering exception.")
                raise Exception("Everything broke in contradictions.")
        self.logger.custom(f"Contradiction result: {contradicting}")
        _contr = [key for key in contradicting.keys() if contradicting[key]]
        dbg_msg = "Instructions found to be contradicting: "
        for _ in _contr:
            dbg_msg = f"{dbg_msg}\n{_}"
        self.logger.custom(dbg_msg)
        return contradicting

    """def filter_contradictions_using_llm(
        self,
        instructions: List[str],
        instruction_scores: List[float],
        baseline_score: float,
    ) -> Tuple[List[str], float]:
        self.logger.custom("Starting contradiction filtering")
        are_pairs_contradicting = self.__identify_contradictions_using_llm(
            instructions=instructions
        )
        self.logger.custom(
            "Starting linear programming optimization of the contradiction graph."
        )
        scores = {
            instruction: score
            for instruction, score in zip(
                instructions, [score - baseline_score for score in instruction_scores]
            )
        }
        contradicting_pairs = {
            key
            for key in are_pairs_contradicting.keys()
            if are_pairs_contradicting[key]
        }
        print(scores)
        print(contradicting_pairs)
        self.logger.custom(
            f"Contradicting pairs entering elimination: {contradicting_pairs}"
        )
        m = pulp.LpProblem("Maximum_Weight_Independent_Set", pulp.LpMaximize)
        keep = {
            instruction: pulp.LpVariable(f"keep_{instruction}", 0, 1, cat="Binary")
            for instruction in scores
        }
        m += pulp.lpSum(
            scores[instruction] * keep[instruction] for instruction in scores
        )
        for i, j in contradicting_pairs:
            m += keep[i] + keep[j] <= 1, f"no_{i}_with{j}"
        print(m)
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [
            instruction for instruction in scores if pulp.value(keep[instruction]) > 0.5
        ]
        total_score = sum(scores[instruction] for instruction in selected)
        self.logger.custom(
            f"Linear optimization of the contradiction graph finished. Best score bonus: {total_score}, \nSelected instructions: {selected}"
        )
        return selected, total_score"""

    def filter_contradictions_using_llm(
        self,
        instructions: List[str],
        instruction_scores: List[float],
    ) -> Tuple[List[str], float]:
        """
        Returns the highest-scoring subset of `instructions` such that no two
        selected instructions contradict.
        """
        if len(instructions) != len(instruction_scores):
            raise ValueError(
                "instructions and instruction_scores must have the same length"
            )

        self.logger.custom("Starting contradiction filtering")

        are_pairs_contradicting = self.__identify_contradictions_using_llm(
            instructions=instructions
        )
        idx_of = {instr: i for i, instr in enumerate(instructions)}
        n = len(instructions)
        weights = {i: instruction_scores[i] for i in range(n)}

        # Normalize contradiction edges: keep only valid, dedup by sorting (i < j)
        edges = set()
        for pair, is_con in are_pairs_contradicting.items():
            if not is_con:
                continue
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            a, b = pair
            if a not in idx_of or b not in idx_of or a == b:
                continue
            i, j = idx_of[a], idx_of[b]
            if i > j:
                i, j = j, i
            edges.add((i, j))

        self.logger.custom(f"Contradicting pairs entering elimination (idx): {edges}")

        # ILP formulation (Maximum Weight Independent Set)
        m = pulp.LpProblem("Maximum_Weight_Independent_Set", pulp.LpMaximize)
        x = {
            i: pulp.LpVariable(f"k_{i}", lowBound=0, upBound=1, cat="Binary")
            for i in range(n)
        }

        # Objective: maximize sum of original scores
        m += pulp.lpSum(weights[i] * x[i] for i in range(n))

        # Constraints: no two contradicting instructions together
        for i, j in edges:
            m += x[i] + x[j] <= 1, f"no_{i}_with_{j}"

        # Solve
        status = m.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            self.logger.custom(f"Solver ended with status {pulp.LpStatus[status]}")

        selected_ids = [i for i in range(n) if pulp.value(x[i]) > 0.5]
        selected = [instructions[i] for i in selected_ids]
        total_score = sum(instruction_scores[i] for i in selected_ids)

        self.logger.custom(
            f"Linear optimization finished. Total score: {total_score}, "
            f"Selected instructions: {selected}"
        )
        return selected, total_score

    def __update_api_statistics_feedback(
        self, api_statistics: "SmartOptimizer.ApiStatistics"
    ):
        self.shared_model_feedback.total_tokens += api_statistics.total_tokens
        self.shared_model_feedback.total_time += api_statistics.total_time
        self.shared_model_feedback.api_calls += api_statistics.api_calls

    def __update_api_statistics_inference(
        self, api_statistics: "SmartOptimizer.ApiStatistics"
    ):
        self.shared_model_inference.total_tokens += api_statistics.total_tokens
        self.shared_model_inference.total_time += api_statistics.total_time
        self.shared_model_inference.api_calls += api_statistics.api_calls
