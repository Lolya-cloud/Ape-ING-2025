from pydantic import BaseModel
from typing import ClassVar
from standalone.data_structures.prompt import Prompt
from standalone.generation_model import GenerationModel
from standalone.data_structures.prompt_result import (
    TrainingStatistics,
)
from standalone.evaluator import Evaluator
from standalone.data_structures.data_handling import OutputData, InputData
import re
import random
from typing import List, Callable, Tuple, Optional, Any
import concurrent.futures
from tqdm import tqdm
import logging
from logging import Logger
import numpy as np
import traceback

from standalone.data_structures.prompt_config import (
    GRADIENT_DESCENT_PROMPT,
    INCORPORATING_GRADIENT_FEEDBACK_PROMPT,
    PARAPHRASE_PROMPT,
)


class ProTeGi(BaseModel):
    model_name_feedback: str
    model_name_inference: str
    response_schema: Any
    shared_model_feedback: "GenerationModel" = None
    shared_model_inference: "GenerationModel" = None
    evaluator: "Evaluator" = None
    # set config for number of repeat calls in case of unexpected generational model behaviour.
    retries_if_error: ClassVar[int] = 5
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
            filename="protegi_info.log",
            format="%(asctime)s %(message)s",
            filemode="w",
            level=logging.CUSTOM,
        )
        self.logger = logging.getLogger()
        self.logger.setLevel("CUSTOM")

    class Config:
        arbitrary_types_allowed = True

    class ProtegiPromptResultWrapper(BaseModel):
        avg_score: float
        incorrect_examples: List[OutputData]
        correct_examples: List[OutputData]
        prompt: Prompt

    def __evaluate_multiple_prompts_bandit(
        self,
        prompts: List[Prompt],
        dataset_batch: List[InputData],
        metric_fn: Callable,
        num_threads: int,
    ) -> List[ProtegiPromptResultWrapper]:
        # handle the case where more prompts than threads.
        num_threads_dedicated_to_prompts = min(len(prompts), num_threads)
        num_document_threads = num_threads // num_threads_dedicated_to_prompts

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads_dedicated_to_prompts
        ) as executor:
            future_to_evaluation = {
                executor.submit(
                    self.evaluator.evaluate_prompt,
                    current_prompt,
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
                    ProTeGi.ProtegiPromptResultWrapper(
                        avg_score=avg_score,
                        incorrect_examples=incorrect_examles,
                        correct_examples=correct_examples,
                        prompt=prompt,
                    )
                )
                self.__update_api_statistics_inference(api_statistics=api_statistics)
        return result

    def __extract_answer(self, response: str) -> str:
        # relaxing regex requirement to deal with flash shenanigans.
        response_extract_pattern = r"<START>(.*?)<END>"
        patterns = [
            response_extract_pattern,
            r"<START>(.*?)</END>",
            r"<START>(.*?)<\\END>",
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

    def __generate_gradients(
        self,
        current_prompt: Prompt,
        error_examples: List[OutputData],
        num_gradients: int,
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
        error_str, docs = self.__make_error_string_to_work_with_pdfs(
            error_examples=error_examples
        )
        meta_prompt = GRADIENT_DESCENT_PROMPT.format(
            prompt_text=current_prompt.text,
            num_gradients=num_gradients,
            error_str=error_str,
        )
        gradient_text = self.shared_model_feedback.query_llm(
            prompt_text=meta_prompt,
            doc=docs,
            config=self.shared_model_feedback.default_diversity_config,
        )
        gradients = self.__extract_answer(
            response=gradient_text,
        )
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
            f"Generating gradients: \nPrompt: {current_prompt.text}. \nGradients: "
        )
        for i, gradient in enumerate(gradients):
            logger_message += f"\nGradient {i + 1}: {gradient}"
        logger_message += "\nGenerated from incorrect examples: "
        for elem in log_list:
            logger_message += f"\n{elem}."
        self.logger.custom(logger_message)
        # logging block end
        return gradients

    def __apply_gradients_to_prompt(
        self,
        current_prompt: Prompt,
        gradients: List,
        steps_per_gradient: int,
        error_examples: List[OutputData],
    ) -> List[Prompt]:
        """
        Apply the gradient to the current prompt and produce
        candidate improved prompts.

        :param current_prompt: The current prompt to be used for edits.
        :param gradients: A list of textual gradients.
        :param steps_per_gradient: Number of different prompt edits to produce for each gradient.
        :param error_examples: a list of dictionaries showing incorrect examples with ground truth and predictions.
        :return: List of new prompts derived from the current_prompt using gradients.
        """
        error_str, docs = self.__make_error_string_to_work_with_pdfs(
            error_examples=error_examples
        )
        prompt_candidates = []
        self.logger.custom("-----APPLYING GRADIENTS-----")
        for gradient in gradients:
            meta_prompt = INCORPORATING_GRADIENT_FEEDBACK_PROMPT.format(
                prompt_text=current_prompt.text,
                error_str=error_str,
                gradient=gradient,
                steps_per_gradient=steps_per_gradient,
            )
            prompt_candidates_raw = self.shared_model_feedback.query_llm(
                prompt_text=meta_prompt,
                doc=docs,
                config=self.shared_model_feedback.default_diversity_config,
            )
            extracted = self.__extract_answer(
                response=prompt_candidates_raw,
            )
            # logger block start
            logger_message = f"Applying gradient: \nParent prompt: {current_prompt.text}. \nGradient: {gradient}. \nNew prompts: "
            for elem in extracted:
                logger_message += f"\n{elem}."
            self.logger.custom(logger_message)
            # logger block end
            prompt_candidates.extend(extracted)
        new_prompts = []
        for candidate in prompt_candidates:
            new_prompts.append(Prompt(text=candidate, parent_prompts=[current_prompt]))
        return new_prompts

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

    def __generate_local_variations(
        self, prompt_candidates: List[Prompt], num_variations: int
    ) -> List[Prompt]:
        """
        Generate local paraphrased or varied versions of each prompt candidate.

        :param prompt_candidates: List of candidate prompts to expand from.
        :param num_variations: Number of variations per prompt candidate.
        :return: List of variations of len(num_variations * len(prompt_candidates)). Note that original prompts are not included.
        """
        if num_variations <= 0:
            return []
        varied_candidates = []
        for candidate in prompt_candidates:
            meta_prompt = PARAPHRASE_PROMPT.format(
                prompt_text=candidate.text, num_paraphrases=num_variations
            )
            paraphrased_prompts = self.shared_model_feedback.query_llm(
                prompt_text=meta_prompt,
                config=self.shared_model_feedback.default_diversity_config,
            )
            paraphrased_prompts = self.__extract_answer(paraphrased_prompts)
            for text_candidate in paraphrased_prompts:
                varied_candidates.append(
                    Prompt(text=text_candidate, parent_prompts=[candidate])
                )
        return varied_candidates

    def __expand_candidates(
        self,
        current_prompt: Prompt,
        training_batch: List[InputData],
        metric_fn: Callable,
        batch_size: int,
        num_gradients: int,
        steps_per_gradient: int,
        num_variations: int,
        max_num_examples: int,
        num_threads: int,
    ) -> List[Prompt]:
        """
        This is the "Expansion" step. We:
        1) Evaluate current_prompt on a mini-batch, gather error examples.
        2) Generate natural language gradients.
        3) Apply each gradient to produce new candidate prompts.
        4) Locally vary each new candidate with Monte Carlo paraphrasing.

        :param current_prompt: Prompt on the beam we want to expand.
        :param training_batch: Larger set of training data from which we draw errors.
        :param metric_fn: Function to measure performance.
        :param batch_size: How many items to sample to find errors.
        :param num_gradients: How many gradients to generate per prompt.
        :param steps_per_gradient: How many improved prompts to produce for each gradient.
        :param num_variations: How many paraphrased variations per newly edited prompt.
        :param max_num_examples: Maximum number of incorrect examples to include in the learning.
        If len(error_examples) > max_num_examples, some docs will be dropped.
        :param num_threads: Number of threads to use.
        :return: A list of new prompt candidates. The original prompt is not included. Note: if no errors are
        found in training_batch using batch_size, an empty list will be returned. To fix this, increase the batch size.
        """
        self.logger.custom("-------Attempting beam expansion-------")
        # 1) Evaluate prompt on a random mini-batch to find errors
        sampled_batch = random.sample(training_batch, batch_size)

        # Identify which examples were predicted incorrectly
        _, incorrect_examples, _, api_statistics = self.evaluator.evaluate_prompt(
            current_prompt=current_prompt,
            dataset_batch=sampled_batch,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        self.__update_api_statistics_inference(api_statistics=api_statistics)
        resampling_retries = 0
        while not incorrect_examples and resampling_retries <= 5:
            log_message = "No incorrect examples in the batch sample, resampling and reevaluating."
            print(log_message)
            self.logger.custom(log_message)
            sampled_batch = random.sample(training_batch, batch_size)

            # Identify which examples were predicted incorrectly
            _, incorrect_examples, _, api_statistics = self.evaluator.evaluate_prompt(
                current_prompt=current_prompt,
                dataset_batch=sampled_batch,
                metric_fn=metric_fn,
                num_threads=num_threads,
            )
            self.__update_api_statistics_inference(api_statistics=api_statistics)
            resampling_retries += 1
        # If no incorrect examples, exit quickly
        if not incorrect_examples:
            return []

        # If too many incorrect examples, trim the list. This is an edge case, but could happen with a really big batch size.
        # This is done to not breach the context size later down the line for gradient generation and application.
        if len(incorrect_examples) > max_num_examples:
            incorrect_examples = incorrect_examples[:max_num_examples]
        # 2) Generate textual gradients explaining the mistakes
        for i in range(self.retries_if_error):
            try:
                gradients = self.__generate_gradients(
                    current_prompt=current_prompt,
                    error_examples=incorrect_examples,
                    num_gradients=num_gradients,
                )
                # 3) Apply each gradient to get new prompt candidates
                new_candidates = self.__apply_gradients_to_prompt(
                    current_prompt=current_prompt,
                    gradients=gradients,
                    steps_per_gradient=steps_per_gradient,
                    error_examples=incorrect_examples,
                )
                # 4) Local Monte Carlo variations
                varied_prompts = self.__generate_local_variations(
                    prompt_candidates=new_candidates, num_variations=num_variations
                )
                break
            except Exception:
                self.logger.custom("Exception, couldn't generate or apply gradients.")
        else:
            raise Exception(
                "Couldn't expand, probably regex failed because flash is a pile of rubbish."
            )

        # Return combined set: direct edits + variations
        return new_candidates + varied_prompts

    def __bandit_selection_v2(
        self,
        prompt_candidates: List[Prompt],
        training_data: List[InputData],
        metric_fn: Callable,
        budget: int,
        top_k: int,
        c: float,
        sample_size: int,
        num_threads: int,
        num_prompts_per_round: int = 1,
    ) -> List[Prompt]:
        # logger block start
        self.logger.custom("-----STARTING UCB BANDIT SELECTION-----")
        logger_message = f"""Budget: {budget}. Sample size: {sample_size}. Prompts per round: {num_prompts_per_round}. C: {c} \nPrompts entering survival: """
        for i, elem in enumerate(prompt_candidates):
            logger_message += f"\nPrompt {i + 1}: {elem.text}"
        self.logger.custom(logger_message)
        # logger block end
        # exit quickly if less prompts than the size of the beam
        if top_k >= len(prompt_candidates):
            self.logger.custom(
                "Beam size enough to fit all the prompts, skipping UCB evaluation."
            )
            return prompt_candidates
        # here they seem to cound number of times the prompt was selected instead of the number of evaluated data points for the prompt.
        # so they consider one exploration/exploitation as a test of the prompt on the subsample of the data.
        counts = np.zeros(len(prompt_candidates))
        scores = np.zeros(len(prompt_candidates))
        num_prompts_per_round = min(num_prompts_per_round, len(prompt_candidates))

        def get_scaled_scores(scores, counts):
            # Some counts may be 0, so we need to avoid division by 0.
            return np.divide(
                scores, counts, out=np.zeros_like(scores), where=counts != 0
            )

        def choose(num_prompts_per_round, round_index, scores, counts):
            # choose which prompts to explore based on ucb values. Returns indices of the prompt candidates list.
            if np.sum(counts) == 0:
                # If all counts are 0, choose randomly.
                return random.sample(
                    range(len(prompt_candidates)), num_prompts_per_round
                )
            scaled_scores = get_scaled_scores(scores, counts)
            ucb_counts = counts + 1e-3
            ucb_scores = scaled_scores + c * np.sqrt(np.log(round_index) / ucb_counts)

            # Choose the prompts with the highest UCB scores
            return np.argsort(ucb_scores)[::-1][:num_prompts_per_round]

        for ri in tqdm(
            range(budget),
            desc=f"Evaluating {len(prompt_candidates)} prompts in batches of {num_prompts_per_round} on {sample_size} datapoints.",
        ):
            # Sample the prompts
            sampled_prompts_idx = choose(
                num_prompts_per_round=num_prompts_per_round,
                round_index=ri,
                scores=scores,
                counts=counts,
            )
            sampled_prompts = [prompt_candidates[i] for i in sampled_prompts_idx]
            assert len(sampled_prompts) == num_prompts_per_round
            dataset_batch = random.sample(training_data, sample_size)
            assert len(dataset_batch) == sample_size
            result = self.__evaluate_multiple_prompts_bandit(
                prompts=sampled_prompts,
                dataset_batch=dataset_batch,
                metric_fn=metric_fn,
                num_threads=num_threads,
            )
            new_scores = []
            # threads can return results in a different order than we passed in, so we need to sort it to match the
            # prompts sample.
            for prompt in sampled_prompts:
                result_for_prompt = [
                    res.avg_score for res in result if res.prompt == prompt
                ][0]
                new_scores.append(result_for_prompt)
            for i, score in zip(sampled_prompts_idx, new_scores):
                counts[i] += sample_size
                scores[i] += score * sample_size
        [scores, candidates] = list(
            zip(
                *sorted(
                    list(
                        zip(
                            get_scaled_scores(scores=scores, counts=counts),
                            prompt_candidates,
                        )
                    ),
                    reverse=True,
                    key=lambda x: x[0],
                )
            )
        )
        best_k_prompts = list(candidates)[:top_k]
        # logger block start
        self.logger.custom("-----UCB BANDIT SELECTION COMPLETE-----")
        diagnostics = {}
        for prompt, score in zip(candidates, scores):
            diagnostics[prompt.text] = {
                "examples": counts[prompt_candidates.index(prompt)],
                "scaled_reward": score,
                "rank": candidates.index(prompt) + 1,
            }
        logger_message = "Bandit selection complete. Diagnostics per prompt: "
        for key, val in diagnostics.items():
            logger_message += f"\n Prompt: {key}. Result: {val}."

        self.logger.custom(logger_message)
        logger_message = "Best prompts: "
        for elem in best_k_prompts:
            logger_message += f"\n {elem.text}"
        self.logger.custom(logger_message)
        # logger block end
        return best_k_prompts

    def prompt_optimization_with_beam_search(
        self,
        initial_prompt: Prompt,
        training_data: List[InputData],
        beam_width: int,
        num_iterations: int,
        bandit_budget: int,
        batch_size_for_errors: int,
        max_num_examples: int,
        num_gradients: int,
        steps_per_gradient: int,
        num_variations: int,
        num_threads: int,
        c: float,
        bandit_sample_size: int,
        num_prompts_per_round: int,
        beam_save: Optional[List[Prompt]] = None,
        metric_fn: Callable = None,
    ) -> Tuple[Prompt, TrainingStatistics]:
        """
        Main driver for the entire discrete prompt optimization procedure.
        1) Maintain a beam (list of n best prompts).
        2) For each prompt in the beam, we "expand" it to produce new candidates.
        3) We gather all new candidates and run bandit_selection to pick the best.
        4) Repeat for num_iterations.
        5) Return the single best final prompt from the beam.

        :param initial_prompt: Starting prompt we want to improve.
        :param training_data: Full training data used to refine the prompt.
        :param beam_width: How many best prompt candidates to keep at each iteration.
        :param num_iterations: How many times to iterate expanding+selecting.
        :param bandit_budget: Total number of iterations of the bandit selection algorithm.
        :param batch_size_for_errors: Training batch size, this many data points will be sampled looking for errors.
        Please note that this does not guarantee presence of errors in the sample.
        :param max_num_examples: Maximum number of error_examples to use for gradient generation. Please note that in case of bigger documents,
        more than one can cause model instability (llm model stops giving proper instructions, etc.)
        If the number of errors in the sampled data point batch is greater, it will be trimmed to match the parameter.
        :param num_gradients: Number of textual gradients to generate per prompt in the beam for an error if the error is found.
        :param steps_per_gradient: Number of new prompts produced per gradient.
        :param num_variations: Number of local paraphrases to generate per each new candidate prompt.
        :param num_threads: Number of threads to use for training. Used as is for initial evaluation, but for bandit algorithm split into
        two parameters. A pool of threads for the prompts in num_prompts_per_round (ideally each prompt has at least 1 thread),
        and a pool of evaluation threads spawned per prompt thread (evaluating the same prompt, different data points.)
        :param c: a hyperparameter for bandit selection to balance exploration vs exploitation. Higher values - more exploration, less
        paying less attention to already good prompts.
        :param bandit_sample_size: number of samples used during each bandit evaluation.
        :param num_promps_per_round: number of prompts evaluated simulateneously per single bandit iteration.
        :return: The best final prompt from the beam and corresponding training statistics data.
        """
        if metric_fn is None:
            metric_fn = self.__metric_fn
        beams = []
        if beam_save:
            print("Starting from a save")
            beam = beam_save
        else:
            print("Starting from scratch")
            beam = [initial_prompt]
        beams.append(beam)
        self.logger.custom(
            "----------------------STARTING PROTEGI----------------------"
        )
        self.logger.custom(f"Starting beam: {beam}")
        for iteration in tqdm(
            range(num_iterations),
            desc="Optimizing prompts for number of iterations specified.",
        ):
            self.logger.custom(f"----------PROTEGI iteration {iteration + 1}----------")
            # 1) Expand each prompt in the beam
            for _ in range(self.retries_if_error):
                try:
                    new_candidates = []
                    for candidate_prompt in beam:
                        expanded = self.__expand_candidates(
                            current_prompt=candidate_prompt,
                            training_batch=training_data,
                            metric_fn=metric_fn,
                            batch_size=batch_size_for_errors,
                            num_gradients=num_gradients,
                            steps_per_gradient=steps_per_gradient,
                            num_variations=num_variations,
                            max_num_examples=max_num_examples,
                            num_threads=num_threads,
                        )
                        new_candidates.extend(expanded)
                    break
                except Exception:
                    self.logger.custom("Something broke, see below. Re-trying.")
                    self.logger.custom(traceback.format_exc())
            else:
                raise Exception("Everything broke.")

            # Include old beam to allow existing candidates to remain
            updated_prompts = []
            for prompt in beam:
                updated_prompts.append(
                    Prompt(text=prompt.text, parent_prompts=[prompt])
                )
            beam = updated_prompts
            all_candidates = beam + new_candidates
            # 2) Use bandit selection to pick the top beam_width
            top_candidates = self.__bandit_selection_v2(
                prompt_candidates=all_candidates,
                training_data=training_data,
                metric_fn=metric_fn,
                budget=bandit_budget,
                top_k=beam_width,
                c=c,
                sample_size=bandit_sample_size,
                num_threads=num_threads,
                num_prompts_per_round=num_prompts_per_round,
            )
            # 3) Update the beam
            beam = top_candidates
            self.logger.custom(f"New beam: {beam}")
            beams.append(beam)
        print("Running final evaluation of the best prompt candidates.")
        self.logger.custom(
            "-----RUNNING FINAL EVALUATION OF THE BEAM USING THE ENTIRE DATASET-----"
        )
        result = self.__evaluate_multiple_prompts_bandit(
            prompts=beam,
            dataset_batch=training_data,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        new_scores = []
        # threads can return results in a different order than we passed in, so we need to sort it to match the
        # prompts sample.
        for prompt in beam:
            result_for_prompt = [
                res.avg_score for res in result if res.prompt == prompt
            ][0]
            new_scores.append(result_for_prompt)
        # pick the best prompt
        best_prompt_idx = new_scores.index(max(new_scores))
        best_prompt = beam[best_prompt_idx]
        training_statistics = TrainingStatistics.factory(
            inference_model=self.shared_model_inference,
            feedback_model=self.shared_model_feedback,
            final_beam_prompts=beam,
            final_beam_scores=new_scores,
            beams=beams,
        )
        self.logger.custom(
            "----------------------PROTEGI COMPLETE----------------------"
        )
        self.logger.custom(f"Best prompt: {best_prompt}")
        self.logger.custom(f"Training statistics: {training_statistics}")
        return best_prompt, training_statistics

    def __update_api_statistics_inference(
        self, api_statistics: "Evaluator.ApiStatistics"
    ):
        self.shared_model_inference.total_tokens += api_statistics.total_tokens
        self.shared_model_inference.total_time += api_statistics.total_time
        self.shared_model_inference.api_calls += api_statistics.api_calls

    def prompt_optimization_with_beam_search_and_grad_verification(
        self,
        initial_prompt: Prompt,
        training_data: List[InputData],
        beam_width: int,
        num_iterations: int,
        bandit_budget: int,
        batch_size_for_errors: int,
        num_gradients: int,
        num_threads: int,
        c: float,
        bandit_sample_size: int,
        num_prompts_per_round: int,
        beam_save: Optional[List[Prompt]] = None,
        metric_fn: Callable = None,
        num_doc_retries: int = 5,
        threshold_score: float = 1,
        num_doc_copies: int = 1,
    ) -> Tuple[Prompt, TrainingStatistics]:
        if metric_fn is None:
            metric_fn = self.__metric_fn
        beams = []
        if beam_save:
            print("Starting from a save")
            beam = beam_save
        else:
            print("Starting from scratch")
            beam = [initial_prompt]
        beams.append(beam)
        self.logger.custom(
            "----------------------STARTING PROTEGI----------------------"
        )
        self.logger.custom(f"Starting beam: {beam}")
        for iteration in tqdm(
            range(num_iterations),
            desc="Optimizing prompts for number of iterations specified.",
        ):
            self.logger.custom(f"----------PROTEGI iteration {iteration + 1}----------")
            # 1) Expand each prompt in the beam
            for _ in range(self.retries_if_error):
                try:
                    new_candidates = []
                    for candidate_prompt in beam:
                        expanded = self.__expand_candidates_with_grad_verification(
                            current_prompt=candidate_prompt,
                            training_batch=training_data,
                            metric_fn=metric_fn,
                            batch_size=batch_size_for_errors,
                            num_gradients=num_gradients,
                            max_num_examples=1,
                            num_threads=num_threads,
                            num_doc_retries=num_doc_retries,
                            threshold_score=threshold_score,
                            num_doc_copies=num_doc_copies,
                        )
                        new_candidates.extend(expanded)
                    break
                except Exception:
                    self.logger.custom("Something broke, see below. Re-trying.")
                    self.logger.custom(traceback.format_exc())
            else:
                raise Exception("Everything broke.")

            # Include old beam to allow existing candidates to remain
            updated_prompts = []
            for prompt in beam:
                updated_prompts.append(
                    Prompt(text=prompt.text, parent_prompts=[prompt])
                )
            beam = updated_prompts
            all_candidates = beam + new_candidates
            # 2) Use bandit selection to pick the top beam_width
            top_candidates = self.__bandit_selection_v2(
                prompt_candidates=all_candidates,
                training_data=training_data,
                metric_fn=metric_fn,
                budget=bandit_budget,
                top_k=beam_width,
                c=c,
                sample_size=bandit_sample_size,
                num_threads=num_threads,
                num_prompts_per_round=num_prompts_per_round,
            )
            # 3) Update the beam
            beam = top_candidates
            self.logger.custom(f"New beam: {beam}")
            beams.append(beam)
        print("Running final evaluation of the best prompt candidates.")
        self.logger.custom(
            "-----RUNNING FINAL EVALUATION OF THE BEAM USING THE ENTIRE DATASET-----"
        )
        result = self.__evaluate_multiple_prompts_bandit(
            prompts=beam,
            dataset_batch=training_data,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        new_scores = []
        # threads can return results in a different order than we passed in, so we need to sort it to match the
        # prompts sample.
        for prompt in beam:
            result_for_prompt = [
                res.avg_score for res in result if res.prompt == prompt
            ][0]
            new_scores.append(result_for_prompt)
        # pick the best prompt
        best_prompt_idx = new_scores.index(max(new_scores))
        best_prompt = beam[best_prompt_idx]
        training_statistics = TrainingStatistics.factory(
            inference_model=self.shared_model_inference,
            feedback_model=self.shared_model_feedback,
            final_beam_prompts=beam,
            final_beam_scores=new_scores,
            beams=beams,
        )
        self.logger.custom(
            "----------------------PROTEGI COMPLETE----------------------"
        )
        self.logger.custom(f"Best prompt: {best_prompt}")
        self.logger.custom(f"Training statistics: {training_statistics}")
        return best_prompt, training_statistics

    def __expand_candidates_with_grad_verification(
        self,
        current_prompt: Prompt,
        training_batch: List[InputData],
        metric_fn: Callable,
        batch_size: int,
        num_gradients: int,
        max_num_examples: int,
        num_threads: int,
        num_doc_retries: int,
        threshold_score: float,
        num_doc_copies: int,
    ) -> List[Prompt]:
        self.logger.custom(
            "-------Attempting beam expansion with grad verification-------"
        )
        # 1) Evaluate prompt on a random mini-batch to find errors
        sampled_batch = random.sample(training_batch, batch_size)

        # Identify which examples were predicted incorrectly
        _, incorrect_examples, _, api_statistics = self.evaluator.evaluate_prompt(
            current_prompt=current_prompt,
            dataset_batch=sampled_batch,
            metric_fn=metric_fn,
            num_threads=num_threads,
        )
        self.__update_api_statistics_inference(api_statistics=api_statistics)
        resampling_retries = 0
        while not incorrect_examples and resampling_retries <= 5:
            log_message = "No incorrect examples in the batch sample, resampling and reevaluating."
            print(log_message)
            self.logger.custom(log_message)
            sampled_batch = random.sample(training_batch, batch_size)

            # Identify which examples were predicted incorrectly
            _, incorrect_examples, _, api_statistics = self.evaluator.evaluate_prompt(
                current_prompt=current_prompt,
                dataset_batch=sampled_batch,
                metric_fn=metric_fn,
                num_threads=num_threads,
            )
            self.__update_api_statistics_inference(api_statistics=api_statistics)
            resampling_retries += 1
        # If no incorrect examples, exit quickly
        if not incorrect_examples:
            return []

        # If too many incorrect examples, trim the list. This is an edge case, but could happen with a really big batch size.
        # This is done to not breach the context size later down the line for gradient generation and application.
        if len(incorrect_examples) > max_num_examples:
            incorrect_examples = incorrect_examples[:max_num_examples]
        # 2) Generate textual gradients explaining the mistakes
        for i in range(self.retries_if_error):
            try:
                new_candidates = self.__generate_grad_with_verification(
                    current_prompt=current_prompt,
                    incorrect_examples=incorrect_examples,
                    training_batch=training_batch,
                    num_gradients=num_gradients,
                    metric_fn=metric_fn,
                    num_threads=num_threads,
                    num_doc_retries=num_doc_retries,
                    threshold_score=threshold_score,
                    num_doc_copies=num_doc_copies,
                )
                break
            except Exception:
                self.logger.custom("Exception, couldn't generate or apply gradients.")
        else:
            raise Exception(
                "Couldn't expand, probably regex failed because flash is a pile of rubbish."
            )

        # Return combined set: direct edits + variations
        return new_candidates

    def __generate_grad_with_verification(
        self,
        current_prompt,
        training_batch,
        incorrect_examples,
        num_gradients,
        metric_fn,
        num_threads,
        num_doc_retries,
        threshold_score,
        num_doc_copies,
    ):
        assert num_doc_copies >= 1
        assert num_doc_retries >= 1
        assert len(incorrect_examples) == 1
        for _ in range(num_doc_retries):
            gradients = self.__generate_gradients(
                current_prompt=current_prompt,
                error_examples=incorrect_examples,
                num_gradients=num_gradients,
            )
            new_candidates = self.__apply_gradients_to_prompt(
                current_prompt=current_prompt,
                gradients=gradients,
                steps_per_gradient=1,
                error_examples=incorrect_examples,
            )
            incorrect_doc_inputs = self.__get_input_data_for_output_data(
                output_data=incorrect_examples, all_input_data=training_batch
            )
            assert len(incorrect_doc_inputs) == 1
            result = self.__evaluate_multiple_prompts_bandit(
                prompts=new_candidates,
                dataset_batch=incorrect_doc_inputs * num_doc_copies,
                metric_fn=metric_fn,
                num_threads=num_threads,
            )
            confirmed_prompts = []
            # threads can return results in a different order than we passed in, so we need to sort it to match the
            # prompts sample.
            for prompt in new_candidates:
                result_for_prompt = [
                    res.avg_score for res in result if res.prompt == prompt
                ][0]
                if result_for_prompt >= threshold_score:
                    confirmed_prompts.append(prompt)
            if len(confirmed_prompts) > 0:
                return confirmed_prompts
        else:
            return []

    def __get_input_data_for_output_data(
        self, output_data: List[OutputData], all_input_data: List[InputData]
    ) -> List[InputData]:
        input_docs = []
        for doc in output_data:
            input_docs.append(
                [elem for elem in all_input_data if elem.doc_id == doc.doc_id][0]
            )
        return input_docs
