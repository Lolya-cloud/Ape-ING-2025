import numpy as np
import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Any
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest, wilcoxon
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from standalone.data_structures.prompt import Prompt
from standalone.evaluator import Evaluator
from standalone.data_structures.data_handling import InputData
from standalone.data_structures.prompt_result import PromptResultLite, PromptResult


class AnalysisPipeline(BaseModel):
    config: Any
    training_dataset: List[InputData]
    validation_dataset: List[InputData]
    test_dataset: List[InputData]
    num_threads: int
    model_name_inference: str
    evaluator: "Evaluator" = None
    best_prompt: "Prompt" = None
    results_folder: str = None

    def __init__(self, **data):
        super().__init__(**data)
        if not os.path.exists(self.config["results_path"]):
            raise FileNotFoundError(
                "The results directory doesn't exist. Create one and save optimized prompt there."
            )
        else:
            self.results_folder = self.config["results_path"]
        self.evaluator = Evaluator(
            model_name_inference=self.model_name_inference,
            response_schema=self.config["schema"],
        )
        self.best_prompt = self.__load_optimized_prompt()

    def run_all_inferences(self):
        """If you don't care and want to run the inferences for the entire pipeline at once."""
        self.run_basic_inferences()
        self.run_inference_for_parents()
        self.run_inference_persistence()

    def run_basic_inferences(
        self,
        starting_prompt_eval_exists=False,
        optimized_prompt_eval_exists=False,
        manual_prompt_eval_exists=False,
    ):
        """Perform inference to get accuracies and other data for analysis without the parent prompts. If some of the prompts have already been
        evaluated and have result files existing, you can use the tag to skip evaluation of that part."""
        if not starting_prompt_eval_exists:
            self.__run_inference_starting_prompt()
        if not optimized_prompt_eval_exists:
            self.__run_inference_optimized_prompt()
        if not manual_prompt_eval_exists:
            self.__run_inference_manual_prompt()

    def run_inference_for_parents(self):
        self.__run_inference_optimized_prompt_parents()

    def run_inference_persistence(self):
        dir_starting, dir_optimized, dir_manual = (
            self.__get_persistence_dirs_create_if_not_exist()
        )
        for i in tqdm(range(10)):
            file_name = f"{i + 1}.json"
            starting_res = self.__run_inference_train_test(
                prompt=self.config["starting_prompt"]
            )
            starting_res.save_to_json(os.path.join(dir_starting, file_name))
            optimized_res = self.__run_inference_train_test(prompt=self.best_prompt)
            optimized_res.save_to_json(os.path.join(dir_optimized, file_name))
            manual_res = self.__run_inference_train_test(
                prompt=self.config["manual_prompt"]
            )
            manual_res.save_to_json(os.path.join(dir_manual, file_name))

    def idgaf_analyze_everything(self):
        # basic analysis
        df = self.analyze_starting_vs_optimized_vs_manual_prompt()
        print("Prompts performance and hypotheses testing: ")
        print(df.to_string(index=False))
        # parents plot
        # self.save_plot_parent_accuracy()

        # persistence in eval and response
        self.analyze_persistence_in_response_and_eval()
        # variability across 10 samples
        self.analyze_proportions_variability_accross_samples()
        # uncertainty in eval plot
        self.save_plot_eval_certainty()
        self.analyze_doc_variability_accross_samples()

    def analyze_starting_vs_optimized_vs_manual_prompt_print_paper_results(self):
        df = self.analyze_starting_vs_optimized_vs_manual_prompt()
        df = df[df["Dataset"] != "Validation"]
        df = df.drop(columns=["H2 p-1", "H1 p-2"])
        print(f"Prompts performance and hypotheses testing for {self.config["name"]}: ")
        print(df.to_string(index=False))

    def analyze_starting_vs_optimized_vs_manual_prompt(self):
        """Analysis pipeline for H1. Assumes the files exist, does not perform inference"""
        prompt_results_agg = self.__load_prompt_results()
        starting_prompt_evaluations = self.__get_prompt_evaluations(
            prompt_results_agg.starting_prompt_tr, prompt_results_agg.starting_prompt_t
        )
        best_prompt_evaluations = self.__get_prompt_evaluations(
            prompt_results_agg.best_prompt_tr, prompt_results_agg.best_prompt_t
        )
        manual_prompt_evaluations = self.__get_prompt_evaluations(
            prompt_results_agg.manual_prompt_tr, prompt_results_agg.manual_prompt_t
        )
        # accuracy (formally - proportion)
        starting_prompt_accuracy = self.__get_accuracy(starting_prompt_evaluations)
        best_prompt_accuracy = self.__get_accuracy(best_prompt_evaluations)
        manual_prompt_accuracy = self.__get_accuracy(manual_prompt_evaluations)
        # confidence intervals using wilson scores
        ci_starting_train, ci_starting_val, ci_starting_test = (
            self.__get_confidence_interval(starting_prompt_accuracy)
        )
        ci_best_train, ci_best_val, ci_best_test = self.__get_confidence_interval(
            best_prompt_accuracy
        )
        ci_manual_train, ci_manual_val, ci_manual_test = self.__get_confidence_interval(
            manual_prompt_accuracy
        )
        # starting prompt
        baseline_tr = np.array(self.__get_scores(starting_prompt_evaluations, "Train"))
        baseline_t = np.array(self.__get_scores(starting_prompt_evaluations, "Test"))
        baseline_v = np.array(
            self.__get_scores(starting_prompt_evaluations, "Validation")
        )
        # optimized prompt
        new_tr = np.array(self.__get_scores(best_prompt_evaluations, "Train"))
        new_t = np.array(self.__get_scores(best_prompt_evaluations, "Test"))
        new_v = np.array(self.__get_scores(best_prompt_evaluations, "Validation"))
        # manual prompt
        manual_tr = np.array(self.__get_scores(manual_prompt_evaluations, "Train"))
        manual_t = np.array(self.__get_scores(manual_prompt_evaluations, "Test"))
        manual_v = np.array(self.__get_scores(manual_prompt_evaluations, "Validation"))
        # optimized vs starting
        train_tests_h1 = self.__binomial_test(new=new_tr, baseline=baseline_tr)
        val_tests_h1 = self.__binomial_test(new=new_v, baseline=baseline_v)
        test_tests_h1 = self.__binomial_test(new=new_t, baseline=baseline_t)
        # optimized vs manual
        train_tests_h2 = self.__binomial_test(new=new_tr, baseline=manual_tr)
        val_tests_h2 = self.__binomial_test(new=new_v, baseline=manual_v)
        test_tests_h2 = self.__binomial_test(new=new_t, baseline=manual_t)

        display_df = pd.DataFrame(
            {
                "Dataset": ["Train", "Validation", "Test"],
                "Starting acc": [
                    starting_prompt_accuracy["Train"],
                    starting_prompt_accuracy["Validation"],
                    starting_prompt_accuracy["Test"],
                ],
                "S 95%-CI": [ci_starting_train, ci_starting_val, ci_starting_test],
                "Optimized acc": [
                    best_prompt_accuracy["Train"],
                    best_prompt_accuracy["Validation"],
                    best_prompt_accuracy["Test"],
                ],
                "O 95%-CI": [ci_best_train, ci_best_val, ci_best_test],
                "H1 p-2": [train_tests_h1[2], val_tests_h1[2], test_tests_h1[2]],
                "H1 p-1": [train_tests_h1[3], val_tests_h1[3], test_tests_h1[3]],
                "H1 n01/n10": [
                    f"{train_tests_h1[0]}/{train_tests_h1[1]}",
                    f"{val_tests_h1[0]}/{val_tests_h1[1]}",
                    f"{test_tests_h1[0]}/{test_tests_h1[1]}",
                ],
                "Manual acc": [
                    manual_prompt_accuracy["Train"],
                    manual_prompt_accuracy["Validation"],
                    manual_prompt_accuracy["Test"],
                ],
                "M 95%-CI": [ci_manual_train, ci_manual_val, ci_manual_test],
                "H2 p-2": [train_tests_h2[2], val_tests_h2[2], test_tests_h2[2]],
                "H2 p-1": [train_tests_h2[3], val_tests_h2[3], test_tests_h2[3]],
                "H2 n01/n10": [
                    f"{train_tests_h2[0]}/{train_tests_h2[1]}",
                    f"{val_tests_h2[0]}/{val_tests_h2[1]}",
                    f"{test_tests_h2[0]}/{test_tests_h2[1]}",
                ],
            },
        )
        # round accuracies for display purposes
        display_df["Starting acc"] = display_df["Starting acc"].apply(
            lambda x: round(x, 2)
        )
        display_df["Optimized acc"] = display_df["Optimized acc"].apply(
            lambda x: round(x, 2)
        )
        display_df["Manual acc"] = display_df["Manual acc"].apply(lambda x: round(x, 2))

        return display_df

    def idgaf_analyse_all_persistence(self):
        print("Doc variability across samples")
        self.analyze_doc_variability_accross_samples()
        print("Accuracy across samples")
        self.analyze_proportions_variability_accross_samples()
        print("Evaluation and response persistence")
        self.analyze_persistence_in_response_and_eval()
        print("Plots")
        self.save_plot_eval_certainty()

    def save_plot_parent_accuracy(self):
        """Loads, calculates and saves a plot of parent accuracies with 95% CI based on wilson score."""
        training_accuracies, _, test_accuracies = self.__load_parents()
        # Calculate confidence intervals
        train_intervals = [
            self.__wilson_score_interval(p, len(self.training_dataset))
            for p in training_accuracies
        ]
        test_intervals = [
            self.__wilson_score_interval(p, len(self.test_dataset))
            for p in test_accuracies
        ]

        # Extract lower and upper bounds
        train_lower, train_upper = zip(*train_intervals)
        test_lower, test_upper = zip(*test_intervals)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            range(len(training_accuracies)),
            training_accuracies,
            yerr=[
                np.array(training_accuracies) - np.array(train_lower),
                np.array(train_upper) - np.array(training_accuracies),
            ],
            label="Training Accuracy",
            marker="o",
            capsize=5,
        )
        plt.errorbar(
            range(len(test_accuracies)),
            test_accuracies,
            yerr=[
                np.array(test_accuracies) - np.array(test_lower),
                np.array(test_upper) - np.array(test_accuracies),
            ],
            label="Test Accuracy",
            marker="o",
            capsize=5,
        )

        plt.xlabel("Best prompt evolution")
        plt.ylabel("Proportion of correct docs")
        plt.title(
            "Training and Test Accuracies Over Prompt Lifespan with 95% Confidence Intervals"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, "prompt_evolution.png"))

    def analyze_proportions_variability_accross_samples(self):
        results_df_starting, results_df_optimized, results_df_manual = (
            self.__load_persistence_variability_results()
        )
        starting_tr_accs, starting_t_accs = (
            results_df_starting["training_accuracy"].values,
            results_df_starting["test_accuracy"].values,
        )
        optimized_tr_accs, optimized_t_accs = (
            results_df_optimized["training_accuracy"].values,
            results_df_optimized["test_accuracy"].values,
        )
        manual_tr_accs, manual_t_accs = (
            results_df_manual["training_accuracy"].values,
            results_df_manual["test_accuracy"].values,
        )
        data = {
            "Prompt": ["Start.", "Start.", "Opt.", "Opt.", "Man.", "Man."],
            "Dataset": [
                "Training",
                "Testing",
                "Training",
                "Testing",
                "Training",
                "Testing",
            ],
            "Sample mean": [
                np.mean(starting_tr_accs),
                np.mean(starting_t_accs),
                np.mean(optimized_tr_accs),
                np.mean(optimized_t_accs),
                np.mean(manual_tr_accs),
                np.mean(manual_t_accs),
            ],
            "Sample std.": np.array([
                np.std(starting_tr_accs, ddof=1),
                np.std(starting_t_accs, ddof=1),
                np.std(optimized_tr_accs, ddof=1),
                np.std(optimized_t_accs, ddof=1),
                np.std(manual_tr_accs, ddof=1),
                np.std(manual_t_accs, ddof=1),
            ]),
        }

        df = pd.DataFrame(data)
        print("Proportions variability across 10 samples.")
        print(df.to_string(index=False))

    def analyze_doc_variability_accross_samples(self):
        results_df_starting, results_df_optimized, results_df_manual = (
            self.__load_persistence_variability_results()
        )
        cert_starting_tr, cert_starting_t = self.__get_eval_changes_per_doc(
            results_df=results_df_starting
        )
        cert_optimized_tr, cert_optimized_t = self.__get_eval_changes_per_doc(
            results_df=results_df_optimized
        )
        cert_manual_tr, cert_manual_t = self.__get_eval_changes_per_doc(
            results_df=results_df_manual
        )
        tr_vars_st = []
        tr_vars_o = []
        tr_vars_m = []
        for doc in self.training_dataset:
            for cert_tuple in cert_starting_tr:
                if cert_tuple["doc_id"] == doc.doc_id:
                    doc_cert = cert_tuple["certainty"]
                    p = ((doc_cert * 5) + 5) / 10
                    doc_var = p * (1 - p)
                    tr_vars_st.append(doc_var)
            for cert_tuple in cert_optimized_tr:
                if cert_tuple["doc_id"] == doc.doc_id:
                    doc_cert = cert_tuple["certainty"]
                    p = ((doc_cert * 5) + 5) / 10
                    doc_var = p * (1 - p)
                    tr_vars_o.append(doc_var)
            for cert_tuple in cert_manual_tr:
                if cert_tuple["doc_id"] == doc.doc_id:
                    doc_cert = cert_tuple["certainty"]
                    p = ((doc_cert * 5) + 5) / 10
                    doc_var = p * (1 - p)
                    tr_vars_m.append(doc_var)
        assert len(tr_vars_m) == len(tr_vars_o)
        assert len(tr_vars_st) == len(tr_vars_o)
        assert len(tr_vars_st) == len(self.training_dataset)
        t_vars_st = []
        t_vars_o = []
        t_vars_m = []
        for doc in self.test_dataset:
            for cert_tuple in cert_starting_t:
                if cert_tuple["doc_id"] == doc.doc_id:
                    doc_cert = cert_tuple["certainty"]
                    p = ((doc_cert * 5) + 5) / 10
                    doc_var = p * (1 - p)
                    t_vars_st.append(doc_var)
            for cert_tuple in cert_optimized_t:
                if cert_tuple["doc_id"] == doc.doc_id:
                    doc_cert = cert_tuple["certainty"]
                    p = ((doc_cert * 5) + 5) / 10
                    doc_var = p * (1 - p)
                    t_vars_o.append(doc_var)
            for cert_tuple in cert_manual_t:
                if cert_tuple["doc_id"] == doc.doc_id:
                    doc_cert = cert_tuple["certainty"]
                    p = ((doc_cert * 5) + 5) / 10
                    doc_var = p * (1 - p)
                    t_vars_m.append(doc_var)
        assert len(t_vars_st) == len(t_vars_o)
        assert len(t_vars_m) == len(t_vars_o)
        assert len(t_vars_st) == len(self.test_dataset)

        # testing optimized better than starting
        tr_dif = np.array(tr_vars_st) - np.array(tr_vars_o)
        t_dif = np.array(t_vars_st) - np.array(t_vars_o)
        print(tr_dif[tr_dif != 0])
        print(t_dif[t_dif != 0])
        tr_h1_p2 = wilcoxon(x=tr_dif, alternative="two-sided").pvalue
        t_h1_p2 = wilcoxon(x=t_dif, alternative="two-sided").pvalue
        tr_h1_p1 = wilcoxon(x=tr_dif, alternative="greater").pvalue
        t_h1_p1 = wilcoxon(x=t_dif, alternative="greater").pvalue

        # testing optimized vs manual
        tr_dif = np.array(tr_vars_m) - np.array(tr_vars_o)
        t_dif = np.array(t_vars_m) - np.array(t_vars_o)
        print(tr_dif[tr_dif != 0])
        print(t_dif[t_dif != 0])
        tr_h2_p2 = wilcoxon(x=tr_dif, alternative="two-sided").pvalue
        t_h2_p2 = wilcoxon(x=t_dif, alternative="two-sided").pvalue
        tr_h2_p1 = wilcoxon(x=tr_dif, alternative="greater").pvalue
        t_h2_p1 = wilcoxon(x=t_dif, alternative="greater").pvalue

        # means of variances
        tr_vars_mu_st = np.mean(tr_vars_st)
        tr_vars_mu_o = np.mean(tr_vars_o)
        tr_vars_mu_m = np.mean(tr_vars_m)
        t_vars_mu_st = np.mean(t_vars_st)
        t_vars_mu_o = np.mean(t_vars_o)
        t_vars_mu_m = np.mean(t_vars_m)
        display_df = pd.DataFrame(
            {
                "Dataset": ["Train", "Test"],
                "Starting mean var": [tr_vars_mu_st, t_vars_mu_st],
                "S. 95%-CI": [
                    self.__calc_t_ci(tr_vars_st),
                    self.__calc_t_ci(t_vars_st),
                ],
                "Optimized mean var": [tr_vars_mu_o, t_vars_mu_o],
                "O 95%-CI": [self.__calc_t_ci(tr_vars_o), self.__calc_t_ci(t_vars_o)],
                "H1 p-2": [tr_h1_p2, t_h1_p2],
                "H1 p-1": [tr_h1_p1, t_h1_p1],
                "Manual var": [tr_vars_mu_m, t_vars_mu_m],
                "M 95%-CI": [self.__calc_t_ci(tr_vars_m), self.__calc_t_ci(t_vars_m)],
                "H2 p-2": [tr_h2_p2, t_h2_p2],
                "H2 p-1": [tr_h2_p1, t_h2_p1],
            },
        )
        df = pd.DataFrame(display_df)
        print("Proportions variability across 10 samples.")
        print(df.to_string(index=False))
        self.__plot_doc_variability_boxplot(
            tr_vars_st=tr_vars_st,
            t_vars_st=t_vars_st,
            tr_vars_o=tr_vars_o,
            t_vars_o=t_vars_o,
            tr_vars_m=tr_vars_m,
            t_vars_m=t_vars_m,
        )

    def __plot_doc_variability_boxplot(
        self, tr_vars_st, t_vars_st, tr_vars_o, t_vars_o, tr_vars_m, t_vars_m
    ):
        data = [
            tr_vars_st,
            t_vars_st,
            tr_vars_o,
            t_vars_o,
            tr_vars_m,
            t_vars_m,
        ]
        labels = [
            "Starting - Train",
            "Starting - Test",
            "Optimized - Train",
            "Optimized - Test",
            "Manual - Train",
            "Manual - Test",
        ]
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=labels, patch_artist=True)
        plt.title(
            "Variance Distributions per doc across 10 iterations by Prompt and Dataset"
        )
        plt.ylabel("Variance")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_folder, "variance_across_10it_boxplot.png")
        )

    def __calc_t_ci(self, array, confidence=0.95):
        n = len(array)
        mu = np.mean(array)
        sem = stats.sem(array)
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * sem
        return (round(mu - margin, 3), round(mu + margin, 3))

    def analyze_persistence_in_response_and_eval(self):
        results_df_starting, results_df_optimized, results_df_manual = (
            self.__load_persistence_variability_results()
        )
        df_eval_persistence = self.__get_evaluation_persistence(
            results_df_starting=results_df_starting,
            results_df_optimized=results_df_optimized,
            results_df_manual=results_df_manual,
        )
        df_response_persistence = self.__get_response_persistence()
        print("Evaluation persistence: ")
        print(df_eval_persistence.to_string(index=False))
        print("Response persistence: ")
        print(df_response_persistence.to_string(index=False))

    def save_plot_eval_certainty(self):
        results_df_starting, results_df_optimized, results_df_manual = (
            self.__load_persistence_variability_results()
        )
        cert_starting_tr, cert_starting_t = self.__get_eval_changes_per_doc(
            results_df=results_df_starting
        )
        cert_optimized_tr, cert_optimized_t = self.__get_eval_changes_per_doc(
            results_df=results_df_optimized
        )
        cert_manual_tr, cert_manual_t = self.__get_eval_changes_per_doc(
            results_df=results_df_manual
        )

        plot_data = {
            "Certainty": [elem["certainty"] for elem in cert_starting_tr]
            + [elem["certainty"] for elem in cert_starting_t]
            + [elem["certainty"] for elem in cert_optimized_tr]
            + [elem["certainty"] for elem in cert_optimized_t]
            + [elem["certainty"] for elem in cert_manual_tr]
            + [elem["certainty"] for elem in cert_manual_t],
            "Dataset": [f"Train: {len(self.training_dataset)}"] * len(cert_starting_tr)
            + [f"Test: {len(self.test_dataset)}"] * len(cert_starting_t)
            + [f"Train: {len(self.training_dataset)}"] * len(cert_optimized_tr)
            + [f"Test: {len(self.test_dataset)}"] * len(cert_optimized_t)
            + [f"Train: {len(self.training_dataset)}"] * len(cert_manual_tr)
            + [f"Test: {len(self.test_dataset)}"] * len(cert_manual_t),
            "Prompt": ["Starting"] * (len(cert_starting_tr) + len(cert_starting_t))
            + ["Optimized"] * (len(cert_optimized_tr) + len(cert_optimized_t))
            + ["Manual"] * (len(cert_manual_tr) + len(cert_manual_t)),
        }
        df = pd.DataFrame(plot_data)
        sns.set_theme(style="ticks", palette="pastel")

        # Draw a stripplot
        plt.figure(figsize=(12, 8))
        sns.stripplot(
            x="Prompt",
            y="Certainty",
            hue="Dataset",
            data=df,
            dodge=True,
            jitter=True,
        )
        sns.despine(offset=10, trim=True)
        plt.title("Estimated evaluation certainty across 3 prompts")
        plt.savefig(os.path.join(self.results_folder, "persistence_striplot.png"))

    def __get_eval_changes_per_doc(self, results_df):
        certainties_training = []
        for doc in self.training_dataset:
            correct = results_df[
                results_df["training_evaluation_correct"].apply(
                    lambda x: doc.doc_id in [elem["doc_id"] for elem in x]
                )
            ].shape[0]
            certainties_training.append({
                "doc_id": doc.doc_id,
                "certainty": abs(correct - 5) / 5,
            })
        certainties_validation = []
        for doc in self.test_dataset:
            correct = results_df[
                results_df["test_evaluation_correct"].apply(
                    lambda x: doc.doc_id in [elem["doc_id"] for elem in x]
                )
            ].shape[0]
            certainties_validation.append({
                "doc_id": doc.doc_id,
                "certainty": abs(correct - 5) / 5,
            })
        return certainties_training, certainties_validation

    def __get_persistence_dirs_create_if_not_exist(self):
        dir_persistence = os.path.join(self.results_folder, "persistence")
        if not os.path.exists(dir_persistence):
            os.makedirs(dir_persistence)
        dir_manual = os.path.join(dir_persistence, "manual_prompt")
        dir_starting = os.path.join(dir_persistence, "starting_prompt")
        dir_optimized = os.path.join(dir_persistence, "optimized_prompt")
        if not os.path.exists(dir_manual):
            os.makedirs(dir_manual)
        if not os.path.exists(dir_starting):
            os.makedirs(dir_starting)
        if not os.path.exists(dir_optimized):
            os.makedirs(dir_optimized)
        return dir_starting, dir_optimized, dir_manual

    def __run_inference_train_test(self, prompt):
        result_train_test = self.evaluator.evaluate(
            prompt=prompt,
            training_data=self.training_dataset,
            test_data=self.test_dataset,
            num_threads=self.num_threads,
        )
        return result_train_test

    def __load_persistence_variability_results(self):
        dir_starting, dir_optimized, dir_manual = (
            self.__get_persistence_dirs_create_if_not_exist()
        )
        results_starting = []
        results_optimized = []
        results_manual = []
        for i in range(10):
            file_name = f"{i + 1}.json"
            result_starting = PromptResultLite.load_from_json(
                os.path.join(dir_starting, file_name)
            ).to_dataframe()
            result_optimized = PromptResultLite.load_from_json(
                os.path.join(dir_optimized, file_name)
            ).to_dataframe()
            result_manual = PromptResultLite.load_from_json(
                os.path.join(dir_manual, file_name)
            ).to_dataframe()
            result_starting["iteration"] = f"{i + 1}"
            result_optimized["iteration"] = f"{i + 1}"
            result_manual["iteration"] = f"{i + 1}"
            results_starting.append(result_starting)
            results_optimized.append(result_optimized)
            results_manual.append(result_manual)

        results_df_starting = pd.concat(results_starting, ignore_index=True)
        results_df_optimized = pd.concat(results_optimized, ignore_index=True)
        results_df_manual = pd.concat(results_manual, ignore_index=True)

        return results_df_starting, results_df_optimized, results_df_manual

    def __load_optimized_prompt(self):
        try:
            file_path = os.path.join(
                self.results_folder, "best_prompt_no_evaluation.json"
            )
            best_prompt = Prompt.from_json(filepath=file_path)
            return best_prompt
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No optimized prompt under the directory {file_path}"
            )

    def __run_inference(self, prompt):
        """
        note: we run inference on pairs of datasets to separate train and val from test and reuse the eval code.
        so first return of the method is the result on training and validation dataset (test dataset field in the file actually contains validation).
        The second return has the train set to the same train as before (to remove computation overhead) and test set to real test dataset.
        """
        result_train_val = self.evaluator.evaluate(
            prompt=prompt,
            training_data=self.training_dataset,
            test_data=self.validation_dataset,
            num_threads=self.num_threads,
        )
        result_train_test = self.evaluator.evaluate(
            prompt=prompt,
            training_data=self.training_dataset[0:1],
            test_data=self.test_dataset,
            num_threads=self.num_threads,
        )
        result_train_test.training_accuracy = result_train_val.training_accuracy
        result_train_test.training_evaluation_correct = (
            result_train_val.training_evaluation_correct
        )
        result_train_test.training_evaluation_incorrect = (
            result_train_val.training_evaluation_incorrect
        )
        return result_train_val, result_train_test

    def __run_inference_optimized_prompt(self):
        best_prompt = self.best_prompt
        result_best_prompt_train, result_best_prompt_test = self.__run_inference(
            prompt=best_prompt
        )
        result_best_prompt_train.save_to_json(
            os.path.join(self.results_folder, "best_prompt_train_set.json")
        )
        result_best_prompt_test.save_to_json(
            os.path.join(self.results_folder, "best_prompt_test_set.json")
        )

    def __run_inference_starting_prompt(self):
        starting_prompt = self.config["starting_prompt"]
        result_starting_prompt_train, result_starting_prompt_test = (
            self.__run_inference(prompt=starting_prompt)
        )
        result_starting_prompt_train.save_to_json(
            os.path.join(self.results_folder, "starting_prompt_train_set.json")
        )
        result_starting_prompt_test.save_to_json(
            os.path.join(self.results_folder, "starting_prompt_test_set.json")
        )

    def __run_inference_manual_prompt(self):
        manual_prompt = self.config["manual_prompt"]
        result_manual_prompt_train, result_manual_prompt_test = self.__run_inference(
            prompt=manual_prompt
        )
        result_manual_prompt_train.save_to_json(
            os.path.join(self.results_folder, "manual_prompt_train_set.json")
        )
        result_manual_prompt_test.save_to_json(
            os.path.join(self.results_folder, "manual_prompt_test_set.json")
        )

    def __run_inference_optimized_prompt_parents(self):
        prompts = self.__extract_prompt_parents(prompt=self.best_prompt)
        folder_path = os.path.join(self.results_folder, "parent_results")
        # if no dir for parent results, make one
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # run inference iteratively over each parent.
        for i, p in enumerate(prompts):
            result_prompt_train, result_prompt_test = self.__run_inference(prompt=p)
            result_prompt_train.save_to_json(
                os.path.join(folder_path, f"parent_prompt_train_set_{i}.json")
            )
            result_prompt_test.save_to_json(
                os.path.join(folder_path, f"parent_prompt_test_set_{i}.json")
            )

    def __extract_prompt_parents(self, prompt: Prompt) -> List[Prompt]:
        prompts = [prompt]
        while prompt.parent_prompts:
            prompt = Prompt(**prompt.parent_prompts[0])
            prompts.append(prompt)
        assert not prompt.parent_prompts
        prompts.reverse()
        return prompts

    class ParameterObjectPromptsResult(BaseModel):
        starting_prompt_tr: PromptResultLite
        starting_prompt_t: PromptResultLite
        best_prompt_tr: PromptResultLite
        best_prompt_t: PromptResultLite
        manual_prompt_tr: PromptResultLite
        manual_prompt_t: PromptResultLite

    def __load_old_res(self, fname):
        path = os.path.join(self.results_folder, fname)
        res = PromptResult.load_from_json(path).to_lite()
        return res

    def test(self):
        res = self.__load_old_res(fname="best_prompt_evaluation.json")
        res1 = self.__load_old_res(fname="best_prompt_test_set.json")
        res1.training_accuracy = res.training_accuracy
        res1.training_evaluation_correct = res.training_evaluation_correct
        res1.training_evaluation_incorrect = res.training_evaluation_incorrect
        self.__save_old_new(res, fname="best_prompt_train_set.json")
        self.__save_old_new(res1, fname="best_prompt_test_set.json")

    def __save_old_new(self, res, fname):
        path = os.path.join(self.results_folder, fname)
        res.save_to_json(path)

    def convert_old_results_to_new(self):
        n_st = "starting_prompt_evaluation.json"
        n_st_t = "starting_prompt_test_set.json"
        n_m = "manual_prompt_evaluation.json"
        n_m_t = "manual_prompt_test_set.json"

        new_st_tr = "starting_prompt_train_set.json"
        new_st_t = "starting_prompt_test_set.json"
        new_m_tr = "manual_prompt_train_set.json"
        new_m_t = "manual_prompt_test_set.json"
        res_st_tr = self.__load_old_res(n_st)
        res_st_t = self.__load_old_res(n_st_t)
        res_st_t.training_accuracy = res_st_tr.training_accuracy
        res_st_t.training_evaluation_correct = res_st_tr.training_evaluation_correct
        res_st_t.training_evaluation_incorrect = res_st_tr.training_evaluation_incorrect
        self.__save_old_new(res_st_tr, fname=new_st_tr)
        self.__save_old_new(res_st_t, new_st_t)
        res_m_tr = self.__load_old_res(n_m)
        res_m_t = self.__load_old_res(n_m_t)
        res_m_t.training_accuracy = res_m_tr.training_accuracy
        res_m_t.training_evaluation_correct = res_m_tr.training_evaluation_correct
        res_m_t.training_evaluation_incorrect = res_m_tr.training_evaluation_incorrect
        self.__save_old_new(res_m_tr, fname=new_m_tr)
        self.__save_old_new(res_m_t, fname=new_m_t)

    def __load_prompt_results(self) -> ParameterObjectPromptsResult:
        result_starting_prompt_train = PromptResultLite.load_from_json(
            os.path.join(self.results_folder, "starting_prompt_train_set.json")
        )
        result_starting_prompt_test = PromptResultLite.load_from_json(
            os.path.join(self.results_folder, "starting_prompt_test_set.json")
        )
        result_best_prompt_train = PromptResultLite.load_from_json(
            os.path.join(self.results_folder, "best_prompt_train_set.json")
        )
        result_best_prompt_test = PromptResultLite.load_from_json(
            os.path.join(self.results_folder, "best_prompt_test_set.json")
        )
        result_manual_prompt_train = PromptResultLite.load_from_json(
            os.path.join(self.results_folder, "manual_prompt_train_set.json")
        )
        result_manual_prompt_test = PromptResultLite.load_from_json(
            os.path.join(self.results_folder, "manual_prompt_test_set.json")
        )
        parameter_object = AnalysisPipeline.ParameterObjectPromptsResult(
            starting_prompt_tr=result_starting_prompt_train,
            starting_prompt_t=result_starting_prompt_test,
            best_prompt_tr=result_best_prompt_train,
            best_prompt_t=result_best_prompt_test,
            manual_prompt_tr=result_manual_prompt_train,
            manual_prompt_t=result_manual_prompt_test,
        )
        return parameter_object

    def __get_prompt_evaluations(self, train_results, test_results):
        best_prompt_evaluations = {
            "Train": [],
            "Validation": [],
            "Test": [],
        }
        # get train results from train dataset
        docs_correct = [doc.doc_id for doc in train_results.training_evaluation_correct]
        for doc in self.training_dataset:
            res_per_doc = {
                "doc_id": doc.doc_id,
                "correct": 1 if doc.doc_id in docs_correct else 0,
            }
            best_prompt_evaluations["Train"].append(res_per_doc)
        # get validation results from train dataset (saved under test variable)
        docs_correct = [doc.doc_id for doc in train_results.test_evaluation_correct]
        for doc in self.validation_dataset:
            res_per_doc = {
                "doc_id": doc.doc_id,
                "correct": 1 if doc.doc_id in docs_correct else 0,
            }
            best_prompt_evaluations["Validation"].append(res_per_doc)
        # get test results from test dataset (saved under test variable)
        docs_correct = [doc.doc_id for doc in test_results.test_evaluation_correct]
        for doc in self.test_dataset:
            res_per_doc = {
                "doc_id": doc.doc_id,
                "correct": 1 if doc.doc_id in docs_correct else 0,
            }
            best_prompt_evaluations["Test"].append(res_per_doc)
        return best_prompt_evaluations

    def __get_confidence_interval(self, accuracies_dict):
        ci_train = self.__wilson_score_interval(
            p=accuracies_dict["Train"], n=len(self.training_dataset)
        )
        ci_val = self.__wilson_score_interval(
            p=accuracies_dict["Validation"], n=len(self.validation_dataset)
        )
        ci_test = self.__wilson_score_interval(
            p=accuracies_dict["Test"], n=len(self.test_dataset)
        )
        ci_train = self.__round_tuple_to_two_digits(ci_train)
        ci_val = self.__round_tuple_to_two_digits(ci_val)
        ci_test = self.__round_tuple_to_two_digits(ci_test)
        return ci_train, ci_val, ci_test

    def __round_tuple_to_two_digits(self, original_tuple):
        return tuple(round(x, 2) for x in original_tuple)

    def __get_scores(self, prompt_evaluations, key):
        scores = [elem["correct"] for elem in prompt_evaluations[key]]
        return scores

    def __get_accuracy(self, prompt_evaluations):
        accuracy = {"Train": 0, "Validation": 0, "Test": 0}
        for key in prompt_evaluations.keys():
            scores = self.__get_scores(prompt_evaluations, key)
            accuracy[key] = scores.count(1) / len(scores)
        return accuracy

    def __wilson_score_interval(self, p, n, alpha=0.05):
        return proportion_confint(count=p * n, nobs=n, alpha=alpha, method="wilson")

    def __binomial_test(self, new, baseline):
        # n11 = np.sum((baseline == 1) & (new == 1))
        n10 = np.sum((baseline == 1) & (new == 0))
        n01 = np.sum((baseline == 0) & (new == 1))
        # n00 = np.sum((baseline == 0) & (new == 0))
        # contingency table = [[n11, n10], [n01, n00]]
        discordant = n01 + n10
        if discordant == 0:
            two_sided = 1.0
            one_sided = 1.0
        else:
            # two-sided test
            two_sided = binomtest(n01, n01 + n10, p=0.5, alternative="two-sided").pvalue
            # One-sided version (binomial test, we n01 observations as successes, n10 as failures.)
            # we are testing whether the chance to get n01 is higher than n10.(null hypothesis: p(n01) = p(n10))
            # alternative hypothesis: p(n01) > p(n10)
            one_sided = binomtest(n01, discordant, p=0.5, alternative="greater").pvalue
        return [n01, n10, two_sided, one_sided]

    def __load_parents(self):
        prompts = self.__extract_prompt_parents(prompt=self.best_prompt)
        folder_path = os.path.join(self.results_folder, "parent_results")
        parent_prompts_evaluations = []
        for i, p in enumerate(prompts):
            prompt_res_train = PromptResultLite.load_from_json(
                os.path.join(folder_path, f"parent_prompt_train_set_{i}.json")
            )
            prompt_res_test = PromptResultLite.load_from_json(
                os.path.join(folder_path, f"parent_prompt_test_set_{i}.json")
            )
            parent_prompts_evaluations.append(
                self.__get_accuracy(
                    self.__get_prompt_evaluations(prompt_res_train, prompt_res_test)
                )
            )
        training_accuracies = [elem["Train"] for elem in parent_prompts_evaluations]
        test_accuracies = [elem["Test"] for elem in parent_prompts_evaluations]
        validation_accuracies = [
            elem["Validation"] for elem in parent_prompts_evaluations
        ]
        return training_accuracies, validation_accuracies, test_accuracies

    def __get_evaluation_persistence(
        self, results_df_starting, results_df_optimized, results_df_manual
    ):
        cert_starting_tr, cert_starting_t = self.__get_eval_changes_per_doc(
            results_df=results_df_starting
        )
        cert_optimized_tr, cert_optimized_t = self.__get_eval_changes_per_doc(
            results_df=results_df_optimized
        )
        cert_manual_tr, cert_manual_t = self.__get_eval_changes_per_doc(
            results_df=results_df_manual
        )

        uncert_starting_tr = [
            elem for elem in cert_starting_tr if elem["certainty"] != 1.0
        ]
        uncert_starting_t = [
            elem for elem in cert_starting_t if elem["certainty"] != 1.0
        ]
        uncert_optimized_tr = [
            elem for elem in cert_optimized_tr if elem["certainty"] != 1.0
        ]
        uncert_optimized_t = [
            elem for elem in cert_optimized_t if elem["certainty"] != 1.0
        ]
        uncert_manual_tr = [elem for elem in cert_manual_tr if elem["certainty"] != 1.0]
        uncert_manual_t = [elem for elem in cert_manual_t if elem["certainty"] != 1.0]

        tr_len = len(self.training_dataset)
        t_len = len(self.test_dataset)
        # Create a DataFrame to store the results
        data = {
            "Prompt": ["Start.", "Start.", "Opt.", "Opt.", "Man.", "Man."],
            "Dataset": [
                "Training",
                "Testing",
                "Training",
                "Testing",
                "Training",
                "Testing",
            ],
            "Count": [
                len(uncert_starting_tr),
                len(uncert_starting_t),
                len(uncert_optimized_tr),
                len(uncert_optimized_t),
                len(uncert_manual_tr),
                len(uncert_manual_t),
            ],
            "Relative Percentage %": np.round(
                np.array([
                    len(uncert_starting_tr) / tr_len,
                    len(uncert_starting_t) / t_len,
                    len(uncert_optimized_tr) / tr_len,
                    len(uncert_optimized_t) / t_len,
                    len(uncert_manual_tr) / tr_len,
                    len(uncert_manual_t) / t_len,
                ])
                * 100,
                decimals=1,
            ),
        }

        df = pd.DataFrame(data)
        return df

    def __get_response_persistence(self):
        starting_tr_count, starting_tr_percentage = self.__get_response_changes_per_doc(
            prompt_folder_name="starting_prompt", train=True
        )
        starting_t_count, starting_t_percentage = self.__get_response_changes_per_doc(
            prompt_folder_name="starting_prompt", train=False
        )

        optimized_tr_count, optimized_tr_percentage = (
            self.__get_response_changes_per_doc(
                prompt_folder_name="optimized_prompt", train=True
            )
        )
        optimized_t_count, optimized_t_percentage = self.__get_response_changes_per_doc(
            prompt_folder_name="optimized_prompt", train=False
        )

        manual_tr_count, manual_tr_percentage = self.__get_response_changes_per_doc(
            prompt_folder_name="manual_prompt", train=True
        )
        manual_t_count, manual_t_percentage = self.__get_response_changes_per_doc(
            prompt_folder_name="manual_prompt", train=False
        )

        data = {
            "Model": ["Starting", "Starting", "Opt", "Opt", "Manual", "Manual"],
            "Dataset": [
                "Training",
                "Test",
                "Training",
                "Test",
                "Training",
                "Test",
            ],
            "Count": [
                starting_tr_count,
                starting_t_count,
                optimized_tr_count,
                optimized_t_count,
                manual_tr_count,
                manual_t_count,
            ],
            "Relative Percentage %": np.round(
                np.array([
                    starting_tr_percentage,
                    starting_t_percentage,
                    optimized_tr_percentage,
                    optimized_t_percentage,
                    manual_tr_percentage,
                    manual_t_percentage,
                ])
                * 100,
                decimals=1,
            ),
        }

        df = pd.DataFrame(data)
        return df

    def __get_response_changes_per_doc(self, prompt_folder_name, train=True):
        if train:
            column_correct = "training_evaluation_correct"
            column_incorrect = "training_evaluation_incorrect"
        else:
            column_correct = "test_evaluation_correct"
            column_incorrect = "test_evaluation_incorrect"
        res_dict = {}
        results_flash = []
        for i in range(10):
            result_i = PromptResultLite.load_from_json(
                os.path.join(
                    self.results_folder,
                    f"persistence/{prompt_folder_name}/{i + 1}.json",
                )
            ).to_dataframe()
            result_i["iteration"] = f"{i + 1}"
            results_flash.append(result_i)

            for doc_res in result_i[column_correct].values[0]:
                doc_id = doc_res["doc_id"]
                value = doc_res["predicted"]
                if doc_id not in res_dict:
                    res_dict[doc_id] = [value]
                else:
                    res_dict[doc_id] += [value]

            for doc_res in result_i[column_incorrect].values[0]:
                doc_id = doc_res["doc_id"]
                value = doc_res["predicted"]
                if doc_id not in res_dict:
                    res_dict[doc_id] = [value]
                else:
                    res_dict[doc_id] += [value]

        doc_ids = list(res_dict.keys())
        num_docs = len(doc_ids)
        unique_answers = []
        for doc_id in doc_ids:
            answer_list = []
            num_unique_answers = 0
            for answer in res_dict[doc_id]:
                if answer not in answer_list:
                    answer_list.append(answer)
                    num_unique_answers += 1
            unique_answers.append(num_unique_answers)

        changes = len(np.where(np.array(unique_answers) > 1)[0])
        changes_percentage = changes / num_docs
        return changes, changes_percentage
