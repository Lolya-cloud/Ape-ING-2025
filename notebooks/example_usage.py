# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: prompt-optimization
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from pydantic import BaseModel
import os

from standalone.protegi import ProTeGi
from standalone.smart_grads_only import SmartOptimizer
from standalone.analysis_pipeline import AnalysisPipeline
from standalone.data_structures.prompt import Prompt
from standalone.data_structures.data_handling import DataLoader, InputData
from standalone.data_structures.response_schemas import AbsoluteScope1EmissionsSchema
from standalone.data_structures.config import configs

# %% [markdown]
# Step 1: load the data. The algorithm expects a training set as a list of InputData objects, where ground truth is formated with the same schema as the expected output. The default metric function does an exact match between the ground truth and LLM output schema objects. The example code below loads the files for the specified target variable (Scope 1 absolute emissions), and automatically converts the ground truths to the schema. For other two variables, swap the load function (loader provides two others), schemas and starting prompts. See the config file for all the promtps and details. 

# %% [markdown]
# For custom variables, specify a schema, a metric function if needed (can be passed as a parameter into the optimization functions), and format training data as List[InputData], where ground truth parameter is formated in the same response schema as the expected output. For details of the input object class, see data_handling.py. Then set starting prompt and results path down below (I load them from the config dictionary, you can set them yourself in the code below to what's needed)

# %%
bucket_name_gt = ""
bucket_name_pdfs = ""
csv_filename = ""

data_loader = DataLoader(bucket_name_gt=bucket_name_gt, bucket_name_pdf=bucket_name_pdfs, csv_filename=csv_filename)
# # copy the ground truth csv from the cloud. Should create a csv file in the notebooks directory.
data_loader.copy_gt_file_gsutil()

# %%
# load the reports for scope 1 emissions
input_data = data_loader.get_input_data_for_scope1_emissions()

# %%
# split into train/test if needed. This split is a dummy, using indices, not a random. 
# replace with your persistent split (note that just setting seed is not enough, 
# gcp returns reports in different order each time).
training_data, test_data = input_data[0:10], input_data[10:20]
# only used for analysis pipeline. Set to 1 document if not needed.
validation_data = training_data[0:1]

# %%
# load config for scope 1
config = configs["s1"]

# %%
# starting prompt and schema.
starting_prompt = config["starting_prompt"]
response_schema = config["schema"]

# %% [markdown]
# Optimizers section. Note that all save to the same results folder specified in config, all save the optimized prompts under the same name. Hence, run only one of 3. To run others, modify the results folder (you can use the corresponding function from config or do it manually by setting config["results_path"] = new path)

# %% [markdown]
# Prompt Enrichment optimizer. NOTE: RESULTS PATH SHOULD EXIST

# %%
# parameters
num_gradients = 2
num_threads = 10
doc_skip_threshold = 2
results_path = config["results_path"]
os.makedirs(name=results_path, exist_ok=True)
# must be a valid name of a model, passed to the google genai sdk. See generation_model.py for the downstream usage.
model_name_inference = "gemini-2.0-flash-001"
model_name_feedback = "gemini-2.0-flash-001"

# %% [markdown]
# Run training with the above parameters.

# %%
smart = SmartOptimizer(
        model_name_feedback=model_name_feedback, 
        model_name_inference=model_name_inference, 
        response_schema=response_schema,
        results_path=results_path,
        doc_skip_threshold=doc_skip_threshold
        )

optimized_prompt, training_statistics = smart.optimize_prompt_with_all_shenanigans(
        prompt=starting_prompt, 
        training_data=training_data, 
        num_gradients=num_gradients,
        num_threads=num_threads
)

# %% [markdown]
# Save the resulting prompts and training statistics

# %%
optimized_prompt.dump_json(os.path.join(results_path, "best_prompt_no_evaluation.json"))
training_statistics.save_to_json(os.path.join(results_path, "smart_training_statistics.json"))

# %%
print(optimized_prompt.text)

# %% [markdown]
# ProTeGi optimizer with/without GradV

# %%
response_schema = config["schema"]
starting_prompt = config["starting_prompt"]
model_name_inference = "gemini-2.0-flash-001"
model_name_feedback = "gemini-2.0-flash-001"

# ProTeGi parameters (for defaults see the config file)
beam_width = 2
num_iterations = 2
bandit_budget = 2
batch_size_for_errors = 10
max_num_examples = 1
bandit_sample_size = 2
c = 1
num_gradients = 2
steps_per_gradient = 1
num_variations = 0
num_threads = 20
num_prompts_per_round = 2

# additional parameters for GradV (ProTeGi + GradV)
num_doc_retries = 2
threshold_score = 1
num_doc_copies = 2

optimizer = ProTeGi(model_name_feedback=model_name_feedback, model_name_inference=model_name_inference, response_schema=response_schema)

# %% [markdown]
# ProTeGi with GradV optimizer

# %%
protegi_optimized_prompt_gv, protegi_training_statistics_gv = optimizer.prompt_optimization_with_beam_search_and_grad_verification(
        initial_prompt=starting_prompt,
        training_data=training_data,
        beam_width=beam_width,
        num_iterations=num_iterations,
        bandit_budget=bandit_budget,
        batch_size_for_errors=batch_size_for_errors,
        num_gradients=num_gradients,
        num_threads=num_threads,
        c=c,
        bandit_sample_size=bandit_sample_size,
        num_prompts_per_round=num_prompts_per_round,
        num_doc_retries=num_doc_retries,
        threshold_score=threshold_score,
        num_doc_copies=num_doc_copies)

# %%
print(protegi_optimized_prompt_gv.text)

# %%
protegi_optimized_prompt_gv.dump_json(os.path.join(results_path, "best_prompt_no_evaluation.json"))
protegi_training_statistics_gv.save_to_json(os.path.join(results_path, "protegi_training_statistics.json"))

# %% [markdown]
# ProTeGi without GradV

# %%
protegi_optimized_prompt, protegi_training_statistics = optimizer.prompt_optimization_with_beam_search(
        initial_prompt=starting_prompt,
        training_data=training_data,
        beam_width=beam_width,
        num_iterations=num_iterations,
        bandit_budget=bandit_budget,
        batch_size_for_errors=batch_size_for_errors,
        num_gradients=num_gradients,
        num_threads=num_threads,
        c=c,
        bandit_sample_size=bandit_sample_size,
        num_prompts_per_round=num_prompts_per_round,
        steps_per_gradient=steps_per_gradient,
        max_num_examples=max_num_examples,
        num_variations=num_variations
        )

# %%
protegi_optimized_prompt.dump_json(os.path.join(results_path, "best_prompt_no_evaluation.json"))
protegi_training_statistics.save_to_json(os.path.join(results_path, "protegi_training_statistics.json"))

# %% [markdown]
# An example on how to use analysis pipeline for the evaluation of the new, starting and manual prompts. Not needed for the optimization, replace with your own if needed.

# %%
anal_pipeline = AnalysisPipeline(
    config=config, 
    training_dataset=training_data, 
    validation_dataset=validation_data, 
    test_dataset=test_data,
    num_threads=30,
    model_name_inference=model_name_inference
    )

# %% [markdown]
# Run inferences. Creates files for each of the 3 prompts needed for the analysis. If exists set to True, assumes the file already exists and does not run inference.

# %%
anal_pipeline.run_basic_inferences(
    starting_prompt_eval_exists=False,
    optimized_prompt_eval_exists=False,
    manual_prompt_eval_exists=False,
    )

# %%
anal_pipeline.analyze_starting_vs_optimized_vs_manual_prompt_print_paper_results()
