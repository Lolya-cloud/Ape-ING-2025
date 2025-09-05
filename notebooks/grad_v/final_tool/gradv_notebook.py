# %%
import os
from src.scripts.prompt_enrichment_gradv import SmartOptimizer
from src.data_structures.prompt import Prompt
from src.scripts.evaluator import Evaluator
from src.data_structures.config import *
from src.scripts.analysis_pipeline import AnalysisPipeline

# define target variable
target_variable = "s1"
config = configs[target_variable]
config = modify_results_paths(config=config, new_subfolder_path="results_4grad")

# %%
# load data from gcp
input_data, input_data_new = load_data(config=config)

# %%
# load data split
training_data, validation_data, test_data = get_train_val_test_data(input_data, input_data_new)


# %%
# load starting prompt and schema from the config
starting_prompt = config["starting_prompt"]
response_schema = config["schema"]
training_data = training_data
num_gradients = 3
num_threads = 24
doc_skip_threshold = 5

# %% [markdown]
# Run training with the above parameters.

# %%
smart = SmartOptimizer(model_name_feedback=model_name_feedback, model_name_inference=model_name_inference, response_schema=response_schema,
results_path=config["results_path"],
doc_skip_threshold=doc_skip_threshold)

optimized_prompt, training_statistics = smart.optimize_prompt_with_all_shenanigans(
        prompt=starting_prompt, 
        training_data=training_data, 
        num_gradients=num_gradients,
        num_threads=num_threads
)

# %%
optimized_prompt.dump_json(os.path.join(config["results_path"], "best_prompt_no_evaluation.json"))
training_statistics.save_to_json(os.path.join(config["results_path"], "smart_training_statistics.json"))

# %%
print(optimized_prompt)

# %%
anal_pipeline = AnalysisPipeline(
    config=config, 
    training_dataset=training_data, 
    validation_dataset=validation_data, 
    test_dataset=test_data,
    num_threads=30,
    model_name_inference=model_name_inference
    )

# %%
anal_pipeline.run_basic_inferences(
    starting_prompt_eval_exists=True,
    optimized_prompt_eval_exists=False,
    manual_prompt_eval_exists=True,
    )

# %% [markdown]
# protegi with grads verification

# %%
from src.scripts.protegi import ProTeGi

model_name_feedback = model_name_feedback
model_name_inference = model_name_inference
response_schema = config["schema"]

starting_prompt = config["starting_prompt"]
training_data = training_data
beam_width = beam_width
num_iterations = 5
bandit_budget = bandit_budget
batch_size_for_errors = batch_size_for_errors
num_gradients = 4
num_threads = 20
c = c
bandit_sample_size = bandit_sample_size
num_prompts_per_round = num_prompts_per_round
num_doc_retries = 5
threshold_score = 1
num_doc_copies = 2

optimizer = ProTeGi(model_name_feedback=model_name_feedback, model_name_inference=model_name_inference, response_schema=response_schema)

# %%
protegi_optimized_prompt, protegi_training_statistics = optimizer.prompt_optimization_with_beam_search_and_grad_verification(
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
protegi_optimized_prompt.dump_json(os.path.join(config["results_path"], "best_prompt_no_evaluation.json"))
protegi_training_statistics.save_to_json(os.path.join(config["results_path"], "protegi_training_statistics.json"))

# %%
anal_pipeline = AnalysisPipeline(
    config=config, 
    training_dataset=training_data, 
    validation_dataset=validation_data, 
    test_dataset=test_data,
    num_threads=50,
    model_name_inference=model_name_inference
    )

anal_pipeline.run_basic_inferences(
    starting_prompt_eval_exists=True,
    optimized_prompt_eval_exists=False,
    manual_prompt_eval_exists=True,
    )

# %%



