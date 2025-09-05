
# %%
import numpy as np
import os
from src.scripts.protegi import ProTeGi
from src.data_structures.prompt import Prompt
from src.scripts.evaluator import Evaluator
from src.data_structures.config import *

# define target variable
target_variable = "s1"
config = configs[target_variable]
#config = modify_results_paths(config=config, new_subfolder_path="results")
print(config)

# %%
# load data from gcp
input_data, input_data_new = load_data(config=config)
# load data split
training_data, validation_data, test_data = get_train_val_test_data(input_data, input_data_new)

# %%
# load starting prompt and schema from the config
starting_prompt = config["starting_prompt"]
response_schema = config["schema"]

# %%
protegi = ProTeGi(
    model_name_feedback=model_name_feedback,
    model_name_inference=model_name_inference,
    response_schema=response_schema,
)

best_prompt, training_statistics = protegi.prompt_optimization_with_beam_search(
    initial_prompt=starting_prompt,
    training_data=training_data,
    beam_width=beam_width,
    num_iterations=num_iterations,
    bandit_budget=bandit_budget,
    batch_size_for_errors=batch_size_for_errors,
    max_num_examples=max_num_examples,
    num_gradients=num_gradients,
    steps_per_gradient=steps_per_gradient,
    num_variations=num_variations,
    num_threads=num_threads,
    bandit_sample_size=bandit_sample_size,
    c=c,
    num_prompts_per_round=num_prompts_per_round,
)

print(f"Best prompt: {best_prompt}")

# %% [markdown]
# Save optimized prompt and training statistics.

# %%
best_prompt.dump_json(os.path.join(config["results_path"], "best_prompt_no_evaluation.json"))
training_statistics.save_to_json(os.path.join(config["results_path"], "protegi_training_statistics.json"))

# %% [markdown]
# Evaluation pipeline

# %%
from src.scripts.analysis_pipeline import AnalysisPipeline

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
# Run all possible inferences. Will take a long time.

# %%
anal_pipeline.run_all_inferences()

# %% [markdown]
# # run only persistence inferences

# %%
anal_pipeline.run_inference_persistence()

# %% [markdown]
# Run all analyisis pipelines. Uses files produced by inference, can be run without inference in the current session if the files already exist (inference was performed before). Takes seconds because no inference, only analysis.

# %%
anal_pipeline.idgaf_analyze_everything()

# %%
