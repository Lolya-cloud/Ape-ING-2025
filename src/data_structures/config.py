import numpy as np
import os
from data_structures.response_schemas import (
    Scope3EmissionsAssuranceSchema,
    AbsoluteScope1EmissionsSchema,
    ReportingPeriodSchema,
)
from data_structures.data_handling import DataLoader, InputData
from data_structures.prompt import Prompt

# if using gcp
bucket_name_gt = ""
bucket_name_old = ""
bucket_name_new = ""

data_split_location = "../../data_split"

configs = {
    "s3": {
        "results_path": "s3/results",
        "name": "s3",
        "bucket_name_gt": bucket_name_gt,
        "bucket_name_old": bucket_name_old,
        "bucket_name_new": bucket_name_new,
        "csv_gt": "",
        "csv_gt_new": "",
        "starting_prompt": Prompt(
            text="Does the company have scope 3 assurance for the reporting year 2023?"
        ),
        "manual_prompt": Prompt(
            text="""Extract assurance information from the following file for the year 2023.      
                        **Instructions:**
                        * First, check if the file mentions the word 'assurance' or not. If there are no mentions of 'assurance': set has_scope_3_assurance to null.
                        * Second, if the file mentions the word 'assurance', check if an assurance was done by an independent professional assurance provider, such as an auditing firm. 
                        * If an assurance was done (limited or full): set has_scope_3_assurance.answer to True.
                        * If an assurance was NOT done: set has_scope_3_assurance.answer to False.
                        * If the company does not seek external assurance: set has_scope_3_assurance.answer to False.
                        
                        **Note:**
                        * In order to have an assurance, the greenhouse gas (GHG) emission data must be subject to audit or review by an independent professional assurance provider, such as an auditing firm.
                        """
        ),
        "schema": Scope3EmissionsAssuranceSchema,
    },
    "s1": {
        "results_path": "s1/results",
        "name": "s1",
        "bucket_name_gt": bucket_name_gt,
        "bucket_name_old": bucket_name_old,
        "bucket_name_new": bucket_name_new,
        "csv_gt": "",
        "csv_gt_new": "",
        "starting_prompt": Prompt(
            text="What is the Scope 1 greenhouse gas emission of this company in 2023?"
        ),
        "manual_prompt": Prompt(
            text="""Extract absolute scope 1 emissions from the following file for the year 2023.
                        10,000 tCO2 is 10000.0 tCO2e. 
                        35,125 tCO2e is 35125.0 tCO2e.

                        Pay attention to the unit and the multiplier of the value, convert it to tCO2e and return as a float.
                        For example, if the absolute scope 1 emissions are 3.12 kilo tCO2e in the file, return 3120.0.
                        If none is mentioned, return 0.0. If the emissions are 0, return 0.0
                        """
        ),
        "schema": AbsoluteScope1EmissionsSchema,
    },
    "rp": {
        "results_path": "rp/results",
        "name": "rp",
        "bucket_name_gt": bucket_name_gt,
        "bucket_name_old": bucket_name_old,
        "bucket_name_new": bucket_name_new,
        "csv_gt": "",
        "csv_gt_new": "",
        "starting_prompt": Prompt(
            text="Extract the start and end reporting date from the following file."
        ),
        "manual_prompt": Prompt(
            text="""Extract the following fields from the following file:
                        "start_date": "Starting date of the reporting period of the report. Return in format YYYY-MM. 
                        If the report covers year 2020, the start date is 2020-01. If the report covers the year until end of April 2020, the start date is 2019-05",
                        "end_date": "Ending date of the reporting period of the report. Return in format YYYY-MM. If the report covers year 2020, the end date is 2020-12. 
                        If the report covers the year until end of April 2020, the end date is 2020-04",
                        """
        ),
        "schema": ReportingPeriodSchema,
    },
}

# default training parameters
beam_width = 4
num_iterations = 5
bandit_budget = 10
batch_size_for_errors = 10
max_num_examples = 1
bandit_sample_size = 10
c = 1
num_gradients = 4
steps_per_gradient = 1
num_variations = 0
num_threads = 20
num_prompts_per_round = 4

model_name_inference = "gemini-2.0-flash-001"
model_name_feedback = "gemini-2.0-flash-001"


def load_data(config):
    data_loader = DataLoader(
        bucket_name_gt=config["bucket_name_gt"],
        bucket_name_pdf=config["bucket_name_old"],
        csv_filename=config["csv_gt"],
    )
    data_loader_extra = DataLoader(
        bucket_name_gt=config["bucket_name_gt"],
        bucket_name_pdf=config["bucket_name_new"],
        csv_filename=config["csv_gt_new"],
    )
    if config["name"] == "s1":
        input_data = data_loader.get_input_data_for_scope1_emissions()
        input_data_new = data_loader_extra.get_input_data_for_scope1_emissions()
    elif config["name"] == "s3":
        input_data = data_loader.get_input_data_for_scope3_assurance()
        input_data_new = data_loader_extra.get_input_data_for_scope3_assurance()
    elif config["name"] == "rp":
        input_data = data_loader.get_input_data_for_reporting_period()
        input_data_new = data_loader_extra.get_input_data_for_reporting_period()
    else:
        raise ValueError("Incorrect config provided")

    return input_data, input_data_new


def get_train_val_test_data(input_data, input_data_new):
    training_ids = np.load(os.path.join(data_split_location, "training_ids.npy"))
    validation_ids = np.load(os.path.join(data_split_location, "validation_ids.npy"))
    test_ids = np.load(os.path.join(data_split_location, "test_ids.npy"))
    test_ids_extension = np.load(
        os.path.join(data_split_location, "test_ids_extension.npy")
    )
    training_data = [elem for elem in input_data_new if elem.doc_id in training_ids]
    validation_data = [elem for elem in input_data_new if elem.doc_id in validation_ids]
    test_data = [elem for elem in input_data_new if elem.doc_id in test_ids]
    test_data_extension = [
        elem for elem in input_data_new if elem.doc_id in test_ids_extension
    ]
    for id in training_ids:
        included_ids = [elem.doc_id for elem in training_data]
        if id not in included_ids:
            leftover_elem = [elem for elem in input_data if elem.doc_id == id][0]
            training_data.append(leftover_elem)
    for id in validation_ids:
        included_ids = [elem.doc_id for elem in validation_data]
        if id not in included_ids:
            leftover_elem = [elem for elem in input_data if elem.doc_id == id][0]
            validation_data.append(leftover_elem)
    for id in test_ids:
        included_ids = [elem.doc_id for elem in test_data]
        if id not in included_ids:
            leftover_elem = [elem for elem in input_data if elem.doc_id == id][0]
            test_data.append(leftover_elem)

    test_data = test_data + test_data_extension
    print(f"Training data size: {len(training_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Validation data size: {len(validation_data)}")
    print(f"Test extension size: {len(test_data_extension)}")
    return training_data, validation_data, test_data


def modify_results_paths(config, new_subfolder_path):
    copy = config
    new_path = os.path.join(copy["name"], new_subfolder_path)
    os.makedirs(new_path, exist_ok=True)
    copy["results_path"] = new_path
    return copy


def get_decoy_data():
    training_ids = np.load(os.path.join(data_split_location, "training_ids.npy"))
    validation_ids = np.load(os.path.join(data_split_location, "validation_ids.npy"))
    test_ids = np.load(os.path.join(data_split_location, "test_ids.npy"))
    test_ids_extension = np.load(
        os.path.join(data_split_location, "test_ids_extension.npy")
    )
    training_data = [
        InputData(doc_id=doc_id, ground_truth=0, doc="") for doc_id in training_ids
    ]
    validation_data = [
        InputData(doc_id=doc_id, ground_truth=0, doc="") for doc_id in validation_ids
    ]
    test_data = [
        InputData(doc_id=doc_id, ground_truth=0, doc="") for doc_id in test_ids
    ]
    test_data_extension = [
        InputData(doc_id=doc_id, ground_truth=0, doc="")
        for doc_id in test_ids_extension
    ]
    test_data = test_data + test_data_extension
    print(f"Training data size: {len(training_data)}")
    print(f"Test data size: {len(test_data)}")
    print(f"Validation data size: {len(validation_data)}")
    print(f"Test extension size: {len(test_data_extension)}")
    return training_data, validation_data, test_data
