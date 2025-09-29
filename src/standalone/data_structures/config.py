from standalone.data_structures.response_schemas import (
    Scope3EmissionsAssuranceSchema,
    AbsoluteScope1EmissionsSchema,
    ReportingPeriodSchema,
)
from standalone.data_structures.prompt import Prompt


configs = {
    "s3": {
        "results_path": "s3/results",
        "name": "s3",
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

num_doc_retries = 5
threshold_score = 1
num_doc_copies = 2
