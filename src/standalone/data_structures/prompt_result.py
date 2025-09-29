from pydantic import BaseModel
from standalone.data_structures.prompt import Prompt
from standalone.data_structures.data_handling import OutputData
from standalone.generation_model import GenerationModel
from typing import List, Any
import json
import pandas as pd


class ProcessedDocument(BaseModel):
    doc_id: str
    ground_truth: Any
    predicted: Any

    @classmethod
    def create_processed_document(cls, data: "OutputData") -> "ProcessedDocument":
        return cls(
            doc_id=data.doc_id, ground_truth=data.ground_truth, predicted=data.predicted
        )


class TrainingStatistics(BaseModel):
    inference_model: str
    inference_tokens: int
    inference_api_calls: int
    inference_time: float

    feedback_model: str
    feedback_tokens: int
    feedback_api_calls: int
    feedback_time: float

    final_beam_prompts: List[Prompt] | None
    final_beam_scores: List[float] | None

    beams: List[List[Prompt]] | None

    @classmethod
    def factory(
        cls,
        inference_model: "GenerationModel",
        feedback_model: "GenerationModel",
        final_beam_prompts: List[Prompt] | None = None,
        final_beam_scores: List[float] | None = None,
        beams: List[List[Prompt]] | None = None,
    ) -> "TrainingStatistics":
        return cls(
            inference_model=inference_model.model_name,
            inference_tokens=inference_model.total_tokens,
            inference_api_calls=inference_model.api_calls,
            inference_time=inference_model.total_time,
            feedback_model=feedback_model.model_name,
            feedback_tokens=feedback_model.total_tokens,
            feedback_api_calls=feedback_model.api_calls,
            feedback_time=feedback_model.total_time,
            final_beam_prompts=final_beam_prompts,
            final_beam_scores=final_beam_scores,
            beams=beams,
        )

    def to_dictionary(self):
        return self.model_dump()

    def to_dataframe(self):
        model_dict = self.to_dictionary()
        df = pd.DataFrame([model_dict])
        return df

    def save_to_json(self, path: str):
        model_dict = self.to_dictionary()
        with open(path, "w") as json_file:
            json.dump(model_dict, json_file, indent=4)

    @classmethod
    def load_from_json(cls, path: str):
        with open(path, "r") as json_file:
            model_dict = json.load(json_file)
        return cls(**model_dict)


class PromptResultLite(BaseModel):
    prompt: Prompt
    training_accuracy: float
    test_accuracy: float
    training_evaluation_correct: List[ProcessedDocument]
    training_evaluation_incorrect: List[ProcessedDocument]
    test_evaluation_correct: List[ProcessedDocument]
    test_evaluation_incorrect: List[ProcessedDocument]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "forbid"

    def to_dictionary(self):
        return self.model_dump()

    def to_dataframe(self):
        model_dict = self.to_dictionary()
        df = pd.DataFrame([model_dict])
        return df

    def save_to_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def save_to_json(self, path: str):
        model_dict = self.to_dictionary()
        with open(path, "w") as json_file:
            json.dump(model_dict, json_file, indent=4)

    @classmethod
    def load_from_csv(cls, path: str):
        df = pd.read_csv(path)
        model_dict = df.to_dict(orient="records")[0]
        print(model_dict["prompt"])
        model_dict["prompt"] = Prompt(**model_dict["prompt"])
        return cls(**model_dict)

    @classmethod
    def load_from_json(cls, path: str):
        with open(path, "r") as json_file:
            model_dict = json.load(json_file)
            model_dict["prompt"] = Prompt(**model_dict["prompt"])
        return cls(**model_dict)


class PromptResult(BaseModel):
    prompt: Prompt
    training_accuracy: float
    test_accuracy: float
    total_tokens: int
    total_api_calls: int
    feedback_model: str
    total_feedback_tokens: int
    total_feedback_calls: int
    total_feedback_time: float
    inference_model: str
    total_inference_tokens: int
    total_inference_calls: int
    total_inference_time: float
    training_evaluation_correct: List[ProcessedDocument]
    training_evaluation_incorrect: List[ProcessedDocument]
    test_evaluation_correct: List[ProcessedDocument]
    test_evaluation_incorrect: List[ProcessedDocument]

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "forbid"

    def to_lite(self):
        lite_res = PromptResultLite(
            prompt=self.prompt,
            training_accuracy=self.training_accuracy,
            test_accuracy=self.test_accuracy,
            training_evaluation_correct=self.training_evaluation_correct,
            training_evaluation_incorrect=self.training_evaluation_incorrect,
            test_evaluation_correct=self.test_evaluation_correct,
            test_evaluation_incorrect=self.test_evaluation_incorrect,
        )
        return lite_res

    def to_dictionary(self):
        return self.model_dump()

    def to_dataframe(self):
        model_dict = self.to_dictionary()
        df = pd.DataFrame([model_dict])
        return df

    def save_to_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)

    def save_to_json(self, path: str):
        model_dict = self.to_dictionary()
        with open(path, "w") as json_file:
            json.dump(model_dict, json_file, indent=4)

    @classmethod
    def load_from_csv(cls, path: str):
        df = pd.read_csv(path)
        model_dict = df.to_dict(orient="records")[0]
        print(model_dict["prompt"])
        model_dict["prompt"] = Prompt(**model_dict["prompt"])
        return cls(**model_dict)

    @classmethod
    def load_from_json(cls, path: str):
        with open(path, "r") as json_file:
            model_dict = json.load(json_file)
            model_dict["prompt"] = Prompt(**model_dict["prompt"])
        return cls(**model_dict)
