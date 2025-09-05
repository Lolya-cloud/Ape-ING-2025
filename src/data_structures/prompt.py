from pydantic import BaseModel, Field
from typing import List
import json


class Prompt(BaseModel):
    text: str = Field(..., description="Current prompt text")
    parent_prompts: List = Field(
        default_factory=lambda: [],
        description="List of parent prompts, each as a list of strings",
    )

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "forbid"

    def dump_json(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def from_json(cls, filepath: str) -> "Prompt":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
