from pydantic import BaseModel, Field


class AbsoluteScope1EmissionsSchema(BaseModel):
    total_scope1_emissions: float | None

    def model_post_init(self, __context) -> None:
        if self.total_scope1_emissions is None:
            self.total_scope1_emissions = 0.0


class Scope3EmissionsAssuranceSchema(BaseModel):
    has_scope_3_assurance: bool | None


class ReportingPeriodSchema(BaseModel):
    start_date: str | None = Field(
        ...,
        description="Starting date of the reporting period of the report. Return in format YYYY-MM. Return None if not found.",
    )
    end_date: str | None = Field(
        ...,
        description="Ending date of the reporting period of the report. Return in format YYYY-MM. Return None if not found.",
    )
