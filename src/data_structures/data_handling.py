from pydantic import BaseModel
from typing import Any, List, Tuple
import pandas as pd
import os
from cloudpathlib import CloudPath
from data_structures.bytes import read_bytes
from data_structures.response_schemas import (
    AbsoluteScope1EmissionsSchema,
    Scope3EmissionsAssuranceSchema,
    ReportingPeriodSchema,
)
from google import genai


class InputData(BaseModel):
    doc_id: str
    doc: Any
    ground_truth: Any


class OutputData(InputData):
    predicted: Any


class DataLoader(BaseModel):
    bucket_name_gt: str
    bucket_name_pdf: str
    csv_filename: str

    def copy_gt_file_gsutil(self):
        command = f"gsutil -m cp -r gs://{self.bucket_name_gt}/{self.csv_filename} ."
        os.system(command)

    def check_bytes(self, file_path: str) -> bool:
        """
        This function returns True if the file exists. Otherwise, it returns False.
        """
        try:
            # option 1: google cloud storage
            if file_path.startswith("gs://"):
                return CloudPath(file_path).is_file()
            # option 2: local file storage
            else:
                return os.path.exists(file_path)

        except Exception as exc:
            print(exc)
            raise exc

    def get_overlapping_filepaths_csv_gcp(self) -> list:
        # We need this helper function to identify which anonymised repors appear in ground truth.
        try:
            # Load CSV file and extract filepaths
            df = pd.read_csv(self.csv_filename)
            filepaths = df["file_path"].tolist()
            print(f"{len(filepaths)} filepaths in {self.csv_filename} file.")
            filepaths = list(set(filepaths))  # unique filepaths
            print(f"{len(filepaths)} unique filepaths in {self.csv_filename} file.")

            # Check cloud storage bucket to see if files exist
            valid_filepaths, invalid_filepaths = [], []
            for file_path in filepaths:
                if self.check_bytes(f"gs://{self.bucket_name_pdf}/{file_path}"):
                    valid_filepaths.append(file_path)
                else:
                    invalid_filepaths.append(file_path)
            print(
                f"{len(valid_filepaths)} files exist in {self.bucket_name_pdf} bucket. {len(invalid_filepaths)} files do not exist in bucket."
            )

            return valid_filepaths, df

        except Exception as exc:
            print(exc)

    def get_input_data_for_scope1_emissions(self) -> List[InputData]:
        # self.copy_gt_file_gsutil()
        common_filepaths, df_ground_truth = self.get_overlapping_filepaths_csv_gcp()
        input_list = []

        for filepath in common_filepaths:
            # print(f"\nWorking with file: {filepath}")

            # Display ground truth values
            gt_entry = df_ground_truth.loc[
                df_ground_truth["file_path"] == filepath
            ].iloc[0]
            # drop part files and not 2023 files.
            if gt_entry.reporting_year != 2023:
                continue
            filename = filepath.split("/")[-1]
            if "_part" in filename:
                continue
            # print(f"* Ground truth. reporting_year: {gt_entry.reporting_year}, scope1_emissions: {gt_entry.scope1_emissions}, scope1_comments: {gt_entry.scope1_comments}")
            # we will use filepath as unique id.
            path = f"gs://{self.bucket_name_pdf}/{filepath}"
            file_size = CloudPath(cloud_path=path).stat().st_size
            # skip if file size is bigger than 40 mb.
            if file_size > 48 * 1024 * 1024:
                continue
            ground_truth = float(gt_entry.scope1_emissions)
            if pd.isna(ground_truth):
                ground_truth = None
            ground_truth = AbsoluteScope1EmissionsSchema(
                total_scope1_emissions=ground_truth
            )
            doc = genai.types.Part.from_bytes(
                data=read_bytes(path),
                mime_type="application/pdf",
            )
            doc_id = filepath
            # create a new input data object
            input_data = InputData(doc_id=doc_id, doc=doc, ground_truth=ground_truth)
            input_list.append(input_data)
        return input_list

    def get_input_data_for_reporting_period(self) -> List[InputData]:
        # self.copy_gt_file_gsutil()
        common_filepaths, df_ground_truth = self.get_overlapping_filepaths_csv_gcp()
        input_list = []

        for filepath in common_filepaths:
            # print(f"\nWorking with file: {filepath}")

            # Display ground truth values
            gt_entry = df_ground_truth.loc[
                df_ground_truth["file_path"] == filepath
            ].iloc[0]
            # drop part files and not 2023 files.
            if gt_entry.reporting_year != 2023:
                continue
            filename = filepath.split("/")[-1]
            if "_part" in filename:
                continue
            # print(f"* Ground truth. reporting_year: {gt_entry.reporting_year}, scope1_emissions: {gt_entry.scope1_emissions}, scope1_comments: {gt_entry.scope1_comments}")
            # we will use filepath as unique id.
            path = f"gs://{self.bucket_name_pdf}/{filepath}"
            file_size = CloudPath(cloud_path=path).stat().st_size
            # skip if file size is bigger than 40 mb.
            if file_size > 48 * 1024 * 1024:
                continue

            start = gt_entry.reporting_period_start
            end = gt_entry.reporting_period_end
            ground_truth = [None, None]
            if not pd.isna(start):
                ground_truth[0] = str(start)[:-3]
            if not pd.isna(end):
                ground_truth[1] = str(end)[:-3]
            ground_truth = ReportingPeriodSchema(
                start_date=ground_truth[0], end_date=ground_truth[1]
            )

            doc = genai.types.Part.from_bytes(
                data=read_bytes(path),
                mime_type="application/pdf",
            )
            doc_id = filepath
            # create a new input data object
            input_data = InputData(doc_id=doc_id, doc=doc, ground_truth=ground_truth)
            input_list.append(input_data)
        return input_list

    def get_input_data_for_scope3_assurance(self) -> List[InputData]:
        # self.copy_gt_file_gsutil()
        common_filepaths, df_ground_truth = self.get_overlapping_filepaths_csv_gcp()
        input_list = []

        for filepath in common_filepaths:
            # print(f"\nWorking with file: {filepath}")

            # Display ground truth values
            gt_entry = df_ground_truth.loc[
                df_ground_truth["file_path"] == filepath
            ].iloc[0]
            # drop part files and not 2023 files.
            if gt_entry.reporting_year != 2023:
                continue
            filename = filepath.split("/")[-1]
            if "_part" in filename:
                continue
            # print(f"* Ground truth. reporting_year: {gt_entry.reporting_year}, scope1_emissions: {gt_entry.scope1_emissions}, scope1_comments: {gt_entry.scope1_comments}")
            # we will use filepath as unique id.
            path = f"gs://{self.bucket_name_pdf}/{filepath}"
            file_size = CloudPath(cloud_path=path).stat().st_size
            # skip if file size is bigger than 40 mb.
            if file_size > 48 * 1024 * 1024:
                continue

            has_assurance = gt_entry.has_scope_3_assurance
            ground_truth = None

            if not pd.isna(has_assurance):
                ground_truth = bool(has_assurance)
            ground_truth = Scope3EmissionsAssuranceSchema(
                has_scope_3_assurance=ground_truth
            )
            doc = genai.types.Part.from_bytes(
                data=read_bytes(path),
                mime_type="application/pdf",
            )
            doc_id = filepath
            # create a new input data object
            input_data = InputData(doc_id=doc_id, doc=doc, ground_truth=ground_truth)
            input_list.append(input_data)
        return input_list

    def get_input_data_for_scope1_emissions_with_filesizes(
        self,
    ) -> Tuple[List[InputData], Any, Any]:
        # self.copy_gt_file_gsutil()
        common_filepaths, df_ground_truth = self.get_overlapping_filepaths_csv_gcp()
        input_list = []
        file_sizes = {}
        pages = {}

        import io
        from PyPDF2 import PdfReader

        for filepath in common_filepaths:
            # print(f"\nWorking with file: {filepath}")
            # Display ground truth values
            gt_entry = df_ground_truth.loc[
                df_ground_truth["file_path"] == filepath
            ].iloc[0]
            # drop part files and not 2023 files.
            if gt_entry.reporting_year != 2023:
                continue
            filename = filepath.split("/")[-1]
            if "_part" in filename:
                continue
            # print(f"* Ground truth. reporting_year: {gt_entry.reporting_year}, scope1_emissions: {gt_entry.scope1_emissions}, scope1_comments: {gt_entry.scope1_comments}")
            # we will use filepath as unique id.
            path = f"gs://{self.bucket_name_pdf}/{filepath}"
            file_size = CloudPath(cloud_path=path).stat().st_size
            # skip if file size is bigger than 40 mb.
            if file_size > 48 * 1024 * 1024:
                continue
            ground_truth = float(gt_entry.scope1_emissions)
            if pd.isna(ground_truth):
                ground_truth = None
            ground_truth = AbsoluteScope1EmissionsSchema(
                total_scope1_emissions=ground_truth
            )
            dt = read_bytes(path)
            doc = genai.types.Part.from_bytes(
                data=dt,
                mime_type="application/pdf",
            )
            doc_id = filepath
            # create a new input data object
            input_data = InputData(doc_id=doc_id, doc=doc, ground_truth=ground_truth)
            input_list.append(input_data)
            reader = PdfReader(io.BytesIO(dt))
            num_pages = len(reader.pages)
            pages[doc_id] = num_pages
            file_sizes[doc_id] = file_size
        return input_list, file_sizes, pages
