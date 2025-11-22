import pandas as pd
from pathlib import Path

from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator


class ValidateCommonArgs(BaseModel):
    db_location: Path
    collection_name: str

    @field_validator("db_location")
    def validate_db_exists(cls, value: Path):
        if not value.exists():
            raise PydanticCustomError(
                "db_doesn't_exist_error",
                "DB path doesn't exist.",
                {"path": value}
            )


class ValidateQueryArgs(ValidateCommonArgs):
    num_results_to_retrieve: int


class ValidateImageDBArgs(ValidateCommonArgs):
    data_csv_path: Path

    @field_validator("data_csv_path")
    def validate_dataset_exists(cls, value: Path):
        if not value.exists():
            raise PydanticCustomError(
                "dataset_doesn't_exist_error",
                "Dataset path doesn't exist.",
                {"path": value}
            )
    

class ValidateDataDF(BaseModel):
    data_df: pd.DataFrame

    @field_validator("data_df")
    def validate_data_df_cols(cls, value: pd.DataFrame):
        if not list(value.columns) == ["id", "image path", "image description"]:
            raise PydanticCustomError(
                "invalid_dataset_columns_error",
                "Dataset doesn't contain the expected columns.",
                {"dataset columns": list(value.columns)}
            )
