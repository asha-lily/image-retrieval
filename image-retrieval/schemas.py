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

    