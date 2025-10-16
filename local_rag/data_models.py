from pydantic_core import PydanticCustomError
from pydantic import BaseModel, field_validator


class ValidateVectorDBVars(BaseModel):
    db_location: Path
    embedding_model: str
    collection_name: str
    csv_path: Path
    num_docs_to_retrieve: int

    @field_validator("csv_path")
    def validate_csv_path(cls, value: Path):
        if not value.exists():
            raise PydanticCustomError(
                "path_doesn't_exist_error",
                "CSV path doesn't exist.",
                {"path": value}
            )