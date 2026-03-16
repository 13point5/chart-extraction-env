from pydantic import BaseModel, ConfigDict


class NormalizedBaseModel(BaseModel):
    """Base model that strips surrounding whitespace from all strings once."""

    model_config = ConfigDict(str_strip_whitespace=True)
