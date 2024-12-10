"""Sample endpoints to IPA using FastAPI."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from aif.ipa import Request, Response, ipa


class SampleModel(BaseModel):
    """Model class for sample endpoint."""

    my_status: int = Field(alias="my sample endpoint status")

    @field_validator("my_status", mode="before")
    @classmethod
    def validate_my_status(cls, my_status: int) -> int:
        """Validate my_status."""
        if my_status != 200:
            raise ValueError('"my_status" not equal 200!!!.')
        else:
            return my_status


@ipa.app.get("/my_ipa_sample_endpoint/{status}", response_model=SampleModel)
@ipa.unauthenticated
async def my_ipa_sample_endpoint(request: Request, response: Response, status: int) -> Any:
    """Get status."""
    return SampleModel(**{"my sample endpoint status": status})


@ipa.app.get("/hello_world")
@ipa.unauthenticated
async def hello_world(request: Request, response: Response) -> Any:
    """Be greeted."""
    return {"hello": "world"}
