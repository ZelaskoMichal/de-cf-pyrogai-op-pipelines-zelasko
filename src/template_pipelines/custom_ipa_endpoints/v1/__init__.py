"""Add scoped endpoints to IPA using FastAPI."""

from typing import Any

from aif.ipa import Request, Response, ipa

scopedapp = ipa.scopedapp()


@scopedapp.get("/hi")
@ipa.unauthenticated
async def hi(request: Request, response: Response) -> Any:
    """Get a useless JSON."""
    return {"hi": "world"}
