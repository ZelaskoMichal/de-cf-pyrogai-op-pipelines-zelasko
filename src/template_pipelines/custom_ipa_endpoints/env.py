"""Output the environment name."""

import os
from typing import Any

from aif.ipa import Request, Response, ipa


@ipa.app.get("/environment")
@ipa.unauthenticated
async def environment(request: Request, response: Response) -> Any:
    """Output environment name."""
    return {"deployment_name": os.environ.get("IPA_DEPLOYMENT_NAME", "<unknown>")}
