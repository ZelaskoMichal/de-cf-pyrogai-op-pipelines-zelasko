"""Basic IPA tests.

This file contains basic IPA tests that are marked using the `ipa` marker.

* Tests here are not executed during normal pytest runs, as the `ipa` marker is excluded by default:
  have a look at the `pytest.ini` file at the root of your repository
* Tests here are executed on CI/CD when the IPA deployment appears to be up and running.

All tests on IPA are supposed to be able to access the `IPA_URL_PREFIX` environment variable, that
contains the name of the IPA URL prefix that you can access.
"""
from __future__ import annotations

import os

import pytest
import requests


@pytest.fixture(scope="module")
def ipa_url_prefix() -> str:
    """Read IPA URL from env var and strip trailing '/' if exists."""
    url = os.environ["IPA_URL_PREFIX"]
    return url.removesuffix("/")


@pytest.fixture(scope="module")
def headers() -> dict[str, str] | None:
    """Request headers."""
    if gcp_id_token := os.environ.get("GOOGLE_ID_ACCESS_TOKEN"):
        return {"Authorization": f"Bearer {gcp_id_token}"}
    return None


@pytest.mark.ipa
def test_livez(ipa_url_prefix: str, headers: dict[str, str]) -> None:
    """Test the /livez endpoint."""
    try:
        resp = requests.get(f"{ipa_url_prefix}/livez", headers=headers)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        pytest.fail(msg=str(e), pytrace=False)

    assert "alive" in resp.json()["message"]
