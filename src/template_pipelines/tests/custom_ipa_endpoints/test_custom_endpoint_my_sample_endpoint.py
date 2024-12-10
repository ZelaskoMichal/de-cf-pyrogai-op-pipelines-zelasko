"""Test for custom_endpoint_my_schedule_endpoint."""

import json
from io import StringIO

import pytest
from fastapi.testclient import TestClient

from aif.ipa import IPA
from aif.ipa.config import IpaConfig

SECRET = "fake_secret"
CONFIG_DICT = {
    "backends": {
        "azureml2": {
            "azure": {
                "tenant_id": "fake_tenant",
                "service_principal": {"sp_id": "fake_sp_id", "sp_pass": "fake_sp_pass"},
                "subscription_id": "fake_sub_id",
                "resource_group": "fake_group",
                "workspace": "fake_ws",
                "include_published_pipelines": ["fake_pipeline"],
            }
        }
    }
}


with StringIO(json.dumps(CONFIG_DICT)) as fp:
    config = IpaConfig(parameters_fp=fp)

ipa = IPA(
    api_secret_key=SECRET,
    host="localhost",
    port=8080,
    config=config,
    test=True,
    log_level=None,
    auto_reload=None,
)
client = TestClient(ipa.app)


def test_my_ipa_sample_endpoint():
    """Test my_ipa_sample_endpoint endpoint."""
    from template_pipelines.custom_ipa_endpoints import custom_endpoint_my_sample_endpoint

    response = client.get("/my_ipa_sample_endpoint/200")
    assert response.status_code == 200
    assert custom_endpoint_my_sample_endpoint.SampleModel.model_validate(response.json())

    with pytest.raises(ValueError):
        client.get("/my_ipa_sample_endpoint/222")


def test_hello_world():
    """Test hello_world endpoint."""
    from template_pipelines.custom_ipa_endpoints import custom_endpoint_my_sample_endpoint  # noqa

    response = client.get("/hello_world")
    assert response.status_code == 200
    assert response.json() == {"hello": "world"}
