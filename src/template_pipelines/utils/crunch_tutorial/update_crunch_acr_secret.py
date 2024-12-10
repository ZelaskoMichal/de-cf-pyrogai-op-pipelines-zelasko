"""This script updates CRUNCH-ACR secret in your secrets.yaml file.

Functions:
    load_secrets(file_path): Loads secrets from a JSON file.
    save_secrets(file_path, secrets): Saves secrets to a JSON file.
    validate_secret(secrets_json_path): Validates the CRUNCH-ACR secret in the secrets.json file.
    main(): Main function that loads secrets, generates CRUNCH-ACR secret, and saves the updated secrets.

Usage:
    Fill in path to your secrets.json file in the secrets_json_path variable.
    Run this script directly to execute the main function.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from nas_crunch_client import Crunch


def load_secrets(file_path):
    """Load secrets from a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")  # noqa
        raise
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from the file {file_path}.")  # noqa
        raise


def save_secrets(file_path, secrets):
    """Save secrets to a JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(secrets, f, indent=4)
    except IOError:
        print(f"Error: Failed to write to the file {file_path}.")  # noqa
        raise


def validate_secret(secrets_json_path):
    """Validate the CRUNCH-ACR secret in the secrets.json file."""
    secrets = load_secrets(secrets_json_path)
    now = datetime.now(timezone.utc)
    creation_time = datetime.fromisoformat(secrets["CRUNCH-ACR"]["token"]["creation_time"])
    if abs((now - creation_time).total_seconds()) <= 3600:
        print("CRUNCH-ACR secret is valid.")  # noqa
    else:
        print(  # noqa
            "CRUNCH-ACR secret is not valid, "
            "check the creation time and expiry of CRUNCH-ACR secret in your secrets.json file."
        )


def main(secrets_json_path):
    """Main function that loads secrets, generates CRUNCH-ACR secret, and saves the updated secrets."""
    secrets = load_secrets(secrets_json_path)

    user_id = secrets["CRUNCH-APP-USER-NAME"]
    secret = secrets["CRUNCH-APP-USER-SECRET"]
    crunch_url = secrets["CRUNCH-URL"]

    crunch = Crunch.from_app_credentials(url=crunch_url, user_id=user_id, secret=secret)

    secrets["CRUNCH-ACR"] = crunch.generate_registry_token()

    save_secrets(secrets_json_path, secrets)
    validate_secret(secrets_json_path)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "crunch_tutorial" and script_dir.parent.name == "utils":
        secrets_json_path = script_dir.parent.parent / "config" / "secrets.json"
    else:
        parser = argparse.ArgumentParser(
            description="Update CRUNCH-ACR secret in your secrets.json file."
        )
        parser.add_argument("secrets_json_path", type=str, help="Path to the secrets.json file")
        args = parser.parse_args()
        secrets_json_path = Path(args.secrets_json_path)

    main(secrets_json_path)
