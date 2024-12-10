import json
import os
from pathlib import Path

import requests

API_URL = os.environ.get('API_URL')
API_CI_ACCESS_KEY = os.environ.get('API_CI_ACCESS_KEY')


def send_request(url, json_file_path):
    """Module called from action '.github/workflows/update_aiapps_db_from_index_file.yaml'.

    Send the request to API with index.json file as a body
    """
    # Read the content of index.json file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Access-Key": API_CI_ACCESS_KEY,
    }

    # Make a POST request with the JSON data
    response = requests.post(url, headers=headers, json=json_data)
    response.raise_for_status()


json_file_path = Path(os.getcwd()).resolve() / 'index.json'
send_request(API_URL, json_file_path)
