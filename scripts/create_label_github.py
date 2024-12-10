"""Creating github labels."""

import json
import os
import random
from pathlib import Path

from github import Auth, Github

USER_GITHUB_TOKEN = os.environ.get('USER_GITHUB_TOKEN')


class GithubWithRepo:
    """Github class."""

    def __init__(self, github: Github, repo_str: str) -> None:
        self._github = github
        self._repo_str = repo_str

    @property
    def github(self) -> Github:
        """github property."""
        return self._github

    @property
    def repo_str(self) -> str:
        """repo_str property."""
        return self._repo_str


class JsonEditor:
    """JsonEditor class."""

    def __init__(
        self, github_with_repo: GithubWithRepo, index_json_path: Path, auto_label_json_path: Path
    ) -> None:
        self.gh_with_repo = github_with_repo
        self.gh_repo = self.gh_with_repo.github.get_repo(self.gh_with_repo.repo_str)
        self.index_json_path = index_json_path
        self.auto_label_json_path = auto_label_json_path

    def _get_labels(self) -> set:
        return set([label.name for label in self.gh_repo.get_labels()])

    def _get_labels_form_auto_label_json(self):
        with open(self.auto_label_json_path, 'r', encoding="utf-8") as file:
            auto_label_json_data = json.load(file)
        pipelines_from_auto_label_json = set(auto_label_json_data["rules"].keys())
        return pipelines_from_auto_label_json

    def _get_unlabeled_pipelines(self) -> list:
        with open(self.index_json_path, 'r', encoding="utf-8") as file:
            json_data = json.load(file)
        pipelines_from_index_json = set(json_data["pipelines"].keys())
        return list(pipelines_from_index_json - self._get_labels_form_auto_label_json())

    def create_new_labels_for_pipelines(self) -> None:
        """Cretes new label using Github cli."""
        for label in self._get_unlabeled_pipelines():
            random_color = f"{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}"
            self.gh_repo.create_label(label, random_color)

    def validate_auto_label_json(self) -> None:
        """Validates if label already exist in github, compared to pipelines in index.js."""
        if self._get_unlabeled_pipelines():
            raise ValueError(
                f"The auto-label.json have missing pipelines: {self._get_unlabeled_pipelines()} from index.js"
            )


auth = Auth.Token(USER_GITHUB_TOKEN)
github = Github(auth=auth)
gh_with_repo = GithubWithRepo(github, "procter-gamble/de-cf-pyrogai-op-pipelines")
json_editor = JsonEditor(
    gh_with_repo,
    Path(os.getcwd()).resolve() / 'index.json',
    Path(os.getcwd()).resolve() / '.github/auto-label.json',
)

json_editor.validate_auto_label_json()
json_editor.create_new_labels_for_pipelines()
