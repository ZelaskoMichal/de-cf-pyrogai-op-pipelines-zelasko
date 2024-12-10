"""This module provides functionality for sending email notifications."""

import argparse
import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Union

logging.basicConfig(level=logging.INFO)


def read_config():
    """Read configuration data from a JSON file and returns it as a dictionary."""
    with open("./email_config.json", "r") as infile:
        config = json.load(infile)
    return config


class EmailSender:
    """Email Sender class."""

    def __init__(
        self,
        report: str,
        server: str,
        port: int,
        user_name: str,
        password: str,
        recipients: list[str],
        subject: str,
    ) -> None:
        """Initialize an EmailSender instance with the provided configuration."""
        self.report = report
        self.server = server
        self.port = port
        self.user_name = user_name
        self.password = password
        self.recipients = recipients
        self.subject = subject
        self.msg = self._create_message()

    def _create_message(self) -> MIMEMultipart:
        """Create an email message formatted with the sender's details, subject, and report content."""
        msg = MIMEMultipart()
        msg["From"] = self.user_name
        msg["To"] = ",".join(self.recipients)
        msg["Subject"] = self.subject
        msg.attach(MIMEText(self.report, "plain"))

        return msg

    def send_email(self):
        """Send an email."""
        try:
            with smtplib.SMTP(self.server, self.port) as server:
                server.starttls()
                server.login(self.user_name, self.password)
                server.send_message(self.msg)
            logging.info("Email sent successfully!")
        except Exception as e:
            logging.error(f"Failed to send an email: {e}")


def main(
    username: str,
    password: str,
    pr_url: str,
    workflows: str,
    config: Dict[str, Union[str, int]]
) -> None:
    """Processes workflow information and sends an email with the results."""
    workflow_list = workflows.split(",")
    formatted_workflow_str = "\n".join([f"- {workflow.strip()}" for workflow in workflow_list])
    email_sender = EmailSender(
        report=config["body"].format(pr_url, formatted_workflow_str),
        server=config["server"],
        port=config["port"],
        user_name=username,
        password=password,
        recipients=config["recipients"],  # TODO Need to update it,
        subject=config["subject"],
    )
    email_sender.send_email()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--pr-url")
    parser.add_argument("--workflows")
    args = parser.parse_args()

    config = read_config()
    main(args.username, args.password, args.pr_url, args.workflows, config)
