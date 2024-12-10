"""Resuable methods and classes."""

import logging
from datetime import datetime
from functools import wraps

from azure.identity import ChainedTokenCredential, DefaultAzureCredential, EnvironmentCredential
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever

logging.basicConfig(level=logging.INFO)


def create_chat_prompt(system_prompt, user_prompt):
    """Creates a chat prompt from system and user prompts."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
    user_message_prompt = HumanMessagePromptTemplate.from_template(user_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, user_message_prompt])
    return chat_prompt


def get_azure_token(scope):
    """Get Azure access token within a specific scope."""
    credential_chain = ChainedTokenCredential(
        EnvironmentCredential(),
        DefaultAzureCredential(exclude_interactive_browser_credential=False),
    )
    return credential_chain.get_token(scope)


class AzureOpenAIClient:
    """Define a customized azure openai class."""

    def __init__(self, openai_settings):
        """Initialize AzureOpenAIClient class."""
        self.genai_proxy = openai_settings["genai_proxy"]
        self.cognitive_services = openai_settings["cognitive_services"]
        self.open_api_version = openai_settings["open_api_version"]
        self.headers = openai_settings["headers"]

        token = get_azure_token(self.cognitive_services)
        self.token_str = token.token
        self.token_expiration = datetime.fromtimestamp(token.expires_on)
        self.token_refresh_thre = openai_settings["token_refresh_thre"]

        self.chat_specs = {
            "chat_engine": openai_settings["chat_engine"],
            "temperature": openai_settings["temperature"],
        }
        self.embedding_specs = {"embedding_engine": openai_settings["embedding_engine"]}
        self.initialize_genai_models()

    def initialize_genai_models(self):
        """Initialize GenAI model objects."""
        self.chat_model = AzureChatOpenAI(
            azure_endpoint=self.genai_proxy,
            azure_deployment=self.chat_specs["chat_engine"],
            api_version=self.open_api_version,
            api_key=self.token_str,
            temperature=self.chat_specs["temperature"],
            default_headers=self.headers,
        )
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=self.genai_proxy,
            deployment=self.embedding_specs["embedding_engine"],
            api_version=self.open_api_version,
            api_key=self.token_str,
            default_headers=self.headers,
        )

    def refresh_access_token(f):  # noqa: N805
        """A wrapper function to refresh access token."""

        @wraps(f)
        def wrapper(self, *args, **kargs):
            current_time = datetime.now()
            time_difference = (self.token_expiration - current_time).total_seconds()
            if time_difference < self.token_refresh_thre:
                logging.info("Refreshing Azure access token...")
                token = get_azure_token(self.cognitive_services)
                self.token_str = token.token
                self.token_expiration = datetime.fromtimestamp(token.expires_on)
                self.initialize_genai_models()
            return f(self, *args, **kargs)

        return wrapper

    @refresh_access_token  # type: ignore
    def get_chat_response(self, messages):
        """Get a chat response."""
        messages = [
            SystemMessage(content=messages["system"]),
            HumanMessage(content=messages["human"]),
        ]
        return self.chat_model(messages).content

    @refresh_access_token  # type: ignore
    def get_embedding(self, document_list):
        """Get a list of embeddings."""
        return self.embedding_model.embed_documents(document_list)


class VectoreStoreRetrieverFilter(VectorStoreRetriever):
    """Define a customized vectorstore retriever with a filtering option."""

    def _get_relevant_documents2(self, query, _filter=None):
        """Search for relevant documents with optional filtering functionality."""
        if _filter is not None:
            docs = self.vectorstore.similarity_search(
                query, filter=_filter, **self.search_kwargs, fetch_k=30
            )
        else:
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        return docs

    def _get_relevant_documents(self, _dict):
        """Override _get_relevant_documents method with optional functionalities."""
        return self._get_relevant_documents2(**_dict)
