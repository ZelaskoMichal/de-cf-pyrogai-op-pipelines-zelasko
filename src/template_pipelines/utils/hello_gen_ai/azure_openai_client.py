"""AzureOpenAIClient class."""

from azure.identity import ChainedTokenCredential, DefaultAzureCredential, EnvironmentCredential
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage


class AzureOpenAIClient:
    """Define a customized azure openai class."""

    def __init__(self, openai_settings, chat_engine, embedding_engine):
        """Initialize AzureOpenAIClient class."""
        self.genai_proxy = openai_settings["genai_proxy"]
        self.cognitive_services = openai_settings["cognitive_services"]
        self.open_api_version = openai_settings["open_api_version"]
        self.headers = openai_settings["headers"]

        credential = ChainedTokenCredential(
            EnvironmentCredential(),
            DefaultAzureCredential(exclude_interactive_browser_credential=False),
        )
        self.token = credential.get_token(self.cognitive_services).token

        self.chat_engine = chat_engine
        self.embedding_engine = embedding_engine

    def get_chat_response(self, messages, temperature=0.2):
        """Get a chat response."""
        chat = AzureChatOpenAI(
            azure_endpoint=self.genai_proxy,
            azure_deployment=self.chat_engine,
            api_version=self.open_api_version,
            api_key=self.token,
            temperature=temperature,
            default_headers=self.headers,
        )

        messages = [
            SystemMessage(content=messages["system"]),
            HumanMessage(content=messages["human"]),
        ]

        return chat(messages).content

    def get_embedding(self, document_list):
        """Get a list of embeddings."""
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=self.genai_proxy,
            deployment=self.embedding_engine,
            api_version=self.open_api_version,
            api_key=self.token,
            default_headers=self.headers,
        )

        return embeddings.embed_documents(document_list)
