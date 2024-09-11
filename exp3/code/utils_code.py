
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def initialize_openai_creds():
    """Load environment variables and set API keys."""
    dotenv_path = find_dotenv()
    if dotenv_path == "":
        print("No .env file found. Make sure the .env file is in the correct directory.")
    else:
        print(f".env file found at: {dotenv_path}")

    load_dotenv(dotenv_path)

    # General Azure OpenAI settings for gpt35 and gpt-4o-mini
    general_creds = {
        "api_key": os.getenv('AZURE_OPENAI_API_KEY'),
        "api_version": os.getenv("AZURE_API_VERSION"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "temperature": 0,  # Default temperature for models
        "gpt35_deployment_name": os.getenv("AZURE_DEPLOYMENT_NAME"),
        "gpt4o_mini_deployment_name": os.getenv("GPT4O_MINI_DEPLOYMENT_NAME")
    }

    # GPT-4o specific settings
    gpt4o_creds = {
        "api_key": os.getenv('GPT4O_API_KEY'),
        "api_version": os.getenv("GPT4O_API_VERSION"),
        "endpoint": os.getenv("GPT4O_AZURE_ENDPOINT"),
        "deployment_name": os.getenv("GPT4O_DEPLOYMENT_NAME"),
        "temperature": os.getenv("GPT4O_TEMPERATURE", 0)  # Default temperature for GPT-4o
    }

    return general_creds, gpt4o_creds



def initialize_openai_creds():
    """Load environment variables and set API keys."""
    dotenv_path = find_dotenv()
    if dotenv_path == "":
        print("No .env file found. Make sure the .env file is in the correct directory.")
    else:
        print(f".env file found at: {dotenv_path}")

    load_dotenv(dotenv_path)

    # GPT-3.5 Credentials
    gpt35_creds = {
        "api_key": os.getenv('AZURE_OPENAI_API_KEY_GPT35'),
        "api_version": os.getenv("AZURE_API_VERSION"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT_GPT35"),
        "temperature": 0,  # Default temperature for models
        "deployment_name": os.getenv("AZURE_DEPLOYMENT_NAME_GPT35")
    }

    # GPT-4o-mini Credentials (shares the same API key as GPT-3.5 but different deployment name and endpoint)
    gpt4o_mini_creds = {
        "api_key": os.getenv('AZURE_OPENAI_API_KEY_GPT4O_MINI'),
        "api_version": os.getenv("AZURE_API_VERSION"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT_GPT4O_MINI"),
        "temperature": 0,  # Default temperature for models
        "deployment_name": os.getenv("GPT4O_MINI_DEPLOYMENT_NAME")
    }

    # GPT-4o specific credentials
    gpt4o_creds = {
        "api_key": os.getenv('GPT4O_API_KEY'),
        "api_version": os.getenv("GPT4O_API_VERSION"),
        "endpoint": os.getenv("GPT4O_AZURE_ENDPOINT"),
        "deployment_name": os.getenv("GPT4O_DEPLOYMENT_NAME"),
        "temperature": os.getenv("GPT4O_TEMPERATURE", 0)  # Default temperature for GPT-4o
    }

    return gpt35_creds, gpt4o_mini_creds, gpt4o_creds



def create_llm(model: str, gpt35_creds: dict, gpt4o_mini_creds: dict, gpt4o_creds: dict):
    """
    Initialize and return the Azure OpenAI LLM based on the selected model.
    
    :param model: The model to initialize ("gpt35", "gpt4o", or "gpt-4o-mini").
    :param gpt35_creds: Credentials for gpt35.
    :param gpt4o_mini_creds: Credentials for gpt-4o-mini.
    :param gpt4o_creds: Credentials for gpt4o.
    """
    if model == "gpt35":
        return AzureOpenAI(
            deployment_name=gpt35_creds["deployment_name"],
            temperature=gpt35_creds["temperature"],
            api_key=gpt35_creds["api_key"],
            azure_endpoint=gpt35_creds["endpoint"],
            api_version=gpt35_creds["api_version"]
        )
    elif model == "gpt-4o-mini":
        return AzureOpenAI(
            deployment_name=gpt4o_mini_creds["deployment_name"],
            temperature=gpt4o_mini_creds["temperature"],
            api_key=gpt4o_mini_creds["api_key"],
            azure_endpoint=gpt4o_mini_creds["endpoint"],
            api_version=gpt4o_mini_creds["api_version"]
        )
    elif model == "gpt4o":
        return AzureOpenAI(
            deployment_name=gpt4o_creds["deployment_name"],
            temperature=gpt4o_creds["temperature"],
            api_key=gpt4o_creds["api_key"],
            azure_endpoint=gpt4o_creds["endpoint"],
            api_version=gpt4o_creds["api_version"]
        )
    else:
        raise ValueError(f"Invalid model: {model}. Choose from 'gpt35', 'gpt4o', or 'gpt-4o-mini'.")

    
    
def create_chat_engine(retriever, memory, llm):
    """Create and return the ContextChatEngine using the provided retriever and memory."""
    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        llm=llm
    )
    return chat_engine


def load_documents(filepaths):
    """
    Load and return documents from specified file paths.
    
    :param filepaths: A string (single file path) or a list of strings (multiple file paths).
    :return: A list of loaded documents.
    """
    loader = PyMuPDFReader()

    # If a single string is passed, convert it to a list for consistent handling
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    # Load and accumulate documents
    all_documents = []
    for filepath in filepaths:
        documents = loader.load(file_path=filepath)
        all_documents += documents

    return all_documents


def create_kg_index(
    documents,
    storage_context,
    llm,
    max_triplets_per_chunk=10,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5"),
    include_embeddings=True,
    chunk_size=512
):
    splitter = SentenceSplitter(chunk_size=chunk_size)
    graph_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=max_triplets_per_chunk,
        llm=llm,
        embed_model=embed_model,
        include_embeddings=include_embeddings,
        transformations=[splitter]
    )
    return graph_index


from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex


def create_pg_index(
    llm,
    documents,
    graph_store,
    max_triplets_per_chunk=10,
    num_workers=4,
    embed_kg_nodes=True,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
):
    
    splitter = SentenceSplitter(chunk_size=512)
    # Initialize the LLM path extractor
    kg_extractor = DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=max_triplets_per_chunk,
        num_workers=num_workers
    )


    # Create the Property Graph Index
    graph_index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        embed_model=embed_model,
        embed_kg_nodes=embed_kg_nodes,
        kg_extractors=[kg_extractor],
        transformations=[splitter]
    )

    return graph_index