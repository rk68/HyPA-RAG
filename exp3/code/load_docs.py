import argparse
from utils_code import load_documents, create_kg_index, create_pg_index
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv
import os
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.node_parser import SentenceSplitter

# Paths for different corpora
LL144_path = ["../../legal_data/LL144/LL144.pdf", "../../legal_data/LL144/LL144_Definitions.pdf"]
EUAIACT_path = ["../../legal_data/EU_AI_ACT/EUAIACT.pdf"]

# Load environment variables for Neo4j, Pinecone, etc.
dotenv_path = find_dotenv()
print(f"Dotenv Path: {dotenv_path}")
load_dotenv(dotenv_path)

# Initialize environment variables for storage contexts like Pinecone or Neo4j
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_url = os.getenv("NEO4J_URI")
neo4j_database = os.getenv("NEO4J_DATABASE")

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("GSK_AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("GSK_AZURE_OPENAI_ENDPOINT")
os.environ["GSK_LLM_MODEL"] = "gpt-4o-mini"

llm_gpt35 = AzureOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"), temperature=0, 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_version=os.getenv("AZURE_API_VERSION")
)

llm_gpt4o_mini = AzureOpenAI(
    deployment_name="gpt-4o-mini", temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION")
)

llm_gpt4o = AzureOpenAI(
    deployment_name="gpt4o", temperature=0,
    api_key=os.getenv("GPT4O_API_KEY"), azure_endpoint=os.getenv("GPT4O_AZURE_ENDPOINT"),
    api_version=os.getenv("GPT4O_API_VERSION")
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
ll144_index_name = 'll144'
euaiact_index_name = 'euaiact'

pc = Pinecone(api_key=pinecone_api_key)

# Main function to handle index creation and storage
def main(args):
    # Select the corpus to load based on user input
    if args.corpus == "LL144":
        documents = load_documents(LL144_path)
        pinecone_index_name = ll144_index_name
    elif args.corpus == "EUAIACT":
        documents = load_documents(EUAIACT_path)
        pinecone_index_name = euaiact_index_name
    else:
        raise ValueError("Unknown corpus specified. Please choose either 'LL144' or 'EUAIACT'.")

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    splitter = SentenceSplitter(chunk_size=512)

    # Based on the index type, create the appropriate index and store it in the specified database
    if args.index_type == "vector":
        # Create vector index
        
        pinecone_index = pc.Index(pinecone_index_name)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model,storage_context=storage_context, transformations=[splitter])
        print(f"Vector index stored in Pinecone at index '{pinecone_index_name}'")


    elif args.index_type == "kg":
        # Knowledge Graph index creation
        if args.database == "neo4j":
            # Neo4j graph store for KG
            graph_store = Neo4jPropertyGraphStore(username=neo4j_username, password=neo4j_password, url=neo4j_url)
            storage_context = StorageContext.from_defaults(graph_store=graph_store)
            kg_index = create_kg_index(documents=documents, storage_context=storage_context, llm=llm_gpt4o_mini)  # LLM can be passed
            print("Knowledge Graph index stored in Neo4j.")
        else:
            raise ValueError("Unsupported database for KG index. Use 'neo4j'.")

    elif args.index_type == "pg":
        # Property Graph index creation
        if args.database == "neo4j":
            # Neo4j property graph store for PG
            graph_store = Neo4jPropertyGraphStore(username=neo4j_username, password=neo4j_password, url=neo4j_url)
            pg_index = create_pg_index(documents=documents, graph_store=graph_store, llm=llm_gpt4o)  # LLM can be passed
            print("Property Graph index stored in Neo4j.")
        else:
            raise ValueError("Unsupported database for PG index. Use 'neo4j'.")

    else:
        raise ValueError("Unsupported index type. Use 'vector', 'kg', or 'pg'.")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Create and store indices for specified corpora.")
    parser.add_argument("--corpus", type=str, required=True, help="Corpus name: either 'LL144' or 'EUAIACT'.")
    parser.add_argument("--index_type", type=str, required=True, choices=["vector", "kg", "pg"], 
                        help="Type of index to create: 'vector', 'kg' (Knowledge Graph), or 'pg' (Property Graph).")
    parser.add_argument("--database", type=str, required=False, choices=["neo4j", "pinecone"],
                        help="Database to store the index: 'neo4j' or 'pinecone'.")

    args = parser.parse_args()
    main(args)
