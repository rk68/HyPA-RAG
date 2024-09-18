import argparse
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from giskard.rag import KnowledgeBase, QATestset, AgentAnswer, evaluate
from giskard.rag.metrics.ragas_metrics import ragas_faithfulness, ragas_answer_relevancy
from utils_code import create_kg_index, create_pg_index, load_documents, create_chat_engine
from retrievers import HyPARetriever, PARetriever, FixedParamRetriever
from correctness_eval import process_correctness_scores
from metrics import compute_metrics
import pandas as pd
from urllib.parse import urlparse
from giskard.llm import set_llm_model, set_llm_api
from giskard.llm.client import get_default_client
from giskard.llm.embeddings import try_get_fastembed_embeddings, set_default_embedding, get_default_embedding
from giskard.llm.embeddings.fastembed import FastEmbedEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Load environment variables from the .env file
dotenv_path = find_dotenv()
print(f"Dotenv Path: {dotenv_path}")
load_dotenv(dotenv_path)

# Remove OPENAI_API_KEY if it exists to avoid conflicts
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# Set API keys for Azure and OpenAI services
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("GSK_AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("GSK_AZURE_OPENAI_ENDPOINT")
os.environ["GSK_LLM_MODEL"] = "gpt-4o-mini"

username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
url = os.getenv("NEO4J_URI")
database = os.getenv("NEO4J_DATABASE")

from fastembed import TextEmbedding
text_embedding=TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embedding_model = FastEmbedEmbedding(text_embedding)
set_default_embedding(embedding_model)

set_llm_api("azure")
set_llm_model(os.getenv("GSK_LLM_MODEL"))


# Initialize the HuggingFace embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.embed_model = embed_model

# Init Giskard Client
client = get_default_client()

# Normalize URLs and ensure the correct Azure configuration
actual_base_url = urlparse(str(client._client._base_url)).netloc
expected_base_url = urlparse(os.getenv("AZURE_OPENAI_ENDPOINT")).netloc
assert actual_base_url == expected_base_url, f"Base URL mismatch: {actual_base_url} != {expected_base_url}"
assert client._client.api_key == os.getenv("AZURE_OPENAI_API_KEY"), "API Key mismatch"
assert client._client._api_version == os.getenv("AZURE_API_VERSION"), "API Version mismatch"


# Step 1: Create the LLM based on environment variables
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

Settings.llm = llm_gpt35


# Function to initialize indices once
def initialize_indices(documents, kg_index=False, property_index=False):
    splitter = SentenceSplitter(chunk_size=512)
    
    # Create vector index
    vector_index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model, transformations=[splitter])
    
    if kg_index:
        graph_store = Neo4jGraphStore(username=username, password=password, url=url, database=database)
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        graph_index = create_kg_index(documents=documents, storage_context=storage_context, llm=llm_gpt35, max_triplets_per_chunk=10)
    elif property_index:
        graph_store = Neo4jPropertyGraphStore(username=username, password=password, url=url)
        graph_index = create_pg_index(documents=documents, graph_store=graph_store, llm=llm_gpt35, max_triplets_per_chunk=10, num_workers=4, embed_kg_nodes=True)
    else:
        graph_index = None

    return vector_index, graph_index

# Function to initialize retrievers
def initialize_retriever(retriever_type, llm, vector_retriever, bm25_retriever, kg_index=None, property_index=False,
                         mode="OR", rewriter=True, classifier_model=None, device='mps', reranker_model_name=None, 
                         verbose=False, fixed_params=None, categories_list=None, param_mappings=None, top_k=3, num_query_rewrites=3):

    # @TODO: Complete this for the fixed parameter retriever
    if retriever_type == "Fixed":
        return FixedParamRetriever(
            # Add all parameters here
            llm=llm, vector_retriever=vector_retriever, bm25_retriever=bm25_retriever,
            rewriter=rewriter, device=device, reranker_model_name=reranker_model_name,
            top_k=top_k, num_query_rewrites=num_query_rewrites
        )

    if retriever_type == "HyPA":
        return HyPARetriever(
            llm=llm, vector_retriever=vector_retriever, bm25_retriever=bm25_retriever,
            kg_index=kg_index, property_index=property_index, mode=mode, rewriter=rewriter,
            classifier_model=classifier_model, device=device, reranker_model_name=reranker_model_name,
            verbose=verbose, fixed_params=fixed_params, categories_list=categories_list,
            param_mappings=param_mappings
        )
    elif retriever_type == "PA":
        return PARetriever(
            llm=llm, vector_retriever=vector_retriever, bm25_retriever=bm25_retriever,
            mode=mode, rewriter=rewriter, classifier_model=classifier_model, device=device,
            reranker_model_name=reranker_model_name, verbose=verbose, fixed_params=fixed_params,
            categories_list=categories_list, param_mappings=param_mappings
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

# Function to run the evaluation with different retriever configurations
def run_evaluation(results_base_path: str, vector_index, graph_index, retriever_type: str = "HyPA", test_set_path: str = "../giskard_test_sets/LL144_275_New.jsonl", 
                   rewriter: bool = False, classifier_model: str = "rk68/distilbert-q-classifier-3", verbose: bool = False, property_index: bool = False, 
                   mode: str = "OR", device: str = 'mps', fixed_params: dict = None, categories_list: list = None, param_mappings: dict = None, top_k: int=5, num_query_rewrites: int=3,
                   reranker_model_name=None):
    
    # Initialize LLM and retrievers
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(index=vector_index, similarity_top_k=10)
    
    retriever = initialize_retriever(retriever_type=retriever_type, llm=llm_gpt35, vector_retriever=vector_retriever, 
                                     bm25_retriever=bm25_retriever, kg_index=graph_index, property_index=property_index, 
                                     mode=mode, rewriter=rewriter, classifier_model=classifier_model, 
                                     device=device, fixed_params=fixed_params, categories_list=categories_list, param_mappings=param_mappings,
                                     top_k=top_k, num_query_rewrites=num_query_rewrites, reranker_model_name=reranker_model_name)

    # Initialize Chat Engine
    memory = ChatMemoryBuffer.from_defaults(token_limit=8192)
    chat_engine = create_chat_engine(llm=llm_gpt35, retriever=retriever, memory=memory)

    # Prepare knowledge base
    splitter = SentenceSplitter(chunk_size=512)
    text_nodes = splitter(vector_index.docstore.docs.values())
    knowledge_base_df = pd.DataFrame([node.text for node in text_nodes], columns=['text'])
    knowledge_base = KnowledgeBase(knowledge_base_df)

    # Define functions to process evaluation
    def answer_fn(question, history=None):
        chat_history = [ChatMessage(role=MessageRole.USER if msg['role'] == 'user' else MessageRole.ASSISTANT, content=msg['content']) for msg in history] if history else []
        return str(chat_engine.chat(question, chat_history=chat_history))

    def get_answer_fn(question: str, history=None) -> str:
        messages = history if history else []
        messages.append({'role': 'user', 'content': question})
        answer = answer_fn(question, history)
        #print(f"answer: {answer}")
        retrieved_nodes = retriever.retrieve(question)
        documents = [node.node.text for node in retrieved_nodes]
        return AgentAnswer(message=answer, documents=documents)

    # Load test set and run evaluation
    try:
        testset = QATestset.load(test_set_path)
    except ValueError as e:
        print(f"Error loading test set: {e}. (Did you specify the filepath correctly?)")
        return

    results_path = f'{results_base_path}_{retriever_type}'
    report = evaluate(get_answer_fn, testset=testset, knowledge_base=knowledge_base, metrics=[ragas_faithfulness, ragas_answer_relevancy])
    results = report.to_pandas()
    csv_path = results_path + '.csv'
    results.to_csv(csv_path, index=False)

    # Step 7: Process correctness scores
    scores_csv_path = results_path + '_metrics.csv'
    process_correctness_scores(file_path=csv_path, llm=llm_gpt4o_mini, threshold=4.0)

    # Compute final metrics based on the correctness scores CSV
    compute_metrics(file_path=csv_path)

# Main function
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run evaluation with specified retriever and index setup.")
    parser.add_argument("--results_base_path", type=str,  help="Base path to store results.")
    parser.add_argument("--test_set_path", type=str, default="../../giskard_test_sets/LL144_275_New.jsonl", help="Path to the test set file.")
    parser.add_argument("--corpus_paths", type=list, default=["../../eval_and_data/legal_data/EU_AI_ACT"])
    parser.add_argument("--kg_index", action="store_true", help="Use knowledge graph index.")
    parser.add_argument("--property_index", action="store_true", help="Use property graph index.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--mode", type=str, default="OR", help="Mode of operation for retriever (AND/OR).")
    parser.add_argument("--rewriter", action="store_true", help="Enable query rewriting.")
    parser.add_argument("--classifier_model", type=str, default=None, help="Optional classifier model for query classification.")
    parser.add_argument("--device", type=str, default="mps", help="Device for running model (e.g., 'cpu', 'cuda', 'mps').")
    parser.add_argument("--fixed_params", type=dict, default=None, help="Fixed parameters for retriever.")
    parser.add_argument("--categories_list", type=list, default=None, help="List of categories for classification.")
    parser.add_argument("--param_mappings", type=dict, default=None, help="Custom parameter mappings based on classifier labels.")

    args = parser.parse_args()
    #ll144_paths = ["../../eval_and_data/legal_data/LL144/LL144.pdf", "../../eval_and_data/legal_data/LL144/LL144_Definitions.pdf"]
    eu_data_path = ["../../eval_and_data/legal_data/EU_AI_ACT/EUAIACT.pdf"]
    test_path = "../../eval_and_data/giskard_test_sets/EUAIACT/eu_ai_act_test_300_new.jsonl"
    #test_path_ex = "../../eval_and_data/giskard_test_sets/LL144/LL144_Sample.jsonl"


    # Load documents and initialize indices
    documents = load_documents(eu_data_path)#(args.corpus_paths)
    vector_index, graph_index = initialize_indices(documents, kg_index=True, property_index=False)

    # @TODO: Edit to run evals for PA + HyPA + Fixed Param Configs for EU Act

    # Eval Config: Fixed Top-K
    #run_evaluation(results_base_path="eu_fixed_k3", vector_index=vector_index, graph_index=None, retriever_type="Fixed", test_set_path=test_path, top_k=3, rewriter=False)
    #run_evaluation(results_base_path="eu_fixed_k5", vector_index=vector_index, graph_index=None, retriever_type="Fixed", test_set_path=test_path, top_k=5, rewriter=False)
    #run_evaluation(results_base_path="eu_fixed_k7", vector_index=vector_index, graph_index=None, retriever_type="Fixed", test_set_path=test_path, top_k=7, rewriter=False)
    #run_evaluation(results_base_path="eu_fixed_k10", vector_index=vector_index, graph_index=None, retriever_type="Fixed", test_set_path=test_path, top_k=10, rewriter=False)

    #run_evaluation(results_base_path="eu_PA_no_rewriter", vector_index=vector_index, graph_index=None, retriever_type="PA", rewriter=False, test_set_path=test_path)
    #run_evaluation(results_base_path="eu_HyPA_no_rewriter", vector_index=vector_index, graph_index=graph_index, retriever_type="HyPA", rewriter=False, test_set_path=test_path)
    #run_evaluation(results_base_path=args.results_base_path, vector_index=vector_index, graph_index=graph_index, retriever_type="PA", test_set_path=args.test_set_path)

    run_evaluation(results_base_path="eu_PA_reranker_no_rewriter", vector_index=vector_index, graph_index=None, retriever_type="PA", rewriter=False, test_set_path=test_path, reranker_model_name="BAAI/bge-reranker-large")
    run_evaluation(results_base_path="eu_HyPA_reranker_no_rewriter", vector_index=vector_index, graph_index=graph_index, retriever_type="HyPA", rewriter=False, test_set_path=test_path, reranker_model_name="BAAI/bge-reranker-large")

    run_evaluation(results_base_path="eu_PA_reranker", vector_index=vector_index, graph_index=None, retriever_type="PA", rewriter=True, test_set_path=test_path, reranker_model_name="BAAI/bge-reranker-large")
    run_evaluation(results_base_path="eu_HyPA_reranker", vector_index=vector_index, graph_index=graph_index, retriever_type="HyPA", rewriter=True, test_set_path=test_path, reranker_model_name="BAAI/bge-reranker-large")

