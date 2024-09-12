from prompts import get_classification_prompt, get_query_generation_prompt
from utils_code import initialize_openai_creds, create_llm
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from transformers import pipeline
from typing import List, Optional
import asyncio
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.property_graph import LLMSynonymRetriever
from llama_index.core.indices.property_graph import VectorContextRetriever, PGRetriever
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever
import os

class FixedParamRetriever(BaseRetriever):
    """Custom retriever that performs query rewriting and Vector/BM25 search with fixed parameters."""

    def __init__(
        self,
        llm,  # LLM for query generation
        vector_retriever: Optional[VectorIndexRetriever] = None,
        bm25_retriever: Optional[BaseRetriever] = None,
        top_k: int = 5,  # Fixed top-k
        num_query_rewrites: int = 3,  # Fixed number of query rewrites
        rewriter: bool = True,
        device: str = 'mps',  # Set to 'mps' as the default device
        reranker_model_name: Optional[str] = None,  # Model name for SentenceTransformerRerank
        verbose: bool = False  # Verbose flag
    ) -> None:
        """Initialize FixedParamRetriever parameters."""
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._llm = llm
        self._rewriter = rewriter
        self._top_k = top_k
        self._num_query_rewrites = num_query_rewrites
        self._reranker_model_name = reranker_model_name
        self._reranker = None  # Initialize reranker as None
        self.verbose = verbose

        if self._reranker_model_name:
            self._reranker = SentenceTransformerRerank(model=self._reranker_model_name, top_n=self._top_k)
            if self.verbose:
                print(f"Initialized reranker with top_n: {self._top_k}")

    def generate_queries(self, query_str: str, num_queries: int) -> List[str]:
        """Generate query variations using the LLM."""
        query_gen_prompt = get_query_generation_prompt(query_str, num_queries)
        response = self._llm.complete(query_gen_prompt)
        queries = response.text.split("\n")
        return [query.strip() for query in queries if query.strip()]

    async def run_queries(self, queries: List[str], retrievers: List[BaseRetriever]) -> dict:
        """Run queries against retrievers."""
        tasks = []
        for query in queries:
            for retriever in retrievers:
                tasks.append(retriever.aretrieve(query))

        task_results = await asyncio.gather(*tasks)

        results_dict = {}
        for i, (query, query_result) in enumerate(zip(queries, task_results)):
            results_dict[(query, i)] = query_result
        return results_dict

    def fuse_vector_and_bm25_results(self, results_dict, similarity_top_k: int) -> List[NodeWithScore]:
        """Fuse results from Vector and BM25 retrievers."""
        k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
        fused_scores = {}
        text_to_node = {}

        for nodes_with_scores in results_dict.values():
            for rank, node_with_score in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                text = node_with_score.node.get_content()
                text_to_node[text] = node_with_score
                if text not in fused_scores:
                    fused_scores[text] = 0.0
                fused_scores[text] += 1.0 / (rank + k)

        reranked_results = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

        reranked_nodes: List[NodeWithScore] = []
        for text, score in reranked_results.items():
            if text in text_to_node:
                node = text_to_node[text]
                node.score = score
                reranked_nodes.append(node)
            else:
                if self.verbose:
                    print(f"Warning: Text not found in `text_to_node`: {text}")

        return reranked_nodes[:similarity_top_k]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        if self._rewriter:
            queries = self.generate_queries(query_bundle.query_str, num_queries=self._num_query_rewrites)
            if self.verbose:
                print(f"Generated Queries: {queries}")
        else:
            queries = [query_bundle.query_str]

        active_retrievers = []
        if self._vector_retriever:
            active_retrievers.append(self._vector_retriever)
        if self._bm25_retriever:
            active_retrievers.append(self._bm25_retriever)

        if not active_retrievers:
            raise ValueError("No active retriever provided!")

        results = asyncio.run(self.run_queries(queries, active_retrievers))
        final_results = self.fuse_vector_and_bm25_results(results, similarity_top_k=self._top_k)

        if self._reranker:
            final_results = self._reranker.postprocess_nodes(final_results, query_bundle)
            if self.verbose:
                print(f"Reranked Results: {final_results}")
        else:
            final_results = final_results[:self._top_k]

        return final_results



class PARetriever(BaseRetriever):
    """Custom retriever that performs query rewriting, Vector search, and BM25 search without Knowledge Graph search."""

    def __init__(
        self,
        llm,  # LLM for query generation
        vector_retriever: Optional[VectorIndexRetriever] = None,
        bm25_retriever: Optional[BaseRetriever] = None,
        mode: str = "OR",
        rewriter: bool = True,
        classifier_model: Optional[str] = None,  # Optional classifier model
        device: str = 'mps',  # Set to 'mps' as the default device
        reranker_model_name: Optional[str] = None,  # Model name for SentenceTransformerRerank
        verbose: bool = False,  # Verbose flag
        fixed_params: Optional[dict] = None,  # New parameter to pass in fixed parameters
        categories_list: Optional[List[str]] = None,  # List of categories for query classification
        param_mappings: Optional[dict] = None  # Custom parameter mappings based on classifier labels
    ) -> None:
        """Initialize PARetriever parameters."""
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._llm = llm
        self._rewriter = rewriter
        self._mode = mode
        self._reranker_model_name = reranker_model_name
        self._reranker = None  # Initialize reranker as None
        self.verbose = verbose
        self.fixed_params = fixed_params
        self.categories_list = categories_list
        self.param_mappings = param_mappings or {  
            "label_0": {"top_k": 5, "max_keywords_per_query": 3, "max_knowledge_sequence": 1},
            "label_1": {"top_k": 7, "max_keywords_per_query": 4, "max_knowledge_sequence": 2},
            "label_2": {"top_k": 10, "max_keywords_per_query": 5, "max_knowledge_sequence": 3}
        }

        # Initialize the classifier if provided
        self.classifier = None
        if classifier_model:
            self.classifier = pipeline("text-classification", model=classifier_model, device=device)

        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")

    def classify_query_and_get_params(self, query: str) -> (str, dict):
        """Classify the query and determine adaptive parameters or use fixed parameters."""
        if self.fixed_params:
            # Use fixed parameters from the dictionary if provided
            params = self.fixed_params
            classification_result = "Fixed"
            if self.verbose:
                print(f"Using fixed parameters: {params}")
        else:
            params = {
                "top_k": 5,  # Default top-k
                "max_keywords_per_query": 4,  # Default max keywords
                "max_knowledge_sequence": 2  # Default max knowledge sequence
            }
            classification_result = None

            if self.classifier:
                classification = self.classifier(query)[0]
                label = classification['label']  # Get the classification label directly
                classification_result = label  # Store the classification result
                if self.verbose:
                    print(f"Query Classification: {classification['label']} with score {classification['score']}")

                # Use custom mappings or default mappings
                if label in self.param_mappings:
                    params = self.param_mappings[label]
                else:
                    if self.verbose:
                        print(f"Warning: No mapping found for label {label}, using default parameters.")

        self._classification_result = classification_result
        return classification_result, params

    def classify_query(self, query_str: str) -> Optional[str]:
        """Classify the query into one of the predefined categories using LLM, or skip if no categories are provided."""
        if not self.categories_list:
            if self.verbose:
                print("No categories provided, skipping query classification.")
            return None

        # Generate the classification prompt using external function
        classification_prompt = get_classification_prompt(self.categories_list) + f" Query: '{query_str}'"

        response = self._llm.complete(classification_prompt)
        category = response.text.strip()

        # Return the category only if it's in the categories list
        return category if category in self.categories_list else None

    def generate_queries(self, query_str: str, category: Optional[str], num_queries: int = 3) -> List[str]:
        """Generate query variations using the LLM, taking into account the category if applicable."""

        # Generate query generation prompt using external function
        query_gen_prompt = get_query_generation_prompt(query_str, num_queries)

        response = self._llm.complete(query_gen_prompt)
        queries = response.text.split("\n")

        queries = [query.strip() for query in queries if query.strip()]

        if category:
            category_query = f"{category}"
            queries.append(category_query)

        return queries

    async def run_queries(self, queries: List[str], retrievers: List[BaseRetriever]) -> dict:
        """Run queries against retrievers."""
        tasks = []
        for query in queries:
            for retriever in retrievers:
                tasks.append(retriever.aretrieve(query))

        task_results = await asyncio.gather(*tasks)

        results_dict = {}
        for i, (query, query_result) in enumerate(zip(queries, task_results)):
            results_dict[(query, i)] = query_result
        return results_dict

    def fuse_vector_and_bm25_results(self, results_dict, similarity_top_k: int) -> List[NodeWithScore]:
        """Fuse results from Vector and BM25 retrievers."""
        k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
        fused_scores = {}
        text_to_node = {}

        for nodes_with_scores in results_dict.values():
            for rank, node_with_score in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                text = node_with_score.node.get_content()
                text_to_node[text] = node_with_score
                if text not in fused_scores:
                    fused_scores[text] = 0.0
                fused_scores[text] += 1.0 / (rank + k)

        reranked_results = dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

        reranked_nodes: List[NodeWithScore] = []
        for text, score in reranked_results.items():
            if text in text_to_node:
                node = text_to_node[text]
                node.score = score
                reranked_nodes.append(node)
            else:
                if self.verbose:
                    print(f"Warning: Text not found in `text_to_node`: {text}")

        return reranked_nodes[:similarity_top_k]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""
        if self._rewriter:
            category = self.classify_query(query_bundle.query_str)
            if self.verbose and category:
                print(f"Classified Category: {category}")

        classification_result, params = self.classify_query_and_get_params(query_bundle.query_str)
        self._classification_result = classification_result

        top_k = params["top_k"]

        if self._reranker_model_name:
            self._reranker = SentenceTransformerRerank(model=self._reranker_model_name, top_n=top_k)
            if self.verbose:
                print(f"Initialized reranker with top_n: {top_k}")

        num_queries = 3 if top_k == 5 else 5 if top_k == 7 else 7
        if self.verbose:
            print(f"Number of Query Rewrites: {num_queries}")

        if self._rewriter:
            queries = self.generate_queries(query_bundle.query_str, category, num_queries=num_queries)
            if self.verbose:
                print(f"Generated Queries: {queries}")
        else:
            queries = [query_bundle.query_str]

        active_retrievers = []
        if self._vector_retriever:
            active_retrievers.append(self._vector_retriever)
        if self._bm25_retriever:
            active_retrievers.append(self._bm25_retriever)

        if not active_retrievers:
            raise ValueError("No active retriever provided!")

        results = {}
        if active_retrievers:
            results = asyncio.run(self.run_queries(queries, active_retrievers))
            if self.verbose:
                print(f"Fusion Results: {results}")

        final_results = self.fuse_vector_and_bm25_results(results, similarity_top_k=top_k)

        if self._reranker:
            final_results = self._reranker.postprocess_nodes(final_results, query_bundle)
            if self.verbose:
                print(f"Reranked Results: {final_results}")
        else:
            final_results = final_results[:top_k]

        if self._rewriter:
            unique_nodes = {}
            for node in final_results:
                content = node.node.get_content()
                if content not in unique_nodes:
                    unique_nodes[content] = node
            final_results = list(unique_nodes.values())

        if self.verbose:
            print(f"Final Results: {final_results}")

        return final_results

    def get_classification_result(self) -> str:
        return getattr(self, "_classification_result", None)


class HyPARetriever(PARetriever):
    """Custom retriever that extends PARetriever with knowledge graph (KG) search."""
    
    def __init__(
        self,
        llm,  # LLM for query generation
        vector_retriever: Optional[VectorIndexRetriever] = None,
        bm25_retriever: Optional[BaseRetriever] = None,
        kg_index=None,  # Pass the knowledge graph index
        property_index: bool = True,  # Whether to use the property graph for retrieval
        pg_filters=None,
        **kwargs,  # Pass any additional arguments to PARetriever
    ):
        # Initialize PARetriever to reuse all its functionality
        super().__init__(
            llm=llm,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            **kwargs
        )

        # Initialize knowledge graph (KG) specific components
        self._pg_filters = pg_filters
        self._kg_index = kg_index
        self.property_index = property_index

    def _initialize_kg_retriever(self, params):
        """Initialize the KG retriever based on retrieval mode."""
        graph_index = self._kg_index
        filters = self._pg_filters

        if self._kg_index and not self.property_index:
            # If not using property index, use KGTableRetriever
            return KGTableRetriever(
                index=self._kg_index,
                retriever_mode='hybrid',
                max_keywords_per_query=params["max_keywords_per_query"],
                max_knowledge_sequence=params["max_knowledge_sequence"]
            )
        
        elif self._kg_index and self.property_index:
            # If using property index, use the simpler graph index retriever
            # Use this for the DEMO 

            vector_retriever = VectorContextRetriever(
                graph_store=graph_index.property_graph_store,
                similarity_top_k=params["max_keywords_per_query"],
                path_depth=params["max_knowledge_sequence"],
                include_text=True,
                filters=filters
            )
            synonym_retriever = LLMSynonymRetriever(
                graph_store=graph_index.property_graph_store,
                llm=self._llm,
                include_text=True,
                filters=filters
            )
            return graph_index.as_retriever(sub_retrievers=[vector_retriever, synonym_retriever])
            #return graph_index.as_retriever(similarity_top_k=params["top_k"])
        
        return None

    def _combine_with_kg_results(self, vector_bm25_results, kg_results):
        """Combine KG results with vector and BM25 results."""
        vector_ids = {n.node.id_ for n in vector_bm25_results}
        kg_ids = {n.node.id_ for n in kg_results}
        
        combined_results = {n.node.id_: n for n in vector_bm25_results}
        combined_results.update({n.node.id_: n for n in kg_results})

        if self._mode == "AND":
            result_ids = vector_ids.intersection(kg_ids)
        else:
            result_ids = vector_ids.union(kg_ids)

        return [combined_results[rid] for rid in result_ids]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes with KG integration."""
        # Call PARetriever's _retrieve to get the vector and BM25 results
        final_results = super()._retrieve(query_bundle)

        # If we have a KG index, initialize the retriever
        if self._kg_index:
            kg_retriever = self._initialize_kg_retriever(self.classify_query_and_get_params(query_bundle.query_str)[1])
            
            if kg_retriever:
                kg_nodes = kg_retriever.retrieve(query_bundle)
                
                # Only combine KG and vector/BM25 results if property_index is True
                if self.property_index:
                    final_results = self._combine_with_kg_results(final_results, kg_nodes)
        
        return final_results



import os
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import KGTableRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core import KnowledgeGraphIndex


def load_documents():
    """Load and return documents from specified file paths."""
    loader = PyMuPDFReader()
    documents1 = loader.load(file_path="../../legal_data/LL144/LL144.pdf")
    documents2 = loader.load(file_path="../../legal_data/LL144/LL144_Definitions.pdf")
    return documents1 + documents2

def create_indices(documents, llm, embed_model):
    """Create and return VectorStoreIndex and KnowledgeGraphIndex from documents."""
    splitter = SentenceSplitter(chunk_size=512)
    
    vector_index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[splitter]
    )
    
    """graph_index = KnowledgeGraphIndex.from_documents(
        documents,
        max_triplets_per_chunk=5,
        llm=llm,
        embed_model=embed_model,
        include_embeddings=True,
        transformations=[splitter]
    )"""

    return vector_index#, graph_index

def create_retrievers(vector_index, graph_index, llm, category_list):
    """Create and return the PA and HyPA retrievers."""
    vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(index=vector_index, similarity_top_k=10)

    PA_retriever = PARetriever(
        llm=llm,
        categories_list=category_list,
        rewriter=True,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        classifier_model="rk68/distilbert-q-classifier-3",
        verbose=False
    )

    HyPA_retriever = HyPARetriever(
        llm=llm,
        categories_list=category_list,
        rewriter=True,
        kg_index=graph_index,
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        classifier_model="rk68/distilbert-q-classifier-3",
        verbose=False,
        property_index=False
    )
    
    return PA_retriever, HyPA_retriever

def create_chat_engine(retriever, memory):
    """Create and return the ContextChatEngine using the provided retriever and memory."""
    return ContextChatEngine.from_defaults(
        retriever=retriever,
        verbose=False,
        chat_mode="context",
        memory_cls=memory,
        memory=memory
    )

def main():
    # Initialize environment and LLM
    gpt35_creds, gpt4o_mini_creds, gpt4o_creds = initialize_openai_creds()
    llm_gpt35 = create_llm(gpt35_creds=gpt35_creds, gpt4o_mini_creds=gpt4o_mini_creds, gpt4o_creds=gpt4o_creds)
    
    # Set global settings for embedding model and LLM
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
    Settings.embed_model = embed_model
    Settings.llm = llm_gpt35
    
    category_list = [
        '5-301 Bias Audit',
        '5-302 Data Requirements',
        '§ 5-303 Published Results',
        '§ 5-304 Notice to Candidates and Employees'
    ]
    
    # Load documents and create indices
    documents = load_documents()
    vector_index, graph_index = create_indices(documents, llm_gpt35, embed_model)
    
    # Create retrievers
    PA_retriever, HyPA_retriever = create_retrievers(vector_index, graph_index, llm_gpt35, category_list)
    
    # Initialize chat memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=8192)
    
    # Create chat engines
    PA_chat_engine = create_chat_engine(PA_retriever, memory)
    HyPA_chat_engine = create_chat_engine(HyPA_retriever, memory)
    
    # Sample question and response
    question = "What is a bias audit?"
    PA_response = PA_chat_engine.chat(question)
    HyPA_response = HyPA_chat_engine.chat(question)
    
    # Output responses in a nicely formatted manner
    print("\n" + "="*50)
    print(f"Question: {question}")
    print("="*50)
    
    print("\n------- PA Retriever Response -------")
    print(PA_response)
    
    print("\n------- HyPA Retriever Response -------")
    print(HyPA_response)
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
