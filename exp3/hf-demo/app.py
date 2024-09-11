import streamlit as st
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from code.retrievers import PARetriever
from code.utils_code import  create_chat_engine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
from llama_index.llms.azure_openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv
from code.retrievers import HyPARetriever, PARetriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.retrievers.bm25 import BM25Retriever

# Load environment variables from the .env file
dotenv_path = find_dotenv()
#print(f"Dotenv Path: {dotenv_path}")
load_dotenv(dotenv_path)


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.embed_model = embed_model

# Set Azure OpenAI keys for Giskard if needed
#os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("GSK_AZURE_OPENAI_API_KEY")
#os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("GSK_AZURE_OPENAI_ENDPOINT")
os.environ["GSK_LLM_MODEL"] = "gpt-4o-mini"

# Pinecone and Neo4j credentials
pinecone_api_key = os.getenv("PINECONE_API_KEY")
ll144_index_name = 'll144'
euaiact_index_name = 'euaiact'

# Initialize Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=pinecone_api_key)


def metadata_filter(corpus_name):

    if corpus_name == "EUAIACT":

        # Filter for 'EUAIACT.pdf'
        filter = MetadataFilters(filters=[MetadataFilter(key="filepath", value="'EUAIACT.pdf'", operator=FilterOperator.CONTAINS)])

    elif corpus_name == "LL144":
    # Filter for 'LLL144.pdf' or 'LL144_Definitions.pdf'
        filter = MetadataFilters(filters=[
            MetadataFilter(key="filepath", value="'LL144.pdf'", operator=FilterOperator.CONTAINS),
            MetadataFilter(key="filepath", value="'LL144_Definitions.pdf'", operator=FilterOperator.CONTAINS)
        ])

    return filter
        

# Load vector index
#@st.cache_data(ttl=None, persist=None)
def load_vector_index(corpus_name):
    if corpus_name == "LL144":
        pinecone_index = pc.Index(ll144_index_name)
    elif corpus_name == "EUAIACT":
        pinecone_index = pc.Index(euaiact_index_name)

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    vector_index = VectorStoreIndex.from_vector_store(vector_store)
    
    return vector_index

# Load property graph index
#@st.cache_data(ttl=None, persist=None)
def load_pg_index():
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_url = os.getenv("NEO4J_URI")

    graph_store = Neo4jPropertyGraphStore(username=neo4j_username, password=neo4j_password, url=neo4j_url)
    pg_index = PropertyGraphIndex.from_existing(property_graph_store=graph_store)
    return pg_index

# Initialize the retriever (HyPA or PA)
def init_retriever(retriever_type, corpus_name, use_reranker, use_rewriter, classifier_model):
    # Check if vector index is cached, if not, load it
    if "vector_index" not in st.session_state:
        st.session_state.vector_index = load_vector_index(corpus_name)

    # Check if property graph index is cached, if not, load it
    if "pg_index" not in st.session_state:
        st.session_state.pg_index = load_pg_index()

    vector_index = st.session_state.vector_index
    graph_index = st.session_state.pg_index
    llm = st.session_state.llm

    filter = metadata_filter(corpus_name=corpus_name)
    # Set the reranker model if selected
    reranker_model_name = "BAAI/bge-reranker-large" if use_reranker else None

    # Choose the appropriate retriever based on user selection
    if retriever_type == "HyPA":
        retriever = HyPARetriever(
            llm=llm,
            vector_retriever=vector_index.as_retriever(similarity_top_k=10),
            bm25_retriever=None,#BM25Retriever.from_defaults(index=vector_index, similarity_top_k=10),
            kg_index=graph_index,  # Include KG for HyPA
            rewriter=use_rewriter,  # Set rewriter option
            classifier_model=classifier_model,  # Use the selected classifier model
            verbose=False,
            property_index=True,  # Use property graph index
            reranker_model_name=reranker_model_name,  # Use reranker if selected
            pg_filters=filter
        )
    else:
        retriever = PARetriever(
            llm=llm,
            vector_retriever=vector_index.as_retriever(similarity_top_k=10),
            bm25_retriever=None,#BM25Retriever.from_defaults(index=vector_index, similarity_top_k=10),
            rewriter=use_rewriter,  # Set rewriter option
            classifier_model=classifier_model,  # Use the selected classifier model
            verbose=False,
            reranker_model_name=reranker_model_name  # Use reranker if selected
        )

    memory = ChatMemoryBuffer.from_defaults(token_limit=8192)
    chat_engine = create_chat_engine(retriever=retriever, memory=memory, llm=llm)
    st.session_state.chat_engine = chat_engine
    #return chat_engine


def process_query(query):
    """Processes the input query and displays it along with the response in the main chat area."""
    # Append the user query to the message history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Ensure the chat engine is initialized
    chat_engine = st.session_state.get('chat_engine', None)
    if chat_engine:
        # Process the query through the chat engine
        with st.chat_message("assistant"):
            with st.spinner("Retrieving Knowledge..."):
                response = chat_engine.stream_chat(query)
                response_str = ""
                response_container = st.empty()
                for token in response.response_gen:
                    response_str += token
                    response_container.write(response_str)
                # Append the assistant's response to the message history
                st.session_state.messages.append({"role": "assistant", "content": response_str})
                
            # Expander for additional info
            with st.expander("Source Nodes"):
                # Display source nodes
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    
                    for idx, node in enumerate(response.source_nodes):
                        st.markdown(f"#### Source Node {idx + 1}")
                        st.write(f"**Node ID:** {node.node_id}")
                        st.write(f"**Node Score:** {node.score}")
                        
                        st.write("**Metadata:**")
                        for key, value in node.metadata.items():
                            st.write(f"- **{key}:** {value}")
                        
                        st.write("**Content:**")
                        st.write(node.node.get_content())

                        # Add a horizontal line to separate nodes
                        st.markdown("---")
                else:
                    st.write("No additional source nodes available.")

        st.session_state.messages.append({"role": "assistant", "content": str(response)})





# Streamlit App
def main():

    
    # Sidebar for retriever options
    with st.sidebar:
        st.image('holisticai.svg', use_column_width=True)
        st.title("Retriever Settings")

        # Azure OpenAI credentials input fields (start with blank fields)
        azure_api_key = st.text_input("Azure OpenAI API Key", value="", type="password")
        azure_endpoint = st.text_input("Azure OpenAI Endpoint", value="", type="password")

        llm_model_choice = st.selectbox("Select LLM Model", ["gpt-4o-mini", "gpt35"])

        # Let the user make selections without updating session state yet
        retriever_type = st.selectbox("Select Retriever Method", ["PA", "HyPA"])
        corpus_name = st.selectbox("Select Corpus", ["LL144", "EUAIACT"])
        temperature = st.slider("Set LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        # Display a red warning about non-zero temperature
        if temperature > 0:
            st.markdown(
                "<p style='color:red;'>Warning: A non-zero temperature may lead to hallucinations in the generated responses.</p>",
                unsafe_allow_html=True
            )

        # Checkboxes for reranker and rewriter options
        use_reranker = st.checkbox("Use Reranker")
        use_rewriter = st.checkbox("Use Rewriter")

        # Radio buttons for classifier model
        classifier_type = st.radio("Select Classifier Type", ["2-Class", "3-Class"])
        classifier_model = "rk68/distilbert-q-classifier-2" if classifier_type == "2-Class" else "rk68/distilbert-q-classifier-3"



        # When the user clicks "Initialize", store everything in session state
        if st.button("Initialize"):
            st.session_state.retriever_type = retriever_type
            st.session_state.corpus_name = corpus_name
            st.session_state.temperature = temperature
            st.session_state.use_reranker = use_reranker
            st.session_state.use_rewriter = use_rewriter
            st.session_state.classifier_type = classifier_type
            st.session_state.classifier_model = classifier_model

            # Store the user inputs in session state
            st.session_state.azure_api_key = azure_api_key
            st.session_state.azure_endpoint = azure_endpoint

            # Set the environment variables from user inputs
            os.environ["AZURE_OPENAI_API_KEY"] = azure_api_key
            os.environ["AZURE_OPENAI_ENDPOINT"] = azure_endpoint

            llm = AzureOpenAI(
                deployment_name=llm_model_choice, temperature=temperature,
                api_key=azure_api_key, azure_endpoint=azure_endpoint,
                api_version=os.getenv("AZURE_API_VERSION")
            )
            Settings.llm = llm
            st.session_state.llm = llm

            # Initialize retriever after storing the settings
            init_retriever(retriever_type, corpus_name, use_reranker, use_rewriter, classifier_model)
            st.success("Retriever Initialized")

        # Example questions based on selected corpus
        st.markdown("### Example Queries")
        # Example questions with unique button handling
        example_questions = {
            "LL144": [
                "What is a bias audit?",
                "When does it come into effect?",
                "Summarise Local Law 144"
            ],
            "EUAIACT": [
                "What is an AI system?",
                "What are the key takeaways?",
                "Explain the key provisions of EUAIACT."
            ]
        }


            # Display buttons for the example queries
        for idx, question in enumerate(example_questions.get(corpus_name, [])):
            if st.button(f"{question} [{idx}]"):
                process_query(question)




        
        # Add a disclaimer at the bottom
        st.markdown("---")  # Horizontal line for separation
        
        st.markdown(
            """
            <p style="color:grey; font-size:12px;">
            <strong>Disclaimer:</strong> This system is an academic prototype demonstration of our hybrid parameter-adaptive retrieval-augmented generation system. It is <strong>NOT</strong> a production-ready application. All outputs should be considered experimental and may not be fully accurate. This system should not be used for making important legal decisions. For complete, specific, and tailored legal advice, please consult a licensed legal professional.<br><br>
            </p>
            """, 
            unsafe_allow_html=True
        )



    # Check if the retriever is initialized
    if "chat_engine" in st.session_state:
        chat_engine = st.session_state.chat_engine
    else:
        st.warning("Please initialize the retriever from the sidebar.")


    # Initialize session state for chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate a response if the last message is from the user
        if st.session_state.messages[-1]["role"] == "user":
            with st.chat_message("assistant"):
                with st.spinner("Retrieving Knowledge..."):
                    response = chat_engine.stream_chat(prompt)
                    response_str = ""
                    response_container = st.empty()
                    for token in response.response_gen:
                        response_str += token
                        response_container.write(response_str)
                # Expander for additional info
                with st.expander("Source Nodes"):
                    # Display source nodes
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        
                        for idx, node in enumerate(response.source_nodes):
                            st.markdown(f"#### Source Node {idx + 1}")
                            st.write(f"**Node ID:** {node.node_id}")
                            st.write(f"**Node Score:** {node.score}")
                            
                            st.write("**Metadata:**")
                            for key, value in node.metadata.items():
                                st.write(f"- **{key}:** {value}")
                            
                            st.write("**Content:**")
                            st.write(node.node.get_content())

                            # Add a horizontal line to separate nodes
                            st.markdown("---")
                    else:
                        st.write("No additional source nodes available.")

            st.session_state.messages.append({"role": "assistant", "content": str(response)})

if __name__ == "__main__":
    main()
