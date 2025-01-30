
import openai


from qdrant_client import QdrantClient
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI  # Updated to use OpenAI
import warnings
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore


warnings.filterwarnings("ignore")

# Set your OpenAI API key
openai.api_key = "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"


class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        
        # Configure global settings
        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)  # Use OpenAI's GPT model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self._index = None
        self._create_kb()
        self._create_chat_engine()

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(input_files=[r"/Users/jayantsankhi/Documents/GitHub/RAG-AI-Voice-assistant-/rag/restaurant_file.txt"])
            documents = reader.load_data()
            
            # Initialize QdrantVectorStore
            vector_store = QdrantVectorStore(client=self._client, collection_name="kitchen_db")

            docstore = SimpleDocumentStore()
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=docstore
            )
            self._index = VectorStoreIndex.from_documents(
                documents, 
                storage_context=storage_context  # Correct parameter
            )
            
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def _create_chat_engine(self):
        if self._index is not None:  # Check if the knowledge base was created successfully
            memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            self._chat_engine = self._index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt=self._prompt,
            )
        else:
            print("Knowledge base creation failed. Chat engine cannot be created.")

    def interact_with_llm(self, customer_query):
        response = self._chat_engine.chat(customer_query)
        return response.response

    @property
    def _prompt(self):
        return (
            '''“You are a professional AI assistant working for Government of Maharashtra to help individuals with their queries regarding the Slum Rehabilitation Authority, Brihanmumbai”
            “Whatever questions people ask to you, process them properly, and answer in simple and understandable language. and in 40 words only ”

            “If you do not know the answer, just say so - don’t make up information. In that case, also mention that you will pass on the query to our human expert of officers and get back to them.”

            “If the question is outside of your knowledge base dataset then ask the person to just stick to the specific use case of the bot”

            "Provide concise responses and do not hallucinate or chat with yourself”'''
        )