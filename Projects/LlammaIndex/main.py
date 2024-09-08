
import os

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from load_data.load_data import DataLoader
from configurations.configurations import Configs
from vector_store.vector_store import Vector_store
from questions import Questions


# Settings
Settings.llm = OpenAI(temperature=0, model="gpt-4o")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=512)

# API Key
OPENAI_API_KEY =  "sk-"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if __name__ == '__main__':

    pdf_file = "Projects/DataScience Interview Questions.pdf"

    configs     = Configs()
    data_loader = DataLoader(pdf_file, configs)
    data_loader.fill_nodes()

    vector_store = Vector_store(configs.selected_vector_store, data_loader.nodes)

    fusion_retriever = QueryFusionRetriever([vector_store.index.as_retriever()],
                                             similarity_top_k = 5,
                                             num_queries = 3,
                                             use_async = True)
    
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-large", top_n = 2)

    query_engine = RetrieverQueryEngine.from_args(fusion_retriever,
                                                  node_postprocessors = [reranker])
    
    chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine = query_engine,)

    for i, question in enumerate(Questions):
        response = chat_engine.chat(question)
        print("{}:\n Query: {}\n Answer: {}\n".format(i, question, response))