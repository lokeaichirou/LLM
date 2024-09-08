
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import qdrant_client
from llama_index.core.node_parser import LanguageConfig
import chromadb

embed_model = OpenAIEmbedding()

class Configs:

    # Reader
    selected_reader_type = 'PyMuPDF_reader'

    # Parser
    selected_node_parser = 'SemanticSplitter'

    Node_parsers_parameters = {'SentenceSplitter': {'chunk_size': 1024, 'chunk_overlap': 20},
                               'SemanticSplitter': {'buffer_size': 1, 'breakpoint_percentile_threshold': 95, 'embed_model': embed_model},
                               'SemanticDoubleMergingSplitter': {'language_config': LanguageConfig(language = "english", spacy_model = "en_core_web_md"),
                                                                 'initial_threshold': 0.4,
                                                                 'appending_threshold': 0.5,
                                                                 'merging_threshold': 0.5,
                                                                 'max_chunk_size': 5000}
                               }

    selected_splitter_parameters = Node_parsers_parameters[selected_node_parser]

    # MultiModal LLM model
    selected_LLM_model = 'OpenAIMultiModal'

    # Vector store
    selected_vector_store = 'Chroma'

    vector_store_parameters = {'Chroma': {'chroma_collection': chromadb.EphemeralClient(settings = Settings(allow_reset = True)).create_collection("demo_3")},
                               
                               'Qdrant': {'collection_name': "demo", 
                                          'client': qdrant_client.QdrantClient(host = "localhost", port = 6333), 
                                          'aclient': qdrant_client.AsyncQdrantClient(location=":memory:"), 
                                          'prefer_grpc': True}
                               }
    
    selected_vector_store_parameters = vector_store_parameters[selected_vector_store]