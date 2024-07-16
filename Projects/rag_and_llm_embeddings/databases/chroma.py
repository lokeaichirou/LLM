import os

if os.environ.get('CUR_ENV_IS_STUDENT',False):
    import sys
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings

class VectorDBConnectorSupportingMultimodal:
    
    def __init__(self, collection_name, embedding_fn):
        
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # For demonstration purpose, there is actually no need to reset() each time.
        chroma_client.reset()

        # create a collection
        self.collection = chroma_client.get_or_create_collection(name = collection_name)
        self.embedding_fn = embedding_fn

    def add_texts(self, texts):
        
        '''Add documents and vectors to a collection'''
        
        self.collection.add(embeddings = self.embedding_fn(texts),
                            documents  = texts,
                            ids        = [f"text_id{i}" for i in range(len(texts))]
                            )

    def add_imagettes_description(self, imagettes_description):
        
        '''Add imagettes_description and vectors to a collection'''

        self.collection.add(embeddings = self.embedding_fn(imagettes_description),
                            documents  = imagettes_description,
                            ids        = [f"imagette_id{i}" for i in range(len(imagettes_description))]
                            )

    def add_tables_description(self, tables_description):

        '''Add tables_description and vectors to a collection'''

        self.collection.add(embeddings = self.embedding_fn(tables_description),
                            documents  = tables_description,
                            ids        = [f"table_id{i}" for i in range(len(tables_description))]
                            )

    def search(self, query, top_n):
        
        '''Vector database retrieval'''
        
        results = self.collection.query(query_embeddings = self.embedding_fn([query]),
                                        n_results        = top_n)
        return results