
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext

from maps.maps import vector_stores, supported_vector_stores


class Vector_store:

    def __init__(self, configs, nodes):
        
        """
        Initialize the Vector Store
        
        Parameters
        ----------
        selected_vector_store : str 
                                The selected vector store to use for the Vector Store
        nodes : list 
                The list of nodes to be used for the Vector Store
            
        Returns
        -------
        None

        """

        if configs.selected_vector_store not in supported_vector_stores:
            raise Exception(f"Sorry, {selected_vector_store} is not in supported vector stores")

        # Create vector store
        selected_vector_store = vector_stores[configs.selected_vector_store](**configs.selected_vector_store_parameters)
        
        # Storage Context is the storage container of Vector Store, which is used to store text, index, vector and other data.
        storage_context       = StorageContext.from_defaults(vector_store = selected_vector_store)

        # Create index through connecting the storage context to the selected vector store
        self.index = VectorStoreIndex(nodes, storage_context = storage_context)

        self.chat_engine = self.index.as_chat_engine()

    def chat(self, question):

        response = self.chat_engine.chat(question)