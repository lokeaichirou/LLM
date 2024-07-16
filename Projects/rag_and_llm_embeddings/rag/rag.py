from abc import ABC, abstractmethod
from rrf import rrf
from llm.prompting import build_prompt, prompt_template
from llm.llm_invoke.generate_similar_queries import generate_queries_by_llm
from sentence_transformers import CrossEncoder


class RAG_Bot(ABC):

    def __init__(self, llm_api, num_results_retrieved_by_database, num_results_selected_for_prompt, documents, vector_db, es_db, es_index):

        # llm get_completion method
        self.llm_api = llm_api

        # Set the number of results retrieved by the database
        self.num_retrieved_results = num_results_retrieved_by_database

        # Set the number of results selected for the prompt
        self.num_selected_results = num_results_selected_for_prompt

        # Load documents
        self.documents = documents

        # Initialize the chroma database
        self.vector_db = vector_db

        # Initialize the elasticsearch database
        self.es_db = es_db

        # elastic search database index
        self.es_index = es_index

    @abstractmethod
    def db_search(self, user_query):
        pass

    def possible_rrf_for_es_and_vec_db_retrieval_result_mixture(self, db_search_results, k):
        
        search_results = rrf(self.documents, db_search_results, k)        
        return search_results

    def chat(self, user_query, model_type, num_generated_queries=0, k=1):

        user_queries           = [user_query]
        db_search_results_list = []

        # 0. Generate similar queries
        if num_generated_queries != 0:
            generated_queries = generate_queries_by_llm(user_query, num_generated_queries)
            user_queries.extend(generated_queries)

        # 1. Retrieval of context by the database
        for query in user_queries:
            
            db_search_results = self.db_search(query, self.num_retrieved_results)

            if len(db_search_results) == 0:
                continue
            elif len(db_search_results) == 1:
                db_search_results_list.append(db_search_results[0])
            elif len(db_search_results) == 2:
                db_search_results_list.append(self.possible_rrf_for_es_and_vec_db_retrieval_result_mixture(db_search_results, k))

        if num_generated_queries != 0:
            sorted_texts = rrf(self.documents, db_search_results_list, k)
        else:
            sorted_texts = db_search_results_list[0]
            
        # 2. Build Prompt
        prompt = build_prompt(prompt_template, 
                              context = sorted_texts[:self.num_selected_results], 
                              query   = user_query)
        
        # 3. Invoke LLM
        response = self.llm_api(prompt, model_type)

        return response
    

# RAG based vector retrieval

class RAG_Bot_without_ranking(RAG_Bot):

    def db_search(self, user_query, num_results):

        valid_vector_db, valid_es_db = False, False
        
        # 1. Retrieval by the database
        if self.vector_db:
            vector_db_search_results = self.vector_db.search(user_query, num_results)['documents'][0]
            if vector_db_search_results:
                valid_vector_db = True

        if self.es_db:
            es_db_search_results = self.es_db.search(user_query, num_results, self.es_index)
            if es_db_search_results:
                valid_es_db = True

        if valid_vector_db and valid_es_db:
            return [vector_db_search_results, es_db_search_results] 
        elif valid_vector_db:
            return [vector_db_search_results]
        elif valid_es_db:
            return [es_db_search_results]
        else:
            return []
        

class RAG_Bot_with_ranking(RAG_Bot):

    def __init__(self, llm_api, num_results_retrieved_by_database, num_results_selected_for_prompt, documents, vector_db, es_db, es_index, rank_model_type = 'BAAI/bge-reranker-large', max_length_for_ranker = 512):

        super().__init__(llm_api, num_results_retrieved_by_database, num_results_selected_for_prompt, documents, vector_db, es_db, es_index)
        self.model = CrossEncoder(rank_model_type, max_length = max_length_for_ranker)

    def db_search(self, user_query, num_results):

        valid_vector_db, valid_es_db = False, False
        
        # 1. Retrieval by the database
        if self.vector_db:
            vector_db_search_results = self.vector_db.search(user_query, num_results)['documents'][0]
            if vector_db_search_results:
                valid_vector_db                 = True
                scores                          = self.model.predict([(user_query, doc) for doc in vector_db_search_results])
                ranked_vector_db_search_results = sorted(zip(vector_db_search_results, scores), key=lambda x: x[1], reverse=True)

        if self.es_db:
            es_db_search_results = self.es_db.search(user_query, num_results, self.es_index)
            if es_db_search_results:
                valid_es_db                     = True
                scores                          = self.model.predict([(user_query, doc) for doc in es_db_search_results])
                ranked_es_db_search_results     = sorted(zip(es_db_search_results, scores), key=lambda x: x[1], reverse=True)

        if valid_vector_db and valid_es_db:
            return [ranked_vector_db_search_results, ranked_es_db_search_results] 
        elif valid_vector_db:
            return [ranked_vector_db_search_results]
        elif valid_es_db:
            return [ranked_es_db_search_results]
        else:
            return []