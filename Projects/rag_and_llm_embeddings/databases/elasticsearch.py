from elasticsearch7 import Elasticsearch, helpers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import os
import time
import warnings
warnings.simplefilter("ignore")  # Shield some ES Warnings

nltk.download('punkt')  # English words segmentation, word root, sentence segmentation and other methods
nltk.download('stopwords')  # English stop words

# Import configuration files
ELASTICSEARCH_BASE_URL = os.getenv('ELASTICSEARCH_BASE_URL')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')
ELASTICSEARCH_NAME     = os.getenv('ELASTICSEARCH_NAME')

def to_keywords(input_string):
        
    '''(English) Only retains keywords in the text'''

    # Use regular expression to replace all non-alphanumeric characters with spaces
    no_symbols  = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string)
    word_tokens = word_tokenize(no_symbols)
    
    # Load stop word list
    stop_words = set(stopwords.words('english'))
    ps         = PorterStemmer()
    
    # Remove stop words and take root
    filtered_sentence = [ps.stem(w) for w in word_tokens if not w.lower() in stop_words]

    return ' '.join(filtered_sentence)


class ElasticsearchDatabase:

    def __init__(self) -> None:
        
        self.es = Elasticsearch(hosts     = [ELASTICSEARCH_BASE_URL],  # Service address and port
                                http_auth = (ELASTICSEARCH_NAME, ELASTICSEARCH_PASSWORD),  # Username Password
                                )
        self.index_names = []
    
    def create_index(self, index_name = "teacher_demo_index_tmp"):

        self.index_names.append(index_name)

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        
        self.es.indices.create(index=index_name)
        time.sleep(1)

    def add_texts_and_descriptions_of_imagettes_and_tables(self, docs, index_name):

        actions = [{"_index": index_name,
                    "_source": {"keywords": to_keywords(doc),
                                "text": doc}
                   } for doc in docs]

        helpers.bulk(self.es, actions)

    def search(self, query_string, top_n, index_name = None):
        
        search_query = {"match": {"keywords": to_keywords(query_string)}}

        if index_name:
            res          = self.es.search(index = index_name, 
                                          query = search_query,
                                          size  = top_n)
        else:
            res          = self.es.search(index = self.index_names[-1], 
                                          query = search_query,
                                          size  = top_n)
        
        return [hit["_source"]["text"] for hit in res["hits"]["hits"]]