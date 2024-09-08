import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from data_preprocessing.text_processing import extract_questions_from_pdf
from data_preprocessing.doc_page_processing import pdf2images
from data_preprocessing.doc_page_processing import crop_and_save_image_patches_and_tables
from llm.llm_invoke.generate_image_description import generate_image_description
from llm.embedding import get_embeddings
from llm.llm_invoke.llm_invoke import get_completion
from databases.elasticsearch import ElasticsearchDatabase
from databases.chroma import VectorDBConnectorSupportingMultimodal
from rag.rag import RAG_Bot_without_ranking
from questions.questions import Questions 

from utils import extract_number

from ultralytics import YOLOv10
import supervision as sv

model  = YOLOv10('yolov10x_best.pt')
client = OpenAI()

pdf_file          = "Projects/DataScience Interview Questions.pdf"
corpus            = extract_questions_from_pdf(pdf_file)
original_corpus   = extract_questions_from_pdf(pdf_file)

doc_pages_output_directory_path = pdf2images(pdf_file)

png_files                    = sorted([os.path.join(doc_pages_output_directory_path, file) for file in os.listdir(doc_pages_output_directory_path) if file.endswith('.png')], key=extract_number)
imagettes_files, table_files = crop_and_save_image_patches_and_tables(doc_pages_output_directory_path, model)
imagettes_description        = generate_image_description(imagettes_files, corpus)
tables_description           = generate_image_description(table_files, corpus)

es_db = ElasticsearchDatabase()
es_db.create_index(index_name = 'es_docs_images')
es_db.add_texts_and_descriptions_of_imagettes_and_tables(corpus, 'es_docs_images')

vector_db = VectorDBConnectorSupportingMultimodal("multimodal", get_embeddings)
vector_db.add_texts(original_corpus)
vector_db.add_imagettes_description(imagettes_description)
vector_db.add_tables_description(tables_description)


if __name__ == '__main__':

    naive_rag_bot =  RAG_Bot_without_ranking(llm_api = get_completion,
                                            num_results_retrieved_by_database = 6,
                                            num_results_selected_for_prompt = 3,
                                            documents = corpus,
                                            vector_db = vector_db,
                                            es_db = es_db,
                                            es_index = 'es_docs_images')

    for i, query in enumerate(Questions):
            
        response = naive_rag_bot.chat(user_query = query,
                                    model_type = 'gpt-4o')
        
        print("{}:\n Query: {}\n Answer: {}\n".format(i, query,response))