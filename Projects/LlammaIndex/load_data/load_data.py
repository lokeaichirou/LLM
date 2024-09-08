import os
from pathlib import Path

from llama_index.core.schema import TextNode
from llama_index.load_data.readers import reader_map
from llama_index.load_data.node_parsers import node_parser_map

from pdf2image import convert_from_path

from maps.maps import LLM_models, SimpleDirectoryReader
                                

def pdf2images(pdf_file):

    '''Convert each PDF page into a PNG image'''

    # The saved path = original PDF file name (without the extension)
    output_directory_path, _        = os.path.splitext(pdf_file)
    doc_pages_output_directory_path = output_directory_path + "/doc_pages"

    if not os.path.exists(doc_pages_output_directory_path):
        os.makedirs(doc_pages_output_directory_path)

    # Convert PDF to images
    images = convert_from_path(pdf_file)

    # Save images as PNG files
    for page_number, image in enumerate(images):
        image.save(f"{doc_pages_output_directory_path}/page_{page_number + 1}.png")

    return doc_pages_output_directory_path


class DataLoader:

    def __init__(self, file_path, configs, if_image=True):

        self.if_image    = if_image
        self.configs     = configs
        self.text_chunks = []
        self.doc_idxs    = []
        self.nodes       = []

        # For image understanding and reasoning
        if if_image:
            self.doc_pages_output_directory_path = pdf2images(file_path)
            self.LLM_model                       = LLM_models[configs.selected_LLM_model]

        # For text loading
        loader         = reader_map[configs.selected_reader_type]()
        self.documents = loader.load(file_path = file_path)
        self.splitter  = node_parser_map[configs.selected_node_parser](**configs.selected_splitter_parameters)

    def fill_nodes_for_text(self):

        nodes = self.splitter.get_nodes_from_documents(self.documents)
        self.nodes.extend(nodes)

    def fill_nodes_for_image_description(self):
        # Fill image description chunks
        image_docs = []
        for img_file in Path(self.doc_pages_output_directory_path).glob("*.png"):

            image_documents = SimpleDirectoryReader(input_files = [img_file]).load_data()
            response        = self.LLM_model.complete(prompt          = "Please briefly describe the information in the image",
                                                      image_documents = image_documents,)
            self.nodes.append(TextNode(text=str(response)))

    def fill_nodes(self):
        # Fill nodes
        self.fill_nodes_for_text()
        if self.if_image:
            self.fill_nodes_for_image_description()