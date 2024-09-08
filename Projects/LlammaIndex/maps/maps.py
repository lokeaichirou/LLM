
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (DocxReader,
                                      HWPReader,
                                      PDFReader,
                                      EpubReader,
                                      FlatReader,
                                      HTMLTagReader,
                                      ImageCaptionReader,
                                      ImageReader,
                                      ImageVisionLLMReader,
                                      IPYNBReader,
                                      MarkdownReader,
                                      MboxReader,
                                      PptxReader,
                                      PandasCSVReader,
                                      VideoAudioReader,
                                      UnstructuredReader,
                                      PyMuPDFReader,
                                      ImageTabularChartReader,
                                      XMLReader,
                                      PagedCSVReader,
                                      CSVReader,
                                      RTFReader)

from llama_index.core.node_parser import (SentenceSplitter,
                                          SemanticSplitterNodeParser,
                                          SemanticDoubleMergingSplitterNodeParser,
                                          LanguageConfig)

from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.opensearch import (OpensearchVectorStore, OpensearchVectorClient)
from llama_index.vector_stores.qdrant import QdrantVectorStore

OPENAI_API_KEY = "sk-"

reader_map = {'docx_reader': DocxReader,
              'HWP_reader':  HWPReader,
              'PDF_reader':  PDFReader,
              'Epub_reader': EpubReader,
              'Flat_reader': FlatReader,
              'HTMLTag_reader': HTMLTagReader,
              'ImageCaption_reader': ImageCaptionReader,
              'Image_reader': ImageReader,
              'ImageVision_reader': ImageVisionLLMReader,
              'IPYNB_reader': IPYNBReader,
              'Markdown_reader': MarkdownReader,
              'Mbox_reader': MboxReader,
              'Pptx_reader': PptxReader,
              'PandasCSV_reader': PandasCSVReader,
              'VideoAudio_reader': VideoAudioReader,
              'Unstructured_reader': UnstructuredReader,
              'PyMuPDF_reader': PyMuPDFReader,
              'ImageTabularChart_reader': ImageTabularChartReader,
              'XML_reader': XMLReader,
              'PagedCSV_reader': PagedCSVReader,
              'CSV_reader': CSVReader,
              'RTF_reader': RTFReader}

# Parser
node_parser_map = {'SentenceSplitter': SentenceSplitter,
                   'SemanticSplitter': SemanticSplitterNodeParser,
                   'SemanticDoubleMergingSplitter': SemanticDoubleMergingSplitterNodeParser}

# Multi-modal LLM
LLM_models = {'OpenAIMultiModal': OpenAIMultiModal(model="gpt-4-turbo-2024-04-09", api_key = OPENAI_API_KEY, max_new_tokens=4096),
              'AzureOpenAIMultiModal': AzureOpenAIMultiModal(engine         = "gpt-4-vision-preview",
                                                             api_version    = "2023-12-01-preview",
                                                             model          = "gpt-4-vision-preview",
                                                             max_new_tokens = 300)}

# vector_stores
supported_vector_stores   = ['Chroma', 'Opensearch', 'Qdrant']

vector_stores = {'Chroma': ChromaVectorStore,
                 'Opensearch': OpensearchVectorStore,
                 'Qdrant': QdrantVectorStore}