# Pre-processing<a href="#Pre-processing" class="anchor-link">¶</a>

## Text Processing<a href="#Text-Processing" class="anchor-link">¶</a>


```python
import fitz  # PyMuPDF

def extract_questions_from_pdf(pdf_file):

    questions = []
    current_question = ""

    # Open the PDF file
    pdf_document = fitz.open(pdf_file)

    for page_num in range(pdf_document.page_count):

        page = pdf_document[page_num]

        # Extract text with formatting information
        text_with_formatting = page.get_text("dict")

        for block in text_with_formatting["blocks"]:

            #if page_num % 10 == 0:
            #    print("block: ", block)

            # Text block
            if block["type"] == 0:

                for line in block["lines"]:

                    line_text = " ".join([span["text"] for span in line["spans"]])

                    # Check for excluding the footer text
                    if "https://lnkd.in/gZu463X" not in line_text:

                        # Check for question ending with a question mark
                        if line_text.strip().endswith("?"):

                            if current_question:
                                questions.append(current_question.strip())

                            current_question = line_text

                        else:
                            if current_question:
                                current_question += " " + line_text

    # Add the last question
    if current_question:
        questions.append(current_question.strip())

    pdf_document.close()

    return questions
```


```python
# Provide the path to the PDF file
pdf_file = 'DataScience Interview Questions.pdf'
corpus   = extract_questions_from_pdf(pdf_file)
original_corpus   = extract_questions_from_pdf(pdf_file)
```

## Doc page processing<a href="#Doc-page-processing" class="anchor-link">¶</a>


```python
import os
from PIL import Image

def pdf2images(pdf_file):
    
    '''Convert each PDF page into a PNG image'''
    
    # The saved path = original PDF file name (without the extension)
    output_directory_path, _ = os.path.splitext(pdf_file)
    doc_pages_output_directory_path = output_directory_path + "/doc_pages"
    
    if not os.path.exists(doc_pages_output_directory_path):
        os.makedirs(doc_pages_output_directory_path)
    
    # Load PDF files
    pdf_document = fitz.open(pdf_file)
    
    for page_number in range(pdf_document.page_count):
        
        page  = pdf_document[page_number]
        pix   = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
        image.save(f"./{doc_pages_output_directory_path}/page_{page_number + 1}.png")
    
    pdf_document.close()

    return doc_pages_output_directory_path
```


```python
doc_pages_output_directory_path = pdf2images("DataScience Interview Questions.pdf")
```

### Extract pictures and tables from the doc<a href="#Extract-pictures-and-tables-from-the-doc"


```python
!pip install -q git+https://github.com/THU-MIG/yolov10.git
!pip install -q supervision
```


```python
import cv2
from ultralytics import YOLOv10
import supervision as sv

model = YOLOv10('yolov10x_best.pt')
```

    /usr/local/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
import re

def extract_number(file_name):
    # Extract the numeric portion of a file name，assume the file name is in the format "page_<number>.png"
    
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else -1
```


```python
png_files = sorted([os.path.join(doc_pages_output_directory_path, file) for file in os.listdir(doc_pages_output_directory_path) if file.endswith('.png')], key=extract_number)
```


```python
def crop_and_save_image_patches_and_tables(doc_pages_output_directory_path):
        
    parental_directory_path  = os.path.dirname(doc_pages_output_directory_path)
    table_images_output_path = os.path.join(parental_directory_path, "table_images")
    pic_images_output_path   = os.path.join(parental_directory_path, "pic_images")
    
    if not os.path.exists(table_images_output_path):
        os.mkdir(table_images_output_path)
    
    if not os.path.exists(pic_images_output_path):
        os.mkdir(pic_images_output_path)
    
    png_files       = sorted([os.path.join(doc_pages_output_directory_path, file) for file in os.listdir(doc_pages_output_directory_path) if file.endswith('.png')], key=extract_number)
    table_png_files = []
    pic_png_files   = []
    
    for file_path in png_files:
        
        results      = model(source = file_path, conf=0.2, iou=0.8)[0]
        detections   = sv.Detections.from_ultralytics(results)
        table        = 0
        pic          = 0
    
        for i, (class_name, confidence) in enumerate(zip(detections.data['class_name'], detections.confidence)):
        
            if (class_name == 'Table' or class_name == 'Picture') and confidence > 0.7:
                
                image_pach_bbox = detections.xyxy[i]
                image           = Image.open(file_path)
    
                cropped_table = image.crop(image_pach_bbox)
    
                if class_name == 'Table':
                    table_image_path = os.path.join(table_images_output_path, f"{os.path.basename(file_path).split('.')[0]}_{table}.png")
                    table_png_files.append(table_image_path)
                    cropped_table.save(table_image_path)
                    table += 1
                else:
                    pic_image_path = os.path.join(pic_images_output_path, f"{os.path.basename(file_path).split('.')[0]}_{pic}.png")
                    pic_png_files.append(pic_image_path)
                    cropped_table.save(pic_image_path)
                    pic += 1

    return pic_png_files, table_png_files
```


```python
doc_pages_output_directory_path
```




    'DataScience Interview Questions/doc_pages'




```python
imagettes_files, table_files = crop_and_save_image_patches_and_tables(doc_pages_output_directory_path)
```

    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_1.png: 640x512 1 List-item, 1 Page-header, 10 Section-headers, 8 Texts, 4783.7ms
    Speed: 4.3ms preprocess, 4783.7ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_2.png: 640x512 1 Picture, 6 Section-headers, 1 Table, 8 Texts, 5177.9ms
    Speed: 1.5ms preprocess, 5177.9ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_3.png: 640x512 4 List-items, 1 Page-footer, 1 Picture, 1 Section-header, 7 Texts, 5103.2ms
    Speed: 2.3ms preprocess, 5103.2ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_4.png: 640x512 2 List-items, 1 Page-footer, 2 Pictures, 3 Section-headers, 1 Table, 4 Texts, 4601.8ms
    Speed: 1.6ms preprocess, 4601.8ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_5.png: 640x512 12 List-items, 3 Pictures, 2 Section-headers, 5 Texts, 4616.2ms
    Speed: 1.6ms preprocess, 4616.2ms inference, 77.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_6.png: 640x512 2 Captions, 7 List-items, 1 Picture, 5 Section-headers, 2 Tables, 9 Texts, 4604.8ms
    Speed: 2.1ms preprocess, 4604.8ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_7.png: 640x512 1 Page-footer, 1 Picture, 2 Section-headers, 7 Texts, 4588.4ms
    Speed: 1.6ms preprocess, 4588.4ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_8.png: 640x512 4 List-items, 3 Section-headers, 2 Tables, 17 Texts, 4682.3ms
    Speed: 1.6ms preprocess, 4682.3ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_9.png: 640x512 3 List-items, 1 Page-footer, 1 Page-header, 4 Section-headers, 2 Tables, 11 Texts, 4699.2ms
    Speed: 1.5ms preprocess, 4699.2ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_10.png: 640x512 1 Caption, 1 Page-footer, 2 Pictures, 5 Section-headers, 5 Texts, 4784.2ms
    Speed: 1.5ms preprocess, 4784.2ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_11.png: 640x512 3 List-items, 2 Page-footers, 6 Section-headers, 10 Texts, 4599.7ms
    Speed: 2.2ms preprocess, 4599.7ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_12.png: 640x512 2 Page-footers, 1 Picture, 2 Section-headers, 4 Texts, 4778.4ms
    Speed: 1.5ms preprocess, 4778.4ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_13.png: 640x512 7 List-items, 1 Picture, 4 Section-headers, 8 Texts, 4563.5ms
    Speed: 1.5ms preprocess, 4563.5ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_14.png: 640x512 2 List-items, 1 Page-footer, 5 Section-headers, 11 Texts, 4596.3ms
    Speed: 1.8ms preprocess, 4596.3ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_15.png: 640x512 1 Page-footer, 6 Section-headers, 14 Texts, 4591.9ms
    Speed: 1.4ms preprocess, 4591.9ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_16.png: 640x512 4 Captions, 1 Page-footer, 7 Pictures, 5 Section-headers, 1 Table, 11 Texts, 4678.0ms
    Speed: 1.6ms preprocess, 4678.0ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_17.png: 640x512 5 List-items, 2 Pictures, 2 Section-headers, 4 Texts, 4479.4ms
    Speed: 1.5ms preprocess, 4479.4ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_18.png: 640x512 1 Caption, 2 Page-footers, 4 Pictures, 4 Section-headers, 1 Table, 5 Texts, 4451.0ms
    Speed: 1.5ms preprocess, 4451.0ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_19.png: 640x512 1 Page-footer, 1 Picture, 4 Section-headers, 4 Tables, 7 Texts, 1 Title, 4599.3ms
    Speed: 1.5ms preprocess, 4599.3ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_20.png: 640x512 1 Caption, 3 List-items, 2 Pictures, 6 Section-headers, 8 Texts, 4576.9ms
    Speed: 1.6ms preprocess, 4576.9ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_21.png: 640x512 8 List-items, 1 Page-footer, 3 Section-headers, 1 Table, 10 Texts, 4683.4ms
    Speed: 1.5ms preprocess, 4683.4ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_22.png: 640x512 4 List-items, 1 Page-footer, 2 Pictures, 4 Section-headers, 7 Texts, 4978.8ms
    Speed: 1.5ms preprocess, 4978.8ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_23.png: 640x512 1 Caption, 1 Page-footer, 2 Pictures, 2 Section-headers, 2 Texts, 5246.6ms
    Speed: 1.7ms preprocess, 5246.6ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_24.png: 640x512 4 List-items, 1 Picture, 3 Section-headers, 10 Texts, 4897.4ms
    Speed: 1.5ms preprocess, 4897.4ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_25.png: 640x512 7 List-items, 1 Picture, 2 Section-headers, 8 Texts, 4799.8ms
    Speed: 2.5ms preprocess, 4799.8ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_26.png: 640x512 7 List-items, 1 Page-footer, 1 Picture, 5 Section-headers, 8 Texts, 5194.4ms
    Speed: 1.7ms preprocess, 5194.4ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_27.png: 640x512 2 List-items, 2 Page-footers, 1 Picture, 3 Section-headers, 5 Texts, 4869.7ms
    Speed: 1.4ms preprocess, 4869.7ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_28.png: 640x512 2 List-items, 1 Page-footer, 1 Picture, 5 Section-headers, 7 Texts, 4479.3ms
    Speed: 1.5ms preprocess, 4479.3ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_29.png: 640x512 5 List-items, 2 Page-footers, 1 Picture, 4 Section-headers, 4 Texts, 4486.8ms
    Speed: 1.5ms preprocess, 4486.8ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_30.png: 640x512 2 Captions, 4 List-items, 1 Page-footer, 1 Picture, 3 Section-headers, 8 Texts, 5202.4ms
    Speed: 2.2ms preprocess, 5202.4ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_31.png: 640x512 3 Page-footers, 2 Pictures, 1 Section-header, 7 Texts, 4606.2ms
    Speed: 2.6ms preprocess, 4606.2ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_32.png: 640x512 1 Caption, 12 List-items, 1 Picture, 6 Section-headers, 7 Texts, 4878.4ms
    Speed: 1.5ms preprocess, 4878.4ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_33.png: 640x512 3 List-items, 2 Page-footers, 1 Picture, 5 Section-headers, 1 Table, 3 Texts, 4499.8ms
    Speed: 1.6ms preprocess, 4499.8ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_34.png: 640x512 1 Page-footer, 2 Pictures, 3 Section-headers, 1 Table, 5 Texts, 1 Title, 4554.0ms
    Speed: 1.6ms preprocess, 4554.0ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_35.png: 640x512 3 Captions, 1 Page-footer, 2 Pictures, 3 Section-headers, 1 Table, 3 Texts, 4449.0ms
    Speed: 1.6ms preprocess, 4449.0ms inference, 82.9ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_36.png: 640x512 5 List-items, 1 Page-footer, 2 Pictures, 1 Section-header, 2 Texts, 5200.8ms
    Speed: 25.3ms preprocess, 5200.8ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_37.png: 640x512 13 List-items, 1 Picture, 3 Section-headers, 5 Texts, 4601.7ms
    Speed: 1.5ms preprocess, 4601.7ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_38.png: 640x512 1 Page-footer, 1 Page-header, 2 Section-headers, 1 Table, 7 Texts, 4579.8ms
    Speed: 1.5ms preprocess, 4579.8ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_39.png: 640x512 1 Footnote, 1 Page-footer, 2 Tables, 3 Texts, 4789.3ms
    Speed: 1.5ms preprocess, 4789.3ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_40.png: 640x512 5 List-items, 1 Page-footer, 2 Section-headers, 2 Tables, 9 Texts, 4954.7ms
    Speed: 1.5ms preprocess, 4954.7ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_41.png: 640x512 1 Page-footer, 1 Table, 9 Texts, 4863.4ms
    Speed: 1.5ms preprocess, 4863.4ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_42.png: 640x512 1 Page-footer, 2 Section-headers, 2 Tables, 15 Texts, 4693.0ms
    Speed: 1.5ms preprocess, 4693.0ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_43.png: 640x512 1 Section-header, 1 Table, 9 Texts, 4482.8ms
    Speed: 1.6ms preprocess, 4482.8ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_44.png: 640x512 1 Page-footer, 1 Picture, 3 Section-headers, 1 Table, 12 Texts, 4574.9ms
    Speed: 1.6ms preprocess, 4574.9ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_45.png: 640x512 3 List-items, 1 Page-footer, 3 Section-headers, 12 Texts, 1 Title, 4596.7ms
    Speed: 1.4ms preprocess, 4596.7ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_46.png: 640x512 1 Caption, 4 Pictures, 3 Section-headers, 15 Texts, 4690.3ms
    Speed: 1.6ms preprocess, 4690.3ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_47.png: 640x512 2 Page-footers, 2 Pictures, 2 Section-headers, 12 Texts, 1 Title, 5109.1ms
    Speed: 2.0ms preprocess, 5109.1ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_48.png: 640x512 2 List-items, 1 Page-footer, 1 Picture, 1 Section-header, 16 Texts, 4790.5ms
    Speed: 1.7ms preprocess, 4790.5ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_49.png: 640x512 8 List-items, 1 Page-footer, 1 Page-header, 1 Section-header, 13 Texts, 4403.7ms
    Speed: 1.5ms preprocess, 4403.7ms inference, 77.2ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_50.png: 640x512 14 List-items, 1 Page-footer, 9 Texts, 4492.5ms
    Speed: 1.5ms preprocess, 4492.5ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_51.png: 640x512 8 List-items, 7 Section-headers, 6 Texts, 1 Title, 4594.2ms
    Speed: 1.5ms preprocess, 4594.2ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_52.png: 640x512 3 List-items, 1 Page-footer, 7 Section-headers, 8 Texts, 4500.0ms
    Speed: 1.5ms preprocess, 4500.0ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_53.png: 640x512 10 List-items, 1 Page-footer, 5 Section-headers, 6 Texts, 4800.5ms
    Speed: 1.5ms preprocess, 4800.5ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 512)
    
    image 1/1 /home/jovyan/lecture-notes/05-rag-embeddings/DataScience Interview Questions/doc_pages/page_54.png: 640x512 13 List-items, 1 Page-footer, 3 Section-headers, 4 Texts, 4476.0ms
    Speed: 1.5ms preprocess, 4476.0ms inference, 0.8ms postprocess per image at shape (1, 3, 640, 512)


# Questions<a href="#Questions" class="anchor-link">¶</a>


```python
Questions = \
["What is the bias-variance trade-off?",
"Which factors dominate the model's error at which stages?",
"What is the bias-variance trade-off?",
"Illustrate how the specific model achieves the bias-variance trade-off with examples.",
"What is a confusion matrix?",
"What kind of problems are the confusion matrix used for?",
"What can be derived from the confusion matrix?",
"What is the correlation and covariance in statistics?",
"What is the plot of large negative covariance be like?",
"What is the plot of nearly zero covariance be like?",
"What is the plot of large positive covariance be like?",
"What is the plot of positive correlation be like?",
"What is the plot of zero correlation be like?",
"What is the plot of negative correlation be like?",
"What is the p-value?"
"Based on what kind of the behavior of p-value, we accept and reject the null hypothesis?",
"How to avoid the overfitting and underfitting?",
"What is the selection bias?",
"What is the best ideal ROC curve be like?",
"What is the formula of Softmax function?",
"How much of the time does data pre-processing take?" ,
"What classification algorithms does classifier include? List concrete algorithm(s) under each classification algorithm class.",
"Which algorithm can solve the case with both numerical and categorical data being involved?",
"What is the separating hyperlane?",
"What's the equation of the separating hyperlane for SVM?",
"Give a concrete example of the decision tree.",
"What is the containment relationship between the artificial inteligence, machine learning and deep learning?",
 ]
```

# LLM


```python
from openai import OpenAI
import os
# 加载环境变量
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 OPENAI_API_KEY

client = OpenAI()
```

## Embedding by LLM<a href="#Embedding-by-LLM" class="anchor-link">¶</a>


```python
import time

def get_embeddings(texts, model="text-embedding-ada-002", dimensions=None, delay=1):

    '''Encapsulate OpenAI's Embedding model interface'''

    embeddings = []

    if model == "text-embedding-ada-002":
        dimensions = None
    if dimensions:
        data = client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
    else:
        data = client.embeddings.create(input=texts, model=model).data

    embeddings.append(data[0].embedding)
    time.sleep(delay)  # Wait for 'delay' seconds before the next request

    return [x.embedding for x in data]
```

## Prompting<a href="#Prompting" class="anchor-link">¶</a>


```python
prompt_template = """
You are a question-answering robot.
Your task is to answer user questions based on the given known information as follows.

Known Information:
{context}

User Question：
{query}

If the known information does not contain the answer to the user's question, or the known information is insufficient to answer the user's question, please directly reply "I can't answer your question".
Please do not output information or answers that are not included in the known information.
Please answer user questions in English.
"""

def build_prompt(prompt_template, **kwargs):

    ''' Assign the Prompt template '''

    inputs = {}

    for k, v in kwargs.items():
        if isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n\n'.join(v)
        else:
            val = v
        inputs[k] = val

    return prompt_template.format(**inputs)
```

## Invoke LLM<a href="#Invoke-LLM" class="anchor-link">¶</a>

### GPT-3.5 & GPT-4<a href="#GPT-3.5-&amp;-GPT-4" class="anchor-link">¶</a>


```python
def get_completion(prompt, model="gpt-3.5-turbo-1106"):

    '''Encapsulate openai interface'''

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model = model,
                                              messages = messages,
                                              temperature = 0,  # The randomness of the model output, 0 means the least randomness
                                              )
    return response.choices[0].message.content
```

### For image description<a href="#For-image-description" class="anchor-link">¶</a>


```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_qa(query, image_path):
    
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(model       = "gpt-4o",
                                              temperature = 0,
                                              seed        = 42,
                                              messages    = [{"role": "user",
                                                              "content": [{"type": "text", "text": query},
                                                                          {"type": "image_url",
                                                                           "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",},
                                                                           },
                                                                          ],
                                                               } 
                                                              ],
                                               )

    if response and response.choices:
        if response.choices[0]:
            return response.choices[0].message.content
    
    return None
```


```python
def generate_image_description(image_file_paths, corpus):

    images_descriptions = []
    for image_path in image_file_paths:
        image_description = image_qa("Please briefly describe the information in the image", image_path)
        if image_description:
            images_descriptions.append(image_description)

    corpus.extend(images_descriptions)

    return images_descriptions
```


```python
imagettes_description = generate_image_description(imagettes_files, corpus)
```


```python
tables_description = generate_image_description(table_files, corpus)
```

### Function to generate queries using OpenAI's ChatGPT<a href="#Function-to-generate-queries-using-OpenAI&#39;s-ChatGPT"


```python
# Function to generate queries using OpenAI's ChatGPT

def generate_queries_by_llm(original_query, num_generated_queries, model="gpt-3.5-turbo-1106"):

    response = client.chat.completions.create(model=model,
                                              messages=[{"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
                                                        {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
                                                        {"role": "user", "content": f"OUTPUT ({num_generated_queries} queries):"}]
                                              )

    generated_queries = response.choices[0].message.content.strip().split("\n")

    return generated_queries
```

# Database<a href="#Database" class="anchor-link">¶</a>

## Elasticsearch database<a href="#Elasticsearch-database" class="anchor-link">¶</a>


```python
from elasticsearch7 import Elasticsearch, helpers
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import os
import warnings
warnings.simplefilter("ignore")  # Shield some ES Warnings

#nltk.download('punkt')  # English words segmentation, word root, sentence segmentation and other methods
#nltk.download('stopwords')  # English stop words

# Import configuration files
ELASTICSEARCH_BASE_URL = os.getenv('ELASTICSEARCH_BASE_URL')
ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')
ELASTICSEARCH_NAME     = os.getenv('ELASTICSEARCH_NAME')
```






```python
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
```


```python
import time
class ElasticsearchDatabase:

    def __init__(self) -> None:
        
        self.es = Elasticsearch(hosts     = ['http://39.106.15.22:9200/'],  # Service address and port
                                http_auth = ("elastic", "FKaB1Jpz0Rlw0l6G"),  # Username Password
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
```


```python
es_db = ElasticsearchDatabase()
es_db.create_index(index_name = 'es_docs_images')
```


```python
es_db.add_texts_and_descriptions_of_imagettes_and_tables(corpus, 'es_docs_images')
```

## Vector database<a href="#Vector-database" class="anchor-link">¶</a>


```python
import os
if os.environ.get('CUR_ENV_IS_STUDENT',False):
    import sys
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```


```python
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
```


```python
vector_db = VectorDBConnectorSupportingMultimodal("multimodal", get_embeddings)
```


```python
vector_db.add_texts(original_corpus)
```


```python
vector_db.add_imagettes_description(imagettes_description)
```


```python
vector_db.add_tables_description(tables_description)
```

# RAG<a href="#RAG" class="anchor-link">¶</a>

## Reciprocal Rank Fusion<a href="#Reciprocal-Rank-Fusion" class="anchor-link">¶</a>


```python
def rrf(documents, db_search_context_results_list, k=1):
        
    # Iterate db_search_results_list to generate the {"doc_{num}": {"text": , "rank":}} 
    # for each db_search_results in db_search_results_list, save them in all_search_results
    all_search_context_results = []
    
    for db_search_context_results in db_search_context_results_list:
        
        search_results = {"doc_"+str(documents.index(doc)): {"text": doc, "rank": i} 
                          for i, doc in enumerate(db_search_context_results)}

        all_search_context_results.append(search_results)

    # Iterate all_search_results and compute reciprocal rank fusion score
    ret = {}
    
    for rank in all_search_context_results:

        for id, val in rank.items():
            
            if id not in ret:
                ret[id] = {"score": 0, 
                           "text":  val["text"]}
            
            ret[id]["score"] += 1.0/(k+val["rank"])
    
    ret_sorted   = sorted(ret.items(), key=lambda item: item[1]["score"], reverse=True)
    sorted_texts = [item[1]["text"] for item in ret_sorted]

    return sorted_texts
```

## RAG parental class<a href="#RAG-parental-class" class="anchor-link">¶</a>


```python
from abc import ABC, abstractmethod

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
```

## Naive RAG<a href="#Naive-RAG" class="anchor-link">¶</a>


```python
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
```


```python
# RAG based vector retrieval

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

```


```python
naive_rag_bot =  RAG_Bot_without_ranking(llm_api = get_completion,
                                         num_results_retrieved_by_database = 6,
                                         num_results_selected_for_prompt = 3,
                                         documents = corpus,
                                         vector_db = vector_db,
                                         es_db = es_db,
                                         es_index = 'es_docs_images')
```


```python
for i, query in enumerate(Questions):
        
    response = naive_rag_bot.chat(user_query = query,
                                  model_type = 'gpt-4o')
    
    print("{}:\n Query: {}\n Answer: {}\n".format(i, query,response))
```

    0:
     Query: What is the bias-variance trade-off?
     Answer: The bias-variance trade-off is a concept in machine learning that refers to the balance between two types of errors that affect the performance of a model:
    
    1. **Bias**: This is the error introduced due to oversimplification of the machine learning algorithm. High bias can lead to underfitting, where the model makes simplified assumptions and fails to capture the underlying patterns in the data. Examples of high bias algorithms include Linear Regression and Logistic Regression.
    
    2. **Variance**: This is the error introduced due to the complexity of the machine learning algorithm. High variance can lead to overfitting, where the model learns noise from the training data and performs poorly on new, unseen data. Examples of low bias but high variance algorithms include Decision Trees, k-NN, and SVM.
    
    The goal of any supervised machine learning algorithm is to achieve low bias and low variance to ensure good prediction performance. However, there is a trade-off between bias and variance:
    
    - Increasing the complexity of the model reduces bias but increases variance.
    - Decreasing the complexity of the model increases bias but reduces variance.
    
    The optimal model complexity is where the total error, which is the sum of bias squared and variance, is minimized. This balance is crucial for developing models that generalize well to new data.
    
    1:
     Query: Which factors dominate the model's error at which stages?
     Answer: The known information indicates that:
    
    - At low model complexity, the error is dominated by high bias (as shown by the red curve).
    - At high model complexity, the error is dominated by high variance (as shown by the blue curve).
    
    Therefore, at different stages of model complexity, either bias or variance dominates the model's error.
    
    2:
     Query: What is the bias-variance trade-off?
     Answer: The bias-variance trade-off is a concept in machine learning that refers to the balance between two types of errors that affect the performance of a model:
    
    1. **Bias**: This is the error introduced due to oversimplification of the machine learning algorithm. High bias can lead to underfitting, where the model makes simplified assumptions and fails to capture the underlying patterns in the data. Examples of high bias algorithms include Linear Regression and Logistic Regression.
    
    2. **Variance**: This is the error introduced due to the complexity of the machine learning algorithm. High variance can lead to overfitting, where the model learns noise from the training data and performs poorly on new, unseen data. Examples of low bias, high variance algorithms include Decision Trees, k-NN, and SVM.
    
    The goal of any supervised machine learning algorithm is to achieve low bias and low variance to ensure good prediction performance. However, there is a trade-off between bias and variance:
    
    - Increasing model complexity typically reduces bias but increases variance.
    - Decreasing model complexity typically increases bias but reduces variance.
    
    The optimal model complexity is where the total error (sum of bias squared and variance) is minimized. This balance is crucial for developing models that generalize well to new data.
    
    3:
     Query: Illustrate how the specific model achieves the bias-variance trade-off with examples.
     Answer: I can't answer your question.
    
    4:
     Query: What is a confusion matrix?
     Answer: A confusion matrix is a 2x2 table that contains 4 outputs provided by a binary classifier. It is used to evaluate the performance of the classifier by comparing the actual class labels with the predicted class labels. The four outcomes in the confusion matrix are:
    
    1. True Positive (TP) — Correct positive prediction
    2. False Positive (FP) — Incorrect positive prediction
    3. True Negative (TN) — Correct negative prediction
    4. False Negative (FN) — Incorrect negative prediction
    
    Various measures, such as error rate, accuracy, specificity, sensitivity, precision, and recall, are derived from the confusion matrix.
    
    5:
     Query: What kind of problems are the confusion matrix used for?
     Answer: The confusion matrix is used for evaluating the performance of a binary classifier.
    
    6:
     Query: What can be derived from the confusion matrix?
     Answer: Various measures can be derived from the confusion matrix, including:
    
    1. Error Rate
    2. Accuracy
    3. Sensitivity (Recall or True Positive Rate)
    4. Specificity (True Negative Rate)
    5. Precision (Positive Predicted Value)
    6. F-Score (Harmonic mean of precision and recall)
    
    These measures help evaluate the performance of a binary classifier.
    
    7:
     Query: What is the correlation and covariance in statistics?
     Answer: Covariance and Correlation are two mathematical concepts; these two approaches are widely used in statistics. Both Correlation and Covariance establish the relationship and also measure the dependency between two random variables. Though the work is similar between these two in mathematical terms, they are different from each other.
    
    **Correlation:**
    Correlation is considered or described as the best technique for measuring and also for estimating the quantitative relationship between two variables. Correlation measures how strongly two variables are related.
    
    **Covariance:**
    In covariance, two items vary together and it’s a measure that indicates the extent to which two random variables change in cycle. It is a statistical term; it explains the systematic relation between a pair of random variables, wherein changes in one variable are reciprocated by a corresponding change in another variable.
    
    8:
     Query: What is the plot of large negative covariance be like?
     Answer: The plot of large negative covariance would show a scatter plot with a downward trend, indicating that as one variable increases, the other decreases.
    
    9:
     Query: What is the plot of nearly zero covariance be like?
     Answer: The plot of nearly zero covariance would show a scatter plot with no clear trend, indicating no linear relationship between the variables.
    
    10:
     Query: What is the plot of large positive covariance be like?
     Answer: The plot of large positive covariance would show a scatter plot with an upward trend, indicating that as one variable increases, the other also increases.
    
    11:
     Query: What is the plot of positive correlation be like?
     Answer: A plot of positive correlation would show a clear upward trend, indicating a strong positive linear relationship between the variables.
    
    12:
     Query: What is the plot of zero correlation be like?
     Answer: The plot of zero correlation would show a scatter plot with no clear trend, indicating no linear relationship between the variables.
    
    13:
     Query: What is the plot of negative correlation be like?
     Answer: The plot of negative correlation would show a clear downward trend, indicating a strong negative linear relationship between the variables.
    
    14:
     Query: What is the p-value?Based on what kind of the behavior of p-value, we accept and reject the null hypothesis?
     Answer: A p-value is a number between 0 and 1 that helps you determine the strength of your results when you perform a hypothesis test in statistics. The claim which is on trial is called the Null Hypothesis.
    
    - Low p-value (≤ 0.05) indicates strength against the null hypothesis, which means we can reject the null hypothesis.
    - High p-value (≥ 0.05) indicates strength for the null hypothesis, which means we can accept the null hypothesis.
    - A p-value of 0.05 indicates the hypothesis could go either way.
    
    To put it another way:
    - High p-values: your data are likely with a true null.
    - Low p-values: your data are unlikely with a true null.
    
    15:
     Query: How to avoid the overfitting and underfitting?
     Answer: To avoid overfitting and underfitting, you can:
    
    1. **Resample the data to estimate the model accuracy** using techniques like k-fold cross-validation.
    2. **Have a validation dataset** to evaluate the model.
    
    Additionally, to specifically avoid overfitting, you can:
    
    1. **Keep the model simple** by taking fewer variables into account, thereby removing some of the noise in the training data.
    2. **Use cross-validation techniques**, such as k-fold cross-validation.
    3. **Use regularization techniques**, such as LASSO, that penalize certain model parameters if they're likely to cause overfitting.
    
    16:
     Query: What is the selection bias?
     Answer: Selection bias is a kind of error that occurs when the researcher decides who is going to be studied. It is usually associated with research where the selection of participants isn’t random. It is sometimes referred to as the selection effect. It is the distortion of statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may not be accurate. The types of selection bias include:
    
    1. Sampling bias: It is a systematic error due to a non-random sample of a population causing some members of the population to be less likely to be included than others resulting in a biased sample.
    2. Time interval: A trial may be terminated early at an extreme value (often for ethical reasons), but the extreme value is likely to be reached by the variable with the largest variance, even if all variables have a similar mean.
    3. Data: When specific subsets of data are chosen to support a conclusion or rejection of bad data on arbitrary grounds, instead of according to previously stated or generally agreed criteria.
    4. Attrition: Attrition bias is a kind of selection bias caused by attrition (loss of participants) discounting trial subjects/tests that did not run to completion.
    
    17:
     Query: What is the best ideal ROC curve be like?
     Answer: The best ideal ROC curve would be one that is closest to the top-left corner of the plot. This indicates a model with a high True Positive Rate and a low False Positive Rate, representing excellent performance.
    
    18:
     Query: What is the formula of Softmax function?
     Answer: The formula of the Softmax function is:
    
    \[ P(y=j \mid \Theta^{(i)}) = \frac{e^{\Theta^{(i)}}}{\sum_{j=0}^{k} e^{\Theta_k^{(i)}}} \]
    
    where \(\Theta = w_0 x_0 + w_1 x_1 + \ldots + w_k x_k = \sum_{i=0}^{k} w_i x_i = w^T x\).
    
    19:
     Query: How much of the time does data pre-processing take?
     Answer: I can't answer your question.
    
    20:
     Query: What classification algorithms does classifier include? List concrete algorithm(s) under each classification algorithm class.
     Answer: The classification algorithms under each classification algorithm class are as follows:
    
    1. **Linear**
       - Naive Bayes
       - Logistic Regression
    
    2. **Decision Trees**
    
    3. **SVM (Support Vector Machines)**
    
    4. **Kernel Estimation**
    
    5. **Neural Networks**
       - Linear
       - Non-Linear
       - Recurrent Neural Network (RNN)
       - Modular Neural Network
    
    6. **Quadratic**
    
    21:
     Query: Which algorithm can solve the case with both numerical and categorical data being involved?
     Answer: The Decision Tree algorithm can solve the case with both numerical and categorical data being involved.
    
    22:
     Query: What is the separating hyperlane?
     Answer: The separating hyperplane is the line labeled \( w \cdot x + b = 0 \) that separates the two classes of data points in the Support Vector Machine (SVM) classification model.
    
    23:
     Query: What's the equation of the separating hyperlane for SVM?
     Answer: The equation of the separating hyperplane for SVM is \( w \cdot x + b = 0 \).
    
    24:
     Query: Give a concrete example of the decision tree.
     Answer: A concrete example of a decision tree is one used for making decisions based on weather conditions. The root node is labeled "Outlook" and branches into three categories: Sunny, Overcast, and Rainy.
    
    - If the outlook is Sunny, the next decision node is "Windy," which branches into False and True:
      - If Windy is False, the decision is Yes.
      - If Windy is True, the decision is No.
    
    - If the outlook is Overcast, the decision is Yes.
    
    - If the outlook is Rainy, the next decision node is "Humidity," which branches into High and Normal:
      - If Humidity is High, the decision is No.
      - If Humidity is Normal, the decision is Yes.
    
    This decision tree helps in determining whether the final decision is Yes or No based on the given conditions.
    
    25:
     Query: What is the containment relationship between the artificial inteligence, machine learning and deep learning?
     Answer: The containment relationship is as follows: Deep Learning (DL) is a subset of Machine Learning (ML), which in turn is a subset of Artificial Intelligence (AI). This means that DL is contained within ML, and ML is contained within AI.
    



```python
for i, query in enumerate(Questions):
        
    response = naive_rag_bot.chat(user_query = query,
                                  model_type = 'gpt-4o',
                                  num_generated_queries = 2)
    
    print("{}:\n Query: {}\n Answer: {}\n".format(i, query,response))
```

    0:
     Query: What is the bias-variance trade-off?
     Answer: The bias-variance trade-off is a concept in machine learning that refers to the balance between two types of errors that affect the performance of a model:
    
    1. **Bias**: This is the error introduced by oversimplifying the machine learning algorithm. High bias can lead to underfitting, where the model makes simplified assumptions and fails to capture the underlying patterns in the data. Examples of high bias algorithms include Linear Regression and Logistic Regression.
    
    2. **Variance**: This is the error introduced by making the model too complex, causing it to learn noise from the training data and perform poorly on new, unseen data. High variance can lead to overfitting, where the model is highly sensitive to the training data. Examples of low bias, high variance algorithms include Decision Trees, k-NN, and SVM.
    
    The goal of any supervised machine learning algorithm is to achieve low bias and low variance to ensure good prediction performance. However, there is a trade-off between bias and variance:
    
    - Increasing the complexity of the model reduces bias but increases variance.
    - Decreasing the complexity of the model reduces variance but increases bias.
    
    The optimal model complexity is where the total error (sum of bias squared and variance) is minimized, balancing the trade-off between bias and variance. This balance is crucial to create a model that generalizes well to new data, avoiding both under-fitting and over-fitting.
    
    1:
     Query: Which factors dominate the model's error at which stages?
     Answer: At different stages of model complexity, different factors dominate the model's error:
    
    1. **Low Model Complexity**: At this stage, the model is too simple and makes oversimplified assumptions. The dominant factor contributing to the model's error is **high bias**. This can lead to underfitting, where the model fails to capture the underlying patterns in the data.
    
    2. **High Model Complexity**: At this stage, the model is very complex and captures not only the underlying patterns but also the noise in the training data. The dominant factor contributing to the model's error is **high variance**. This can lead to overfitting, where the model performs well on the training data but poorly on the test data.
    
    The goal is to find an optimal model complexity where the total error, which is the sum of bias squared and variance, is minimized. This balance is known as the bias-variance trade-off.
    
    2:
     Query: What is the bias-variance trade-off?
     Answer: The bias-variance trade-off is a concept in machine learning that refers to the balance between two types of errors that affect the performance of a model:
    
    1. **Bias**: This is the error introduced by oversimplifying the machine learning algorithm. High bias can lead to underfitting, where the model makes overly simplified assumptions and fails to capture the underlying patterns in the data. Low bias machine learning algorithms include Decision Trees, k-NN, and SVM, while high bias algorithms include Linear Regression and Logistic Regression.
    
    2. **Variance**: This is the error introduced by the model's sensitivity to small fluctuations in the training data. High variance can lead to overfitting, where the model captures noise and performs poorly on new, unseen data. As model complexity increases, variance typically increases while bias decreases.
    
    The goal of any supervised machine learning algorithm is to achieve low bias and low variance to ensure good prediction performance. However, there is a trade-off between the two: increasing bias will decrease variance and vice versa. The optimal model complexity is where the total error, which is the sum of bias squared and variance, is minimized.
    
    For example:
    - The k-nearest neighbour algorithm has low bias and high variance, but increasing the value of k can increase the bias and decrease the variance.
    - The support vector machine algorithm has low bias and high variance, but increasing the C parameter can increase the bias and decrease the variance.
    
    The image provided illustrates this trade-off, showing that there is an optimal point where the total error is minimized, balancing the trade-off between bias and variance.
    
    3:
     Query: Illustrate how the specific model achieves the bias-variance trade-off with examples.
     Answer: The known information provides examples of how specific models achieve the bias-variance trade-off:
    
    1. **k-Nearest Neighbour (k-NN) Algorithm:**
       - **Low Bias and High Variance:** By default, k-NN has low bias because it makes few assumptions about the data. However, it has high variance because it is sensitive to the specific training data points.
       - **Adjusting the Trade-off:** The trade-off can be managed by increasing the value of \( k \). When \( k \) is increased, more neighbors contribute to the prediction, which smooths out the decision boundary. This increases the bias but decreases the variance, helping to avoid overfitting.
    
    2. **Support Vector Machine (SVM) Algorithm:**
       - **Low Bias and High Variance:** SVMs typically have low bias because they aim to find the optimal hyperplane that separates the classes with maximum margin. However, they can have high variance, especially with a small margin, making them sensitive to the training data.
       - **Adjusting the Trade-off:** The trade-off can be adjusted by increasing the \( C \) parameter. A higher \( C \) value allows fewer margin violations, leading to a more complex model with lower bias but higher variance. Conversely, a lower \( C \) value allows more margin violations, increasing the bias but reducing the variance, thus helping to avoid overfitting.
    
    These examples illustrate how adjusting specific parameters in machine learning models can help achieve a balance between bias and variance, leading to better generalization and prediction performance.
    
    4:
     Query: What is a confusion matrix?
     Answer: A confusion matrix is a 2x2 table that contains 4 outputs provided by a binary classifier. It is used to evaluate the performance of the classifier by comparing the actual class labels with the predicted class labels. The four outcomes in the confusion matrix are:
    
    1. True Positive (TP) — Correct positive prediction
    2. False Positive (FP) — Incorrect positive prediction
    3. True Negative (TN) — Correct negative prediction
    4. False Negative (FN) — Incorrect negative prediction
    
    Various measures, such as error rate, accuracy, specificity, sensitivity, precision, and recall, are derived from the confusion matrix.
    
    5:
     Query: What kind of problems are the confusion matrix used for?
     Answer: The confusion matrix is used for evaluating the performance of a binary classifier.
    
    6:
     Query: What can be derived from the confusion matrix?
     Answer: Various measures can be derived from the confusion matrix, including:
    
    1. Error Rate = (FP + FN) / (P + N)
    2. Accuracy = (TP + TN) / (P + N)
    3. Sensitivity (Recall or True Positive Rate) = TP / P
    4. Specificity (True Negative Rate) = TN / N
    5. Precision (Positive Predicted Value) = TP / (TP + FP)
    6. F-Score (Harmonic Mean of Precision and Recall) = (1 + b) * (PREC * REC) / (b² * PREC + REC), where b is commonly 0.5, 1, or 2.
    
    7:
     Query: What is the correlation and covariance in statistics?
     Answer: Covariance and Correlation are two mathematical concepts; these two approaches are widely used in statistics. Both Correlation and Covariance establish the relationship and also measure the dependency between two random variables. Though the work is similar between these two in mathematical terms, they are different from each other.
    
    **Correlation:**
    Correlation is considered or described as the best technique for measuring and also for estimating the quantitative relationship between two variables. Correlation measures how strongly two variables are related.
    
    **Covariance:**
    In covariance, two items vary together and it’s a measure that indicates the extent to which two random variables change in cycle. It is a statistical term; it explains the systematic relation between a pair of random variables, wherein changes in one variable reciprocal by a corresponding change in another variable.
    
    8:
     Query: What is the plot of large negative covariance be like?
     Answer: The plot of large negative covariance would be a scatter plot showing a downward trend, indicating that as one variable increases, the other decreases.
    
    9:
     Query: What is the plot of nearly zero covariance be like?
     Answer: The plot of nearly zero covariance would show a scatter plot with no clear trend, indicating no linear relationship between the variables.
    
    10:
     Query: What is the plot of large positive covariance be like?
     Answer: The plot of large positive covariance would show a scatter plot with an upward trend, indicating that as one variable increases, the other also increases.
    
    11:
     Query: What is the plot of positive correlation be like?
     Answer: A plot of positive correlation would show a clear upward trend, indicating a strong positive linear relationship between the variables. This means that as one variable increases, the other variable also increases.
    
    12:
     Query: What is the plot of zero correlation be like?
     Answer: The plot of zero correlation would show a scatter plot with no clear trend, indicating no linear relationship between the variables.
    
    13:
     Query: What is the plot of negative correlation be like?
     Answer: A plot of negative correlation would show a clear downward trend, indicating a strong negative linear relationship between the variables. This means that as one variable increases, the other variable decreases.
    
    14:
     Query: What is the p-value?Based on what kind of the behavior of p-value, we accept and reject the null hypothesis?
     Answer: A p-value is a number between 0 and 1 that helps you determine the strength of your results when you perform a hypothesis test in statistics. The claim which is on trial is called the Null Hypothesis. 
    
    - A low p-value (≤ 0.05) indicates strong evidence against the null hypothesis, which means we can reject the null hypothesis.
    - A high p-value (≥ 0.05) indicates weak evidence against the null hypothesis, which means we can accept the null hypothesis.
    - A p-value of 0.05 indicates that the hypothesis could go either way.
    
    In summary, based on the behavior of the p-value:
    - If the p-value is ≤ 0.05, we reject the null hypothesis.
    - If the p-value is > 0.05, we accept the null hypothesis.
    
    15:
     Query: How to avoid the overfitting and underfitting?
     Answer: To avoid overfitting and underfitting, you can:
    
    1. **Resample the data to estimate the model accuracy (k-fold cross-validation)**: This helps in assessing how the model will generalize to an independent dataset.
    2. **Have a validation dataset to evaluate the model**: This allows you to check the model's performance on unseen data and adjust accordingly.
    
    Additionally, to specifically avoid overfitting, you can:
    1. **Keep the model simple**: Take fewer variables into account to remove some of the noise in the training data.
    2. **Use regularization techniques**: Techniques like LASSO penalize certain model parameters that are likely to cause overfitting.
    
    These methods help in balancing the model complexity and ensuring it generalizes well to new data.
    
    16:
     Query: What is the selection bias?
     Answer: Selection bias is a kind of error that occurs when the researcher decides who is going to be studied. It is usually associated with research where the selection of participants isn’t random. It is sometimes referred to as the selection effect. It is the distortion of statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may not be accurate.
    
    17:
     Query: What is the best ideal ROC curve be like?
     Answer: The best ideal ROC curve would be one that is closest to the top-left corner of the plot. This indicates a high true positive rate and a low false positive rate, representing the best performance of a classification model.
    
    18:
     Query: What is the formula of Softmax function?
     Answer: The formula of the Softmax function is:
    
    \[ P(y=j \mid \Theta^{(i)}) = \frac{e^{\Theta^{(i)}}}{\sum_{j=0}^{k} e^{\Theta_k^{(i)}}} \]
    
    where \(\Theta = w_0 x_0 + w_1 x_1 + \ldots + w_k x_k = \sum_{i=0}^{k} w_i x_i = w^T x\).
    
    19:
     Query: How much of the time does data pre-processing take?
     Answer: I can't answer your question.
    
    20:
     Query: What classification algorithms does classifier include? List concrete algorithm(s) under each classification algorithm class.
     Answer: I can't answer your question.
    
    21:
     Query: Which algorithm can solve the case with both numerical and categorical data being involved?
     Answer: The algorithm that can handle both numerical and categorical data is the Decision Tree algorithm.
    
    22:
     Query: What is the separating hyperlane?
     Answer: The separating hyperplane is the line labeled \( w \cdot x + b = 0 \) that separates the two classes in the Support Vector Machine (SVM) classification model.
    
    23:
     Query: What's the equation of the separating hyperlane for SVM?
     Answer: The equation of the separating hyperplane for SVM is \( w \cdot x + b = 0 \).
    
    24:
     Query: Give a concrete example of the decision tree.
     Answer: A concrete example of a decision tree is one used for making decisions based on weather conditions. The root node is labeled "Outlook" and branches into three categories: Sunny, Overcast, and Rainy.
    
    - If the outlook is Sunny, the next decision node is "Windy," which branches into False and True:
      - If Windy is False, the decision is Yes.
      - If Windy is True, the decision is No.
    
    - If the outlook is Overcast, the decision is Yes.
    
    - If the outlook is Rainy, the next decision node is "Humidity," which branches into High and Normal:
      - If Humidity is High, the decision is No.
      - If Humidity is Normal, the decision is Yes.
    
    This decision tree helps in determining whether the final decision is Yes or No based on the given conditions.
    
    25:
     Query: What is the containment relationship between the artificial inteligence, machine learning and deep learning?
     Answer: The containment relationship between Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) is as follows:
    
    - Artificial Intelligence (AI) is the broadest category, represented by the outermost circle.
    - Machine Learning (ML) is a subset of AI, depicted as a smaller circle within the AI circle.
    - Deep Learning (DL) is a subset of ML, represented as the innermost circle within the ML circle.
    
    This means that Deep Learning is a specialized area within Machine Learning, which in turn is a specialized area within the broader field of Artificial Intelligence.
    



```python

```
