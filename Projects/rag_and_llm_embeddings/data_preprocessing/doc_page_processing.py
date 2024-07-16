import os
import fitz
from PIL import Image
import supervision as sv

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


def crop_and_save_image_patches_and_tables(doc_pages_output_directory_path, image_detection_model):
        
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
        
        results      = image_detection_model(source = file_path, conf=0.2, iou=0.8)[0]
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
