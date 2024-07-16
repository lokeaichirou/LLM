import re

def extract_number(file_name):
    # Extract the numeric portion of a file name，assume the file name is in the format "page_<number>.png"
    
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else -1
