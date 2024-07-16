import fitz  # PyMuPDF

def extract_questions_from_pdf(pdf_file):
    
    """ Extract questions from <<DataScience Interview Questions>> PDF file."""

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
