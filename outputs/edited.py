import os
import re
import pandas as pd
import PyPDF2
import pdfplumber
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from moviepy import VideoFileClip
import fitz  # PyMuPDF for extracting images

# Step 1: Extract Text from PDF, Excluding Table Content
def extract_text_from_pdf(pdf_path):
    text = ""
    tables_text = set()  # To store text from tables
    try:
        # Extract text and tables simultaneously to filter overlapping content
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract raw text from page
                page_text = page.extract_text() or ""
    
                text += page_text
                
                # Extract tables from the same page
                page_tables = page.extract_tables()
                for table in page_tables:
                    for row in table:
                        tables_text.update(" ".join(str(cell) for cell in row).split())

        # Remove any table-related text from the extracted text
        filtered_text = " ".join(word for word in text.split() if word not in tables_text)
        return filtered_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# Step 2: Extract Tables from PDF
def extract_tables_from_pdf(pdf_path):
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    tables.append(table)
        return tables
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

def save_tables_to_excel(tables, output_path):
    try:
        # Ensure the directory exists for saving the Excel file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with pd.ExcelWriter(output_path) as writer:
            for i, table in enumerate(tables):
                df = pd.DataFrame(table)
                df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)
        print(f"Tables saved to {output_path}")
    except Exception as e:
        print(f"Error saving tables: {e}")

# Step 3: Preprocess Text
def preprocess_text(text):
    """
    Clean the extracted text by removing non-printable characters and excessive whitespace.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)           # Replace multiple spaces and newlines with a single space
    return text.strip()

# Step 4: Summarize Text, Excluding Table Content
model_name = "allenai/led-large-16384-arxiv"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def chunk_text(text, max_tokens=16000):
    """
    Split text into smaller chunks, ensuring each chunk has fewer than max_tokens tokens.
    """
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def summarize_text(text_output_path, summary_output_path):
    try:
        with open(text_output_path, 'r', encoding='utf-8') as file:
            text = file.read()

        text = preprocess_text(text)
        text_chunks = chunk_text(text, max_tokens=16000)
        summaries = []

        for i, chunk in enumerate(text_chunks):
            print(f"Summarizing chunk {i + 1}/{len(text_chunks)}...")
            input_ids = tokenizer(chunk, return_tensors="pt", truncation=True).input_ids
            summary_ids = model.generate(input_ids, max_length=512, min_length=50, no_repeat_ngram_size=3)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        # Filter out content already found in tables
        tables = extract_tables_from_pdf(text_output_path)  # Reuse table extraction logic
        table_content = set(" ".join(str(cell) for row in table for cell in row).split() for table in tables)
        summary_text = " ".join(word for word in " ".join(summaries).split() if word not in table_content)

        # Ensure the directory exists for saving the summary file
        os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

        with open(summary_output_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary_text)
        print(f"Summary saved to {summary_output_path}")
        return summary_text
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return ""

# Step 5: Question and Answer Generation
def generate_questions_and_answers(summary, qa_output_path):
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    summary = preprocess_text(summary)
    questions = question_generator(summary, max_length=128, num_beams=5, num_return_sequences=5)  # Use beam search

    # Ensure the directory exists for saving the Q&A file
    os.makedirs(os.path.dirname(qa_output_path), exist_ok=True)

    with open(qa_output_path, "w", encoding="utf-8") as qa_file:
        for q in questions:
            question = q['generated_text']
            answer = qa_model(question=question, context=summary)['answer']
            qa_file.write(f"Q: {question}\nA: {answer}\n\n")
    print(f"Questions and Answers saved to {qa_output_path}")

# Step 6: Extract Important Points
def extract_important_points(text, output_path, num_points=5):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarization_pipeline(text, max_length=130, min_length=30, do_sample=False)
    points = summary[0]['summary_text'].split('. ')

    # Ensure the directory exists for saving important points file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as points_file:
        for i, point in enumerate(points[:num_points], start=1):
            points_file.write(f"{i}. {point.strip()}.\n")
    print(f"Important Points saved to {output_path}")

# Step 7: Video Summarization
'''def summarize_video(video_path, summary_output_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        summarized_clip = clip.subclip(0, min(60, duration))  # Example: First 60 seconds
        os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)  # Ensure directory exists
        summarized_clip.write_videofile(summary_output_path, codec="libx264")
        print(f"Video summarized and saved to {summary_output_path}")
    except Exception as e:
        print(f"Error summarizing video: {e}")'''

# Step 8: Extract Images from PDF
def extract_images_from_pdf(pdf_path, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory for images exists

        pdf_document = fitz.open(pdf_path)
        for i in range(len(pdf_document)):
            page = pdf_document[i]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_filename = os.path.join(output_dir, f"page_{i+1}_img_{img_index+1}.png")
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
        print(f"Images extracted to {output_dir}")
    except Exception as e:
        print(f"Error extracting images: {e}")

# Main Function
def main():
    confirm=True
    while(confirm):

        print("1) TEXT EXTRACTION AND TABLES EXTRACTION \n 2)SUMMARISE THE TEXT \n 3)QUESTIONS AND ANSWERS GENERATION \n 4)EXTRACT IMPORTANT POINTS \n 5)EXTRACT IMAGES FROM PDF")
        number=eval(input("enter the correct key to perform any of the operation"))
        if number==1:
            pdf_path = input("Enter the path to the PDF file: ")
            output_excel_path = input("Enter the path to save the tables into Excel file: ")
            text_output_path = os.path.splitext(output_excel_path)[0] + "_text.txt"
            if not os.path.exists(pdf_path):
                print(f"Error: File not found at {pdf_path}")
                return
            text = extract_text_from_pdf(pdf_path)
            os.makedirs(os.path.dirname(text_output_path), exist_ok=True)  # Ensure directory exists for text output
            with open(text_output_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)
            print(f"Text extracted and saved to {text_output_path}")
            output_dir = os.path.dirname(output_excel_path)
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory for output Excel file exists
            tables = extract_tables_from_pdf(pdf_path)
            if tables:
                save_tables_to_excel(tables, output_excel_path)
            else:
                print("No tables found in the PDF.")
      
        elif number==2:
            pdf_path = input("Enter the path to the PDF file: ")
            output_excel_path = input("Enter the path to separate  tables from the text file: ")
            text_output_path = os.path.splitext(output_excel_path)[0] + "_text.txt"
            if not os.path.exists(pdf_path):
                print(f"Error: File not found at {pdf_path}")
                return
            text = extract_text_from_pdf(pdf_path)
            os.makedirs(os.path.dirname(text_output_path), exist_ok=True)  # Ensure directory exists for text output
            with open(text_output_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)
            #print(f"Text extracted and saved to {text_output_path}")
            output_dir = os.path.dirname(output_excel_path)
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory for output Excel file exists
            tables = extract_tables_from_pdf(pdf_path)
            if tables:
                save_tables_to_excel(tables, output_excel_path)
            else:
                print("No tables found in the PDF.")
                
            summary_output_path = os.path.splitext(output_excel_path)[0] + "_summary.txt"
            summary = summarize_text(text_output_path, summary_output_path)
            print("Summary generated and saved.")
        elif number==3:
            pdf_path = input("Enter the path to the PDF file: ")
            output_excel_path = input("Enter the path to separate  tables from the text file: ")
            text_output_path = os.path.splitext(output_excel_path)[0] + "_text.txt"
            if not os.path.exists(pdf_path):
                print(f"Error: File not found at {pdf_path}")
                return
            text = extract_text_from_pdf(pdf_path)
            os.makedirs(os.path.dirname(text_output_path), exist_ok=True)  # Ensure directory exists for text output
            with open(text_output_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)
            #print(f"Text extracted and saved to {text_output_path}")
            output_dir = os.path.dirname(output_excel_path)
            os.makedirs(output_dir, exist_ok=True)  # Ensure the directory for output Excel file exists
            tables = extract_tables_from_pdf(pdf_path)
            if tables:
                save_tables_to_excel(tables, output_excel_path)
            else:
                #print("No tables found in the PDF.")
                pass
            summary_output_path = os.path.splitext(output_excel_path)[0] + "_summary.txt"
            summary = summarize_text(text_output_path, summary_output_path)
            #print("Summary generated and saved.")
            qa_output_path = os.path.splitext(output_excel_path)[0] + "_qa.txt"
            generate_questions_and_answers(summary, qa_output_path)
            print("Questions and answers generated and saved.")
        elif number==4:
            pdf_path = input("Enter the path to the PDF file: ")
            output_excel_path = input("Enter the path to save the tables into Excel file: ")
            text_output_path = os.path.splitext(output_excel_path)[0] + "_text.txt"
            if not os.path.exists(pdf_path):
                print(f"Error: File not found at {pdf_path}")
                return
            text = extract_text_from_pdf(pdf_path)
            os.makedirs(os.path.dirname(text_output_path), exist_ok=True)  # Ensure directory exists for text output
            with open(text_output_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)
            #print(f"Text extracted and saved to {text_output_path}")
            output_dir = os.path.dirname(output_excel_path)
            '''os.makedirs(output_dir, exist_ok=True)  # Ensure the directory for output Excel file exists
            tables = extract_tables_from_pdf(pdf_path)
            if tables:
                save_tables_to_excel(tables, output_excel_path)
            else:
                #print("No tables found in the PDF.")
                pass'''
            summary_output_path = os.path.splitext(output_excel_path)[0] + "_summary.txt"
            summary = summarize_text(text_output_path, summary_output_path)
            points_output_path = os.path.splitext(output_excel_path)[0] + "_points.txt"
            extract_important_points(summary, points_output_path)
            print("Important points extracted and saved.")
        elif number==5:
            pdf_path = input("Enter the path to the PDF file: ")
        
            output_image_dir = input("Enter the directory to save extracted images: ")
            extract_images_from_pdf(pdf_path, output_image_dir)
    
    
        print("choose correct key from the following menu")
        print("do you want to continue")
        choice=eval(input("press 1 for continue or 0 for discontinue"))
        if choice==1:
            confirm=True
        else:
            confirm=False

    
            


        

   

if __name__ == "__main__":
    main()


