#project based on extracting the tables and the raw data from the pdf
#step1 SEPARATING THE TEXT AND TABLES FROM THE PDF
import os
import pandas as pd
import PyPDF2
import pdfplumber
pdf_path=""
def extract_text_from_pdf(pdf_path):
    """Extract plain text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF file using pdfplumber."""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    tables.append(table)  #openpyxl
        return tables
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

def save_tables_to_excel(tables, output_path):
    """Save extracted tables to an Excel file."""
    try:
        with pd.ExcelWriter(output_path) as writer:
            for i, table in enumerate(tables):
                df = pd.DataFrame(table)
                df.to_excel(writer, sheet_name=f"Table_{i+1}", index=False)
        print(f"Tables saved to {output_path}")
    except Exception as e:
        print(f"Error saving tables: {e}")

def main():
    # Input and output file paths
    pdf_path = input("Enter the path to the PDF file: ")
    output_excel_path = input("Enter the path to save the Excel file: ")

    # Extract text and save to a text file
    text = extract_text_from_pdf(pdf_path)
    text_output_path = os.path.splitext(output_excel_path)[0] + "_text.txt"
    with open(text_output_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    print(f"Text extracted and saved to {text_output_path}")

    # Extract tables and save to an Excel file
    tables = extract_tables_from_pdf(pdf_path)
    if tables:
        save_tables_to_excel(tables, output_excel_path)
    else:
        print("No tables found in the PDF.")

if __name__ == "__main__":
    main()
'''#STEP2 EXTRACT THE TEXT FROM THE TEXT FILE
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.numPages)):
            page = reader.pages[page_num] #access each page using page attribute
            text += page.eC:\project\data.xlsxtract_text()
    return text

#file_path = input("enter the path of the text file")
pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text[:500])  # Print the first 500 characters to check'''
