import fitz 
import os


# Function to extract text from a single PDF file
def extract_text_from_pdf(pdf_path):
  document = fitz.open(pdf_path)
  entire_text = ""
  
  for page_num in range(len(document)):
    page = document.load_page(page_num)
    page_text = page.get_text("text")
    entire_text += page_text.strip() + "\n" 

  return entire_text.strip() 

# Example PDF file path (replace with your actual path)
pdf_file = 'documents/open_cv.pdf'

# Extract text from the PDF
entire_text = extract_text_from_pdf(pdf_file)

with open("extracted_text2.txt", 'w', encoding='utf-8') as text:
    text.write(entire_text)
