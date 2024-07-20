"""
This file is the logic to extract the text from the PDF file and store in accordance with
the JSON struture mentioned in the ToC_json.py file.

We need to define manually what is the struture of the page number and the title of each
section and chapter in the book.

"""


# This is the logic for the Natural Language Processing with Python book
import json
import re

# Load the initial JSON structure
with open('NLPtoc.json', 'r') as json_file:
    toc = json.load(json_file)

# Define the text file path
text_file_path = 'NLP.txt'

# Function to remove non-alphanumeric characters
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Function to count words in a text
def word_count(text):
    return len(text.split())

# Function to add content to the JSON structure
def add_content_to_json(chapter, section, content, page):
    for part in toc['parts']:
        if part['title'].startswith(chapter):
            for sec in part['sections']:
                if sec['title'].startswith(section):
                    sec['content'] = sec.get('content', '') + content
                    sec['page'] = page
                    sec['word_count'] = sec.get('word_count', 0) + word_count(content)

# Read the text file
with open(text_file_path, 'r') as file:
    lines = file.readlines()

current_chapter = None
current_section = None
current_page = None
content_buffer = ''
total_word_count = 0

for line in lines:
    line = line.strip()

    # Check for chapter title
    chapter_match = re.match(r'CHAPTER\s+(\d+)', line)
    if chapter_match:
        if current_chapter and current_section:
            cleaned_content = clean_text(content_buffer)
            add_content_to_json(current_chapter, current_section, cleaned_content, current_page)
            total_word_count += word_count(cleaned_content)
        current_chapter = f'Chapter {chapter_match.group(1)}'
        current_section = None
        content_buffer = ''
        continue

    # Check for section title
    section_match = re.match(r'(\d+\.\d+)\s+([A-Za-z0-9\s:]+)', line)
    if section_match:
        if current_section:
            cleaned_content = clean_text(content_buffer)
            add_content_to_json(current_chapter, current_section, cleaned_content, current_page)
            total_word_count += word_count(cleaned_content)
        current_section = f'{section_match.group(1)} {section_match.group(2)}'
        content_buffer = ''
        continue

    # Check for page number
    page_match = re.match(r'(\d+)\s+\|\s+.+', line) or re.match(r'.+\|\s+(\d+)', line)
    if page_match:
        current_page = page_match.group(1)
        continue

    # Append content to buffer
    content_buffer += ' ' + line

# Add remaining content if any
if current_chapter and current_section:
    cleaned_content = clean_text(content_buffer)
    add_content_to_json(current_chapter, current_section, cleaned_content, current_page)
    total_word_count += word_count(cleaned_content)

# Add the total word count to the JSON structure
toc['total_word_count'] = total_word_count

# Save the updated JSON structure
with open('updated_toc.json', 'w') as json_file:
    json.dump(toc, json_file, indent=4)

print(f"Content extracted and JSON file updated successfully. Total word count: {total_word_count}")


# ---------------------------------------------------------------------------------------------------


# This is the logic for the book Hands-On Machine Learning with Scikit-Learn and TensorFlow
import json
import re

def load_data(text_file, json_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text_content = file.read()
    with open(json_file, 'r', encoding='utf-8') as file:
        book_structure = json.load(file)
    return text_content, book_structure

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def create_title_list(toc):
    title_list = []
    for part in toc:
        for chapter in part.get('chapters', []):
            title_list.append(chapter['title'])
            for section in chapter.get('sections', []):
                title_list.append(section['title'])
                for subsection in section.get('subsections', []):
                    title_list.append(subsection['title'])
    return title_list

def find_max_title_length(title_list):
    max_length = 0
    for title in title_list:
        max_length = max(max_length, len(title))
    return max_length

def assign_content_to_titles(lines, title_list):
    max_title_length = find_max_title_length(title_list)
    content_dict = {title: '' for title in title_list}
    current_title = None

    for line in lines:
        clean_line = clean_text(line.strip())
        if len(clean_line) <= max_title_length:
            if clean_line in title_list:
                current_title = clean_line
        if current_title:
            content_dict[current_title] += clean_line + ' '

    # Strip trailing spaces from content
    for title in content_dict:
        content_dict[title] = content_dict[title].strip()

    return content_dict

def update_toc_with_content(toc, content_dict):
    for part in toc:
        for chapter in part.get('chapters', []):
            if chapter['title'] in content_dict:
                chapter['content'] = content_dict[chapter['title']]
            for section in chapter.get('sections', []):
                if section['title'] in content_dict:
                    section['content'] = content_dict[section['title']]
                    for subsection in section.get('subsections', []):
                        if subsection['title'] in content_dict:
                            subsection['content'] = content_dict[subsection['title']]

def count_words(text):
    return len(text.split())

def main(text_file, json_file, output_file):
    text_content, book_structure = load_data(text_file, json_file)
    title_list = create_title_list(book_structure['parts'])

    lines = text_content.split('\n')
    start_line = 0
    for i, line in enumerate(lines):
        if re.match(r'^xxiv', line, re.IGNORECASE):
            start_line = i
            break
    lines = lines[start_line:]

    content_dict = assign_content_to_titles(lines, title_list)
    update_toc_with_content(book_structure['parts'], content_dict)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(book_structure, file, indent=2, ensure_ascii=False)

    text_word_count = count_words(text_content)
    json_word_count = count_words(json.dumps(book_structure))

    print(f'Total words in text file: {text_word_count}')
    print(f'Total words in JSON file content: {json_word_count}')
    print(f'Updated book structure saved to {output_file}')

text_file = '/content/extracted_text.txt'
json_file = '/content/book_structure.json'
output_file = '/content/upd_book_structure.json'
main(text_file, json_file, output_file)