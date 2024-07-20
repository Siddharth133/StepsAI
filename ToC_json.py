import json
import re

def parse_toc(toc_text):
    lines = toc_text.split('\n')
    result = {"title": "Natural Language Processing with Python", "chapters": []}
    current_chapter = None
    
    chapter_pattern = re.compile(r'^\d+\.\s+(.*)\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+\.\s+(\d+)$')
    section_pattern = re.compile(r'^\d+\.\d+\s+(.*)\s+(\d+)$')
    
    for line in lines:
        chapter_match = chapter_pattern.match(line)
        section_match = section_pattern.match(line)
        
        if chapter_match:
            if current_chapter:
                result["chapters"].append(current_chapter)
            current_chapter = {
                "title": chapter_match.group(1),
                "page": chapter_match.group(2),
                "sections": []
            }
        elif section_match and current_chapter:
            current_chapter["sections"].append({
                "title": section_match.group(1),
                "page": section_match.group(2),
                "subsections": []
            })
    
    if current_chapter:
        result["chapters"].append(current_chapter)
    
    return result

def main():
    input_filename = 'toc.txt'
    output_filename = 'toc.json'
    
    with open(input_filename, 'r') as file:
        toc_text = file.read()
    
    toc_json = parse_toc(toc_text)
    
    with open(output_filename, 'w') as file:
        json.dump(toc_json, file, indent=2)
    
    print(f'TOC has been saved to {output_filename}')

if __name__ == '__main__':
    main()
# -------------------------------------------------------------------------------------

import re
import json

def parse_toc(text):
    lines = text.split('\n')
    chapters = []
    current_chapter = None
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for chapter
        if re.match(r'^\d+\.', line):
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                title, page = parts
                current_chapter = {"title": title.strip(), "page": page.strip(), "sections": []}
                chapters.append(current_chapter)
                current_section = None
        # Check for section or subsection
        elif line[0].isalpha():
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                title, page = parts
                if line.startswith('    '):  # Subsection
                    if current_section:
                        current_section["subsections"].append({"title": title.strip(), "page": page.strip()})
                else:  # Section
                    current_section = {"title": title.strip(), "page": page.strip(), "subsections": []}
                    if current_chapter:
                        current_chapter["sections"].append(current_section)

    return chapters

# Read the TOC from the file
with open('ToC.txt', 'r') as file:
    toc_text = file.read()

# Parse the TOC
chapters = parse_toc(toc_text)

# Create the book structure
book_structure = {
    "title": "Hands-On Machine Learning with Scikit-Learn and TensorFlow",
    "parts": [
        {
            "title": "Part I. The Fundamentals of Machine Learning",
            "chapters": chapters
        }
    ]
}

# Save the JSON structure to a file
with open('book_structure.json', 'w') as json_file:
    json.dump(book_structure, json_file, indent=2)

print("Book structure has been saved to 'book_structure.json'")