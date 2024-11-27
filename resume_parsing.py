import os
import re
import json
from docx import Document
from pdfminer.high_level import extract_text
import tempfile

def scrape_resumes(files, remove_below_n=4):
    """
    Reads resumes from a list of file objects and returns a dictionary of resumes.
    Each resume is represented as a dictionary containing format, sections, and metadata.
    """
    # Supported extensions
    supported_exts = {".docx", ".pdf", ".txt", ".json"}

    # Initialize the dictionary to store resumes
    resumes = {}

    for file in files:
        filename = file.name
        ext = os.path.splitext(filename)[1].lower()

        if ext not in supported_exts:
            print(f"Unsupported file type: {filename}")
            continue

        # Initialize a dictionary for this resume
        resume_info = {}
        resume_info['format'] = ext
        resume_info['filename'] = filename

        # Read the file according to its format
        if ext == ".docx":
            paragraphs = read_docx_paragraphs(file)
        elif ext == ".pdf":
            paragraphs = read_pdf_paragraphs(file)
        elif ext == ".txt":
            text = file.getvalue().decode("utf-8")
            paragraphs = split_text_into_paragraphs(text)
        elif ext == ".json":
            paragraphs = read_json_resume(file)
        else:
            paragraphs = []

        # Remove sections with fewer than remove_below_n words
        if remove_below_n > 0:
            paragraphs = [p for p in paragraphs if len(p.split()) >= remove_below_n]

        # Store the sections/paragraphs
        resume_info['sections'] = paragraphs

        # Store in the main dictionary using the filename as the key
        resumes[filename] = resume_info

    return resumes

def read_docx_paragraphs(file):
    """
    Reads paragraphs from a DOCX file object.
    """
    try:
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        # Read the DOCX file
        doc = Document(tmp_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip() != ""]

        # Clean up the temporary file
        os.unlink(tmp_path)

        return paragraphs
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
        return []

def read_pdf_paragraphs(file):
    """
    Reads paragraphs from a PDF file object.
    """
    try:
        # Save the file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        # Extract text from the PDF
        text = extract_text(tmp_path)

        # Clean up the temporary file
        os.unlink(tmp_path)

        # Split text into paragraphs
        paragraphs = split_text_into_paragraphs(text)

        return paragraphs
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return []

def read_json_resume(file):
    """
    Reads a resume in JSON format following the JSON Resume schema.
    """
    try:
        json_content = json.loads(file.getvalue().decode("utf-8"))
        sections = []

        # Extract relevant fields
        if 'basics' in json_content:
            basics = json_content['basics']
            basics_text = "\n".join([str(basics.get(field, "")) for field in ['name', 'label', 'summary']])
            sections.append(basics_text.strip())

        # List of possible sections
        section_keys = ['work', 'volunteer', 'education', 'awards', 'publications', 'skills', 'languages', 'interests', 'references']

        for key in section_keys:
            if key in json_content and isinstance(json_content[key], list):
                for item in json_content[key]:
                    item_text = "\n".join([str(value) for value in item.values() if isinstance(value, str)])
                    sections.append(item_text.strip())

        # Remove empty sections
        sections = [s for s in sections if s != ""]

        return sections
    except Exception as e:
        print(f"Error reading JSON resume: {e}")
        return []

def split_text_into_paragraphs(text):
    """
    Splits text into paragraphs based on empty lines.
    """
    # Replace multiple newlines with a placeholder
    text = re.sub(r'\n{2,}', '###NEWPARAGRAPH###', text)

    # Replace remaining single newlines with spaces
    text = re.sub(r'\n', ' ', text)

    # Split on the placeholder to get paragraphs
    paragraphs = text.split('###NEWPARAGRAPH###')

    # Clean up the paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip() != ""]

    return paragraphs