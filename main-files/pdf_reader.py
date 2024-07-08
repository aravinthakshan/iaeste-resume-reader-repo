import PyPDF2
import pycountry
import streamlit as st
import re

# Function to extract text from the PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    return full_text

# Function to check if a country is mentioned in the text
def contains_country(text):
    countries = [country.name for country in pycountry.countries]
    for country in countries:
        if country.lower() in text.lower():
            return True, country
    return False, None

# Function to extract content after a specific keyword
def extract_content_after_keyword(text, keyword):
    lines = text.split('\n')
    extracting = False
    extracted_content = ""

    for line in lines:
        if extracting:
            if line.strip() == "":
                break
            extracted_content += line + "\n"
        elif keyword in line:
            extracting = True
            extracted_content += line + "\n"

    return extracted_content.strip()

# Function to extract content after a specific keyword with different formatting
def extract_content_after_keyword2(text, keyword):
    lines = text.split('\n')
    extracting = False
    extracted_content = ""

    for line in lines:
        if extracting:
            if line.strip() == "":
                break
            extracted_content += line + "\n"
        elif keyword in line:
            extracted_content += line + "\n"
            extracting = False

    extracted_content = extracted_content.strip()
    return extracted_content.split(':')[-1]

# Function to parse the duration string and extract min and max duration
def parse_duration(text):
    match = re.search(r"Number of weeks offered:\s*(\d+)\s*-\s*(\d+)", text)
    if match:
        min_duration = match.group(1)
        max_duration = match.group(2)
        return f"{min_duration}", f"{max_duration}"
    return "Duration information not found", ' '

# Function to display content and provide a copy option
def copy_option(title, content):
    st.write(f"{title}")
    st.code(content, language="text")

# Main function to process the PDF and extract required information
def process_pdf(file):
    full_text = extract_text_from_pdf(file)

    important_keys = [
        'General Discipline',
        'Required Qualifications and Skills',
        'Internship Offered'
    ]

    important_keys2 = [
        'INTERNSHIP OFFER Ref. No.',
        "Location of placement",
        "Number of weeks offered",
        'Gross pay',
        'Estimated cost of living incl. lodging:',
    ]

    # Process and display the content for important_keys2
    for key in important_keys2:
        if key == "Location of placement":
            content = extract_content_after_keyword(full_text, key)
            is_present, country_name = contains_country(full_text)
            if is_present:
                copy_option(key, country_name)
                continue
            else:
                copy_option(key, "No country found in the text.")
                continue
        elif key == "INTERNSHIP OFFER Ref. No.":
            content = extract_content_after_keyword2(full_text, key).split('.')[-1].strip()
        elif key == "Number of weeks offered":
            min_duration, max_duration = parse_duration(full_text)
            if max_duration != ' ':
                copy_option("Project Min Duration", min_duration)
                copy_option("Project Max Duration", max_duration)
                continue
        else:
            content = extract_content_after_keyword2(full_text, key)
        copy_option(key, content)

    # Process and display the content for important_keys
    for key in important_keys:
        content = extract_content_after_keyword(full_text, key)
        copy_option(key, content)

# Streamlit App
st.title("Offer Letter PDF Content Extractor")

# File uploader to upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Process the uploaded PDF file
if uploaded_file is not None:
    process_pdf(uploaded_file)
