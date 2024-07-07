import PyPDF2
import pycountry
import streamlit as st

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    return full_text

def contains_country(text):
    countries = [country.name for country in pycountry.countries]
    for country in countries:
        if country.lower() in text.lower():
            return True, country
    return False, None

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

def pretty_print(content, key):
    if content:
        st.write(f"--- {key} ---\n{content}\n")
    else:
        st.write(f"--- {key} ---\nNot found\n")

def process_pdf(file):
    full_text = extract_text_from_pdf(file)

    important_keys = [
        'General Discipline',
        'Required Qualifications and Skills',
        'Other requirements',
        'Internship Offered',
        "Location of placement",
        "Student Required"
    ]

    important_keys2 = [
        'INTERNSHIP OFFER Ref. No.',
        'Gross pay',
        'Estimated cost of living incl. lodging:',
    ]

    for key in important_keys:
        if key == "Location of placement":
            content = extract_content_after_keyword(full_text, key)
            is_present, country_name = contains_country(full_text)
            if is_present:
                st.write(f"Country found: {country_name}")
                content = country_name
            else:
                st.write("No country found in the text.")
        else:
            content = extract_content_after_keyword(full_text, key)
        pretty_print(content, key)

    for key in important_keys2:
        if key == "INTERNSHIP OFFER Ref. No.":
            content = extract_content_after_keyword2(full_text, key).split('.')[-1].strip()
        else:
            content = extract_content_after_keyword2(full_text, key)
        pretty_print(content, key)

# Streamlit App
st.title("PDF Content Extractor")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    process_pdf(uploaded_file)
