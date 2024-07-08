import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from api_keys import API_KEY  # Assuming you have API_KEY imported
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

# Access your API key as an environment variable.
genai.configure(api_key=API_KEY)

# Function to extract text from the PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    return full_text

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

# Function to compute cosine similarity
def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0, 1]

# Function to interact with the Generative AI model
def interact_with_model(model, resume_text):
    response = model.send_message(resume_text, stream=True)
    return response

# Function to generate improvements using Gemini API
def generate_improvements(qualifications, discipline, internship_offer, resume_text):
    instructions = f"You are a Resume Reviewer, here is the job description for an internship: {qualifications}, {discipline}, {internship_offer}, when prompted with a candidates RESUME CONTENTS; provide detailed suggestions and improvements, give the output in proper markdown format"

    model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=instructions
        )
        
    chat = model.start_chat()

    result = interact_with_model(chat, resume_text)

    for word in result:
        # Clean up the text to prevent messy formatting
        text = word.text
        if text:  # Check if text is not empty
            st.markdown(text)  # Display each response on a new line

# Function to highlight missing keywords (nouns) in resume
def highlight_missing_keywords(offer_text, resume_text):
    # Tokenize and tag offer letter text to get nouns
    offer_tokens = word_tokenize(offer_text)
    offer_tags = nltk.pos_tag(offer_tokens)
    offer_nouns = [word for word, pos in offer_tags if pos.startswith('NN')]

    # Tokenize and tag resume text to get nouns
    resume_tokens = word_tokenize(resume_text)
    resume_tags = nltk.pos_tag(resume_tokens)
    resume_nouns = [word for word, pos in resume_tags if pos.startswith('NN')]

    # Identify missing nouns in resume
    missing_nouns = set(offer_nouns) - set(resume_nouns)
    return missing_nouns

# Streamlit app
def main():

    # File uploaders for offer letter and resume
    uploaded_offer_letter = st.file_uploader("Upload Offer Letter PDF", type="pdf")
    uploaded_resume = st.file_uploader("Upload Resume PDF", type="pdf")

    if uploaded_offer_letter and uploaded_resume:
        offer_letter_text = extract_text_from_pdf(uploaded_offer_letter)
        resume_text = extract_text_from_pdf(uploaded_resume)

        # Compute similarity score
        similarity_score = compute_cosine_similarity(offer_letter_text, resume_text)
        st.write(f"Similarity Score: {similarity_score:.2f}")

        # Extract and display specific sections from offer letter
        qualifications = extract_content_after_keyword(offer_letter_text, 'Required Qualifications and Skills')
        discipline = extract_content_after_keyword(offer_letter_text, 'General Discipline')
        internship_offer = extract_content_after_keyword(offer_letter_text, 'Internship Offered')

        st.write("### Offer Letter Content")
        st.write(f"**Required Qualifications and Skills:**\n{qualifications}")
        st.write(f"**General Discipline:**\n{discipline}")
        st.write(f"**Internship Offered:**\n{internship_offer}")

        # Highlight missing nouns in resume
        st.write("### Missing Nouns in Resume")
        missing_nouns = highlight_missing_keywords(qualifications + ' ' + discipline + ' ' + internship_offer, resume_text)
        st.write(", ".join(missing_nouns))

        # Generate improvements for the resume
        st.write("### Suggested Improvements for Resume")
        generate_improvements(qualifications, discipline, internship_offer, resume_text)

