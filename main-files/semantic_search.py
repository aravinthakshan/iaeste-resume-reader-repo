import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import google.generativeai as genai
from api_keys import API_KEY

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Configure Google Generative AI
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

# Function to highlight missing keywords
def highlight_missing_keywords(offer_text, resume_text):
    stop_words = set(stopwords.words('english'))
    offer_words = set(word_tokenize(offer_text.lower())) - stop_words
    resume_words = set(word_tokenize(resume_text.lower())) - stop_words
    missing_words = offer_words - resume_words
    return missing_words

# Function to generate improvements
def generate_improvements(qualifications, discipline, internship_offer, resume_text):
    instructions = f"You are a resume evaluator, provide short but precise suggestions and improvements in bullet points for the following internship offer letter: {qualifications}, {discipline}, {internship_offer}, when prompted with RESUME CONTENTS"
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=instructions
    )
    response_stream = model.generate_content(f"RESUME CONTENTS: {resume_text}", stream=True)
    
    response_text = ""
    for response in response_stream:
        response_text += response.text
        yield response.text

# Streamlit App
st.title("Semantic Search Between Offer Letter and Resume")

# File uploaders for offer letter and resume
offer_letter_file = st.file_uploader("Upload Offer Letter PDF", type="pdf", key="offers")
resume_file = st.file_uploader("Upload Resume PDF", type="pdf", key="resume")

if offer_letter_file is not None:
    offer_letter_text = extract_text_from_pdf(offer_letter_file)
    st.subheader("Offer Letter Content")
    st.text(offer_letter_text)

if resume_file is not None:
    resume_text = extract_text_from_pdf(resume_file)
    st.subheader("Resume Content")
    st.text(resume_text)

if offer_letter_file is not None and resume_file is not None:
    similarity_score = compute_cosine_similarity(offer_letter_text, resume_text)
    st.write(f"Similarity Score: {similarity_score:.2f}")

    # Extract and display specific sections
    qualifications = extract_content_after_keyword(offer_letter_text, 'Required Qualifications and Skills')
    discipline = extract_content_after_keyword(offer_letter_text, 'General Discipline')
    internship_offer = extract_content_after_keyword(offer_letter_text, 'Internship Offered')

    st.write("### Offer Letter Content")
    st.write(f"**Required Qualifications and Skills:**\n{qualifications}")
    st.write(f"**General Discipline:**\n{discipline}")
    st.write(f"**Internship Offered:**\n{internship_offer}")

    # Highlight missing keywords
    st.write("### Missing Keywords in Resume")
    missing_keywords = highlight_missing_keywords(qualifications + ' ' + discipline + ' ' + internship_offer, resume_text)
    st.write(", ".join(missing_keywords))

    # Generate improvements
    st.write("### Suggested Improvements for Resume")
    placeholder = st.empty()
    improvements_generator = generate_improvements(qualifications, discipline, internship_offer, resume_text)
    for improvement in improvements_generator:
        placeholder.write(improvement)
