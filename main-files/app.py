import streamlit as st
from pdf_reader import process_pdf as offer_letter_process_pdf, extract_content_after_keyword
from semantic_search import compute_cosine_similarity, extract_text_from_pdf, highlight_missing_keywords, generate_improvements,offer_letter_process_pdf_2

# Navigation
page = st.selectbox("Choose a page", ["Offer Letter Extractor", "Semantic Search"])

if page == "Offer Letter Extractor":
    st.title("Offer Letter PDF Content Extractor")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf",key ="offer1")
    if uploaded_file is not None:
        offer_letter_process_pdf(uploaded_file)

elif page == "Semantic Search":
    st.title("Semantic Search Between Offer Letter and Resume")
    
    offer_letter_file = st.file_uploader("Upload Offer Letter PDF", type="pdf", key="offer2")
    resume_file = st.file_uploader("Upload Resume PDF", type="pdf", key="resume2")

    if offer_letter_file is not None and resume_file is not None:
        offer_letter_process_pdf_2(offer_letter_file,resume_file)
