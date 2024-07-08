import streamlit as st
from pdf_reader import process_pdf as offer_letter_process_pdf, extract_content_after_keyword
from semantic_search import compute_cosine_similarity, extract_text_from_pdf, highlight_missing_keywords, generate_improvements,main

# Navigation
page = st.selectbox("Choose a page", ["Offer Letter Extractor", "Resume Review"])

if page == "Offer Letter Extractor":
    st.title("Offer Letter PDF Content Extractor")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf",key ="offer1")
    if uploaded_file is not None:
        offer_letter_process_pdf(uploaded_file)

elif page == "Resume Review":
    st.title("Resume Review")
    main()
