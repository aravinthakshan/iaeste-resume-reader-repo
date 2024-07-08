import streamlit as st
from pdf_reader import process_pdf as offer_letter_process_pdf,extract_content_after_keyword
from semantic_search import compute_cosine_similarity, extract_text_from_pdf, highlight_missing_keywords, generate_improvements

# Navigation
page = st.selectbox("Choose a page", ["Offer Letter Extractor", "Semantic Search"])

if page == "Offer Letter Extractor":
    st.title("Offer Letter PDF Content Extractor")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf",key = "uploader")
    if uploaded_file is not None:
        offer_letter_process_pdf(uploaded_file)

elif page == "Semantic Search":
    st.title("Semantic Search Between Offer Letter and Resume")
    
    offer_letter_file = st.file_uploader("Upload Offer Letter PDF", type="pdf", key="offer")
    resume_file = st.file_uploader("Upload Resume PDF", type="pdf", key="resume")

    if offer_letter_file is not None and resume_file is not None:
        offer_letter_text = extract_text_from_pdf(offer_letter_file)
        resume_text = extract_text_from_pdf(resume_file)

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
       
