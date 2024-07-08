import streamlit as st
import google.generativeai as genai
from api_keys import API_KEY  # Assuming you have API_KEY imported
import re


st.title("Gemini-like clone")

instructions = f"You are a Resume Reviewer, here is the job description for an internship:, when prompted with a candidates RESUME CONTENTS; provide detailed suggestions and improvements"


# Access your API key as an environment variable.
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=instructions
    )
    
chat = model.start_chat()

def LLM_Response(question):
    response = chat.send_message(question,stream=True)
    return response

st.title("Chat Application using Gemini Pro")

user_quest = st.text_input("Ask a question:")
btn = st.button("Ask")

if btn and user_quest:
    result = LLM_Response(user_quest)
    st.subheader("Response : ")
    for word in result:
        st.text(word.text)

"""
# Function to generate improvements using Gemini API
def generate_improvements(qualifications, discipline, internship_offer, resume_text):
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=instructions
    )
    
    response_stream = model.generate_content(f"RESUME CONTENTS: {resume_text}", stream=True)
    response_text = []
    
    for response_chunk in response_stream:
        response_text.append(response_chunk)
    
    return response_text"""