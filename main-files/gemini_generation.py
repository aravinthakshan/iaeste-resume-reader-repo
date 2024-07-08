import pathlib
import os
import google.generativeai as genai
from api_keys import API_KEY
# Access your API key as an environment variable.
genai.configure(api_key=API_KEY)


# Choose a model that's appropriate for your use case.

def generate_improvements(qualifications ,discipline,internship_offer, resume_text):
    instructions = f"You are a resume evaluator, provide short but precise suggestions and improvements in bullet points for the following internship offer letter: \
        {qualifications},{discipline},{internship_offer}, when prompted with RESUME CONTENTS"
    model=genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  system_instruction=instructions)
    response = model.generate_content(f"RESUME CONTENTs: {resume_text}")
    print(response.text)
    return response.text