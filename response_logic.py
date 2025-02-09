from google.generativeai.types import HarmCategory, HarmBlockThreshold
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

import os
import google.generativeai as genai


# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
genai.configure(api_key="AIzaSyBeBKUl5kF_HrhmgxFqyYLsMPpUmj9Frmg")

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)




def prompt_correction(prompt):

  model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="for every input i give to you, return the corrected form of spelling for any typo and misspelling while also correcting potential grammatical error. the input may be english or khmer or a mix of the two but do the same thing for all of the scenario.",
  )
  response = model.generate_content(prompt)
  return response.text
