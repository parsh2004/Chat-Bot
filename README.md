# Chat-Bot: A Conversational AI with Natural Voice Responses
Th Aim of the project to make a chat bot response as voice note with the help of open AI.

## Features

- Enhances accessibility for users with visual impairments by delivering information effectively through voice.
- Provides a more engaging and pleasant interaction for users.
- The chatbot should be capable of recognizing and accurately pronouncing words and phrases in various languages.
- Allows the chatbot to adapt its tone to different scenarios, improving contextual relevance.
- This feature enables the chatbot to adopt different speech styles, such as formal, casual, or enthusiastic, based on the context of the conversation or user preferences.

## Requirements

- Python 3.x
- Jupyter Notebook (Anaconda 3)

## Installiation :
   # 1.	Clone the repositaryyy
      ```
     	https://github.com/parsh2004/Chat-Bot.git
     	```
   # 2.	Install the required packages
        1.Python 3.x
        2.Jupyter Notebook (Anaconda 3)

## Usage

1. Create interactive FAQs on websites or applications where users can ask questions in text, and the chatbot responds with voice-generated answers, providing a more engaging user experience.
2. Chatbots that can convert text-based customer queries into natural-sounding voice responses
3. Improve e-learning experiences by incorporating text-to-voice generation, enabling chatbots to narrate lessons, provide instructions, and engage learners in a more dynamic and interactive manner.

## Program:

# Text to Voice
```
!pip install langchain
!pip install openai
!pip install gradio
!pip install huggingface_hub

import os
import re
import requests
import json
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

OPENAI_API_KEY="OPENAI_API_KEY"
PLAY_HT_API_KEY="PLAY_HT_API_KEY"
PLAY_HT_USER_ID="PLAY_HT_USER_ID"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

play_ht_api_get_audio_url = "https://play.ht/api/v2/tts"
PLAY_HT_VOICE_ID="PLAY_HT_VOICE_ID"

template = """You are a helpful assistant to answer user queries.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature='0.5', model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,)

headers = {
      "accept": "text/event-stream",
      "content-type": "application/json",
      "AUTHORIZATION": "Bearer "+ PLAY_HT_API_KEY,
      "X-USER-ID": PLAY_HT_USER_ID}

def get_payload(text):
  return {
    "text": text,
    "voice": PLAY_HT_VOICE_ID,
    "quality": "medium",
    "output_format": "mp3",
    "speed": 1,
    "sample_rate": 24000,
    "seed": None,
    "temperature": None
  }

def get_generated_audio(text):
  payload = get_payload(text)
  generated_response = {}
  try:
      response = requests.post(play_ht_api_get_audio_url, json=payload, headers=headers)
      response.raise_for_status()
      generated_response["type"]= 'SUCCESS'
      generated_response["response"] = response.text
  except requests.exceptions.RequestException as e:
      generated_response["type"]= 'ERROR'
      try:
        response_text = json.loads(response.text)
        if response_text['error_message']:
          generated_response

def get_text_response(user_message):
    response = llm_chain.predict(user_message = user_message)
    return response

def get_text_response_and_audio_response(user_message):
    response = get_text_response(user_message) # Getting the reply from Open AI
    audio_reply_for_question_response = get_audio_reply_for_question(response)
    final_response = {
        'output_file_path': '',
        'message':''
    }
    audio_url = audio_reply_for_question_response['audio_url']
    if audio_url:
      output_file_path=get_filename_from_url(audio_url)
      download_url_response = download_url(audio_url)
      audio_content = download_url_response['content']
      if audio_content:
        with open(output_file_path, "wb") as audio_file:
          audio_file.write(audio_content)
          final_response['output_file_path'] = output_file_path
      else:
          final_response['message'] = download_url_response['error']
    else:
      final_response['message'] = audio_reply_for_question_response['message']
    return final_response

def chat_bot_response(message, history):
    text_and_audio_response = get_text_response_and_audio_response(message)
    output_file_path = text_and_audio_response['output_file_path']
    if output_file_path:
      return (text_and_audio_response['output_file_path'],)
    else:
      return text_and_audio_response['message']

demo = gr.ChatInterface(chat_bot_response,examples=["How are you doing?","What are your interests?","Which places do you like to visit?"])

if _name_ == "_main_":
    demo.launch() #To create a public link, set `share=True` in `launch()`. To enable errors and logs, set `debug=True` in `launch()`.

from huggingface_hub import notebook_login
notebook_login()

from huggingface_hub import HfApi
api = HfApi()

HUGGING_FACE_REPO_ID = "<<Hugging Face UserName/Repo ID>>"

%mkdir /content/ChatBotWithOpenAILangChainAndPlayHT
!wget -P  /content/ChatBotWithOpenAILangChainAndPlayHT/ https://s3.ap-south-1.amazonaws.com/cdn1.ccbp.in/GenAI-Workshop/ChatBotWithOpenAILangChainPlayHT2/app.py
!wget -P /content/ChatBotWithOpenAILangChainAndPlayHT/ https://s3.ap-south-1.amazonaws.com/cdn1.ccbp.in/GenAI-Workshop/ChatBotWithOpenAILangChainPlayHT/requirements.txt

%cd /content/ChatBotWithOpenAILangChainAndPlayHT

api.upload_file(
    path_or_fileobj="./requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=HUGGING_FACE_REPO_ID,
    repo_type="space")

api.upload_file(
    path_or_fileobj="./app.py",
    path_in_repo="app.py",
    repo_id=HUGGING_FACE_REPO_ID,
    repo_type="space")
```
## Output:
# Promting:
![image](https://github.com/parsh2004/Chat-Bot/assets/95388047/b457fcaa-eba9-4959-811a-dda681db3e5d)
# Outcome:
![image](https://github.com/parsh2004/Chat-Bot/assets/95388047/db479dea-6511-4592-91b2-de97945d1a10)

# Text to Text
```
!pip install langchain
!pip install openai
!pip install gradio
!pip install huggingface_hub

import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

OPENAI_API_KEY="OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

template = """You are a helpful assistant to answer user queries.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

# from langchain.llms import HuggingFacePipeline
# hf = HuggingFacePipeline.from_model_id(
#     model_id="gpt2",
#     task="text-generation",)

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature='0.5', model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

def get_text_response(user_message,history):
    response = llm_chain.predict(user_message = user_message)
    return response

demo = gr.ChatInterface(get_text_response, examples=["How are you doing?","What are your interests?","Which places do you like to visit?"])

if _name_ == "_main_":
    demo.launch() #To create a public link, set `share=True` in `launch()`. To enable errors and logs, set `debug=True` in `launch()`.

from huggingface_hub import notebook_login
notebook_login()

from huggingface_hub import HfApi
api = HfApi()

HUGGING_FACE_REPO_ID = "<Hugging Face User Name/Repo Name>"

%mkdir /content/ChatBotWithOpenAI
!wget -P  /content/ChatBotWithOpenAI/ https://s3.ap-south-1.amazonaws.com/cdn1.ccbp.in/GenAI-Workshop/ChatBotWithOpenAIAndLangChain/app.py
!wget -P /content/ChatBotWithOpenAI/ https://s3.ap-south-1.amazonaws.com/cdn1.ccbp.in/GenAI-Workshop/ChatBotWithOpenAIAndLangChain/requirements.txt

%cd /content/ChatBotWithOpenAI

api.upload_file(
    path_or_fileobj="./requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=HUGGING_FACE_REPO_ID,
    repo_type="space")

api.upload_file(
    path_or_fileobj="./app.py",
    path_in_repo="app.py",
    repo_id=HUGGING_FACE_REPO_ID,
    repo_type="space")
```

## Output 

# Promting:
![image](https://github.com/parsh2004/Chat-Bot/assets/95388047/8874a3fc-86d3-417e-827a-e509ef392fa6)
# Outcome:
![image](https://github.com/parsh2004/Chat-Bot/assets/95388047/e168020e-927c-4f71-bdb9-66077244b404)


## Result:
The project successfully created a chatbot capable of generating voice responses for user prompts, enhancing the interactive and dynamic nature of conversations. This implementation brings a new dimension to user engagement by providing spoken responses tailored to the queries posed. The project's achievement lies in combining chatbot functionality with voice synthesis to create a seamless and natural conversational experience.
