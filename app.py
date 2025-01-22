from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

import streamlit as st
import os
from dotenv import load_dotenv
import requests
from requests.exceptions import ConnectionError, Timeout

# Load environment variables
load_dotenv()

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with OLLAMA"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question:{question}")
    ]
)

# Function to generate response
def generate_response(question, engine, temperature, max_tokens):
    try:
        llm = OllamaLLM(model=engine)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except ConnectionError as e:
        return "Error: Unable to connect to the Ollama API. Please check your network or API configuration."
    except Timeout:
        return "Error: The request to the Ollama API timed out. Please try again later."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Streamlit app setup
st.title("Simple Q&A Chatbot with Ollama")

# Select the model
engine = st.sidebar.selectbox("Select OpenAI model", ["gemma2:2b"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main user interface
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(f"Response: {response}")
else:
    st.write("Please provide user input.")
