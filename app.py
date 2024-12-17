import streamlit as st
import json
import os
from src.agent import TravelAgent
from src.tools import TravelTools
from src.configg import CONFIG
from transformers import AutoTokenizer
import sys
import os

# Ajouter le dossier src au chemin Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
MAX_LENGTH = 512  # Define a constant for maximum length

def truncate_text(text: str, max_length: int = MAX_LENGTH) -> str:
    """
    Truncate text using the tokenizer with a fixed maximum length.
    """
    encoded = tokenizer.encode(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return tokenizer.decode(encoded[0], skip_special_tokens=True)

def prepare_context(docs, max_length: int = MAX_LENGTH) -> str:
    """
    Prepare and combine context documents within token limits.
    """
    combined_text = " ".join(doc["Réponse"] for doc in docs)
    return truncate_text(combined_text, max_length)

def load_data(data_path):
    """
    Load data from a JSON file if it exists.
    """
    if not os.path.exists(data_path):
        st.error(f"Error: File not found at {data_path}.")
        return None
    with open(data_path, "r", encoding="utf-8") as file:
        return json.load(file)

def main():
    st.set_page_config(page_title="Virtual Assistant - Travel Agency", layout="wide")
    st.title("Virtual Assistant - Travel Agency")
    st.markdown("Welcome! I can assist you in French, English, or Arabic.")

    data_path = "C:/Users/Lenovo/Desktop/ChatBot/data/data.json"

    if "agent" not in st.session_state:
        with st.spinner("Initializing assistant..."):
            try:
                data = load_data(data_path)
                if data is None:
                    return

                tools = TravelTools(data_path)
                st.session_state.agent = TravelAgent(tools)
                st.session_state.data = data
                st.session_state.error_count = 0
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
                return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if user_input := st.chat_input("How can I assist you?"):
        processed_input = truncate_text(user_input)

        st.session_state.messages.append({
            "role": "user",
            "content": processed_input
        })
        with st.chat_message("user"):
            st.write(processed_input)

        with st.chat_message("assistant"):
            with st.spinner("Looking for the best response..."):
                try:
                    relevant_docs = [
                        entry for entry in st.session_state.data
                        if user_input.lower() in entry["Question"].lower()
                    ]

                    if relevant_docs:
                        context = prepare_context(relevant_docs)
                        response = st.session_state.agent.get_response(
                            user_input=processed_input,
                            context=[context]
                        )
                    else:
                        response = (
                            "I couldn't find a relevant answer to your question. "
                            "Can you rephrase it?"
                        )

                    st.write(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.session_state.error_count = 0

                except Exception as e:
                    error_msg = str(e)
                    st.session_state.error_count += 1

                    if st.session_state.error_count >= 3:
                        response = (
                            "I am experiencing technical difficulties. Please try again later."
                        )
                    else:
                        response = (
                            "I'm sorry, I couldn't understand. Could you simplify your question?"
                        )

                    st.error(f"Error: {error_msg}")
                    st.write(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })

if __name__ == "__main__":
    main()
