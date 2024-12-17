from pathlib import Path

CONFIG = {
    "model_name": "xlm-roberta-base",
    "fine_tuned_model": "C:/Users/Lenovo/Desktop/ChatBot/models/fine_tuned_model",
    "supported_languages": ["fr", "ar"],
    "max_tokens": 512,
    "temperature": 0.7,
    "vectorstore_path": "C:/Users/Lenovo/Desktop/ChatBot/data/vectorstore",
    "knowledge_base_path": "C:/Users/Lenovo/Desktop/ChatBot/data/data.json"
}

PROMPT_TEMPLATE = '''
Tu es un agent de voyage professionnel pour notre agence.
Contexte de la conversation précédente : {chat_history}
Documents pertinents : {context}
Question du client : {question}

Instructions :
1. Réponds uniquement en {language}
2. Sois professionnel et courtois
3. Utilise les informations des documents si pertinent

Réponse :'''
