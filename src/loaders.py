# Importing necessary libraries
from src.configg import CONFIG  # Importing the CONFIG dictionary
from langchain.embeddings import HuggingFaceEmbeddings  # Import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter  # Import CharacterTextSplitter
from langchain.document_loaders import JSONLoader, CSVLoader  # Import JSON and CSV Loaders
from langchain.vectorstores import Chroma  # Import Chroma for vector store

class DocumentManager:
    def __init__(self):
        # Initialisation des embeddings pour convertir le texte en vecteurs
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["model_name"]
        )
        # Diviser les documents en chunks pour une meilleure recherche
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,  # Taille des morceaux
            chunk_overlap=200  # Recouvrement pour le contexte
        )

    def load_documents(self):
        # Charger les offres commerciales depuis le fichier JSON
        json_loader = JSONLoader(
            file_path=CONFIG["knowledge_base_path"],
            jq_schema='.knowledge_base_path[]',  # Spécifie la structure JSON
            text_content=False
        )

        # Charger la base de connaissances depuis le fichier CSV
        csv_loader = CSVLoader(
            file_path=CONFIG["knowledge_base_path"]
        )

        documents = []
        for loader in [json_loader, csv_loader]:
            # Charger et diviser les documents
            docs = loader.load()
            split_docs = self.text_splitter.split_documents(docs)
            documents.extend(split_docs)

        return documents

    def setup_vectorstore(self):
        # Construire un stockage de vecteurs pour la recherche rapide
        documents = self.load_documents()
        vectorstore = Chroma.from_documents(
            documents,
            self.embeddings,
            persist_directory=CONFIG["vectorstore_path"]  # Sauvegarder pour utilisation future
        )
        return vectorstore
