from langchain.agents import initialize_agent, AgentType
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
from src.configg import CONFIG
import torch
import traceback
from src.tools import TravelTools
import sys
import os

# Ajouter le dossier parent au chemin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

class TravelAgent:
    def __init__(self, tools):
        # Initialiser le tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

        # Détecter si un GPU est disponible
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Assignation des outils
        self.tools = tools

        # Créer le pipeline
        self.pipeline = self._create_pipeline()

        # Initialiser l'agent
        self.agent = initialize_agent(
            tools=self.tools.get_tools(),  # Appelez get_tools() ici pour obtenir la liste des outils
            llm=self.pipeline,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,  # Activer les logs si nécessaire
            agent_kwargs={
                "input_keys": ["input", "context"],
                "max_iterations": 3,  # Limiter le nombre d'itérations
                "max_execution_time": 30  # Timeout en secondes
            }
        )

    def _create_pipeline(self):
        """
        Initialisation du pipeline de génération de texte avec des paramètres optimisés.
        """
        model = pipeline(
            task="text-generation",
            model=CONFIG["fine_tuned_model"],
            device=self.device,
            max_length=512,  # Longueur maximale fixe
            max_new_tokens=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=CONFIG["temperature"],
            num_return_sequences=1,
            do_sample=True,
            truncation=True
        )
        return HuggingFacePipeline(pipeline=model)

    def _prepare_input(self, text: str, max_length: int = 512) -> str:
        """
        Prépare et tronque le texte d'entrée de manière sûre.
        """
        # Encoder avec padding et troncature
        encoded = self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        # Décoder en texte
        return self.tokenizer.decode(encoded[0], skip_special_tokens=True)

    def get_response(self, user_input: str, context: list) -> str:
        """
        Générer une réponse en gérant la taille des entrées.
        """
        try:
            # Prétraiter le contexte
            processed_context = []
            remaining_length = 512  # Longueur maximale totale

            # Réserver de l'espace pour l'entrée utilisateur (environ 1/3)
            user_input_max = 170
            context_max = remaining_length - user_input_max

            # Traiter l'entrée utilisateur
            processed_input = self._prepare_input(user_input, max_length=user_input_max)

            # Traiter chaque élément du contexte
            for ctx in context:
                if context_max > 0:
                    processed_ctx = self._prepare_input(ctx, max_length=context_max)
                    processed_context.append(processed_ctx)
                    context_max -= len(self.tokenizer.encode(processed_ctx))

            # Combiner le contexte et l'entrée
            combined_input = f"{' '.join(processed_context)}\n{processed_input}"

            # Générer la réponse
            return self.agent.run(input=combined_input)

        except Exception as e:
            print(f"Erreur détaillée dans get_response: {type(e).__name__}: {str(e)}")
            print(f"Trace complète:", traceback.format_exc())
            return "Je suis désolé, je n'ai pas pu générer une réponse appropriée. Pourriez-vous reformuler votre question ?"

    def detect_language(self, text: str) -> str:
        """
        Détecte la langue de l'entrée utilisateur.
        """
        # TODO: Implémenter la détection de langue si nécessaire
        pass


# Exemple d'utilisation dans un script principal
if __name__ == "__main__":
    # Chemin vers votre fichier JSON
    data_path = 'C:/Users/Lenovo/Desktop/ChatBot/data/data.json'  # Corrigez le chemin si nécessaire
    travel_tools = TravelTools(data_path)

    # Créer l'agent en passant les outils
    agent = TravelAgent(travel_tools)

    # Tester avec une entrée utilisateur
    user_input = "Quel est le temps à Paris ?"
    context = ["Contexte général sur la météo"]

    response = agent.get_response(user_input, context)
    print(response)
