#TP6
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os
from datetime import datetime

class GenrePredictor:
    """
    Cette classe permet de charger un modèle BERT déjà entraîné
    pour prédire le genre littéraire d'un texte donné.
    Elle gère aussi la sauvegarde des prédictions dans un fichier CSV.
    """

    def __init__(self, model_dir='genre_classifier', results_dir='prediction_results'):
        """
        Constructeur de la classe GenrePredictor.
        Initialise le tokenizer et le modèle BERT, charge le mapping label<->id,
        prépare le dossier pour sauvegarder les résultats.

        Args:
            model_dir (str): chemin du dossier où est stocké le modèle BERT entraîné.
            results_dir (str): dossier où seront enregistrés les résultats des prédictions.
        """


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Chargement du tokenizer BERT pré-entraîné dans model_dir
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        # Chargement du modèle BERT pour classification de séquences, transféré sur le device
        self.model = BertForSequenceClassification.from_pretrained(model_dir).to(self.device)

        # On met le modèle en mode évaluation, ce qui désactive dropout et gradient
        self.model.eval()

        # Chargement du fichier CSV contenant le mapping entre labels et IDs
        # Ce fichier a été créé lors de l'entraînement pour associer les classes
        self.label_mapping = pd.read_csv('label_mapping.csv', index_col=0, header=None).squeeze().to_dict()

        # Inversion du dictionnaire pour obtenir id -> label (plus facile pour prédire)
        self.id2label = {v: k for k, v in self.label_mapping.items()}

        # Création du dossier pour enregistrer les résultats (si n'existe pas)
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir

    def predict(self, text, max_length=256, save_results=True):
        """
        Méthode principale pour prédire le genre d'un texte.

        Args:
            text (str): le texte à classifier.
            max_length (int): longueur maximale des séquences d'entrée (padding/truncation).
            save_results (bool): indique si on doit sauvegarder la prédiction dans un fichier CSV.

        Returns:
            str: le genre prédit pour le texte.
        """

        # Tokenisation du texte avec padding/troncature pour avoir des séquences de longueur fixe
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',    # on complète les séquences courtes avec des tokens
            truncation=True,         # on coupe les séquences trop longues
            return_tensors='pt'
        )

        # On désactive la mise à jour des gradients car on est en phase d'inférence seulement
        with torch.no_grad():
            # On récupère les tenseurs input_ids et attention_mask et on les envoie sur le device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Passage dans le modèle BERT (sans labels, donc juste output des logits)
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # On récupère la classe avec la plus grande valeur parmi les logits (max softmax)
            _, preds = torch.max(outputs.logits, dim=1)

        # On convertit l'id prédit en label texte (ex: 0 -> 'fiction', 1 -> 'policier', etc)
        predicted_genre = self.id2label[preds.item()]

        # Si on souhaite sauvegarder les résultats dans un fichier CSV
        if save_results:
            self._save_prediction(text, predicted_genre)

        # On retourne la prédiction finale sous forme de chaîne de caractères
        return predicted_genre

    def _save_prediction(self, text, prediction):
        """
        Fonction interne pour sauvegarder une prédiction dans un fichier CSV.
        Chaque prédiction est enregistrée avec un timestamp et un extrait du texte.

        Args:
            text (str): texte d'entrée pour lequel on a fait la prédiction.
            prediction (str): label de la classe prédite.
        """

        # Timestamp précis pour différencier les fichiers de sauvegarde
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Nom du fichier de sauvegarde, horodaté pour éviter écrasement
        filename = f"predictions_{timestamp}.csv"

        # Chemin complet vers le fichier dans le dossier des résultats
        filepath = os.path.join(self.results_dir, filename)

        # Préparation d'un dictionnaire pour créer un DataFrame Pandas
        # On limite la longueur du texte à 500 caractères + "..." pour lisibilité
        result = {
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'text': [text[:500] + "..." if len(text) > 500 else text],
            'predicted_genre': [prediction]
        }

        df = pd.DataFrame(result)

        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            df = pd.concat([existing_df, df], ignore_index=True)

        # Sauvegarde du DataFrame en CSV sans index
        df.to_csv(filepath, index=False)

        # Message de confirmation dans la console
        print(f"\nRésultat sauvegardé dans: {filepath}")

# Point d'entrée du script si lancé directement
if __name__ == "__main__":
    # On crée un objet GenrePredictor qui charge tout le nécessaire
    predictor = GenrePredictor()

    # Quelques exemples de textes pour tester la prédiction pris sur internet
    sample_texts = [
        """
        The detective carefully examined the crime scene, noticing the small details
        that others had overlooked. A single fingerprint on the window sill would
        crack the case wide open.
        """,
        """
        The spaceship hovered silently above the alien planet, its crew marveling at
        the strange landscape below. This was humanity's first contact with an
        extraterrestrial civilization.
        """,
        """
        She couldn't stop laughing at his ridiculous joke, even though it wasn't
        really that funny. There was just something about his delivery that made
        everything hilarious.
        """
    ]

    # Pour chaque texte d'exemple, on prédit le genre et on affiche le résultat
    for text in sample_texts:
        predicted_genre = predictor.predict(text)
        print(f"\nTexte: {text[:100]}...")    # Affiche un extrait de 100 caractères
        print(f"Genre prédit: {predicted_genre}")
