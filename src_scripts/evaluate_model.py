#TP6
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from train_transformer import BookGenreDataset, load_data
import os

def evaluate_model():
    # Configuration des paramètres
    # Dossier où est sauvegardé le modèle entraîné
    MODEL_DIR = 'genre_classifier'
    # Taille max des séquences de tokens en entrée (pour BERT)
    MAX_LENGTH = 256
    # Taille des lots (batch size) pour l'évaluation
    BATCH_SIZE = 16
    # Dossier où on sauvegarde les résultats de l'évaluation
    OUTPUT_DIR = 'evaluation_results'
    # Création du dossier s'il n'existe pas encore
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #  Chargement du modèle et du tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Chargement du tokenizer BERT qui correspond au modèle sauvegardé
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

    # Chargement du modèle BERT déjà entraîné pour la classification
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)

    # On met le modèle en mode évaluation (désactive dropout, etc.)
    model.eval()

    #  Chargement du mapping des labels
    # On charge le fichier CSV qui contient la correspondance entre les labels numériques et leurs noms (ex: 0 -> "Fantasy")
    label_mapping = pd.read_csv('label_mapping.csv', index_col=0, header=None).squeeze().to_dict()

    # On inverse le dictionnaire pour avoir un mapping id -> label (ex: 0: "Fantasy")
    id2label = {v: k for k, v in label_mapping.items()}

    #  Chargement des données de test
    # La fonction load_data() nous retourne les données d'entraînement, validation, test
    # Ici on récupère uniquement les textes et labels du test
    (_, _, test_texts, test_labels), _ = load_data()

    # On crée un Dataset personnalisé (défini dans train_transformer.py) avec nos données de test,
    # en appliquant le tokenizer et en tronquant à MAX_LENGTH
    test_dataset = BookGenreDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)

    # On utilise DataLoader pour charger les données par batch, ce qui est plus efficace
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    #  Initialisation des listes pour stocker les résultats
    predictions = []  # Les prédictions du modèle
    true_labels = []  # Les vraies étiquettes des textes

    # - Boucle d'évaluation sans calcul des gradients
    with torch.no_grad():
        # On parcourt chaque batch dans le DataLoader
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # On passe les données dans le modèle
            outputs = model(input_ids, attention_mask=attention_mask)

            # Les sorties logits correspondent aux scores bruts pour chaque classe
            # On récupère la classe avec le score maximal comme prédiction
            _, preds = torch.max(outputs.logits, dim=1)

            # On ajoute les prédictions et vraies labels à nos listes
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())


    # Correction importante pour gérer les classes absentes du test

    # On récupère les classes présentes dans la vérité terrain et dans les prédictions
    unique_labels = np.unique(true_labels + predictions)

    # On récupère la liste des noms des labels correspondants aux IDs trouvés
    available_labels = [id2label[label_id] for label_id in unique_labels]

    #  1. Rapport de classification
    # On utilise sklearn pour avoir un rapport complet avec précision, rappel, f1-score par classe
    # zero_division=0 évite les warnings si une classe n'est pas prédite du tout
    report = classification_report(true_labels, predictions,
                                 target_names=available_labels,
                                 zero_division=0)

    # On écrit ce rapport dans un fichier texte dans OUTPUT_DIR
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write("=== RAPPORT DE CLASSIFICATION ===\n\n")
        f.write(report)

    # 2. Matrice de confusion
    # La matrice de confusion montre les erreurs de classification entre chaque classe
    cm = confusion_matrix(true_labels, predictions, labels=list(unique_labels))

    # On crée une figure avec matplotlib
    plt.figure(figsize=(10, 8))

    # On utilise seaborn pour faire une heatmap  avec annotations
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=available_labels,
                yticklabels=available_labels,
                cmap='Blues')

    # Titres et légendes
    plt.title('Matrice de confusion')
    plt.ylabel('Vrai label')
    plt.xlabel('Label prédit')

    # Ajustement du layout pour que rien ne soit coupé
    plt.tight_layout()

    # On sauvegarde la figure dans le dossier OUTPUT_DIR
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))

    # On ferme la figure
    plt.close()

    #  3. Analyse des erreurs
    # On crée une liste pour stocker les erreurs d'étiquetage
    errors = []

    # On parcourt toutes les prédictions pour identifier où le modèle s'est trompé
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred:
            # On sauvegarde le texte tronqué (pour avoir un aperçu), l'étiquette vraie et celle prédite
            errors.append({
                'text': test_texts[i][:200] + "...",  # On limite à 200 caractères pour éviter les fichiers trop lourds
                'true': id2label[true],
                'predicted': id2label[pred]
            })

    # On convertit cette liste d'erreurs en DataFrame pandas puis on l'enregistre en CSV
    # Utile pour analyser manuellement les erreurs après coup
    pd.DataFrame(errors).to_csv(os.path.join(OUTPUT_DIR, 'classification_errors.csv'), index=False)

    #  Fin de l'évaluation, on affiche un message résumé
    print(f"Évaluation terminée. Résultats sauvegardés dans '{OUTPUT_DIR}'")
    print(f"- classification_report.txt")
    print(f"- confusion_matrix.png")
    print(f"- classification_errors.csv")


if __name__ == "__main__":
    evaluate_model()
