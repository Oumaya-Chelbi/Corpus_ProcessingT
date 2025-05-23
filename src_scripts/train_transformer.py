#TP5
"""
Ce script utilise la bibliothèque Transformers de Hugging Face pour faire du fine-tuning de BERT sur une tâche de classification de textes en genres littéraires.
Le dataset est supposé être organisé en dossiers où chaque dossier correspond à un genre (ex : "romance", "science-fiction", etc.), contenant des fichiers .txt de textes.
On crée un Dataset PyTorch personnalisé qui tokenise à la volée les textes avec le tokenizer BERT.
Le modèle BERT est configuré pour faire une classification multi-classes avec autant de sorties que de genres dans les données.
L’entraînement est fait avec la classe Trainer qui gère automatiquement le batching, la validation, l’évaluation, la sauvegarde, etc.
Les métriques principales sont accuracy et f1-score, avec un suivi précis pendant et après l’entraînement.
Le modèle et le tokenizer finaux sont sauvegardés pour être réutilisés facilement.
On génère des fichiers de résultats (mapping des labels, matrice de confusion, rapport d’évaluation) pour analyser la performance du modèle.
"""



import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import json
from collections import defaultdict



MODEL_NAME = 'bert-base-uncased'  # On utilise le modèle BERT pré-entraîné en anglais, version 'base' non-casée
MAX_LENGTH = 256                  # Longueur maximale des séquences tokenisées (padding/truncation à 256 tokens)
BATCH_SIZE = 8                   # Taille des lots d'entraînement
EPOCHS = 3                       # Nombre d'époques d'entraînement, compromis classique pour éviter overfitting
LEARNING_RATE = 2e-5             # Taux d'apprentissage faible, typique pour fine-tuning de BERT
OUTPUT_DIR = 'model_output'      # Dossier où seront enregistrés les résultats d'entraînement
CORPUS_DIR = 'corpus_augmente_par_genre'  # Dossier contenant les textes classés par genre


# Détection automatique du device de calcul : GPU (cuda), ou Apple Silicon (mps), sinon CPU
# j'ai pris ça d'u  forum parce qu'au début ça me faisait une erreur

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Utilisation du device: {device}")  # Affiche le device utilisé, utile pour vérifier qu'on utilise bien le GPU


# Classe personnalisée Dataset pour PyTorch
# Permet de gérer nos données textuelles et leurs labels sous forme tokenisée

class BookGenreDataset(Dataset):
    """Dataset personnalisé pour la classification de genres de livres"""
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        texts: liste des textes (strings)
        labels: liste des labels (int)
        tokenizer: tokenizer BERT pour transformer texte en tokens
        max_length: taille max des séquences tokenisées
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        # Retourne le nombre d'exemples dans le dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # Récupère le texte et label à l'indice idx
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenisation avec padding et troncature à max_length, retourne tenseurs PyTorch
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',  # padding à droite pour que toutes les séquences aient la même longueur
            truncation=True,       # coupe les séquences trop longues
            return_tensors='pt'    # retourne des tenseurs PyTorch (input_ids, attention_mask)
        )

        # On renvoie un dictionnaire attendu par le Trainer de Hugging Face
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),  # masque d'attention (1 sur token utile, 0 sur padding)
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Fonction pour charger les données à partir du corpus organisé par dossiers de genres
# Retourne les splits train/validation ainsi qu'un dictionnaire label -> index

def load_data():
    """Charge les textes et leurs labels depuis le dossier CORPUS_DIR
    Structure attendue : un dossier par genre, contenant des fichiers .txt"""
    texts = []
    labels = []
    label_dict = {}  # dictionnaire label string -> index int

    # Parcours des dossiers dans CORPUS_DIR
    for label_name in os.listdir(CORPUS_DIR):
        label_path = os.path.join(CORPUS_DIR, label_name)
        if not os.path.isdir(label_path):  # on ne traite que les dossiers
            continue

        # On attribue un index unique à chaque genre rencontré
        if label_name not in label_dict:
            label_dict[label_name] = len(label_dict)

        # Lecture de tous les fichiers .txt dans le dossier du genre
        for filename in os.listdir(label_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(label_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())                   # ajout du texte
                    labels.append(label_dict[label_name])   # ajout du label associé

    # Split des données en train et validation (80%/20%)
    # stratify=labels pour garder la même proportion de classes dans chaque split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)

    return (train_texts, train_labels, val_texts, val_labels), label_dict


# Fonction pour calculer les métriques d'évaluation pendant l'entraînement

def compute_metrics(p):
    """Calcule accuracy et F1-score pondéré à partir des prédictions et labels"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)  # on prend la classe la plus probable

    return {
        'accuracy': accuracy_score(labels, predictions),         # taux de bonnes prédictions
        'f1': f1_score(labels, predictions, average='weighted')  # F1-score pondéré pour multi-classes
    }


# Fonction principale d'entraînement

def train():
    """Entraîne le modèle BERT pour la classification des genres"""

    # Chargement des données
    (train_texts, train_labels, val_texts, val_labels), label_dict = load_data()
    # On sauvegarde le mapping label->index dans un csv pour référence future
    pd.Series(label_dict).to_csv('label_mapping.csv')
    print(f"Mapping des labels : {label_dict}")

    # Initialisation du tokenizer et du modèle BERT pour classification
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_dict)  # nombre de classes à prédire (nombre de genres)
    ).to(device)  # envoi du modèle sur le device

    # Création des datasets PyTorch pour entraînement et validation
    train_dataset = BookGenreDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = BookGenreDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Définition des arguments d'entraînement avec Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,                  # dossier de sortie des checkpoints et logs
        num_train_epochs=EPOCHS,                # nombre d'époques
        per_device_train_batch_size=BATCH_SIZE, # batch size train
        per_device_eval_batch_size=BATCH_SIZE,  # batch size validation
        learning_rate=LEARNING_RATE,            # learning rate fixé
        weight_decay=0.01,                      # poids de régularisation pour éviter overfitting
        eval_strategy="epoch",                  # évaluer à la fin de chaque époque
        save_strategy="epoch",                  # sauvegarder modèle à la fin de chaque époque
        load_best_model_at_end=True,            # charger le meilleur modèle selon la métrique f1
        metric_for_best_model='f1',             # critère pour choisir le meilleur modèle
        logging_dir='logs',
        logging_steps=50,                       # fréquence des logs
        report_to='none'                       # ne pas envoyer de rapports automatiques (ex: wandb)
    )

    # Création du Trainer de Hugging Face avec modèle, données, arguments et métriques
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("Début de l'entraînement...")

    # Lancement réel de l'entraînement
    trainer.train()

    # Récupération de l'historique d'entraînement (loss, accuracy, etc) pour sauvegarde
    history = trainer.state.log_history

    # Initialisation des dictionnaires pour stocker les métriques d'entraînement et de validation
    train_metrics = defaultdict(list)
    eval_metrics = defaultdict(list)

    # Parcours des entrées de l'historique d'entraînement
    for entry in history:
        # Stockage des différentes métriques dans les dictionnaires respectifs
        if 'loss' in entry and 'epoch' in entry:
            train_metrics['loss'].append(entry['loss'])
        if 'accuracy' in entry and 'epoch' in entry:
            train_metrics['accuracy'].append(entry['accuracy'])
        if 'eval_loss' in entry:
            eval_metrics['val_loss'].append(entry['eval_loss'])
        if 'eval_accuracy' in entry:
            eval_metrics['val_accuracy'].append(entry['eval_accuracy'])

    # Gestion d’un cas particulier : si aucune accuracy train, on crée une liste vide pour correspondre
    if not train_metrics['accuracy'] and eval_metrics['val_accuracy']:
        train_metrics['accuracy'] = [None] * len(eval_metrics['val_accuracy'])

    # On prépare un dictionnaire global des résultats à sauvegarder dans un fichier JSON
    resultats = {
        'accuracy': train_metrics['accuracy'],
        'loss': train_metrics['loss'],
        'val_accuracy': eval_metrics['val_accuracy'],
        'val_loss': eval_metrics['val_loss']
    }

    # Sauvegarde des résultats dans un fichier JSON pour analyse ou visualisation ultérieure
    with open('resultats_entrainement.json', 'w') as f:
        json.dump(resultats, f)

    # Évaluation finale du modèle sur le jeu de validation
    test_results = trainer.evaluate(val_dataset)

    # Prédictions sur le jeu de validation pour analyse plus fine
    predictions = trainer.predict(val_dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)

    # Calcul et sauvegarde de la matrice de confusion dans un fichier CSV
    pd.DataFrame(confusion_matrix(y_true, y_pred),
                index=label_dict.keys(),
                columns=label_dict.keys()).to_csv('matrice_confusion.csv')

    # Construction d’un rapport texte détaillé des métriques par classe
    evaluation_report = "=== RAPPORT FINAL ===\n\n"
    evaluation_report += f"Accuracy: {test_results['eval_accuracy']:.4f}\n"
    evaluation_report += f"F1-score: {test_results['eval_f1']:.4f}\n"
    evaluation_report += f"Perte: {test_results['eval_loss']:.4f}\n\n"
    evaluation_report += "Détails par classe:\n"

    # Calcul des métriques précises pour chaque classe (genre)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # Ajout au rapport des mesures par classe, avec précision, rappel et F1-score
    for i, genre in enumerate(label_dict.keys()):
        evaluation_report += f"{genre}:\n"
        evaluation_report += f"  - Precision: {precision[i]:.4f}\n"
        evaluation_report += f"  - Recall: {recall[i]:.4f}\n"
        evaluation_report += f"  - F1: {f1[i]:.4f}\n\n"

    # Écriture du rapport final dans un fichier texte
    with open('evaluation.txt', 'w') as f:
        f.write(evaluation_report)

    # Sauvegarde finale du modèle et du tokenizer dans un dossier
    model.save_pretrained('genre_classifier')
    tokenizer.save_pretrained('genre_classifier')
    print("Modèle sauvegardé dans 'genre_classifier/'")

if __name__ == "__main__":
    train()
