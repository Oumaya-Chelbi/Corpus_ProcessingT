#TP2

import pandas as pd
import os
import re
from collections import defaultdict

#  INPUT_CSV : fichier généré par le script scrape_wikipedia_summaries.py
INPUT_CSV = "gutenberg_books_with_summaries.csv"

#  OUTPUT_DIR : nom du dossier où on va sauvegarder tous les fichiers .txt par genre
OUTPUT_DIR = "corpus_livres_par_genre"

def clean_filename(title):
    """Nettoie le titre pour créer un nom de fichier valide"""
    # Certains caractères ne sont pas autorisés dans les noms de fichiers → on les enlève
    cleaned = re.sub(r'[\\/*?:"<>|]', "", str(title))
    # On remplace les espaces par des underscores
    cleaned = cleaned.replace(" ", "_").strip()
    return cleaned[:100]  # On limite la taille du nom (important pour éviter les erreurs sous Windows)

def save_individual_books():
    # On charge le CSV avec les résumés
    df = pd.read_csv(INPUT_CSV)

    # On vérifie que les colonnes nécessaires sont bien là
    required_columns = ['Genre', 'Summary', 'Titre']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Il manque des colonnes: {missing}")

    # Compteurs pour savoir combien de fichiers on garde par genre
    genre_counts = defaultdict(int)
    ignored_count = 0  # Pour compter les résumés ignorés

    # Certains résumés doivent être ignorés : ceux qui contiennent un message d’échec
    skip_phrases = [
        '[Pas de lien Wikipedia disponible]',
        '[Aucun résumé trouvé sur la page Wikipedia]',
        '[Erreur de chargement du résumé]'
    ]

    # On crée le dossier principal s’il n’existe pas déjà
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # On parcourt chaque ligne du CSV
    for _, row in df.iterrows():
        genre = str(row['Genre']).strip()
        summary = str(row['Summary']).strip()
        title = row['Titre']

        # Si le résumé est vide, trop court ou contient une phrase à ignorer → on le saute
        if pd.isna(summary) or any(phrase in summary for phrase in skip_phrases) or len(summary) < 50:
            ignored_count += 1
            continue

        # On crée le sous-dossier du genre (ex : Fiction/, Humor/, etc.)
        genre_dir = os.path.join(OUTPUT_DIR, genre)
        os.makedirs(genre_dir, exist_ok=True)

        # On crée le nom du fichier à partir du titre
        filename = f"{clean_filename(title)}.txt"
        filepath = os.path.join(genre_dir, filename)

        # On écrit le résumé dans un fichier .txt
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(summary)
            genre_counts[genre] += 1
        except Exception as e:
            print(f"Erreur avec le livre '{title}': {str(e)}")

    # Petit rapport récapitulatif à la fin de l'exécution
    print("\n=== RAPPORT FINAL ===")
    print(f"Livres traités: {len(df)}")
    print(f"Livres ignorés: {ignored_count}\n")

    print("Livres sauvegardés par genre:")
    for genre, count in sorted(genre_counts.items()):
        print(f"- {genre}: {count} livres")

    print(f" Total fichiers créés: {sum(genre_counts.values())}")
    print(f"\nStructure créée:\n{OUTPUT_DIR}/")

#  Fichier de sortie :
# Dossier : corpus_livres_par_genre/
# Contient un sous-dossier par genre (Fiction/, Humor/, etc.)
# Chaque résumé est sauvegardé comme un fichier .txt propre, prêt à être utilisé dans le modèle

# Ce script marque la fin de la phase de collecte et transformation des données.
# Le corpus est enfin utilisable pour des analyses et entraînement de modèles !

if __name__ == "__main__":
    save_individual_books()
