#TP2
#!/usr/bin/env python3
"""
Nettoyage avancé de corpus littéraire
- On enlève des mots trop fréquents et pas utiles (ex: novel, book, etc.)
- On met les mots à leur forme canonique (lemmatisation)
- On filtre aussi avec une liste de mots vides étendue (stopwords)
- On garde la structure de base du texte pour ne pas perdre d’info
"""

import os
import re
import unicodedata
from collections import defaultdict
from bs4 import BeautifulSoup
import pandas as pd
from nltk.tokenize import word_tokenize  # Découpage en mots
from nltk.corpus import stopwords  # Mots vides classiques
from nltk.stem import WordNetLemmatizer  # Pour ramener les mots à leur forme de base
import nltk

# Téléchargements des ressources NLTK nécessaires
# (On fait ça au début pour s'assurer que tout est dispo)
nltk.download('punkt')       # Tokenizer (découpe mots)
nltk.download('stopwords')   # Liste mots vides
nltk.download('wordnet')     # Dictionnaire pour lemmatizer
nltk.download('omw-1.4')     # Compléments WordNet

# CONFIGURATION
INPUT_DIR = "corpus_livres_par_genre"       # Dossier d'entrée avec les livres par genre
OUTPUT_DIR = "corpus_nettoye_par_genre"     # Dossier où on stocke les textes nettoyés
MIN_WORDS = 30                              # Seuls les textes avec au moins 30 mots sont conservés
LOG_FILE = "cleaning_log.txt"               # Fichier où on va écrire le rapport final

# Liste personnalisée de mots qu’on veut absolument supprimer,
# car ils sont très fréquents dans les résumés mais pas informatifs
CUSTOM_STOPWORDS = {
    'novel', 'book', 'one', 'find', 'first', 'said', 'will', 'page',
    'chapter', 'author', 'read', 'like', 'could', 'would', 'also',
    'many', 'must', 'might', 'may', 'shall', 'should','story','character',
    'life','work', 'time','published'
}

# Classe pour nettoyer le texte
class TextCleaner:
    def __init__(self):
        # Initialisation du lemmatizer (pour transformer mots en racines)
        self.lemmatizer = WordNetLemmatizer()
        # Préparer la liste complète des stopwords, y compris nos mots persos
        self.stop_words = self._prepare_stopwords()

    def _prepare_stopwords(self):
        """Prépare la liste finale de stopwords avec nltk + custom"""
        stops = set(stopwords.words('english'))  # Mots vides NLTK
        stops.update(CUSTOM_STOPWORDS)            # On ajoute nos mots à supprimer
        return stops

    def lemmatize_text(self, text):
        """Lemmatise chaque mot du texte (ex: 'running' → 'run')"""
        tokens = word_tokenize(text)               # Découpe en mots
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def clean_text(self, text):
        """Nettoie et prépare le texte en plusieurs étapes"""
        # 1. Normaliser les caractères Unicode (pour uniformiser accents, symboles, etc.)
        text = unicodedata.normalize('NFKC', text)

        text = BeautifulSoup(text, 'html.parser').get_text()

        # 2. Nettoyage spécifique pour les textes style Wikipédia ou autres :
        #    - on enlève les notes en crochets [1], (texte), {texte}, tabulations, retours à la ligne etc.
        text = re.sub(r'\[\d+\]|\([^)]*\)|\{[^}]*\}|\\n|\\t', '', text)
        #    - on supprime aussi les mentions style "Page 123", "Pages 56-60" (pas utiles)
        text = re.sub(r'\bPages?\b.*?\d+', '', text, flags=re.IGNORECASE)

        # 3. On passe le texte en minuscules et on lemmatise (forme canonique)
        tokens = self.lemmatize_text(text.lower())

        # 4. On filtre les mots :
        #    - on garde que les mots alphabétiques (pas de chiffres ni ponctuation)
        #    - on enlève les stopwords (mots vides)
        #    - on enlève les mots trop courts (moins de 3 lettres)
        clean_tokens = [
            word for word in tokens
            if word.isalpha()
            and word not in self.stop_words
            and len(word) > 2
        ]

        # 5. On rassemble tout ça en une chaîne de caractères nettoyée, prête à l’usage
        return ' '.join(clean_tokens)

# Fonction pour compter combien de livres par genre dans un dossier
def compter_livres(dossier):
    """Compte le nombre de fichiers txt par genre dans le dossier"""
    return {
        genre: len(fichiers)
        for genre in os.listdir(dossier)
        if os.path.isdir(os.path.join(dossier, genre))
        for fichiers in [[f for f in os.listdir(os.path.join(dossier, genre)) if f.endswith('.txt')]]
    }

# Fonction qui écrit un rapport après nettoyage
def generer_rapport(avant, apres, fichier_log):
    """
    Écrit un fichier texte résumé :
    - nombre de livres avant et après nettoyage
    - combien ont été supprimés par genre
    """
    with open(fichier_log, 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT DE NETTOYAGE ===\n\n")
        f.write(f"{'GENRE':<20} | {'AVANT':>10} | {'APRES':>10} | {'SUPPRIMES':>12}\n")
        f.write("-"*60 + "\n")

        for genre in sorted(avant.keys()):
            diff = avant[genre] - apres.get(genre, 0)
            f.write(f"{genre:<20} | {avant[genre]:>10} | {apres.get(genre, 0):>10} | {diff:>12}\n")

        total_avant = sum(avant.values())
        total_apres = sum(apres.values())
        f.write(f"\nTOTAL: {total_avant} -> {total_apres} ({total_avant - total_apres} supprimés)")

# Fonction principale qui nettoie tout le corpus
def nettoyer_corpus():
    """
    Parcourt tous les fichiers txt dans INPUT_DIR,
    nettoie chaque texte avec TextCleaner,
    conserve seulement ceux avec au moins MIN_WORDS mots,
    écrit les résultats dans OUTPUT_DIR,
    et produit un rapport final.
    """
    # On crée le dossier de sortie s’il n’existe pas encore
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cleaner = TextCleaner()  # On prépare notre nettoyeur

    comptage_avant = compter_livres(INPUT_DIR)  # Compte avant nettoyage
    comptage_apres = defaultdict(int)            # Compte après nettoyage

    # On parcourt chaque dossier de genre
    for genre in os.listdir(INPUT_DIR):
        in_dir = os.path.join(INPUT_DIR, genre)
        out_dir = os.path.join(OUTPUT_DIR, genre)

        # On saute si ce n’est pas un dossier (ex: fichier caché)
        if not os.path.isdir(in_dir):
            continue

        # On crée le dossier de sortie pour ce genre
        os.makedirs(out_dir, exist_ok=True)

        # On traite chaque fichier texte dans le dossier genre
        for fichier in os.listdir(in_dir):
            if not fichier.endswith('.txt'):
                continue  # On ignore tout ce qui n’est pas txt

            in_path = os.path.join(in_dir, fichier)
            out_path = os.path.join(out_dir, fichier)

            try:
                # Lecture du fichier
                with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
                    texte = f.read()

                # Nettoyage du texte
                texte_propre = cleaner.clean_text(texte)

                # On conserve seulement si le texte a assez de mots (qualité)
                if len(word_tokenize(texte_propre)) >= MIN_WORDS:
                    # On écrit le texte nettoyé dans le dossier de sortie
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(texte_propre)
                    comptage_apres[genre] += 1  # On compte ce texte conservé

            except Exception as e:
                # Si erreur, on affiche pour déboguer mais on continue
                print(f"Erreur avec {in_path}: {str(e)}")

    # À la fin, on fait un rapport pour voir combien on a perdu
    generer_rapport(comptage_avant, comptage_apres, LOG_FILE)
    print(f"\nNettoyage terminé. Voir le rapport dans {LOG_FILE}")

if __name__ == "__main__":
    print("Début du nettoyage du corpus...")
    nettoyer_corpus()

    # Fichier généré :
    # cleaning_log.txt
    # → Ce fichier journalise le nombre de fichiers nettoyés par genre,
    #    ainsi que ceux supprimés car trop courts ou non conformes.
    # → Le dossier 'corpus_nettoye_par_genre' contient les textes nettoyés,
    #    prêts pour une analyse NLP plus fiable et pertinente.
