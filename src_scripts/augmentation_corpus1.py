"""
TP4
"""
import os
import random
import re
from collections import defaultdict
import shutil
from googletrans import Translator  # Pour la traduction automatique (back-translation)
from nltk.corpus import wordnet      # Pour récupérer des synonymes avec WordNet
from nltk.tokenize import word_tokenize  # Pour découper le texte en mots
import nltk
from transformers import pipeline    # Pour le modèle de paraphrase
import pandas as pd                 # Pour manipuler les statistiques et données tabulaires

# -- Initialisation des ressources NLTK nécessaires --
nltk.download('wordnet')  # Dictionnaire de synonymes WordNet
nltk.download('punkt')    # Tokenizer pour découper les phrases en mots

# -- Configurations générales du script --
INPUT_DIR = "corpus_nettoye_par_genre"      # Dossier d'entrée avec corpus original
OUTPUT_DIR = "corpus_augmente_par_genre"    # Dossier de sortie pour corpus augmenté
AUGMENTATION_FACTOR = 3  # Nombre maximum de variantes générées par texte
MIN_TEXT_LENGTH = 50     # Longueur minimale (en mots) pour qu'un texte soit augmenté

# -- Initialisation des outils externes utilisés --
translator = Translator()  # Outil de traduction automatique pour back-translation
paraphrase_pipe = pipeline("text2text-generation", model="t5-small")  # Modèle T5 pour paraphrase


def clean_filename(title):
    """
    Nettoie un titre pour en faire un nom de fichier valide.
    Supprime les caractères interdits et limite la longueur à 100 caractères.
    """
    cleaned = re.sub(r'[\\/*?:"<>|]', "", str(title))  # Supprime caractères invalides pour fichiers
    cleaned = cleaned.replace(" ", "_").strip()       # Remplace espaces par underscore et enlève espaces en début/fin
    return cleaned[:100]  # Limite la longueur du nom de fichier

def get_synonyms(word):
    """
    Récupère une liste de synonymes d'un mot donné avec WordNet.
    On ignore le mot lui-même pour éviter répétition.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            syn_word = lemma.name().replace("_", " ").lower()
            if syn_word != word.lower():
                synonyms.add(syn_word)
    return list(synonyms)

def synonym_replacement(text, n=3):
    """
    Remplace jusqu'à n mots du texte par leurs synonymes.
    On choisit les mots aléatoirement parmi ceux qui ont des synonymes.
    """
    words = word_tokenize(text)
    new_words = words.copy()
    # Liste de mots uniques alphabétiques pour éviter ponctuation
    random_word_list = list(set([word for word in words if word.isalpha()]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            # Remplace toutes les occurrences du mot par un synonyme choisi
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    return ' '.join(new_words)

def back_translation(text, src_lang='en', intermediate_lang='fr'):
    """
    Réalise une augmentation par back-translation :
    - Traduction du texte en langue intermédiaire (ex: français)
    - Puis retraduction dans la langue source (ex: anglais)
    Cela permet d'obtenir une reformulation naturelle.
    """
    try:
        translated = translator.translate(text, src=src_lang, dest=intermediate_lang).text
        back_translated = translator.translate(translated, src=intermediate_lang, dest=src_lang).text
        return back_translated
    except Exception as e:
        print(f"Erreur back-translation: {e}")
        return text  # En cas d'erreur, on retourne le texte original

def random_insertion(text, n=2):
    """
    Insère aléatoirement des synonymes dans le texte.
    Pour chaque insertion, on choisit un mot au hasard, on récupère un synonyme,
    et on insère ce synonyme à une position aléatoire.
    """
    words = word_tokenize(text)
    if len(words) < 2:
        return text

    new_words = words.copy()
    for _ in range(n):
        synonyms = []
        counter = 0
        # On essaie jusqu'à 10 fois de trouver un mot avec synonymes
        while len(synonyms) < 1 and counter < 10:
            random_word = new_words[random.randint(0, len(new_words)-1)]
            synonyms = get_synonyms(random_word)
            counter += 1
        if synonyms:
            random_synonym = random.choice(synonyms)
            insert_pos = random.randint(0, len(new_words)-1)
            new_words.insert(insert_pos, random_synonym)

    return ' '.join(new_words)

def random_swap(text, n=2):
    """
    Échange aléatoirement la position de deux mots du texte, n fois.
    Cela modifie l'ordre des mots sans changer leur contenu.
    """
    words = word_tokenize(text)
    if len(words) < 2:
        return text

    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]

    return ' '.join(new_words)

def random_deletion(text, p=0.1):
    """
    Supprime aléatoirement des mots avec une probabilité p.
    Permet d'obtenir des textes plus courts en enlevant certains mots.
    """
    words = word_tokenize(text)
    if len(words) <= 1:
        return text

    new_words = [word for word in words if random.uniform(0, 1) > p]
    # Si suppression trop importante, on conserve au moins un mot aléatoire
    if len(new_words) == 0:
        return ' '.join(random.choice(words) for _ in range(random.randint(1, len(words))))

    return ' '.join(new_words)

def paraphrase_with_model(text, model=paraphrase_pipe):
    """
    Utilise un modèle T5 (transformer) pour paraphraser le texte.
    Génère une reformulation qui conserve le sens.
    """
    try:
        result = model(f"paraphrase: {text}", max_length=len(text.split())*2)
        return result[0]['generated_text']
    except Exception as e:
        print(f"Erreur paraphrase: {e}")
        return text

def apply_augmentation(text):
    """
    Applique plusieurs méthodes d'augmentation sur un texte donné.
    Retourne une liste de versions augmentées sans doublons,
    filtrées pour éviter les textes trop courts ou identiques à l'original.
    """
    augmented_texts = set()  # Utilisation d'un set pour éviter les doublons

    # On conserve le texte original dans le set
    augmented_texts.add(text)

    # Ajout des différentes méthodes d'augmentation simples
    augmented_texts.add(synonym_replacement(text))
    augmented_texts.add(random_insertion(text))
    augmented_texts.add(random_swap(text))
    augmented_texts.add(random_deletion(text))

    # Méthodes plus avancées avec traduction et paraphrase
    try:
        augmented_texts.add(back_translation(text))
        augmented_texts.add(paraphrase_with_model(text))
    except Exception as e:
        print(f"Erreur méthodes avancées: {e}")

    # Filtrer pour garder uniquement des textes assez longs et différents de l'original
    filtered = []
    for t in augmented_texts:
        if len(t.split()) >= MIN_TEXT_LENGTH and t != text:
            filtered.append(t)

    # On limite le nombre de variantes retournées au facteur d'augmentation choisi
    return filtered[:AUGMENTATION_FACTOR]

def balance_genres(input_dir, output_dir):
    """
    Fonction pour équilibrer le nombre de fichiers par genre.
    Pour les genres sous-représentés, on génère des augmentations pour compenser.
    """
    genre_counts = {}

    # Comptage des fichiers texte par genre
    for genre in os.listdir(input_dir):
        genre_path = os.path.join(input_dir, genre)
        if os.path.isdir(genre_path):
            count = len([f for f in os.listdir(genre_path) if f.endswith('.txt')])
            genre_counts[genre] = count

    if not genre_counts:
        return

    # Trouver le nombre maximal de fichiers dans un genre (pour équilibrer)
    max_count = max(genre_counts.values())

    for genre in genre_counts:
        if genre_counts[genre] < max_count:
            genre_path = os.path.join(input_dir, genre)
            files = [f for f in os.listdir(genre_path) if f.endswith('.txt')]

            # Calcul du nombre d'augmentations nécessaires pour ce genre
            needed = max_count - genre_counts[genre]
            augment_per_file = max(1, needed // len(files))

            print(f"Augmenting genre {genre} - {needed} needed, {augment_per_file} per file")

            # Pour chaque fichier du genre, on crée les versions augmentées
            for filename in files:
                filepath = os.path.join(genre_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

                # On applique les augmentations sur le texte
                augmented = apply_augmentation(text)

                # Sauvegarde des versions augmentées dans le dossier de sortie
                for i, aug_text in enumerate(augmented[:augment_per_file]):
                    new_filename = f"aug_{i}_{filename}"
                    new_path = os.path.join(output_dir, genre, new_filename)

                    with open(new_path, 'w', encoding='utf-8') as f:
                        f.write(aug_text)

def main():
    """
    Fonction principale qui organise le processus complet d'augmentation :
    - Prépare les dossiers
    - Copie les fichiers originaux
    - Applique l'équilibrage des genres par augmentation
    - Calcule et affiche les statistiques
    """
    # Suppression du dossier de sortie s'il existe pour repartir à zéro
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Recopie de la structure originale dans le dossier de sortie
    for genre in os.listdir(INPUT_DIR):
        genre_path = os.path.join(INPUT_DIR, genre)
        if os.path.isdir(genre_path):
            os.makedirs(os.path.join(OUTPUT_DIR, genre), exist_ok=True)

            for filename in os.listdir(genre_path):
                if filename.endswith('.txt'):
                    src = os.path.join(genre_path, filename)
                    dst = os.path.join(OUTPUT_DIR, genre, filename)
                    shutil.copy(src, dst)

    # Appel à la fonction pour équilibrer les genres par augmentation
    balance_genres(INPUT_DIR, OUTPUT_DIR)

    # Calcul des statistiques d'augmentation (nombre fichiers originaux vs augmentés)
    original_counts = defaultdict(int)
    augmented_counts = defaultdict(int)

    for genre in os.listdir(INPUT_DIR):
        genre_path = os.path.join(INPUT_DIR, genre)
        if os.path.isdir(genre_path):
            original_counts[genre] = len([f for f in os.listdir(genre_path) if f.endswith('.txt')])

    for genre in os.listdir(OUTPUT_DIR):
        genre_path = os.path.join(OUTPUT_DIR, genre)
        if os.path.isdir(genre_path):
            augmented_counts[genre] = len([f for f in os.listdir(genre_path) if f.endswith('.txt')])

    # Création d'un DataFrame pandas pour présenter clairement les stats
    stats = pd.DataFrame({
        'Original': original_counts,
        'Augmented': augmented_counts
    }).fillna(0)

    # Calcul des augmentations en valeur absolue et en pourcentage
    stats['Increase'] = stats['Augmented'] - stats['Original']
    stats['Increase_pct'] = (stats['Increase'] / stats['Original']) * 100

    # Affichage des statistiques d'augmentation
    print("\nStatistiques d'augmentation:")
    print(stats)

    # Sauvegarde des statistiques dans un fichier CSV
    stats.to_csv("augmentation_stats.csv")


if __name__ == "__main__":
    main()

# Le but ici c'etait de réequilibré le corpud parce que on avit beaucoup plus de livres fiction
# que les deux autres donc pas équilibré.
