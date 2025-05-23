# TP3
"""
Résumé de ce que fait ce script :

- Il charge tous les fichiers texte nettoyés, rangés par genre, dans un DataFrame.
- Il calcule des statistiques simples (nombre total et unique de mots).
- Il produit plusieurs visualisations (boxplots, histogrammes, nuages de mots, heatmaps de similarité, courbes Zipf).
- Il sauvegarde tous les résultats dans des dossiers dédiés.
"""

# Import des bibliothèques nécessaires pour la manipulation de fichiers, le calcul, et la visualisation
import os
import matplotlib.pyplot as plt  # Visualisations graphiques
import seaborn as sns            # Amélioration esthétique des graphiques
from wordcloud import WordCloud  # Nuages de mots
from collections import Counter  # Compteur d'occurrences d'éléments
import pandas as pd              # Manipulation de données sous forme de tableaux (DataFrame)
import numpy as np               # Calcul numérique (tableaux, vecteurs)
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF pour pondérer mots
from sklearn.metrics.pairwise import cosine_similarity       # Mesure de similarité entre textes

# Configuration esthétique des graphiques
print("Styles disponibles:", plt.style.available)  # Affiche les styles disponibles de matplotlib
plt.style.use('seaborn-v0_8')  # Application d'un style prédéfini pour un rendu plus joli
sns.set_theme(style="whitegrid")  # Thème seaborn avec fond quadrillé blanc
plt.rcParams['figure.dpi'] = 150  # Résolution plus élevée pour figures
plt.rcParams['savefig.bbox'] = 'tight'  # Enregistrement des figures sans marges superflues

# Définition des dossiers d'entrée et sortie
CORPUS_DIR = "corpus_nettoye_par_genre"  # Dossier où se trouvent les fichiers textes nettoyés, classés par genre
OUTPUT_DIR = "visualisations"             # Dossier où seront enregistrées les visualisations globales
GENRE_DIR = "visualisations_par_genre"    # Dossier pour visualisations spécifiques à chaque genre

# Création des dossiers s'ils n'existent pas encore
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENRE_DIR, exist_ok=True)

def charger_corpus():
    """
    Charge tous les fichiers textes nettoyés dans un DataFrame pandas.
    Chaque ligne correspond à un document avec son genre, son titre, le nombre total de mots,
    le nombre de mots uniques, et le texte complet.
    """
    donnees = []

    for genre in os.listdir(CORPUS_DIR):
        genre_path = os.path.join(CORPUS_DIR, genre)
        if not os.path.isdir(genre_path):  # Ignore si ce n'est pas un dossier
            continue

        for fichier in os.listdir(genre_path):
            if not fichier.endswith('.txt'):  # Ne traite que les fichiers .txt
                continue

            with open(os.path.join(genre_path, fichier), 'r', encoding='utf-8') as f:
                texte = f.read()
                mots = texte.split()  # Tokenisation basique par espaces

                donnees.append({
                    'Genre': genre,
                    'Titre': fichier,
                    'Mots': len(mots),                # Nombre total de mots
                    'Mots_uniques': len(set(mots)),  # Nombre de mots distincts
                    'Texte': texte                    # Texte complet
                })

    return pd.DataFrame(donnees)  # Retourne un DataFrame pandas

def visualiser_longueurs(df):
    """
    Visualise la longueur des textes par genre :
    - Un boxplot montrant la distribution des longueurs
    - Un histogramme pour voir la distribution détaillée des longueurs par genre
    """
    plt.figure(figsize=(12, 5))

    # Boxplot : résumé statistique des longueurs par genre
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='Genre', y='Mots')
    plt.title("Longueur des textes par genre")
    plt.xticks(rotation=45)

    # Histogrammes superposés par genre (distribution des longueurs)
    plt.subplot(1, 2, 2)
    for genre in df['Genre'].unique():
        sns.histplot(df[df['Genre'] == genre]['Mots'],
                     bins=30, kde=True, label=genre, alpha=0.6)
    plt.title("Distribution des longueurs")
    plt.xlabel("Nombre de mots")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "longueurs_textes.png"))
    plt.close()

def generer_wordclouds(df):
    """
    Génère un nuage de mots (WordCloud) pour chaque genre, à partir des textes concaténés.
    Les mots les plus fréquents apparaissent en plus grand.
    """
    for genre in df['Genre'].unique():
        textes = ' '.join(df[df['Genre'] == genre]['Texte'])

        wc = WordCloud(width=800, height=400,
                       background_color='white',
                       max_words=200).generate(textes)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.title(f"Mots-clés : {genre}", fontsize=14)
        plt.axis('off')  # Pas d'axes pour les nuages de mots
        plt.savefig(os.path.join(OUTPUT_DIR, f"wordcloud_{genre}.png"))
        plt.close()

def analyser_similarite(df):
    """
    Calcule la similarité entre genres à partir de leurs textes combinés.
    - Utilise TF-IDF pour vectoriser les textes (avec max 1000 mots caractéristiques)
    - Calcule la similarité cosinus entre ces vecteurs
    - Affiche une heatmap des similarités entre genres
    """
    genres_textes = df.groupby('Genre')['Texte'].apply(' '.join)

    vectoriseur = TfidfVectorizer(max_features=1000, stop_words='english')
    matrice_tfidf = vectoriseur.fit_transform(genres_textes)

    similarite = cosine_similarity(matrice_tfidf)

    plt.figure(figsize=(8, 6))
    sns.heatmap(similarite, annot=True, fmt='.2f',
                xticklabels=genres_textes.index,
                yticklabels=genres_textes.index,
                cmap='Blues')
    plt.title("Similarité entre genres (TF-IDF)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "similarite_genres.png"))
    plt.close()

def calculer_statistiques(df):
    """
    Calcule et sauvegarde dans un fichier texte des statistiques descriptives par genre :
    - Moyenne, médiane, écart-type du nombre de mots
    - Nombre de textes par genre
    - Moyenne de mots uniques
    - Ratio mots uniques / mots moyens
    """
    stats = df.groupby('Genre').agg({
        'Mots': ['mean', 'median', 'std', 'count'],
        'Mots_uniques': 'mean'
    })

    # Renommage des colonnes pour plus de clarté
    stats.columns = ['Moyenne mots', 'Médiane mots', 'Écart-type mots',
                     'Nombre textes', 'Moyenne mots uniques']

    stats['Ratio mots uniques'] = stats['Moyenne mots uniques'] / stats['Moyenne mots']

    # Écriture dans un fichier texte
    with open(os.path.join(OUTPUT_DIR, "statistiques.txt"), 'w') as f:
        f.write("=== STATISTIQUES DU CORPUS ===\n\n")
        f.write(stats.round(2).to_string())
        f.write("\n\n=== DESCRIPTION GÉNÉRALE ===\n")
        f.write(str(df.describe()))  # Statistiques globales de toutes les colonnes numériques

def visualiser_zipf(df):
    """
    Génère la courbe de Zipf globale du corpus complet.
    - Affiche la fréquence des mots en fonction de leur rang (ordonnés par fréquence décroissante)
    - Utilisation d'échelles logarithmiques pour rendre la loi Zipf visible
    """
    print("Génération de la courbe de Zipf globale...")
    tous_les_mots = ' '.join(df['Texte']).split()
    freqs = Counter(tous_les_mots)
    sorted_freqs = sorted(freqs.values(), reverse=True)

    rangs = np.arange(1, len(sorted_freqs) + 1)
    frequences = np.array(sorted_freqs)

    plt.figure(figsize=(8, 6))
    plt.loglog(rangs, frequences, marker=".")
    plt.title("Courbe de Zipf (corpus entier)")
    plt.xlabel("Rang (log)")
    plt.ylabel("Fréquence (log)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig(os.path.join(OUTPUT_DIR, "zipf_courbe.png"))
    plt.close()

def visualiser_zipf_par_genre(df):
    """
    Génère une courbe de Zipf individuelle pour chaque genre.
    Même principe que la courbe globale, mais appliquée séparément à chaque genre.
    """
    print("Génération des courbes de Zipf par genre...")
    for genre in df['Genre'].unique():
        mots = ' '.join(df[df['Genre'] == genre]['Texte']).split()
        freqs = Counter(mots)
        sorted_freqs = sorted(freqs.values(), reverse=True)

        rangs = np.arange(1, len(sorted_freqs) + 1)
        frequences = np.array(sorted_freqs)

        plt.figure(figsize=(8, 6))
        plt.loglog(rangs, frequences, marker=".")
        plt.title(f"Zipf : {genre}")
        plt.xlabel("Rang (log)")
        plt.ylabel("Fréquence (log)")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.savefig(os.path.join(GENRE_DIR, f"zipf_{genre}.png"))
        plt.close()

def main():
    """
    Fonction principale :
    - Charge le corpus
    - Lance toutes les visualisations
    - Calcule les statistiques
    - Affiche les messages de progression
    """
    print("Chargement du corpus...")
    df = charger_corpus()
    print(f"Corpus chargé : {len(df)} documents")

    print("Visualisation des longueurs...")
    visualiser_longueurs(df)

    print("Génération des nuages de mots...")
    generer_wordclouds(df)

    print("Analyse des similarités entre genres...")
    analyser_similarite(df)

    print("Calcul des statistiques...")
    calculer_statistiques(df)

    print("Courbe de Zipf globale...")
    visualiser_zipf(df)

    print("Courbes de Zipf par genre...")
    visualiser_zipf_par_genre(df)

    print("Traitement terminé. Les résultats sont dans les dossiers :")
    print(f" - {OUTPUT_DIR}")
    print(f" - {GENRE_DIR}")

if __name__ == "__main__":
    main()
