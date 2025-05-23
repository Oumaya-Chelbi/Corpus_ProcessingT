#TP2

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# On définit un User-Agent personnalisé
# Cela permet de ne pas se faire bloquer par le site Gutenberg on sait jamais (je l'ai
# fait pour goodreads donc je l'ai fait ici aussi)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MasterTAL-Project/1.0; +https://univ-paris3.fr)"
}

# Lien racine utilisé pour compléter les chemins relatifs des livres
BASE_URL = "https://www.gutenberg.org"

# On choisit 3 catégories représentatives pour notre corpus : fiction, polar, humour
# Le but est d’avoir un échantillon équilibré pour l’analyse de style ou de contenu
GENRE_URLS = {
    "Fiction": "/ebooks/bookshelf/486",
    "Crime_Mystery": "/ebooks/bookshelf/433",
    "Humor": "/ebooks/bookshelf/453"
}

# Pour rester courtoise (eviter de ce faire bloquer,honnêtement j'ai toujours fait avec un delai
# pour tout mes projets donc je l'ai mis par reflèxe)
# avec le serveur, j’introduis une pause entre les requêtes
DELAY = 3
PER_PAGE = 25  # Chaque page de Gutenberg contient 25 livres par défaut
MAX_BOOKS = 300  # Je veux 300 livres uniques exactement par genre

# Cette fonction récupère les livres d’une page donnée (elle retourne une liste de dictionnaires)
def get_books_from_page(url):
    try:
        response = requests.get(BASE_URL + url, headers=HEADERS)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        books = []
        # Chaque livre est dans une balise <li class="booklink">
        for book in soup.select('li.booklink'):
            link = book.find('a', href=True)  # Lien vers le livre
            title = book.find('span', class_='title')  # Titre affiché
            if link and title:
                books.append({
                    'url': BASE_URL + link['href'],
                    'title': title.get_text(strip=True)  # Nettoie les retours à la ligne
                })
        return books

    except Exception as e:
        print(f"Erreur: {str(e)}")
        return []

# Cette fonction gère le scraping pour un genre donné, en parcourant les pages jusqu’à en avoir 300
def scrape_genre(genre_name, genre_path):
    print(f"\n=== Scraping {genre_name} ===")
    all_books = []
    start_index = 0
    page = 1

    # Boucle jusqu’à ce que j’aie 300 livres ou qu’il n’y ait plus de pages
    while len(all_books) < MAX_BOOKS:
        # Sur Gutenberg, la pagination se fait avec ?start_index=XX
        page_url = f"{genre_path}?start_index={start_index}" if page > 1 else genre_path

        print(f"Page {page} ({len(all_books)}/{MAX_BOOKS} livres)...")
        books = get_books_from_page(page_url)

        if not books:
            print("Fin des livres disponibles")
            break

        # Je vérifie que l’URL du livre n’a pas déjà été récupérée (évite les doublons)
        for book in books:
            if len(all_books) >= MAX_BOOKS:
                break
            if book['url'] not in [b['url'] for b in all_books]:
                all_books.append(book)

        start_index += PER_PAGE
        page += 1
        # Petite pause avec un peu d’aléatoire pour ne pas être détectée comme un bot
        time.sleep(DELAY + random.uniform(0, 1))

    return all_books[:MAX_BOOKS]  # Je m’assure de retourner exactement 300 livres

# Fonction principale qui pilote tout le processus
def main():
    all_data = []

    # Pour chaque genre défini, je lance la collecte
    for genre_name, genre_path in GENRE_URLS.items():
        books = scrape_genre(genre_name, genre_path)

        for book in books:
            all_data.append({
                'Genre': genre_name,
                'URL': book['url'],
                'Titre': book['title']
            })

        print(f"→ {genre_name}: {len(books)} livres")

    # Je transforme tout en DataFrame pandas
    df = pd.DataFrame(all_data)

    # Suppression éventuelle de doublons par URL
    df = df.drop_duplicates(subset=['URL'])

    # Garantie qu’il y a exactement 300 livres pour chaque genre (utile pour stats équilibrées)
    for genre in GENRE_URLS.keys():
        genre_count = len(df[df['Genre'] == genre])
        if genre_count > MAX_BOOKS:
            df = df.drop(df[df['Genre'] == genre].index[MAX_BOOKS:])

    # Je sauvegarde le fichier avec toutes les infos nécessaires (titre, genre, URL)
    filename = "gutenberg_books_300_par_categorie.csv"
    df.to_csv(filename, index=False)

    # Affichage des stats finales
    print("\n RÉSULTATS ")
    print(f"Fichier généré : {filename}")
    print("\nRépartition par catégorie (doit être 300 pour chaque) :")
    print(df['Genre'].value_counts())

    # Vérification manuelle anti-doublons par genre
    print("\n[VÉRIFICATION ANTI-DOUBLONS]")
    for genre in GENRE_URLS:
        genre_df = df[df['Genre'] == genre]
        duplicates = genre_df.duplicated(subset=['URL']).sum()
        print(f"{genre}: {duplicates} doublons détectés")

#  Fichier généré
# - gutenberg_books_300_par_categorie.csv :
#      Contient exactement 300 livres par genre
#      Colonnes : Genre, URL, Titre
#      Sert de base pour tout le pipeline suivant (notamment extract_wikipedia_links.py)

if __name__ == "__main__":
    main()
