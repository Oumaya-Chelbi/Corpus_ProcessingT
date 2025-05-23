#TP2

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Comme dans le premier script, j’utilise un User-Agent propre pour éviter d’être bloquée par Gutenberg
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MasterTAL-Project/1.0; +https://univ-paris3.fr)"
}

# Délai entre les requêtes pour respecter le site
DELAY = 3

def extract_wikipedia_link(book_url):
    """Essaie de trouver un lien Wikipedia dans la page d’un livre Gutenberg"""
    try:
        response = requests.get(book_url, headers=HEADERS)
        response.raise_for_status()  # Arrête si la réponse est une erreur
        soup = BeautifulSoup(response.text, "html.parser")

        # On cible d'abord les sections qui pourraient contenir des infos qui nous intéresse
        for section in soup.select("div.page_content, div.ebook, div.bibliography"):
            for a in section.find_all("a", href=True):
                href = a['href']
                if "wikipedia.org/wiki/" in href.lower():
                    return normalize_wikipedia_url(href)

        # Si on ne trouve rien dans les sections ciblées, on regarde partout
        for a in soup.find_all("a", href=True):
            href = a['href']
            if "wikipedia.org/wiki/" in href.lower():
                return normalize_wikipedia_url(href)

        return None  # Aucun lien trouvé

    except Exception as e:
        print(f"Erreur pour {book_url} : {str(e)}")
        return None

def normalize_wikipedia_url(href):
    """Nettoie l’URL Wikipedia pour qu’elle soit toujours complète et correcte"""
    if href.startswith("//"):
        return "https:" + href
    elif href.startswith("/"):
        return "https://en.wikipedia.org" + href
    elif not href.startswith("http"):
        return "https://en.wikipedia.org/wiki/" + href
    return href  # Lien déjà complet

def main():
    # Chargement du fichier créé précédemment
    input_file = "gutenberg_books_300_par_categorie.csv"
    output_file = "gutenberg_books_with_wikipedia.csv"

    df = pd.read_csv(input_file)

    # Je vérifie que le fichier contient bien la colonne des URLs nécessaires
    if 'URL' not in df.columns:
        raise ValueError("Le fichier CSV doit contenir une colonne 'URL'")

    # Ajout d’une colonne vide pour les liens Wikipedia
    df['Wikipedia_URL'] = None

    # Affichage de progression
    total_books = len(df)
    print(f"Début de l'extraction des liens Wikipedia pour {total_books} livres...")

    # Boucle principale : pour chaque livre, je cherche le lien Wikipédia
    for index, row in df.iterrows():
        book_url = row['URL']
        print(f"Progression: {index+1}/{total_books} | Livre: {book_url[-20:]}...")

        wiki_url = extract_wikipedia_link(book_url)
        df.at[index, 'Wikipedia_URL'] = wiki_url

        # Pause entre les requêtes (avec un peu d’aléatoire)
        time.sleep(DELAY + random.uniform(0, 1))

    # Sauvegarde dans un nouveau fichier CSV
    df.to_csv(output_file, index=False, encoding='utf-8')

    # Résumé final
    success_count = df['Wikipedia_URL'].notna().sum()
    print(f"\nExtraction terminée !")
    print(f"Liens Wikipedia trouvés: {success_count}/{total_books} ({success_count/total_books:.1%})")
    print(f"Fichier sauvegardé: {output_file}")

#  Fichier généré
# - gutenberg_books_with_wikipedia.csv :
#      Ajoute une colonne "Wikipedia_URL" aux livres récupérés
#      Contient les liens vers les pages Wikipedia pour les livres trouvés
#      Sert de base pour le scraping des résumés (prochain script du pipeline)

if __name__ == "__main__":
    import random  # Utilisé pour ajouter de la variabilité dans les pauses
    main()
