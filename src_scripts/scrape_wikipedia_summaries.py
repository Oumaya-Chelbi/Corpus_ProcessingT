#TP2
# Ce script sert à aller chercher un résumé sur Wikipedia pour chaque livre
# On part d’un fichier CSV contenant les URLs Wikipedia récupérées précédemment
# Et on extrait un paragraphe de résumé pour chacun (si possible !)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# On définit un User-Agent correct pour ne pas se faire bloquer par le site
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MasterTAL-Project/1.0; +https://univ-paris3.fr)"
}

# Pour ne pas spammer le site on attend entre chaque requête (3 secondes + petit random)
DELAY = 3

def get_wikipedia_summary(url):
    """Essaie d'extraire un résumé depuis une page Wikipedia. Utilise plusieurs stratégies (sections, balises, fallback...)."""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Stoppe tout si la page n'est pas accessible
        soup = BeautifulSoup(response.text, "html.parser")

        #  Méthode 1 : on cherche les sections les plus classiques de résumé
        section_ids = ["Plot", "Plot_summary", "Summary", "Synopsis", "Introduction"]
        for section_id in section_ids:
            section = soup.find(id=section_id) or soup.find("span", {"id": section_id})
            if section:
                content = []
                # On prend tous les paragraphes qui suivent cette section jusqu'à la prochaine H2
                for elem in section.parent.find_next_siblings():
                    if elem.name == "h2":
                        break
                    if elem.name == "p":
                        content.append(elem.get_text(strip=True))
                if content:
                    return "\n\n".join(content)

        #  Méthode 2 : si on n'a pas trouvé, on prend le 1er paragraphe long de la page
        for p in soup.select("div.mw-parser-output > p"):
            text = p.get_text(strip=True)
            if len(text) > 100:
                return text

        #  Méthode 3 : si rien du tout, on regarde dans la meta-description
        meta_desc = soup.find("meta", {"property": "og:description"})
        if meta_desc:
            return meta_desc.get("content", "")

        # Si toutes les méthodes échouent :
        return "[Aucun résumé trouvé sur la page Wikipedia]"

    except Exception as e:
        print(f"❌ Erreur lors du scraping de {url} : {str(e)}")
        return "[Erreur de chargement du résumé]"

def main():
    #  Fichier d'entrée = celui produit par extract_wikipedia_links.py
    input_file = "gutenberg_books_with_wikipedia.csv"
    #  Fichier de sortie = résumés inclus
    output_file = "gutenberg_books_with_summaries.csv"

    df = pd.read_csv(input_file)

    # Vérification que la colonne des liens existe bien
    if 'Wikipedia_URL' not in df.columns:
        raise ValueError(" Le fichier CSV doit contenir une colonne 'Wikipedia_URL'")

    df['Summary'] = ""  # Ajoute une colonne vide pour le résumé

    total_books = len(df)
    print(f" Début de l'extraction des résumés pour {total_books} livres...")

    for index, row in df.iterrows():
        wiki_url = row['Wikipedia_URL']

        # Cas sans lien Wikipedia ou lien mal formé
        if pd.isna(wiki_url) or not isinstance(wiki_url, str) or not wiki_url.startswith("http"):
            df.at[index, 'Summary'] = "[Pas de lien Wikipedia disponible]"
            continue

        print(f"→ Livre {index+1}/{total_books} | URL : {wiki_url[:60]}...")
        summary = get_wikipedia_summary(wiki_url)
        df.at[index, 'Summary'] = summary

        # Délai pour pas se faire bannir par Wikipedia
        time.sleep(DELAY + random.uniform(0, 2))

    #  On enregistre le fichier final
    df.to_csv(output_file, index=False, encoding='utf-8')

    #  Statistiques finales : combien de résumés valides ?
    success_count = df[df['Summary'].str.contains(r"\[(Aucun|Erreur|Pas de)", regex=True) == False].shape[0]
    print(f" Extraction terminée !")
    print(f" Résumés valides : {success_count}/{total_books} ({success_count/total_books:.1%})")
    print(f" Fichier sauvegardé : {output_file}")

#  Fichier généré
# gutenberg_books_with_summaries.csv :
#  Contient : Genre, URL, Titre, Wikipedia_URL, Summary
#  Il sera utilisé par le script suivant pour créer un vrai corpus par fichiers textes

if __name__ == "__main__":
    import random
    main()
