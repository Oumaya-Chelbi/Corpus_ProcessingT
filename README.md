#                                 Corpus_ProcessingT

### PS : Initialement, je souhaitais suivre une structure classique de projet avec des dossiers tels que src/, data/, bin/, etc. Cependant, en raison de la nature de mon sujet, cette organisation s’est révélée peu lisible et difficile à maintenir. En effet, plusieurs scripts génèrent chacun leurs propres visualisations (notamment des matrices de confusion), ce qui entraîne une multitude de fichiers similaires. Regrouper toutes ces sorties dans un même dossier aurait nui à la clarté du projet, car il devenait difficile de savoir à quel script correspondait chaque fichier.La majorité des résultats et visualisations sont au format .png, avec quelques fichiers en .txt. Pour alléger le dépôt Git et éviter de dépasser les limites de taille imposées (notamment sur GitHub), je n’ai pas pu inclure tous les fichiers dans le dossier resultats_train_transformer/. Seuls les fichiers essentiels ont été conservés.Concernant le dossier src/, j'avais prévu de créer des sous-dossiers tels que process/ et plot/, mais certains scripts ne rentraient pas clairement dans ces catégories. J’ai donc préféré les laisser directement à la racine de src/ afin de ne pas surcharger la structure avec trop de sous-dossiers.Enfin, les données data/raw/ ont peu dire que ce sont celles issues du crawling/scraping, tandis data/clean/ c'est tout ce qui a été générées après exécution de clean_corpus.py (corpus_nettoyé, corpus_augmenter)j'ai juste pas pu tout mettre dans un même dossier car j'ai eu un petit souci de push. En cas de question, je reste joignable à oumayachelbi@gmail.com.

# Présentation du  projet:

## Dans quel besoin vous inscrivez vous ?

   Le projet s'inscrit dans le besoin d'organisation et d'exploration automatique de grandes collections de textes littéraires, notamment pour faciliter la recherche ou la recommandation de livres. Concrètement, il s'agit ici de développer un outil capable d’identifier automatiquement le genre littéraire (ex: science-fiction, policier, comédie, etc.) d’un livre, uniquement à partir de son résumé. Un tel outil pourrait par exemple aider à indexer automatiquement de nouvelles œuvres dans une base de données littéraire, ou à améliorer des systèmes de recommandation, Bibliothèques en ligne etc ..

## Quel sujet allez vous traiter ?

   Le sujet de ce projet est la classification de résumés de livres selon leur genre littéraire. L’objectif est d’entraîner un modèle de type Transformer (BERT) pour apprendre à prédire le genre d’un livre à partir de son résumé, de façon automatique.

## Quel type de tâche allez vous réaliser ?

   Il s'agit d'une tâche de classification de texte supervisée, plus précisément une classification mono-label (chaque livre appartient à un seul genre dans ce projet). Le pipeline complet comprend :

   - Constitution d’un corpus de résumés de livres,
   - Nettoyage, exploration, visualisation des données,
   - Entraînement d’un modèle de classification (fine-tuning de BERT),
   - Prédiction et évaluation des performances.

## Quel type de données allez vous exploiter ?

   Les données utilisées sont :

   - Titres de livres issus du Projet Gutenberg, qui est un projet de numérisation de textes libres de droit.
   - Résumés de livres récupérés à partir des pages Wikipédia associées à ces titres.
   - Remarque : Le projet Gutenberg fournit également des descriptions, mais elles sont souvent trop génériques ou générées automatiquement, donc moins fiables que les résumés rédigés sur Wikipédia.
   - Genres littéraires des livres : correspond à ceux données sur le site Projet Gutenberg.

## Source et accessibilité des données

   Les livres sont issus du Projet Gutenberg, qui propose des œuvres du domaine public, donc totalement libres d’accès et réutilisables.
   Les résumés proviennent des pages Wikipédia correspondantes aux livres sélectionnés.      
   Wikipédia est également une source ouverte, et les contenus sont publiés sous licence CC BY-SA, donc libres à condition de citer la source.
   J’avais envisagé d’utiliser Goodreads pour récupérer des résumés plus qualitatifs, mais j’ai finalement écarté cette option par précaution : même si j’ai pu scraper le site (le code est fonctionnel), son rachat par Amazon rend l’exploitation juridique des données incertaine. Pour respecter les consignes, j’ai donc préféré ne garder que des données issues de sources explicitement libres.
   


