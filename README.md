#                                 Corpus_ProcessingT



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
   


