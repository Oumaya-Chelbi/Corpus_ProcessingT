# TP6
"""
Visualisation des résultats du modèle de classification

Ce script a pour but de générer des graphiques qui vont nous aider à mieux comprendre
comment notre modèle s'est entraîné, à quel point il est performant, et aussi à analyser
ses erreurs grâce à la matrice de confusion.
"""

# Import des bibliothèques nécessaires
import json  # Pour charger les fichiers JSON contenant l'historique d'entraînement
import matplotlib.pyplot as plt  # Bibliothèque principale pour tracer les graphiques
import seaborn as sns  # Extension de matplotlib, permet de faire des graphiques plus esthétiques
import pandas as pd  # Pour manipuler les données tabulaires (ex : matrice de confusion)
import numpy as np  # Pour manipulations numériques et conversion en arrays
from sklearn.metrics import confusion_matrix

#  Configuration esthétique globale des graphiques

plt.style.use('seaborn-v0_8')
sns.set_palette("dark")
plt.rcParams['figure.dpi'] = 150  # Résolution plus élevée pour des images nettes
plt.rcParams['font.size'] = 10  # Taille de police modérée pour la lisibilité sans surcharger

def charger_resultats():
    """
    Charge les résultats enregistrés durant l'entraînement du modèle.
    Ces résultats sont essentiels pour visualiser et analyser la qualité du modèle.
    """

    try:
        # Lecture du fichier JSON contenant l'historique des métriques (loss, accuracy, etc.)
        with open('resultats_entrainement.json') as f:
            historique = json.load(f)

        # Conversion des listes en arrays numpy pour faciliter le traitement et les tracés
        historique = {k: np.array(v) for k, v in historique.items()}

    except Exception as e:
        # Gestion d'erreur simple : si le fichier est absent ou corrompu, on prévient l'utilisateur
        print(f"Erreur lors du chargement des résultats: {str(e)}")
        historique = None

    try:
        # Lecture du rapport textuel d'évaluation finale
        with open('evaluation.txt') as f:
            evaluation = f.read()
    except:
        # Si le fichier est manquant, on met une chaîne vide pour ne pas casser la suite
        evaluation = ""

    try:
        # Chargement de la matrice de confusion enregistrée sous forme CSV
        df_confusion = pd.read_csv('matrice_confusion.csv', index_col=0)
    except:
        # Si la matrice n'existe pas ou est vide, on retourne un DataFrame vide
        df_confusion = pd.DataFrame()

    # On retourne les trois objets importants pour la suite du script
    return historique, evaluation, df_confusion

def visualiser_apprentissage(historique):
    """
    Affiche les courbes d'évolution des métriques d'entraînement et de validation
    (précision et perte) pour analyser l'apprentissage du modèle sur plusieurs epochs.
    """

    # Vérification que l'historique existe bien pour éviter les erreurs
    if historique is None:
        print("Aucune donnée d'apprentissage à visualiser")
        return

    plt.figure(figsize=(12, 5))  # Création d'une figure large pour placer 2 sous-graphes côte à côte

    # Détection automatique du nombre d'epochs à partir des données chargées
    lengths = [len(v) for v in historique.values() if isinstance(v, np.ndarray)]
    if not lengths:
        print("Aucune donnée valide pour la visualisation")
        return

    num_epochs = max(lengths)  # On prend la plus longue série de métriques
    epochs = np.arange(1, num_epochs + 1)  # Création d'un vecteur [1, 2, ..., num_epochs]

    # Premier subplot : précision (accuracy) par epoch
    plt.subplot(1, 2, 1)
    if 'accuracy' in historique and len(historique['accuracy']) > 0:
        # Tracé de la précision sur les données d'entraînement (en bleu)
        plt.plot(epochs[:len(historique['accuracy'])], historique['accuracy'], 'b-o', label='Entraînement')
    if 'val_accuracy' in historique and len(historique['val_accuracy']) > 0:
        # Tracé de la précision sur les données de validation (en rouge)
        plt.plot(epochs[:len(historique['val_accuracy'])], historique['val_accuracy'], 'r-o', label='Validation')
    plt.title('Précision par epoch')  # Titre du graphique
    plt.xlabel('Epochs')  # Label de l'axe des abscisses
    plt.ylabel('Précision')  # Label de l'axe des ordonnées
    plt.legend()  # Légende pour distinguer les courbes

    # Deuxième subplot : perte (loss) par epoch
    plt.subplot(1, 2, 2)
    if 'loss' in historique and len(historique['loss']) > 0:
        # Tracé de la perte sur les données d'entraînement (en bleu)
        plt.plot(epochs[:len(historique['loss'])], historique['loss'], 'b-o', label='Entraînement')
    if 'val_loss' in historique and len(historique['val_loss']) > 0:
        # Tracé de la perte sur les données de validation (en rouge)
        plt.plot(epochs[:len(historique['val_loss'])], historique['val_loss'], 'r-o', label='Validation')
    plt.title('Perte par epoch')  # Titre du graphique
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    plt.tight_layout()  # Ajuste les espacements entre les subplots pour éviter les chevauchements
    plt.savefig('courbes_apprentissage.png')  # Enregistrement du graphique dans un fichier PNG
    plt.close()  # Ferme la figure pour libérer la mémoire

def visualiser_confusion(df):
    """
    Affiche une matrice de confusion sous forme de carte thermique (heatmap)
    pour visualiser les performances du modèle sur chaque classe.
    """

    if df.empty:
        print("Matrice de confusion vide")  # Cas où on n'a pas de matrice à afficher
        return

    plt.figure(figsize=(8, 6))
    # Heatmap avec annotations des valeurs, format entier, palette bleue, sans barre de couleur
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matrice de confusion')
    plt.ylabel('Vraies étiquettes')  # Axe Y = classes réelles
    plt.xlabel('Prédictions')  # Axe X = classes prédites
    plt.tight_layout()
    plt.savefig('matrice_confusion_visuelle.png')  # Enregistrement de la matrice en image
    plt.close()

def generer_rapport(evaluation):
    """
    Génère un rapport visuel en affichant le texte d'évaluation final
    dans une figure matplotlib, utile pour avoir un résumé clair dans un format image.
    """

    if not evaluation:
        print("Aucune évaluation à afficher")  # Pas de rapport à générer si vide
        return

    plt.figure(figsize=(10, 6))
    plt.text(0.05, 0.5, evaluation,
             fontfamily='monospace',
             fontsize=10,
             verticalalignment='center')
    plt.axis('off')  # On masque les axes pour ne garder que le texte visible
    plt.tight_layout()
    plt.savefig('rapport_evaluation.png')  # Sauvegarde du rapport en image
    plt.close()

def main():
    """
    Fonction principale appelée lors de l'exécution du script.
    Elle organise le processus complet : chargement des données puis création des visualisations.
    """

    print("Chargement des résultats...")  # Message pour indiquer que ça commence
    historique, evaluation, df_confusion = charger_resultats()  # Chargement des fichiers produits par l'entraînement

    print("Création des visualisations...")  # Indication d'avancement
    visualiser_apprentissage(historique)  # Graphique des métriques d'apprentissage
    visualiser_confusion(df_confusion)    # Matrice de confusion graphique
    generer_rapport(evaluation)            # Rapport final sous forme image

    print("Visualisations sauvegardées!")  # Confirmation de la fin du traitement


if __name__ == "__main__":
    main()
