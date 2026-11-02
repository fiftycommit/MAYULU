"""Construire et exécuter le notebook principal, limité aux appels de fonctions."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RACINE = Path(__file__).resolve().parent

INTRODUCTION = """# Détection d’anomalies : résultats

Apprentissage → validation → test. Partitions stratifiées 60/20/20, graine 42.
Prétraitements et rééchantillonnage ajustés uniquement sur l’apprentissage.
Méthodes, hyperparamètres et seuils choisis sur validation, puis figés avant le test.
Aucune validation ni aucun test n’est rééquilibré. Le tableau indique les effectifs utilisés.

Le notebook affiche les évaluations finales ; la recherche sur validation est documentée
séparément dans le Q/R. Mouse commence par des outliers dans l’échantillon observé,
puis utilise un test distinct pour les nouveautés. Les calculs sont dans les fichiers Python.

Les modèles complémentaires ont été ajoutés après une première lecture des tests Credit Card :
leurs réglages sont fixés a priori et leurs seuils choisis sur validation, mais cette extension
reste exploratoire. Une confirmation indépendante demanderait de nouvelles données.
"""

SECTIONS = [
    ("protocole", "Protocole, effectifs et prétraitements",
     "LOF utilise 30 000 transactions d’apprentissage sur Credit Card. Pour KDDCup99, un tirage stratifié de 60 000 vecteurs dédoublonnés précède la partition ; les sept familles partagent un sous-échantillon d’apprentissage de 30 000 lignes, validation et test inchangés."),
    ("mouse", "1. Mouse : outliers, scores et règles de seuil",
     "Les dix dernières lignes sont les anomalies de référence. Leurs étiquettes évaluent les décisions, sans régler IF, LOF, IQR ou KMeans. LOF outlier utilise les scores des points d’apprentissage ; son mode novelty est réservé aux nouveaux points. IF et LOF diffèrent surtout par les faux positifs aux seuils essayés. Dix anomalies rendent le F1 fragile ; AP et AUC-ROC ne changent pas lorsqu’on déplace seulement le seuil."),
    ("nouveautes", "Mouse : nouveautés sur le test réservé",
     "LOF apprend sur 343 points normaux et évalue 147 normaux réservés avec les dix anomalies. L’IQR est calculé sur les scores d’apprentissage, pas sur le test. Constituer cette référence normale utilise les étiquettes : le problème diffère de la détection d’outliers."),
    ("creditcard", "2. Credit Card : comparaison finale des méthodes",
     "La classe positive est la fraude. Le choix marqué dans le tableau vient uniquement de validation : rappel maximal sous précision ≥ 80 %, puis précision et AP. Si la contrainte est inaccessible, le compromis maximise le F1 et l’objectif est signalé non atteint. Les seuils restent inchangés au test, même si la précision baisse. Les colonnes alertes, fausses_alertes et manquees donnent les enquêtes nécessaires, inutiles et les fraudes manquées."),
    ("hbos", "3. HBOS : comparaison avec IF et LOF",
     "HBOS additionne −log(hauteur normalisée + alpha) sur les variables, sous une hypothèse d’indépendance. Trois nombres de cases sont essayés : 10, 50, 200, alpha=0,1. Les composantes d’ACP sont décorrélées, pas nécessairement indépendantes. Le score est peu coûteux et ses contributions sont lisibles, mais il ignore les dépendances ; une combinaison rare de valeurs banales peut lui échapper. Les profils moyens ci-dessous décrivent les réussites et échecs observés, sans prouver leur cause. [Goldstein et Dengel, 2012 ; documentation HBOS](https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.hbos)."),
    ("ai4i", "4. AI4I : transfert à la maintenance",
     "Cible : Machine failure ; identifiants et cinq modes de panne exclus. Le meilleur supervisé Credit Card (ensembles inclus) et IF sont transférés avec leurs hyperparamètres conservés ; le meilleur classique est gardé comme référence. Modèles et prétraitements sont réajustés, seuils choisis sous précision ≥ 90 % sur validation. AI4I est synthétique ; la cible d’origine est conservée malgré 27 discordances avec les modes de panne. Une précision élevée sur peu d’alertes peut masquer un rappel très faible. Les interactions de maintenance et la prévalence diffèrent des fraudes : ne pas comparer directement leurs AP."),
    ("robustesse", "5. Robustesse sur le même test Mouse",
     "On remplace 0, 5 ou 10 % de la référence normale par des points uniformes dans sa boîte élargie de 20 % de chaque côté. Taille d’apprentissage fixe, trois graines, aucun point injecté dans le test. Des points synthétiques peuvent tomber dans les nuages normaux. Les conclusions, notamment l’amélioration d’IF, dépendent donc de ce générateur."),
    ("shap", "SHAP : six transactions de test",
     "SHAP explique la profondeur moyenne d’isolement, pas directement le score final. Une contribution négative raccourcit le chemin et pousse vers l’alerte. Le fond contient 50 transactions d’apprentissage. Deux fraudes détectées, deux fausses alertes et deux fraudes manquées sont choisies autant que possible ; toute catégorie manquante est signalée et complétée. Ces cas ne représentent pas toute la population. SHAP décrit le modèle, sans causalité ni sens métier retrouvé pour les variables anonymisées."),
    ("kdd", "6. KDDCup99 : détection d’intrusions",
     "Corpus officiel à 10 %, intrusion=1 si l’étiquette diffère de normal. Les doublons prédictifs sont retirés avant le tirage ; le seul groupe à cible binaire contradictoire est exclu. Les attaques majoritaires dans le brut deviennent minoritaires après dédoublonnage. Sept familles sont comparées ; la précision minimale de 80 % est un choix pédagogique annoncé, non une exigence spécifique du sujet. RandomOverSampler remplace SMOTE pour conserver des catégories valides après one-hot. Données historiques de connexions simulées : ce test aléatoire ne mesure ni les attaques inédites ni un réseau actuel."),
]


def construire_notebook():
    cellules = [nbformat.v4.new_markdown_cell(INTRODUCTION),
                nbformat.v4.new_code_cell(
                    "from experiences import calculer_resultats, afficher_resultats, exporter_resultats\n\n"
                    "resultats = calculer_resultats()")]
    for nom, titre, commentaire in SECTIONS:
        cellules.extend([
            nbformat.v4.new_markdown_cell(f"## {titre}\n\n{commentaire}"),
            nbformat.v4.new_code_cell(f'afficher_resultats(resultats, "{nom}")'),
        ])
    cellules.extend([
        nbformat.v4.new_markdown_cell(
            "## Livrables\n\nLes cinq parties et le bloc complémentaire non bonus du PDF sont couverts, "
            "dont sous-échantillonnage supervisé, EasyEnsemble, RUSBoost, XGBoost et KDDCup99. "
            "Le bonus autoencodeur n’est pas réalisé. Les analyses détaillées, les sources et la synthèse "
            "de deux pages sont dans `rendus/`. Les fichiers Python nécessaires sont listés dans le README."),
        nbformat.v4.new_code_cell("exporter_resultats(resultats)"),
    ])
    notebook = nbformat.v4.new_notebook(cells=cellules)
    notebook.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


def main():
    notebook = construire_notebook()
    print("Apprentissage, sélection sur validation, puis tests et affichages…", flush=True)
    NotebookClient(notebook, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": str(RACINE)}}).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, RACINE / "TP_detection_anomalies.ipynb")
    print("Notebook exécuté ; résultats numériques exportés dans rendus/resultats.json.")


if __name__ == "__main__":
    main()
