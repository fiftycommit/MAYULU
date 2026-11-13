"""Construire et exécuter le notebook principal, limité aux appels de fonctions."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RACINE = Path(__file__).resolve().parent

INTRODUCTION = """# Détection d’anomalies

M’BEY Max, FERCHICHE Fares — M2 Data Science

Ce notebook suit l’énoncé sous-étape par sous-étape. Ses cellules de code ne contiennent que des appels :
le code est dans `experiences.py`, `experiences_kdd.py`, `modeles_complementaires.py` et `utils.py`.

Apprentissage → validation → test. Partitions stratifiées 60/20/20, graine 42.
Prétraitements et rééchantillonnages sont ajustés uniquement sur l’apprentissage.
Méthodes, hyperparamètres et seuils sont choisis sur validation, puis figés avant le test.
Dans chaque partie, les analyses de validation précèdent donc les résultats de test.

Les modèles complémentaires (RUS, ensembles, KDDCup99) ont été ajoutés après une première lecture du test
Credit Card. Leurs seuils viennent uniquement de la validation, mais cette extension reste exploratoire :
ce test n’est plus un holdout vierge. Une confirmation indépendante demanderait de nouvelles données.
"""

# (étape affichée ou None pour une réponse rédigée, titre, commentaire)
SECTIONS = [
    ("protocole", "## Protocole commun",
     "Classe positive = anomalie, fraude, panne ou intrusion ; un score élevé signifie suspect. LOF utilise 30 000 transactions d’apprentissage sur Credit Card. Pour KDDCup99, un tirage stratifié de 60 000 vecteurs dédoublonnés précède la partition ; les sept familles partagent un sous-échantillon d’apprentissage de 30 000 lignes, validation et test inchangés."),

    (None, "## Partie 1 — Données Mouse", "Les dix dernières lignes sont les anomalies de référence. Leurs étiquettes évaluent les décisions, sans régler IF, LOF, IQR ou KMeans."),
    ("mouse_1a", "### 1.a Charger, vérifier le format, représenter les points", "Deux variables numériques, cinq cents points. Les anomalies de référence apparaissent en rouge."),
    ("mouse_1b", "### 1.b Isolation Forest et LOF : outliers de l’ensemble",
     "Pour chaque méthode : points détectés au seuil par défaut, plan coloré par le score, histogramme. En détection d’outliers, le score LOF d’un point d’apprentissage vient de `negative_outlier_factor_`, calculé en excluant le point de ses propres voisins. `score_samples` et `decision_function` sont réservés au mode `novelty=True`, pour des points nouveaux : on ne les applique pas ici."),
    ("mouse_2a", "### 2.a Seuil sans étiquettes : écart interquartile",
     "Seuil = Q3 + 1,5 × (Q3 − Q1) sur les scores, avec le coefficient usuel de 1,5 fixé sans regarder les étiquettes. Le tableau donne aussi la seconde règle demandée, par clustering : un KMeans à deux groupes sur les scores, dont le seuil est le plus petit score du groupe au centre le plus élevé. Ce sont des heuristiques, pas des procédures optimales."),
    ("mouse_2b", "### 2.b Comparaison avec le seuil par défaut", "Points détectés, matrices de confusion et F1 de la classe anomalie pour chaque méthode et chaque règle. Le F1 est la mesure principale : il pénalise à la fois les anomalies manquées et les points normaux signalés."),
    ("mouse_2c", "### 2.c AP et AUC-ROC",
     "AP et AUC-ROC ne dépendent que de l’ordre des points selon le score. Un seuil choisit seulement un point sur les courbes ROC et précision-rappel : le déplacer ne change pas cet ordre, donc aucune de ces deux mesures. Ces valeurs décrivent la détection dans l’échantillon observé, pas une estimation sur des données futures."),
    ("mouse_3a", "### 3.a Détection de nouveautés avec LOF",
     "Une partie des points normaux et toutes les anomalies forment un test distinct. LOF `novelty=True` apprend sur les seuls normaux restants ; l’IQR est calculé sur les scores d’apprentissage, pas sur ceux du test. Seuls les points réservés sont évalués."),
    (None, "### 3.b Outliers et nouveautés : ce qui change",
     "En détection d’outliers, le modèle reçoit un ensemble contaminé, sans étiquettes, et doit désigner ses points atypiques. En détection de nouveautés, il apprend une référence supposée propre et juge des points qu’il n’a jamais vus. Constituer cette référence utilise les étiquettes : on sait quels points sont normaux, information absente du premier cadre. Les deux résultats ne sont donc pas obtenus à information égale, et leur meilleure qualité apparente ne prouve pas que le second algorithme est meilleur."),
    ("mouse_desaccords", "### Résultats attendus : désaccords entre IF et LOF et fragilité", "Croisement des alertes des deux méthodes pour chaque règle, sans nouvel ajustement."),

    (None, "## Partie 2 — Détection des fraudes", "La classe positive est la fraude. Les choix sont faits sur validation."),
    ("cc_1a", "### 1.a Types, valeurs manquantes, doublons, proportion de fraudes",
     "Les doublons de prédicteurs sont retirés avant le partage : sinon une même transaction pourrait se trouver en apprentissage et en test. Le contrôle porte sur le fichier brut."),
    ("cc_1b", "### 1.b Échelles et prétraitement par famille",
     "Amount est très asymétrique et s’étend sur une échelle bien plus large que les composantes V : on lui applique log(1 + Amount), sans retirer les grands montants. La régression logistique, LOF et SMOTE dépendent de distances ou de coefficients comparables : ils sont standardisés. Les arbres, leurs ensembles, IF et HBOS (histogramme par variable) n’utilisent que des comparaisons par variable : la standardisation ne leur apporte rien. Le tableau indique le prétraitement réellement retenu pour chaque famille. Chaque transformation est ajustée sur l’apprentissage seul."),
    ("cc_2a", "### 2.a Variantes de traitement du déséquilibre",
     "Un arbre et une régression logistique, chacun avec : aucune correction, poids de classes, SMOTE, RandomUnderSampler. SMOTE vise un ratio positifs/négatifs de 0,1 : équilibrer complètement obligerait à créer des centaines de points synthétiques par fraude réelle à partir de quelques centaines de voisins. RUS équilibre les classes (`auto`) : il garde toutes les fraudes, au prix de la quasi-totalité des transactions normales. Validation et test ne sont jamais rééchantillonnés. Le tableau donne les effectifs avant et après."),
    ("cc_2b", "### 2.b Six configurations par modèle et sélection",
     "Deux réglages (profondeur 4 ou 8, C = 0,01 ou 1) croisés avec aucune correction ou poids équilibrés donnent quatre essais ; un SMOTE et un RUS au réglage fort complètent les six. Les petits réglages servent de repère, les grands laissent de la capacité : l’objectif est de comparer des variantes, pas d’épuiser l’espace. Chaque ligne porte le seuil de validation sous précision ≥ 80 %. Le tri suit la règle : rappel maximal, puis précision et AP ; à défaut, le meilleur F1 est signalé comme non conforme."),
    ("cc_2c", "### 2.c Isolation Forest et LOF",
     "IF compare 100/300 arbres × 256/2048 points ; LOF compare 10/20/50/100 voisins en mode nouveauté, sur 30 000 lignes d’apprentissage. Une valeur retenue à l’extrémité d’une grille signale que le meilleur réglage peut se trouver au-delà : ces résultats ne désignent pas un optimum."),
    (None, "### 2.d Ajustement et sélection",
     "IF et LOF sont ajustés sans utiliser la cible. Le choix de leurs hyperparamètres et de leurs seuils se fait en revanche avec les étiquettes de validation, et le sous-échantillon de LOF est stratifié : c’est une sélection supervisée. Le protocole complet n’est donc pas purement non supervisé ; un cas réel sans étiquettes ne disposerait pas de cette étape."),
    ("cc_ensembles", "### Bloc complémentaire : EasyEnsemble, RUSBoost et XGBoost",
     "Ces ensembles sont soumis aux mêmes partitions et à la même règle. EasyEnsemble compare 5 et 10 ensembles de 50 souches, RUSBoost 50 et 100 arbres de profondeur 2, XGBoost les profondeurs 3 et 5 avec 100 arbres, taux 0,1 et un poids positif calculé sur l’apprentissage."),
    ("cc_3a", "### 3.a Validation : AP, AUC-ROC, courbes, F1 et BA selon le seuil",
     "Le F1 et la balanced accuracy optimisent chacun leur propre compromis et n’imposent aucun plancher de précision. Leur maximum peut donc se situer là où la précision est inférieure à 80 %, ou beaucoup plus prudente qu’il n’est nécessaire : ils ne répondent pas à l’objectif opérationnel."),
    ("cc_3b", "### 3.b Seuil retenu et seuil par défaut", "Le seuil retenu maximise le rappel sous précision ≥ 80 % sur validation. Le choix de méthode est fait ici, avant le test."),
    ("cc_3c", "### 3.c Tableau final sur le test", "Les modèles déjà ajustés et les seuils de validation sont appliqués au test sans retouche."),
    ("cc_3d", "### 3.d Interprétation en nombre de transactions", "Alertes à examiner, fraudes trouvées et fraudes qui échappent à la détection, avec les seuils figés."),

    (None, "## Partie 3 — Une méthode absente du cours : HBOS", ""),
    (None, "### 3.a Choix de la méthode", "HBOS (Histogram-Based Outlier Score) n’apparaît ni dans le cours ni parmi les modèles imposés. L’implémentation est écrite dans `utils.py`, pour rendre explicite la construction du score."),
    (None, "### 3.b Source, principe, hypothèses, score",
     "Source : Goldstein et Dengel (2012), *Histogram-based Outlier Score (HBOS): A fast Unsupervised Anomaly Detection Algorithm*, KI-2012 ; voir aussi la [documentation PyOD](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/hbos.html). Un histogramme est construit pour chaque variable sur l’apprentissage. Le score d’une observation additionne −log(hauteur normalisée + alpha) sur les variables : une valeur tombant dans une case rare ou vide augmente le score. L’hypothèse principale est l’indépendance des variables : les interactions ne sont pas modélisées."),
    (None, "### 3.c Intérêt attendu et hyperparamètres",
     "Sur Credit Card, les composantes d’ACP sont décorrélées, ce qui rend l’hypothèse moins déraisonnable sans la garantir : décorrélation n’est pas indépendance. HBOS n’a besoin d’aucune distance et peut utiliser tout l’apprentissage. Le nombre de cases règle la finesse des distributions (trop peu masque les queues, trop rend les histogrammes instables). Alpha évite le logarithme de zéro et règle la pénalité des cases vides ; il reste fixé à 0,1."),
    ("hbos_3d", "### 3.d Intégration aux mêmes fonctions d’évaluation", "Trois configurations (10, 50 et 200 cases), mêmes partitions, même règle de seuil, sans standardisation ni correction du déséquilibre."),
    ("hbos_3e", "### 3.e Comparaison avec IF et LOF : réussite, limite, coût", "Les profils moyens des contributions décrivent les réussites et échecs observés, sans en prouver la cause."),

    (None, "## Partie 4 — Transfert à la maintenance", "AI4I 2020 est un jeu **synthétique** : ses résultats ne décrivent pas directement une usine réelle."),
    ("ai4i_a", "### 4.a Cible et entrées", "`Machine failure` est la cible. Les identifiants et les colonnes TWF, HDF, PWF, OSF, RNF sont retirés : ces modes révéleraient directement la panne."),
    ("ai4i_b", "### 4.b Fréquence des pannes et prétraitement", "`Type` est encodée en one-hot. Les variables numériques sont standardisées pour les modèles sensibles à l’échelle. Tout est ajusté sur l’apprentissage."),
    ("ai4i_c", "### 4.c Modèles réutilisés", "IF, le meilleur modèle supervisé de Credit Card et le classique de référence gardent leurs hyperparamètres ; modèles et transformations sont réajustés sur AI4I. Aucune valeur de seuil n’est reprise d’un jeu à l’autre."),
    ("ai4i_d", "### 4.d Seuil sous précision ≥ 90 %, puis test", "Sur validation, le seuil maximise le rappel sous précision ≥ 90 % ; si c’est inaccessible, la règle commune de signalement s’applique. Fausses alertes = interventions inutiles, cas manqués = pannes manquées."),
    ("ai4i_e", "### 4.e Pourquoi le gagnant peut changer",
     "Les fraudes sont décrites par des composantes anonymisées ; les pannes d’AI4I suivent des règles synthétiques sur le couple, la puissance, la chaleur et l’usure. Variables, interactions, contrainte de précision et proportion de positifs diffèrent : garder une famille et ses hyperparamètres ne garantit pas le même classement. L’AP dépend de la prévalence : celles du tableau ci-dessous empêchent de comparer directement les AP entre les deux jeux."),

    (None, "## Partie 5 — Robustesse et explicabilité", ""),
    ("robustesse", "### 5.1 Contamination de la référence normale Mouse",
     "On remplace 0, 5 ou 10 % de la référence normale par des points uniformes tirés dans sa boîte englobante, élargie de 20 % de chaque côté. Taille d’apprentissage et hyperparamètres fixes, trois graines, aucun point injecté dans le test. Des points uniformes peuvent tomber dans les nuages normaux et sont alors peu atypiques : LOF, sensible au voisinage local, et IF, qui découpe l’espace globalement, n’y réagissent pas de la même manière. Ce mécanisme n’est pas isolé ici, et les conclusions dépendent du générateur choisi."),
    ("shap_cas", "### 5.2 SHAP : six observations du test",
     "SHAP explique la longueur moyenne de chemin corrigée d’IF, pas le score final : une contribution négative raccourcit le chemin et pousse vers l’alerte. Le fond contient 50 transactions d’apprentissage. Deux fraudes détectées, deux fausses alertes et deux fraudes manquées sont choisies, autant que possible ; une catégorie absente est signalée et complétée."),
    ("shap_comparaison", "### 5.2 Variables contribuant aux alertes", "SHAP permet de lire, cas par cas, ce qui a raccourci le chemin dans le modèle. Il ne permet ni d’établir une cause, ni de garantir qu’une alerte est une fraude, ni de redonner un sens métier à des composantes anonymisées ; six cas choisis ne représentent pas la population."),

    (None, "## Bloc complémentaire — KDDCup99", "La méthodologie de Credit Card est appliquée à la détection d’intrusions."),
    ("kdd_a", "### Corpus et doublons", "Corpus officiel à 10 %, intrusion = 1 si l’étiquette diffère de normal. Les doublons prédictifs sont retirés avant le tirage. Les attaques majoritaires dans le fichier brut deviennent minoritaires après dédoublonnage."),
    ("kdd_b", "### Validation", "Sept familles sont comparées ; la précision minimale de 80 % est un choix pédagogique annoncé, pas une exigence propre à ce jeu. RandomOverSampler remplace SMOTE pour conserver des catégories valides après one-hot."),
    ("kdd_c", "### Test", "Modèles et seuils de validation appliqués au test."),
    (None, "### Limites", "Données historiques de connexions simulées : ce test aléatoire ne mesure ni les attaques inédites ni un réseau actuel."),
]


def construire_notebook():
    cellules = [nbformat.v4.new_markdown_cell(INTRODUCTION),
                nbformat.v4.new_code_cell(
                    "from experiences import calculer_resultats, afficher_resultats, exporter_resultats\n\n"
                    "resultats = calculer_resultats()")]
    for etape, titre, commentaire in SECTIONS:
        cellules.append(nbformat.v4.new_markdown_cell(f"{titre}\n\n{commentaire}".rstrip()))
        if etape:
            cellules.append(nbformat.v4.new_code_cell(f'afficher_resultats(resultats, "{etape}")'))
    cellules.extend([
        nbformat.v4.new_markdown_cell(
            "## Livrables\n\nLes cinq parties et le bloc complémentaire non bonus sont couverts. "
            "SHAP et la robustesse sont traités en partie 5 ; seul le bonus autoencodeur n’est pas réalisé. "
            "Ce notebook se remet avec les fichiers Python qu’il importe, listés dans le README. "
            "L’appel ci-dessous écrit les nombres utilisés par les documents de `rendus/`."),
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
