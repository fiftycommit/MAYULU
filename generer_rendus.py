"""Régénérer les documents depuis le JSON et les images du notebook exécuté.

Aucun import des modules d'expériences, aucun entraînement. Lancer depuis n'importe
quel dossier : .venv/bin/python generer_rendus.py. Le PDF est imprimé séparément
à partir de /tmp/tp1-synthese.html ; ce script ne lance aucun navigateur.
"""
from pathlib import Path
import base64
import html
import json
import math
import re
import statistics

import mistune

RACINE = Path(__file__).resolve().parent
RENDUS = RACINE / "rendus"
FIGURES = RENDUS / "figures"
NON_SUPERVISES = {"Isolation Forest", "LOF", "HBOS"}
STYLE = """
body{font-family:Arial,sans-serif;color:#20252b;line-height:1.48;max-width:1120px;margin:32px auto;padding:0 22px}
h1,h2,h3{color:#17354c;line-height:1.18}h2{margin-top:1.7em}summary{cursor:pointer;font-weight:600;padding:9px 0}
details{border-bottom:1px solid #ddd;padding:5px 0}table{border-collapse:collapse;font-size:.87em;margin:14px 0;display:block;overflow-x:auto}
th,td{border:1px solid #ccd2d7;padding:5px 7px;text-align:left}th{background:#edf2f6}img{max-width:100%;height:auto}
code{font-size:.9em}a{color:#175784}pre{white-space:pre-wrap}
"""
PRINT_STYLE = """
@page{size:A4;margin:14mm 12mm}*{box-sizing:border-box}body{font-family:Arial,sans-serif;font-size:10pt;line-height:1.25;color:#171d22;margin:0;max-width:none;padding:0}
h1{font-size:16pt;margin:0 0 5pt}h2{font-size:12pt;margin:9pt 0 4pt;break-after:avoid}h2:nth-of-type(3){break-before:page}p{margin:5pt 0;break-inside:avoid}
table{width:100%;border-collapse:collapse;font-size:8.5pt;margin:5pt 0;break-inside:avoid}th,td{border:1px solid #b9c2c9;padding:2.5pt;text-align:left}th{background:#edf2f6}
a{color:inherit;text-decoration:none}h3{font-size:10pt;margin:6pt 0 3pt}li{margin:2pt 0}ul{margin:4pt 0;padding-left:14pt}
"""


def nombre(value, decimales=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nombre non fini dans les résultats")
        return f"{value:.{decimales}f}".replace(".", ",")
    return str(value).replace("|", " / ").replace("\n", " ")


def tableau(lignes, colonnes):
    """Colonnes : couples (clé JSON, titre lisible), sans dépendre de leur ordre."""
    return "\n".join([
        "| " + " | ".join(titre for _, titre in colonnes) + " |",
        "| " + " | ".join("---" for _ in colonnes) + " |",
        *["| " + " | ".join(nombre(ligne[cle], 6 if cle == "seuil" else 4)
                            for cle, _ in colonnes) + " |" for ligne in lignes],
    ])


COL_VAL = [("nom", "Configuration"), ("n_app", "n apprentissage"),
           ("ap_val", "AP"), ("auc_roc_val", "AUC-ROC"), ("seuil", "Seuil"),
           ("precision_val", "Précision"), ("rappel_val", "Rappel"),
           ("f1_val", "F1"), ("alertes_val", "Alertes"), ("atteint", "Objectif atteint"),
           ("temps_app_s", "Apprentissage (s)"), ("temps_score_val_s", "Score validation (s)")]
COL_TEST = [("famille", "Famille"), ("ap", "AP"), ("auc_roc", "AUC-ROC"),
            ("seuil", "Seuil"), ("precision", "Précision"), ("rappel", "Rappel"),
            ("f1", "F1"), ("balanced_accuracy", "BA"), ("alertes", "Alertes"),
            ("fausses_alertes", "FP"), ("manquees", "FN"), ("objectif test", "Objectif test")]
COL_DECISION = [("famille", "Famille"), ("règle", "Règle"), ("seuil", "Seuil"),
                ("precision", "Précision"), ("rappel", "Rappel"), ("f1", "F1"),
                ("balanced_accuracy", "BA"), ("alertes", "Alertes"),
                ("fausses_alertes", "FP"), ("manquees", "FN")]
COL_MOUSE = [("méthode", "Méthode"), ("règle", "Règle"), ("seuil", "Seuil"),
             ("precision", "Précision"), ("rappel", "Rappel"), ("f1", "F1"),
             ("balanced_accuracy", "BA"), ("alertes", "Alertes"),
             ("fausses_alertes", "FP"), ("manquees", "FN")]


def image(nom, legende):
    return f"![{legende}](figures/{nom}.png)"


def gagnant(jeu):
    choix = [ligne for ligne in jeu["test"] if ligne["sélectionnée sur validation"]]
    if len(choix) != 1:
        raise ValueError("Chaque jeu doit avoir exactement une méthode sélectionnée sur validation")
    return choix[0]


def retenues(jeu, familles=None):
    noms = {ligne["méthode"] for ligne in jeu["test"]
            if familles is None or ligne["famille"] in familles}
    return [ligne for ligne in jeu["validation"] if ligne["nom"] in noms]


def matrices(jeu):
    sections = []
    for famille, matrice in jeu["matrices"].items():
        (vn, fp), (fn, vp) = matrice
        sections.append(f"### {famille}\n\n| Réalité / prédiction | Classe 0 | Classe 1 |\n"
                        f"|---|---:|---:|\n| Classe 0 | {vn} | {fp} |\n| Classe 1 | {fn} | {vp} |")
    return "\n\n".join(sections)


def erreurs(jeu, familles=None):
    lignes = []
    for famille, matrice in jeu["matrices"].items():
        if familles is not None and famille not in familles:
            continue
        (vn, fp), (fn, vp) = matrice
        lignes.append({"famille": famille, "fpr": 100 * fp / (vn + fp),
                       "fnr": 100 * fn / (fn + vp),
                       "inutiles": 100 * fp / (vp + fp) if vp + fp else None,
                       "erreur": 100 * (fp + fn) / (vn + fp + fn + vp)})
    return tableau(lignes, [("famille", "Modèle"), ("fpr", "FPR (%)"),
                           ("fnr", "FNR (%)"), ("inutiles", "Alertes inutiles (%)"),
                           ("erreur", "Erreur globale (%)")])


def bilan(jeu, cible):
    phrases = []
    for ligne in jeu["test"]:
        vp = ligne["alertes"] - ligne["fausses_alertes"]
        phrases.append(f"{ligne['famille']} : {nombre(ligne['alertes'])} alertes, "
                       f"dont {nombre(vp)} {cible} et {nombre(ligne['fausses_alertes'])} fausses alertes ; "
                       f"{nombre(ligne['manquees'])} cas manqués. Précision {nombre(ligne['precision'])}, "
                       f"rappel {nombre(ligne['rappel'])}, objectif au test {ligne['objectif test']}.")
    return "\n\n".join(phrases)


def statistiques_robustesse(resultats):
    groupes = {}
    for ligne in resultats["robustesse"]:
        groupes.setdefault((ligne["méthode"], ligne["contamination"]), []).append(ligne)
    lignes = []
    for (methode, taux), valeurs in sorted(groupes.items()):
        ligne = {"méthode": methode, "taux": taux}
        for metrique in ("ap", "auc_roc"):
            nombres = [v[metrique] for v in valeurs]
            ligne[metrique] = statistics.mean(nombres)
            ligne[metrique + "_std"] = statistics.stdev(nombres)
        lignes.append(ligne)
    return tableau(lignes, [("méthode", "Méthode"), ("taux", "Contamination"),
                           ("ap", "AP moyenne"), ("ap_std", "Écart-type AP"),
                           ("auc_roc", "AUC-ROC moyenne"), ("auc_roc_std", "Écart-type AUC-ROC")])


def extraire_figures(notebook):
    """Les cellules sont identifiées par leur appel nommé, jamais par leur numéro."""
    noms = {
        "mouse": ["p1_mouse", "p1_if_mouse", "p1_lof_mouse", "p1_matrices_confusion", "p1_seuils", "p1_courbes"],
        "nouveautes": ["p1_matrices_nouveautes", "p1_nouveautes", "p1_courbes_nouveautes"],
        "creditcard": ["p2_matrices_confusion", "p2_courbes_test"],
        "hbos": ["p3_matrices_confusion", "p3_courbes_test"],
        "ai4i": ["p4_matrices_confusion", "p4_courbes_test"],
        "robustesse": ["p5_contamination"], "shap": ["p5_shap"],
        "kdd": ["p6_matrices_confusion", "p6_courbes_test"],
    }
    trouvees = set()
    for cellule in notebook["cells"]:
        source = cellule.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        appel = re.search(r'afficher_resultats\(\s*\w+\s*,\s*[\"\x27](\w+)[\"\x27]\s*\)', source)
        if not appel or appel[1] not in noms:
            continue
        partie = appel[1]
        images = [sortie["data"]["image/png"] for sortie in cellule.get("outputs", [])
                  if "image/png" in sortie.get("data", {})]
        if len(images) != len(noms[partie]):
            raise ValueError(f"Cellule {partie} : {len(images)} figures, {len(noms[partie])} attendues")
        for nom, contenu in zip(noms[partie], images):
            contenu = "".join(contenu) if isinstance(contenu, list) else contenu
            (FIGURES / f"{nom}.png").write_bytes(base64.b64decode(contenu, validate=True))
        trouvees.add(partie)
    if trouvees != set(noms):
        raise ValueError(f"Notebook incomplet : cellules manquantes {set(noms) - trouvees}")


GRILLE = """Les familles classiques ont six configurations au total, pas six par variante. Pour l’arbre (profondeurs 4 et 8) et la logistique (C=0,01 et 1), on croise les deux valeurs avec absence de correction ou poids équilibrés, soit quatre essais. On ajoute un seul sur-échantillonnage au réglage fort et un seul RandomUnderSampler au réglage fort. Sur Credit Card, SMOTE vise un ratio positif/négatif de 0,1 et RUS équilibre les classes (`auto`). Sur KDD, RandomOverSampler remplace SMOTE pour préserver les catégories : il copie des lignes réelles au lieu d’interpoler leurs encodages. ROS et RUS utilisent `auto`. Cette correction porte sur la validité des données synthétiques, pas sur le choix d’un meilleur score de test.

EasyEnsemble compare 5 et 10 ensembles, chacun avec un AdaBoost de 50 souches. RUSBoost compare 50 et 100 arbres de profondeur 2. XGBoost compare les profondeurs 3 et 5 avec 100 arbres et un taux d’apprentissage de 0,1 ; son poids positif est le rapport négatifs/positifs de l’apprentissage seulement. IF compare 100/300 arbres × 256/2048 points ; LOF compare 10/20/50/100 voisins en mode nouveauté. HBOS compare 10, 50 et 200 cases, avec alpha=0,1."""
EXPLORATOIRE = "L’extension (RUS, ensembles et KDD) a été réalisée après une première lecture du test Credit Card. Les réglages sont annoncés a priori et les seuils utilisent uniquement la validation, mais cette comparaison reste exploratoire : le test ne peut plus être présenté comme un holdout entièrement vierge. Une confirmation demanderait de nouvelles données."
TRANSFERT = "Le meilleur supervisé global sur Credit Card, ensembles compris, et IF sont transférés à AI4I avec leurs hyperparamètres conservés. Le classique de référence ({classique}) est conservé, sans être la gagnante globale de la comparaison étendue Credit Card. Les transformations et modèles sont réajustés sur l’apprentissage AI4I ; les seuils sont choisis à nouveau sur sa validation sous précision ≥ 90 %. Pour XGBoost, `scale_pos_weight` est conservé depuis Credit Card, conformément au transfert demandé : il n’est ni recalculé sur AI4I ni réglé sur son test."
SOURCES = """- Sujet : *Atelier : Détection d’anomalies avec Python*, 2026–2027, et cours de H. Elghazel, fournis à la racine.
- Mouse : fichier fourni avec le TP, `data/mouse.txt`.
- Credit Card : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- AI4I 2020 : https://doi.org/10.24432/C5HS5C
- KDDCup99, corpus officiel à 10 % : https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html ; archive https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
- Goldstein et Dengel (2012), *HBOS: A Fast Unsupervised Anomaly Detection Algorithm*, KI-2012 ; documentation https://pyod.readthedocs.io/en/latest/_modules/pyod/models/hbos.html
- EasyEnsemble et RUSBoost : https://imbalanced-learn.org/stable/ensemble.html ; XGBoost : https://xgboost.readthedocs.io/en/stable/ ; SHAP : https://shap.readthedocs.io/"""


def classique(jeu):
    """Famille classique de référence réellement présente dans le jeu (arbre ou logistique)."""
    return next(l["famille"] for l in jeu["test"] if l["famille"] in ("Arbre de décision", "Régression logistique"))


def phrase_variante(cc):
    v = cc["variante_marge"]
    g = next(l for l in v["test"] if l["sélectionnée sur validation"])
    cible = int(round(100 * (0.80 + v["marge"])))
    return (f"Variante complémentaire, distincte de la règle de l’énoncé (80 %) : seuils cherchés à {cible} % sur validation "
            f"(marge de {int(round(100 * v['marge']))} points fixée a priori, la validation ne comptant que peu de fraudes). "
            f"Modèle choisi « {g['méthode']} » : précision {nombre(g['precision'])}, rappel {nombre(g['rappel'])} au test, "
            f"objectif de 80 % {g['objectif test']}.")


def texte_variante(cc):
    return (phrase_variante(cc) + " Le résultat principal reste celui de la règle de l’énoncé. "
            "Cette variante est comparée séparément ; le test Credit Card ayant déjà été lu, elle reste exploratoire.\n\n"
            + tableau(cc["variante_marge"]["test"], COL_TEST))


def reponses(resultats):
    cc, ai, kdd = (resultats[nom] for nom in ("creditcard", "ai4i", "kdd"))
    choix_cc, choix_ai = gagnant(cc), gagnant(ai)
    val_cc = tableau(cc["validation"], COL_VAL)
    test_cc = tableau(cc["test"], COL_TEST)
    partitions = tableau(resultats["partitions"], [("jeu", "Jeu"), ("partition", "Partition"),
                                                ("observations", "Observations"), ("positifs", "Positifs")])
    non_sup = [l for l in cc["test"] if l["famille"] in NON_SUPERVISES]
    bilan_cc = bilan(cc, "fraudes")
    r = {
        0: "Comparer le classement des observations et les décisions après seuillage sur Mouse, Credit Card, AI4I et KDDCup99. Le TP couvre les cinq parties, puis le bloc complémentaire non bonus : sous-échantillonnage supervisé, EasyEnsemble, RUSBoost, XGBoost et intrusions. Seul le bonus autoencodeur n’est pas réalisé.\n\n" + EXPLORATOIRE,
        1: "Le notebook exécuté ne contient que des imports, appels et résultats finaux. L’entraînement et la sélection sont dans `experiences.py`, `experiences_kdd.py`, `modeles_complementaires.py` et `utils.py`. Les appels nommés sont protocole, mouse, nouveautes, creditcard, hbos, ai4i, robustesse, shap et kdd. `generer_notebook.py` reconstruit et exécute le notebook ; `generer_rendus.py` produit les documents sans entraînement, depuis `rendus/resultats.json` et les figures du notebook. Les analyses de validation restent dans ce Q/R.",
        2: "Les partitions et modèles aléatoires utilisent la graine 42. La robustesse est répétée sur les graines 0, 1 et 2. Les partitions sont communes aux méthodes et les données locales permettent de travailler hors connexion. Les versions effectives sont affichées dans la rubrique protocole du notebook, les dépendances fixées dans `requirements.txt`. Les temps ci-dessous viennent du JSON de cette exécution : ce ne sont pas des moyennes de benchmarks.",
        3: "Les partitions sont stratifiées 60/20/20. Le dédoublonnage précède le partage pour Credit Card et KDD ; les transformations apprennent seulement sur l’apprentissage. La validation choisit les réglages et les seuils ; le test décrit les résultats des choix figés.\n\n" + partitions + "\n\nSur KDD, la partition d’apprentissage contient 36 000 lignes, mais les sept familles utilisent le même sous-échantillon stratifié de 30 000 lignes. Mouse suit les cadres outliers puis nouveautés.\n\n" + EXPLORATOIRE,
        5: GRILLE + "\n\nLOF Credit Card utilise un sous-échantillon stratifié fixe : " + nombre(cc["informations"]["app_lof"]) + " lignes dont " + nombre(cc["informations"]["positifs_app_lof"]) + " fraudes. Validation et test restent complets. Sur AI4I, on transfère les réglages retenus, sans nouvelle grille.\n\n" + TRANSFERT,
        13: "Pour chaque configuration, la validation choisit le rappel maximal parmi les seuils donnant au moins une alerte et respectant la précision minimale (0,80 en fraude, 0,90 en maintenance). Les égalités sont départagées par précision puis AP. Les configurations conformes sont prioritaires ; à défaut, le meilleur F1, puis rappel, précision et AP désignent un compromis explicitement non conforme. Les choix sont ensuite figés. Une alerte correspond à un score supérieur ou égal au seuil. Pour les détecteurs sklearn, le seuil inclusif représente la décision native stricte grâce à `nextafter(-offset_, +inf)`.",
        15: "Le notebook appelle `calculer_resultats()` puis `afficher_resultats(resultats, nom)` ; aucun apprentissage ou réglage n’est caché dans une cellule de présentation. Les expériences réutilisent `utils.decouper_donnees`, les pipelines de prétraitement, `utils.comparer_configurations`, `utils.choisir_seuil_contraint` et `utils.selectionner`. L’évaluation finale applique les modèles déjà ajustés et les seuils de validation. Les résultats sont exportés par `exporter_resultats` sous des clés nommées, pas des numéros de cellule.",
        17: "IF et LOF sont ajustés sans étiquettes sur les points Mouse, anomalies comprises. IF utilise 100 arbres et 256 points par arbre ; LOF utilise 20 voisins. Une grande valeur indique toujours une observation suspecte : on inverse `score_samples` pour IF et `negative_outlier_factor_` pour LOF outlier. Ce ne sont pas des probabilités. Le mode nouveauté de LOF ne doit pas servir à réévaluer ses propres points d’apprentissage, car les voisinages diffèrent.\n\n" + image("p1_if_mouse", "IF : décisions, scores continus et histogramme") + "\n\n" + image("p1_lof_mouse", "LOF : décisions, scores continus et histogramme"),
        18: "IQR utilise Q3 + 1,5 × (Q3 − Q1), avec coefficient fixé sans étiquettes. KMeans forme deux groupes sur les scores et retient le groupe de centre le plus élevé. Les étiquettes ne servent qu’à évaluer ces règles. Le tableau contient les décisions sur l’échantillon observé, pas une performance prospective.\n\n" + tableau(resultats["mouse"]["seuils"], COL_MOUSE) + "\n\n" + image("p1_seuils", "Décisions Mouse pour chaque méthode et règle") + "\n\n" + image("p1_matrices_confusion", "Matrices Mouse ; chaque panneau porte sa méthode et sa règle") + "\n\nUne règle peut améliorer IF et dégrader LOF : un seuil n’est pas meilleur indépendamment de la distribution des scores.",
        19: tableau(resultats["mouse"]["classement"], [("méthode", "Méthode"), ("ap", "AP"), ("auc_roc", "AUC-ROC")]) + "\n\nCes indicateurs utilisent les scores continus. Déplacer uniquement le seuil ne change pas leur classement ; cela déplace le point de fonctionnement, sans modifier AP ou AUC-ROC. Les résultats concernent l’ensemble ajusté, avec dix anomalies seulement : une anomalie manquée enlève dix points de pourcentage de rappel.",
        20: "LOF en mode nouveauté apprend sur les 343 normaux réservés à l’apprentissage. Le test distinct contient 147 normaux et les dix anomalies. L’IQR vient des scores d’apprentissage, pas des scores du test. Construire cette référence normale utilise les étiquettes : on dispose donc de plus d’information que dans la détection d’outliers.\n\n" + tableau(resultats["mouse"]["nouveautes"], COL_MOUSE[1:] + [("ap", "AP test"), ("auc_roc", "AUC-ROC test")]) + "\n\n" + image("p1_nouveautes", "Référence normale et test réservé LOF") + "\n\n" + image("p1_matrices_nouveautes", "Matrices de test : LOF nouveautés, défaut et IQR") + "\n\n" + image("p1_courbes_nouveautes", "Courbes du test réservé Mouse"),
        21: "FPR = FP/(VN+FP), FNR = FN/(VP+FN), part d’alertes inutiles = FP/(VP+FP), erreur globale = (FP+FN)/N. Les effectifs de FP et FN sont publiés dans les tableaux Mouse précédents ; leurs dénominateurs ne sont pas interchangeables. Avec dix anomalies parmi 500 points, prédire partout normal donne peu d’erreurs globales mais manque toutes les anomalies. Les matrices figurées permettent de lire séparément les deux classes.",
        22: "Credit Card est dédoublonné sur les prédicteurs V1 à V28 et Amount avant la partition. Time est exclue des entrées mais gardée pour discuter la temporalité. Le fichier brut a 284 807 lignes et 492 fraudes ; la population dédoublonnée utilisée compte " + nombre(cc["informations"]["observations"]) + " transactions et " + nombre(cc["informations"]["positifs"]) + " fraudes. Les mêmes prédicteurs ne traversent donc pas apprentissage et test. La population évaluée diffère du fichier brut.\n\n" + partitions,
        23: "Amount est asymétrique : une transformation fixe log(1 + Amount) est appliquée, sans supprimer les montants élevés. La logistique et LOF sont standardisés ; SMOTE utilise également la standardisation avant ses voisins. Les arbres, IF et HBOS n’en ont pas besoin. Les moyennes, écarts-types et encodages s’apprennent uniquement sur l’apprentissage. Les variables ACP anonymisées ne retrouvent pas leur sens métier par cette transformation. Ni validation ni test ne sont rééchantillonnés.",
        24: GRILLE + "\n\nRésultats de validation des deux familles classiques (les lignes non conformes sont des compromis F1) :\n\n" + tableau([l for l in cc["validation"] if l["famille"] in {"Arbre de décision", "Régression logistique"}], COL_VAL) + "\n\nLa pondération modifie les coûts relatifs des classes. SMOTE interpole la minorité ; RUS supprime des observations de la majorité. Aucune correction n’est automatiquement préférable. L’AP maximale et le rappel maximal sous précision minimale ne sélectionnent pas nécessairement la même configuration.",
        25: "IF et LOF ajustent leurs estimateurs sans cible, sur un apprentissage contaminé. Le sous-échantillonnage stratifié de LOF utilise néanmoins la cible pour conserver la composition ; la sélection des réglages et seuils utilise les étiquettes de validation. Le protocole complet n’est donc pas purement non supervisé. LOF emploie `novelty=True` pour scorer validation et test, jamais pour remplacer les scores outliers de l’apprentissage.\n\n" + tableau([l for l in cc["validation"] if l["famille"] in {"Isolation Forest", "LOF"}], COL_VAL),
        26: "Le classement est résumé par AP et AUC-ROC. Une AUC-ROC élevée peut accompagner beaucoup de fausses alertes lorsque les normales dominent. Les maxima de F1 et de BA sont calculés sur validation, pas sur le test ; maximiser ces métriques n’impose pas un plancher de précision. La BA équilibre les rappels des classes, pas la proportion d’enquêtes utiles.\n\n" + tableau(retenues(cc), COL_VAL) + "\n\n" + tableau(cc["maxima_validation"], [("famille", "Famille"), ("critère", "Critère maximisé")] + COL_DECISION[2:8]) + "\n\n" + image("p2_courbes_val", "ROC et précision-rappel de validation Credit Card") + "\n\n" + image("p2_f1_ba_if", "F1 et BA de validation selon le seuil IF"),
        27: "Le choix de la comparaison étendue est « " + choix_cc["méthode"] + " », selon la règle annoncée sur validation. La logistique pondérée C=1 était le choix historique avant l’ajout des ensembles ; elle reste dans la comparaison, sans être présentée comme sélectionnée globale. Le test ne change pas ce choix.\n\n" + tableau(cc["comparaison_seuils_validation"], COL_DECISION) + "\n\nLe défaut d’un classifieur est 0,5. Pour les détecteurs, le seuil inclusif reproduit la frontière native stricte. Relever le seuil limite généralement les alertes au prix d’un rappel plus faible ; les valeurs exactes, non les arrondis imprimés, produisent les décisions.\n\n" + EXPLORATOIRE,
        28: "Les modèles ajustés et seuils de validation sont appliqués au test sans nouvel ajustement.\n\n" + test_cc + "\n\n" + tableau(cc["test"], [("méthode", "Configuration retenue"), ("prétraitement", "Prétraitement"), ("déséquilibre", "Déséquilibre"), ("n apprentissage fourni", "n apprentissage fourni"), ("temps app. (s)", "Apprentissage (s)"), ("temps score test (s)", "Score test (s)")]) + "\n\n" + image("p2_matrices_confusion", "Matrices test Credit Card, modèles légendés par panneau") + "\n\nLes temps viennent de cette exécution et ne sont pas des mesures répétées. Les coûts LOF ne sont pas à effectifs d’apprentissage égaux ; toutes les familles scorent le même test complet.",
        29: bilan_cc + "\n\nUne précision exigée sur validation n’est pas garantie au test. Les échecs restent visibles ; aucun seuil n’est retouché pour les effacer. Le choix historique logistique et le choix global étendu ne doivent pas être confondus.\n\n" + texte_variante(cc),
        30: erreurs(cc) + "\n\nUn faible FPR utilise toutes les normales au dénominateur ; il ne garantit pas une faible proportion d’alertes inutiles. Prédire toujours normale manquerait toutes les fraudes malgré un faible taux global d’erreur.",
        32: "HBOS évite les distances entre observations et peut utiliser tout l’apprentissage Credit Card, soit " + nombre(next(l["n_app"] for l in cc["validation"] if l["famille"] == "HBOS")) + " transactions. Ses contributions s’additionnent par variable. Les composantes ACP sont décorrélées, ce qui motive l’essai sans garantir l’indépendance. Le nombre de cases règle la finesse des distributions, alpha la pénalisation des cases vides. Trois nombres de cases sont comparés, 10, 50 et 200, avec alpha=0,1 fixé ; trop de cases peut rendre les histogrammes instables, trop peu masquer leurs queues.",
        33: "HBOS utilise les partitions communes et log(1 + Amount), sans standardisation ni correction du déséquilibre. Les histogrammes sont ajustés sans étiquettes ; les paramètres et seuils sont ensuite sélectionnés avec les étiquettes de validation. Les objectifs non atteints restent marqués ; leur seuil est un compromis de F1, pas un seuil conforme.\n\n" + tableau([l for l in cc["validation"] if l["famille"] == "HBOS"], COL_VAL),
        34: "Sur validation, les configurations retenues par famille sont :\n\n" + tableau(retenues(cc, NON_SUPERVISES), COL_VAL) + "\n\n" + image("p3_non_supervises", "IF, LOF et HBOS sur validation") + "\n\nAu test, sans modifier les seuils :\n\n" + tableau(non_sup, COL_TEST) + "\n\n" + image("p3_matrices_confusion", "Matrices test des trois détecteurs, légendées par modèle") + "\n\nHBOS permet une somme de contributions marginales lisibles, mais ignore les dépendances. Une combinaison rare de valeurs ordinaires peut lui échapper. Les profils moyens montrés dans le notebook sont descriptifs, pas une preuve de la cause des erreurs. Les temps et effectifs de chaque modèle sont dans la comparaison finale : ne pas transformer un temps ponctuel en rapport de vitesse universel.",
        35: erreurs(cc, {"HBOS"}) + "\n\nLe FNR mesure les fraudes manquées, la proportion d’alertes inutiles le coût des enquêtes : le FPR seul ne répond à aucune de ces deux questions.",
        36: "La cible positive est Machine failure=1. Les identifiants UDI et Product ID ainsi que TWF, HDF, PWF, OSF et RNF sont exclus des prédicteurs : ces modes révèlent directement la panne. Les entrées restantes sont Type, les deux températures, la vitesse de rotation, le couple et l’usure de l’outil. L’audit relève " + nombre(ai["informations"]["discordances"]) + " discordances entre la cible et le OU des modes. On conserve la cible fournie et toutes les lignes : reconstruire une cible ou supprimer ces cas modifierait la tâche. Les modes servent uniquement à l’audit descriptif.",
        37: "AI4I contient " + nombre(ai["informations"]["observations"]) + " observations synthétiques et " + nombre(ai["informations"]["positifs"]) + " pannes. La cible d’origine est conservée. Type est encodé en one-hot ; les variables numériques sont standardisées pour les modèles sensibles à l’échelle, pas pour les arbres ou IF. Les transformations sont ajustées sur l’apprentissage uniquement.\n\n" + tableau([l for l in resultats["partitions"] if l["jeu"] == "ai4i"], [("partition", "Partition"), ("observations", "Observations"), ("positifs", "Pannes")]),
        38: TRANSFERT.replace("{classique}", classique(ai).lower()) + "\n\nConfigurations effectivement réajustées :\n\n" + tableau(ai["validation"], COL_VAL),
        39: "Sur validation AI4I, le choix est « " + choix_ai["méthode"] + " ». La contrainte de 90 % est appliquée avant le test. Une contrainte satisfaite sur très peu d’alertes reste fragile.\n\n" + tableau(ai["validation"], COL_VAL) + "\n\n" + image("p4_ai4i", "Courbes de validation AI4I") + "\n\nRésultats finaux aux seuils conservés :\n\n" + tableau(ai["test"], COL_TEST) + "\n\n" + image("p4_matrices_confusion", "Matrices AI4I : chaque panneau indique son modèle") + "\n\n" + matrices(ai) + "\n\n" + bilan(ai, "pannes") + "\n\nLa référence classique (" + classique(ai).lower() + ") reste visible même si un modèle transféré plus performant est sélectionné. Peu d’interventions toutes correctes ne suffisent pas lorsque la plupart des pannes sont manquées.",
        40: "Les variables, interactions et distributions changent entre fraude et maintenance. Conserver une famille et ses hyperparamètres ne garantit pas le même classement des méthodes. Les mécanismes synthétiques d’AI4I peuvent limiter une frontière linéaire ; les seules métriques ne prouvent cependant pas la cause de ses erreurs. Le transfert inclut désormais le meilleur supervisé global et IF, avec " + classique(ai).lower() + " de référence : l’ancien contrôle isolé d’arbre ne sert plus de résultat complémentaire.\n\nL’AP dépend de la prévalence positive, différente entre les jeux. Les cibles et contraintes diffèrent aussi. On compare les méthodes au sein d’une même partition, sans attribuer leurs écarts entre jeux à une seule cause.",
        41: erreurs(ai) + "\n\nLa fiabilité des alertes et leur couverture sont distinctes. Un résultat de test inférieur à la contrainte reste rapporté, sans nouvelle sélection.",
        42: "La référence normale Mouse de 343 points est contaminée par remplacement, pas par ajout : sa taille reste fixe. Le test réel réservé ne reçoit aucun point injecté. Les points uniformes sont tirés dans la boîte englobante de la référence, élargie de 20 % de chaque côté ; certains peuvent tomber dans un nuage normal. Les taux nominaux ne sont donc pas une garantie de proportion d’anomalies réellement atypiques. IF et LOF gardent leurs hyperparamètres ; les graines déterminent les remplacements et l’aléa des modèles.\n\n" + tableau([{"taux": taux, "injectes": sorted({l["n injectées"] for l in resultats["robustesse"] if l["contamination"] == taux})[0], "graines": ", ".join(str(g) for g in sorted({l["graine"] for l in resultats["robustesse"] if l["contamination"] == taux}))} for taux in sorted({l["contamination"] for l in resultats["robustesse"]})], [("taux", "Taux nominal"), ("injectes", "Normaux remplacés"), ("graines", "Graines")]),
        43: "Les agrégats portent sur le même test réel Mouse ; l’écart-type est l’écart-type échantillonnal sur les trois graines, pas un intervalle de confiance.\n\n" + statistiques_robustesse(resultats) + "\n\n" + image("p5_contamination", "AP moyenne et écart-type, trois graines") + "\n\nUne amélioration sous ce générateur n’implique pas que la contamination soit bénéfique en général. Les points uniformes peuvent modifier les coupures d’IF, mais ce mécanisme n’est pas isolé expérimentalement. Le générateur, sa boîte et les dix anomalies de test limitent les conclusions.",
        44: "On explique IF retenu sur Credit Card, avec son modèle et son seuil de validation figés. Le fond contient 50 transactions d’apprentissage uniquement. Les cas sont choisis pour contraster deux fraudes détectées, deux fausses alertes et deux fraudes manquées, quand ces catégories sont disponibles ; les sorties du notebook signalent toute catégorie manquante. Les scores les plus élevés des deux premières catégories et les plus faibles des fraudes manquées sont privilégiés : ce n’est pas un échantillon représentatif.\n\n" + tableau([{"catégorie": k, "n": v} for k, v in resultats["shap"]["categories"].items()], [("catégorie", "Catégorie disponible sur test"), ("n", "Effectif")]),
        45: "TreeExplainer explique ici la longueur moyenne de chemin corrigée h(x), pas une probabilité ni directement le score d’anomalie. Le score orienté vaut s(x)=2^(−h(x)/c(ψ)). Une contribution SHAP négative raccourcit le chemin et pousse donc vers l’alerte ; une contribution positive allonge le chemin. On compare la base plus la somme des contributions à −c(ψ) log₂(s(x)). L’écart maximal observé sur les six cas vaut " + format(resultats["shap"]["ecart"], ".12g") + ". Les sorties par cas sont conservées dans le notebook exécuté.",
        46: image("p5_shap", "Contributions SHAP : six cas de test, rouge vers l’alerte") + "\n\nLes variables affichées dans chaque panneau sont celles dont les contributions absolues sont les plus fortes pour ce cas. Des contributions vers l’alerte peuvent apparaître dans une fraude comme dans une fausse alerte. La somme avec la valeur de base, pas une barre isolée, détermine la sortie expliquée. Les fraudes manquées choisies sont volontairement les moins suspectes pour IF ; elles ne représentent pas toutes les fraudes manquées. SHAP décrit le modèle relativement au fond, sans causalité, sans garantie de fraude et sans restituer le sens métier des composantes anonymisées.",
        47: "Le seuil change l’utilité des alertes sans changer le classement. Sur Credit Card, le choix étendu sur validation est « " + choix_cc["méthode"] + " » ; la logistique était le choix historique avant l’extension. Sur AI4I, le choix est « " + choix_ai["méthode"] + " ». Les résultats test ne servent pas à réécrire ces décisions. Les limites des jeux et le caractère exploratoire de l’extension empêchent de conclure à une supériorité générale.",
        49: "Aucune garantie de précision future ni supériorité universelle n’est établie. Mouse comporte dix anomalies ; les faibles effectifs positifs fragilisent les seuils. AI4I est synthétique, KDD est un corpus historique simulé et dédoublonné. Les découpages aléatoires ne mesurent ni le futur ni des attaques nouvelles. Trois graines ne couvrent pas toutes les contaminations ; six cas SHAP ne décrivent pas toute la population.\n\n" + EXPLORATOIRE,
        51: "Le notebook exécuté, `experiences.py`, `experiences_kdd.py`, `modeles_complementaires.py`, `utils.py`, `generer_notebook.py`, `generer_rendus.py`, `requirements.txt` et les données de `data/` permettent de reprendre le travail. `rendus/resultats.json` est la source des chiffres du compte rendu, du Q/R Markdown et HTML et de la synthèse. La synthèse PDF de deux pages reste le rendu court demandé ; ce Q/R détaillé ne la remplace pas. Conserver `rendus/figures/` avec les documents. Le bloc complémentaire non bonus est réalisé ; seul le bonus autoencodeur ne l’est pas.",
        52: SOURCES,
    }
    return r


def rubrique_kdd(resultats):
    kdd = resultats["kdd"]
    infos = kdd["informations"]
    def question(titre, corps):
        return f"<details>\n<summary>{titre}</summary>\n\n{corps}\n\n</details>"
    effectifs = [{"étape": cle, **infos[cle]} for cle in
                 ("effectifs_bruts", "effectifs_dedoublonnes", "effectifs_echantillon", "effectifs_apprentissage_effectif")]
    corps = "\n\n".join([
        question("Quels modèles complémentaires sont comparés et selon quelle grille ?", GRILLE + "\n\n" + EXPLORATOIRE + "\n\n" + tableau([l for l in resultats["creditcard"]["validation"] if l["famille"] in {"EasyEnsemble", "RUSBoost", "XGBoost"}], COL_VAL)),
        question("Quels sont les effectifs après rééchantillonnage sur Credit Card ?", tableau(resultats["creditcard"]["effectifs"], [("configuration", "Configuration"), ("avant", "Avant"), ("après rééchantillonnage externe", "Après rééchantillonnage externe"), ("positifs après", "Positifs après")]) + "\n\nLes effectifs fournis à EasyEnsemble et RUSBoost ne décrivent pas leurs tirages internes : ces méthodes rééchantillonnent pendant leur propre apprentissage. XGBoost pondère sans modifier le nombre de lignes."),
        question("6.a. Quel corpus KDDCup99 est utilisé et comment sont traités les doublons ?", "Le fichier local est le vrai corpus officiel à 10 %, réexporté en CSV gzip ASCII depuis `fetch_kddcup99(percent10=True)`, pas une donnée synthétique fabriquée pour le TP. Il est inclus pour une exécution hors connexion. L’étiquette positive est intrusion=1 pour toute attaque différente de normal.\n\n" + tableau(effectifs, [("étape", "Étape"), ("total", "Total"), ("normales", "Normales"), ("intrusions", "Intrusions"), ("proportion_intrusions", "Proportion d’intrusions")]) + "\n\n" + nombre(infos["conflits_binaires_groupes"]) + " groupe à cibles binaires contradictoires, soit " + nombre(infos["conflits_binaires_lignes"]) + " lignes, est exclu. Un représentant par vecteur prédictif univoque est ensuite gardé, avant le tirage et la partition. Les répétitions retirées changent fortement la prévalence ; la rareté statistique et l’étiquette d’intrusion ne sont pas synonymes.\n\nSHA-256 du fichier local fourni : `" + infos["sha256_fichier_fourni"] + "`."),
        question("6.b. Quels prétraitements, partitions et contrainte utilise-t-on ?", "Après dédoublonnage, un tirage stratifié fixe de 60 000 vecteurs précède le partage 60/20/20. Les sept familles partagent exactement le même sous-échantillon d’apprentissage stratifié de 30 000 lignes ; les validations et tests de 12 000 lignes sont inchangés. Trois variables catégorielles sont encodées en one-hot ; les numériques sont standardisées pour logistique et LOF. RandomOverSampler copie des lignes minoritaires réelles : les catégories restent valides, contrairement à une interpolation après one-hot. Les estimateurs IF et LOF n’utilisent pas y, mais leur sélection sur validation est supervisée.\n\nLa précision minimale de 0,80 est un choix pédagogique annoncé a priori, pas une exigence propre à KDD dans l’énoncé. La règle reste rappel maximal sous contrainte, sinon compromis F1.\n\n" + tableau(kdd["validation"], COL_VAL) + "\n\n" + image("p6_courbes_val", "Courbes de validation KDDCup99")),
        question("6.c. Quels résultats d’intrusion obtient-on au test ?", "Le choix de validation est « " + gagnant(kdd)["méthode"] + " ».\n\n" + tableau(kdd["test"], COL_TEST) + "\n\n" + tableau(kdd["test"], [("méthode", "Configuration"), ("n apprentissage fourni", "n apprentissage fourni"), ("temps app. (s)", "Apprentissage (s)"), ("temps score test (s)", "Score test (s)")]) + "\n\n" + image("p6_matrices_confusion", "Matrices test KDDCup99, modèle indiqué sur chaque panneau") + "\n\n" + bilan(kdd, "intrusions") + "\n\n" + erreurs(kdd)),
        question("6.d. Quelles limites empêchent de transposer ces résultats à un réseau actuel ?", "KDDCup99 décrit des connexions d’un environnement militaire simulé historique. Le fichier brut répète massivement certaines attaques, notamment smurf et neptune. Le dédoublonnage et le tirage modifient la population évaluée ; le test aléatoire ne mesure pas des attaques inédites ni une généralisation temporelle. Les sept familles ont le même apprentissage effectif, mais ces données et la contrainte pédagogique de précision ne constituent pas un cahier des charges de cybersécurité actuel."),
    ])
    return "\n\n## Bloc complémentaire et KDDCup99\n\n" + corps + "\n"


def generer_qr(resultats):
    chemin = RENDUS / "rendu_questions_reponses.md"
    texte = chemin.read_text(encoding="utf-8")
    # Cette rubrique est reconstruite à chaque passage, sans doublon.
    texte = texte.split("\n\n## Bloc complémentaire et KDDCup99", 1)[0]
    corps = reponses(resultats)
    index = 0
    def remplacer(match):
        nonlocal index
        numero = index
        index += 1
        if numero not in corps:
            return match[0]
        return match[1] + "\n\n" + corps[numero] + "\n\n</details>"
    texte = re.sub(r"(<details>\s*<summary>.*?</summary>).*?</details>", remplacer, texte, flags=re.S)
    if index != 53:
        raise ValueError(f"Structure du Q/R inattendue : {index} questions, 53 attendues avant extension")
    return texte.rstrip() + rubrique_kdd(resultats)


def generer_compte_rendu(resultats):
    cc, ai, kdd = (resultats[nom] for nom in ("creditcard", "ai4i", "kdd"))
    return "\n\n".join([
        "# Compte rendu du TP Détection d’anomalies\n\nM’BEY Max — M2 Data Science",
        "J’ai séparé les calculs et la présentation. Le notebook montre les résultats finaux et les figures ; les expériences et réglages sont dans les fichiers Python. Le Q/R garde les comparaisons de validation et les explications détaillées.",
        "## Mouse : classer ne suffit pas",
        "J’ai commencé par ajuster IF et LOF sans étiquettes sur Mouse, puis comparé les seuils par défaut, IQR et KMeans. Les étiquettes servent à évaluer, pas à régler ces deux règles. Le choix du seuil change fortement les alertes sans changer l’AP ou l’AUC-ROC.",
        tableau(resultats["mouse"]["seuils"], COL_MOUSE),
        image("p1_matrices_confusion", "Mouse : décisions selon le modèle et le seuil"),
        "Pour les nouveautés, j’ai constitué une référence normale et réservé un test distinct. Cette référence exige les étiquettes, donc la comparaison n’est pas faite à information identique.",
        tableau(resultats["mouse"]["nouveautes"], COL_MOUSE[1:]),
        "## Credit Card : choisir avant le test",
        "J’ai retiré les répétitions de prédicteurs avant le découpage pour éviter de tester une copie d’une transaction apprise. Amount est transformée par log(1 + Amount), puis les méthodes sensibles à l’échelle sont standardisées. Les transformations et rééchantillonnages concernent seulement l’apprentissage. Les mêmes validations et tests servent à toutes les familles.",
        GRILLE,
        "La sélection étendue sur validation retient « " + gagnant(cc)["méthode"] + " ». La logistique pondérée était le choix historique, avant les ensembles ; ses mauvais résultats de test restent visibles. Pour chaque configuration, je maximise le rappel sous précision ≥ 80 %, ou rapporte le meilleur compromis F1 si cette contrainte est inaccessible.",
        EXPLORATOIRE,
        tableau(cc["test"], COL_TEST),
        "Les objectifs de précision ne sont pas des garanties sur le test. Je conserve les seuils de validation, y compris quand les résultats passent sous la contrainte.",
        image("p2_matrices_confusion", "Credit Card : matrices finales des familles comparées"),
        "## HBOS : des raretés marginales",
        "HBOS additionne −log(hauteur normalisée + alpha) sur des histogrammes ajustés par variable. Les composantes d’ACP sont décorrélées, pas nécessairement indépendantes. La méthode est lisible mais ignore les dépendances : une combinaison rare de valeurs banales peut lui échapper. J’ai comparé trois nombres de cases sur validation, avec le même critère de choix que les autres méthodes.",
        tableau([l for l in cc["test"] if l["famille"] in NON_SUPERVISES], COL_TEST),
        "## AI4I : transférer les réglages, pas les modèles appris",
        "J’ai conservé Machine failure comme cible et retiré identifiants et modes de panne des entrées. L’audit relève " + nombre(ai["informations"]["discordances"]) + " discordances avec ces modes ; je les signale sans réétiqueter les lignes. AI4I est synthétique : les résultats ne décrivent pas directement une usine réelle.",
        TRANSFERT,
        "Le choix sur validation AI4I est « " + gagnant(ai)["méthode"] + " ». Les résultats de tous les modèles, y compris la référence historique, restent publiés :",
        tableau(ai["test"], COL_TEST),
        bilan(ai, "pannes"),
        "## Robustesse et SHAP",
        "J’ai remplacé des points de la référence normale Mouse par des points uniformes dans sa boîte élargie, en gardant taille d’apprentissage et test fixes. Trois graines permettent de voir une dispersion, pas de garantir une stabilité générale. Une amélioration avec cette contamination ne vaut pas pour tous les générateurs.",
        statistiques_robustesse(resultats),
        "SHAP explique la longueur moyenne de chemin corrigée d’IF : les contributions négatives poussent vers l’alerte. Le fond vient seulement de l’apprentissage. L’écart maximal de reconstruction observé vaut " + format(resultats["shap"]["ecart"], ".12g") + ". Les six cas sont choisis pour contraster détections, fausses alertes et fraudes manquées ; ce ne sont ni une importance globale ni une explication causale.",
        image("p5_shap", "SHAP : contributions des six transactions de test"),
        "## KDDCup99 : une comparaison d’intrusions historique",
        "J’ai utilisé le corpus officiel à 10 %, disponible localement, après exclusion du groupe à cible binaire contradictoire et dédoublonnage. Un tirage stratifié de 60 000 lignes précède la partition. Les sept familles partagent 30 000 lignes d’apprentissage effectif, et les mêmes validations et tests de 12 000 lignes. Intrusion reste la classe 1. La précision minimale de 80 % est un choix pédagogique, pas une exigence propre à ce jeu.",
        tableau(kdd["test"], COL_TEST),
        "Le choix de validation est « " + gagnant(kdd)["méthode"] + " ». Le corpus est ancien et simulé ; ses répétitions et les proportions modifiées par dédoublonnage empêchent d’assimiler ce test aléatoire à la détection d’attaques nouvelles sur un réseau actuel. Seul le bonus autoencodeur n’est pas réalisé.",
        "## Sources\n\n" + SOURCES,
    ]) + "\n"


def generer_synthese(resultats):
    cc, ai, kdd = (resultats[nom] for nom in ("creditcard", "ai4i", "kdd"))
    colonnes = [("famille", "Modèle"), ("ap", "AP"), ("precision", "Précision"),
                ("rappel", "Rappel"), ("fausses_alertes", "FP"), ("manquees", "FN")]
    nouveaute = resultats["mouse"]["nouveautes"][0]
    return "\n\n".join([
        "# Synthèse — Détection d’anomalies\n\nM’BEY Max — M2 Data Science",
        "## Protocole et données",
        "Classe positive = anomalie, fraude, panne ou intrusion ; score élevé = suspect. Partitions stratifiées 60/20/20, graine 42. Transformations et rééchantillonnages sur apprentissage seulement. La validation sélectionne configurations et seuils : rappel maximal sous précision ≥ 80 % (fraude), ≥ 90 % (maintenance), puis précision et AP ; à défaut, compromis au meilleur F1 signalé non conforme. Le test conserve ces choix. Le notebook présente les résultats ; calculs dans les modules Python, analyses de validation dans le Q/R.",
        "Credit Card : " + nombre(cc["informations"]["observations"]) + " transactions et " + nombre(cc["informations"]["positifs"]) + " fraudes après dédoublonnage des prédicteurs avant partage. Time est exclue ; Amount devient log(1 + Amount). Standardisation pour logistique, LOF et SMOTE. AI4I : " + nombre(ai["informations"]["observations"]) + " observations, " + nombre(ai["informations"]["positifs"]) + " pannes. Identifiants et modes de panne exclus ; cible d’origine conservée malgré " + nombre(ai["informations"]["discordances"]) + " discordances avec les modes. Type est encodé en one-hot.",
        "## Mouse, fraude et méthode recherchée",
        "Sur Mouse, IF et LOF classent les anomalies aux premiers rangs. Changer seulement le seuil ne change pas AP/ROC. IQR réduit les fausses alertes d’IF mais peut dégrader LOF ; aucune règle ne domine. En nouveautés, LOF apprend sur 343 normaux et évalue 157 points distincts : au défaut, " + nombre(nouveaute["alertes"]) + " alertes, " + nombre(nouveaute["fausses_alertes"]) + " fausses alertes, " + nombre(nouveaute["manquees"]) + " anomalie manquée. Cette référence normale nécessite les étiquettes.",
        "Grille classique : six essais par famille (profondeur 4/8 ou C=0,01/1 × aucun/poids, puis un SMOTE et un RUS au réglage fort). EasyEnsemble : 5/10 AdaBoost de 50 souches ; RUSBoost : 50/100 arbres profondeur 2 ; XGBoost : 100 arbres profondeur 3/5, taux 0,1, poids appris sur l’apprentissage. IF : quatre essais ; LOF : quatre voisinages sur 30 000 lignes. HBOS : trois essais.",
        tableau(cc["test"], colonnes),
        "Le choix étendu de validation est « " + gagnant(cc)["méthode"] + " ». La logistique était le choix historique avant les ensembles ; ses métriques restent visibles, sans la présenter comme gagnante globale. Les seuils suivent la règle de l’énoncé (précision ≥ 80 % sur validation) et ne sont jamais retouchés. " + phrase_variante(cc) + " " + EXPLORATOIRE,
        "HBOS somme −log(hauteur normalisée + alpha) sur des histogrammes marginaux ; 10/50/200 cases, alpha=0,1. Décorrélation des composantes ACP ne signifie pas indépendance. Le score est décomposable, mais une combinaison rare de valeurs marginalement ordinaires peut lui échapper. Les coûts mesurés sont ponctuels, à effectifs d’apprentissage différents pour LOF.",
        "## Maintenance et intrusions",
        "AI4I : transfert du meilleur supervisé global Credit Card et d’IF, avec " + classique(ai).lower() + " de référence. Hyperparamètres conservés, dont le poids positif XGBoost Credit Card ; modèles réajustés et seuils propres à AI4I. Choix de validation : " + gagnant(ai)["famille"] + ". Objectif de précision de 90 % au test : " + gagnant(ai)["objectif test"] + ". La référence (" + classique(ai).lower() + ") " + ("conserve un faible rappel." if next(l for l in ai["test"] if l["famille"] == classique(ai))["rappel"] < 0.2 else "est comparée dans le tableau."),
        tableau(ai["test"], colonnes),
        "KDDCup99 : vrai corpus officiel à 10 % local. " + nombre(kdd["informations"]["effectifs_bruts"]["total"]) + " lignes brutes ; " + nombre(kdd["informations"]["conflits_binaires_lignes"]) + " lignes à cible binaire contradictoire exclues, puis " + nombre(kdd["informations"]["effectifs_dedoublonnes"]["total"]) + " vecteurs uniques, dont " + nombre(kdd["informations"]["effectifs_dedoublonnes"]["intrusions"]) + " intrusions. Tirage stratifié 60 000 puis 60/20/20 ; apprentissage effectif commun de 30 000, validation/test 12 000 chacun. Intrusion=1 ; one-hot des catégories, standardisation selon la famille. ROS copie des lignes réelles pour conserver des catégories valides. Précision ≥ 80 % choisie à titre pédagogique. Choix validation : " + gagnant(kdd)["famille"] + ".",
        tableau(kdd["test"], colonnes),
        "## Robustesse, explications et limites",
        "Mouse : remplacement de 0/5/10 % de la référence normale par des points uniformes dans une boîte élargie, trois graines, taille et test fixes. Les moyennes et dispersions du Q/R dépendent de ce générateur ; une amélioration d’IF n’est pas une loi générale. SHAP explique le chemin corrigé d’IF, pas une probabilité : contribution négative vers l’alerte. Fond de 50 transactions d’apprentissage, six cas contrastés ; écart maximal de reconstruction " + format(resultats["shap"]["ecart"], ".3g") + ". Ni causalité ni représentativité globale.",
        "Dix anomalies Mouse et peu de positifs fragilisent les seuils. AI4I est synthétique ; KDD est historique et simulé, avec prévalence modifiée par dédoublonnage. Un partage aléatoire ne mesure ni le futur ni des attaques inédites. Les AP dépendent des prévalences et ne se comparent pas directement entre jeux. Régler IF/LOF avec les étiquettes de validation constitue une sélection supervisée. Bloc complémentaire réalisé ; seul le bonus autoencodeur ne l’est pas.",
        "## Sources",
        "Sujet 2026–2027 et cours de H. Elghazel ; Credit Card : [Kaggle/ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) ; AI4I : [UCI, DOI 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) ; [KDDCup99 officiel](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) ; Goldstein et Dengel (2012), HBOS, et [PyOD](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/hbos.html). Sources des ensembles et SHAP dans le Q/R.",
    ]) + "\n"


def page_html(texte, titre, style):
    # Les détails sont rendus bloc par bloc : le Markdown à l’intérieur d’un
    # <details> brut ne doit pas devenir du texte non interprété.
    markdown = mistune.create_markdown(escape=False, plugins=["table"])
    def details(match):
        return "<details open><summary>" + match[1] + "</summary>" + markdown(match[2]) + "</details>"
    texte = re.sub(r"<details>\s*<summary>(.*?)</summary>(.*?)</details>", details, texte, flags=re.S)
    contenu = markdown(texte)
    return ("<!doctype html>\n<html lang=\"fr\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{html.escape(titre)}</title>"
            f"<style>{style}</style></head><body>{contenu}</body></html>\n")


def main():
    resultats = json.loads((RENDUS / "resultats.json").read_text(encoding="utf-8"))
    notebook = json.loads((RACINE / "TP_detection_anomalies.ipynb").read_text(encoding="utf-8"))
    # Calculer les textes avant d’écrire : un JSON incomplet ne doit pas laisser
    # un mélange de documents anciens et nouveaux.
    qr = generer_qr(resultats)
    compte_rendu = generer_compte_rendu(resultats)
    synthese = generer_synthese(resultats)
    FIGURES.mkdir(parents=True, exist_ok=True)
    extraire_figures(notebook)
    for nom, texte in (("rendu_questions_reponses.md", qr), ("compte_rendu.md", compte_rendu), ("synthese.md", synthese)):
        (RENDUS / nom).write_text(texte, encoding="utf-8")
    (RENDUS / "rendu_questions_reponses.html").write_text(
        page_html(qr, "Détection d’anomalies — Questions et réponses", STYLE), encoding="utf-8")
    Path("/tmp/tp1-synthese.html").write_text(
        page_html(synthese, "Synthèse — Détection d’anomalies", PRINT_STYLE), encoding="utf-8")
    print("Q/R Markdown et HTML, compte rendu et synthèse régénérés sans entraînement.")
    print("HTML A4 à imprimer en PDF : /tmp/tp1-synthese.html")


if __name__ == "__main__":
    main()
