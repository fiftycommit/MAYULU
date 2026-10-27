"""Fonctions réutilisables du TP de détection d'anomalies :
préparation, apprentissage, calcul des scores, choix du seuil et évaluation."""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)


def arrondir(table, decimales=4):
    """
    Rôles : Arrondir un tableau pour l'affichage, en gardant 6 décimales pour les seuils.
    Saisies :
        - table : DataFrame
        - decimales : nombre de décimales des autres colonnes
    Résultat : DataFrame arrondi
    """
    return table.round({c: 6 if c == "seuil" else decimales for c in table.columns})


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

def charger_mouse(chemin="mouse.txt"):
    """
    Rôles : Charger Mouse et construire les étiquettes d'évaluation.
    Saisies :
        - chemin : chemin du fichier (fins de ligne « \\r »)
    Résultat : X (DataFrame x1, x2) et y (1 = anomalie : les 10 dernières lignes)
    """
    X = pd.read_csv(chemin, sep=" ", header=None, names=["x1", "x2"], lineterminator="\r").dropna()
    X = X.reset_index(drop=True)
    # Étiquettes construites avant tout mélange
    y = np.zeros(len(X), dtype=int)
    y[-10:] = 1
    return X, pd.Series(y, name="anomalie")


def charger_creditcard(chemin="creditcard.csv"):
    """
    Rôles : Charger Credit Card et retirer les transactions dupliquées.
    Saisies :
        - chemin : chemin du fichier
    Résultat : X (sans Time ni Class), y (1 = fraude), time (conservée pour l'analyse)
    """
    df = pd.read_csv(chemin)
    variables = [c for c in df.columns if c not in ("Time", "Class")]
    # Un doublon ne doit pas apparaître dans deux partitions : on garde une seule copie
    df = df.drop_duplicates(subset=variables).reset_index(drop=True)
    return df[variables], df["Class"], df["Time"]


def charger_ai4i(chemin="ai4i2020.csv"):
    """
    Rôles : Charger AI4I 2020 sans identifiants ni types de panne.
    Saisies :
        - chemin : chemin du fichier
    Résultat : X (Type + variables numériques) et y (Machine failure)
    """
    df = pd.read_csv(chemin)
    # Les colonnes TWF, HDF, ... décrivent la panne : fuite de la cible
    X = df.drop(columns=["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])
    return X, df["Machine failure"]


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

def decouper_donnees(X, y, graine=42):
    """
    Rôles : Découper en apprentissage / validation / test (60 / 20 / 20), stratifié.
    Saisies :
        - X, y : données et étiquettes
        - graine : graine aléatoire
    Résultat : dictionnaire {"app": (X, y), "val": (X, y), "test": (X, y)}
    """
    X_app, X_reste, y_app, y_reste = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=graine)
    X_val, X_test, y_val, y_test = train_test_split(
        X_reste, y_reste, test_size=0.5, stratify=y_reste, random_state=graine)
    return {"app": (X_app, y_app), "val": (X_val, y_val), "test": (X_test, y_test)}


def sous_echantillon_stratifie(X, y, n=30_000, graine=42):
    """
    Rôles : Tirer un sous-échantillon stratifié fixe (pour les méthodes coûteuses).
    Saisies :
        - X, y : données d'apprentissage
        - n : taille du sous-échantillon
        - graine : graine aléatoire
    Résultat : X et y sous-échantillonnés
    """
    if len(X) <= n:
        return X, y
    X_sous, _, y_sous, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=graine)
    return X_sous, y_sous


# ---------------------------------------------------------------------------
# Prétraitements
# ---------------------------------------------------------------------------

def log_montant(X):
    """
    Rôles : Compresser la queue de distribution de Amount par log(1 + x).
    Saisies :
        - X : DataFrame Credit Card
    Résultat : copie de X avec Amount transformée
    """
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"])
    return X


def pretraitement_creditcard(standardiser):
    """
    Rôles : Construire le prétraitement Credit Card.
    Saisies :
        - standardiser : True pour les méthodes sensibles à l'échelle (distances, logistique)
    Résultat : liste d'étapes de Pipeline
    """
    etapes: list = [("log_amount", FunctionTransformer(log_montant))]
    if standardiser:
        etapes.append(("standard", StandardScaler()))
    return etapes


def pretraitement_ai4i(standardiser):
    """
    Rôles : Construire le prétraitement AI4I (Type encodé, numériques éventuellement standardisées).
    Saisies :
        - standardiser : True pour standardiser les variables numériques
    Résultat : liste d'étapes de Pipeline
    """
    numeriques = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
                  "Torque [Nm]", "Tool wear [min]"]
    transfo_num = StandardScaler() if standardiser else "passthrough"
    colonnes = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Type"]),
        ("num", transfo_num, numeriques),
    ])
    return [("colonnes", colonnes)]


# ---------------------------------------------------------------------------
# Scores, apprentissage et temps de calcul
# ---------------------------------------------------------------------------

def dernier_modele(modele):
    """
    Rôles : Récupérer l'estimateur final (utile si le modèle est un Pipeline).
    Saisies :
        - modele : estimateur ou Pipeline
    Résultat : estimateur final
    """
    return modele[-1] if isinstance(modele, Pipeline) else modele


def score_anomalie(modele, X):
    """
    Rôles : Calculer un score orienté : plus il est élevé, plus l'observation est suspecte.
    Saisies :
        - modele : modèle entraîné (détecteur non supervisé ou classifieur)
        - X : données à scorer
    Résultat : tableau numpy de scores
    """
    final = dernier_modele(modele)
    # LOF classique (novelty=False) : seul le score des points d'apprentissage existe
    if getattr(final, "novelty", True) is False:
        return -final.negative_outlier_factor_
    # Classifieur : probabilité de la classe 1
    if hasattr(modele, "predict_proba"):
        return modele.predict_proba(X)[:, 1]
    # sklearn : plus faible = plus atypique, on inverse
    return -modele.score_samples(X)


def seuil_defaut(modele):
    """
    Rôles : Donner le seuil par défaut du modèle, exprimé sur le score orienté.
    Saisies :
        - modele : modèle entraîné
    Résultat : seuil (0.5 pour un classifieur, -offset_ pour IF / LOF)
    """
    final = dernier_modele(modele)
    if hasattr(final, "offset_"):
        # decision_function = score_samples - offset_ < 0  <=>  -score_samples > -offset_
        return -final.offset_
    return 0.5


def entrainer_et_scorer(modele, X_app, y_app, jeux):
    """
    Rôles : Entraîner un modèle puis scorer plusieurs jeux, en mesurant les temps.
    Saisies :
        - modele : estimateur ou Pipeline non entraîné
        - X_app, y_app : apprentissage (y_app ignoré par les détecteurs non supervisés)
        - jeux : dictionnaire {nom : X} des jeux à scorer
    Résultat : modèle entraîné, dictionnaire des scores, temps d'apprentissage, temps de score
    """
    debut = time.perf_counter()
    modele.fit(X_app, y_app)
    temps_app = time.perf_counter() - debut

    debut = time.perf_counter()
    scores = {nom: score_anomalie(modele, X) for nom, X in jeux.items()}
    temps_score = time.perf_counter() - debut
    return modele, scores, temps_app, temps_score


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluer(y, scores):
    """
    Rôles : Évaluer le classement produit par des scores continus.
    Saisies :
        - y : vraies étiquettes (1 = anomalie)
        - scores : scores orientés (élevé = suspect)
    Résultat : dictionnaire avec l'AP et l'AUC-ROC
    """
    return {
        "ap": average_precision_score(y, scores),
        "auc_roc": roc_auc_score(y, scores),
    }


def rapport_seuil(y, scores, seuil):
    """
    Rôles : Évaluer les décisions obtenues avec un seuil (alerte si score >= seuil).
    Saisies :
        - y : vraies étiquettes (1 = anomalie)
        - scores : scores orientés
        - seuil : seuil de décision
    Résultat : dictionnaire (matrice, précision, rappel, F1, balanced accuracy, alertes, FP, FN)
    """
    y_pred = (np.asarray(scores) >= seuil).astype(int)
    matrice = confusion_matrix(y, y_pred, labels=[0, 1])
    return {
        "matrice": matrice,
        "precision": precision_score(y, y_pred, zero_division=0),
        "rappel": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "alertes": int(y_pred.sum()),
        "fausses_alertes": int(matrice[0, 1]),
        "manquees": int(matrice[1, 0]),
    }


# ---------------------------------------------------------------------------
# Choix du seuil
# ---------------------------------------------------------------------------

def seuil_iqr(scores):
    """
    Rôles : Seuil non supervisé par la règle de l'écart interquartile.
    Saisies :
        - scores : scores orientés
    Résultat : seuil Q3 + 1.5 * IQR
    """
    q1, q3 = np.percentile(scores, [25, 75])
    return q3 + 1.5 * (q3 - q1)


def seuil_kmeans(scores, graine=42):
    """
    Rôles : Seuil non supervisé par clustering des scores en deux groupes (KMeans 1D).
    Saisies :
        - scores : scores orientés
        - graine : graine aléatoire
    Résultat : plus petit score du groupe au centre le plus élevé
    """
    s = np.asarray(scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init="auto", random_state=graine).fit(s)
    groupe_haut = np.argmax(kmeans.cluster_centers_.ravel())
    return s[kmeans.labels_ == groupe_haut].min()


def choisir_seuil_contraint(y, scores, precision_min):
    """
    Rôles : Choisir le seuil qui maximise le rappel sous la contrainte précision >= precision_min.
    Saisies :
        - y : étiquettes de validation
        - scores : scores orientés de validation
        - precision_min : précision minimale exigée
    Résultat : dictionnaire (seuil, précision, rappel, alertes, objectif atteint ou non).
               Si la contrainte est inaccessible, renvoie le meilleur compromis (F1 maximal).
    """
    precision, rappel, seuils = precision_recall_curve(y, scores)
    # Le dernier point (rappel 0, sans seuil) est retiré : on veut au moins une alerte
    precision, rappel = precision[:-1], rappel[:-1]
    valides = np.flatnonzero(precision >= precision_min)
    atteint = len(valides) > 0
    if atteint:
        # Meilleur rappel, puis meilleure précision
        i = valides[np.lexsort((-precision[valides], -rappel[valides]))[0]]
    else:
        f1 = 2 * precision * rappel / np.maximum(precision + rappel, 1e-12)
        i = np.argmax(f1)
    return {
        "seuil": seuils[i],
        "precision": precision[i],
        "rappel": rappel[i],
        "alertes": int((np.asarray(scores) >= seuils[i]).sum()),
        "atteint": atteint,
    }


def selectionner(resultats):
    """
    Rôles : Classer les configurations selon la règle imposée.
    Saisies :
        - resultats : DataFrame avec atteint, f1_val, rappel_val, precision_val, ap_val
    Résultat : DataFrame trié, meilleure configuration en première ligne
    """
    # Sous contrainte : rappel d'abord. Sinon : meilleur compromis F1.
    atteints = resultats[resultats["atteint"]].sort_values(
        ["rappel_val", "precision_val", "ap_val"], ascending=False, kind="stable")
    non_atteints = resultats[~resultats["atteint"]].sort_values(
        ["f1_val", "rappel_val", "precision_val", "ap_val"], ascending=False, kind="stable")
    return pd.concat([atteints, non_atteints])


def comparer_configurations(configurations, app, val, precision_min):
    """
    Rôles : Entraîner chaque configuration, la scorer sur la validation et y choisir son seuil.
    Saisies :
        - configurations : liste de dictionnaires (famille, nom, pretraitement, desequilibre, modele,
                           et éventuellement "app" pour un apprentissage propre, ex. sous-échantillon)
        - app : (X, y) d'apprentissage commun
        - val : (X, y) de validation
        - precision_min : contrainte de précision de l'objectif opérationnel
    Résultat : DataFrame des résultats de validation et dictionnaire {nom : modèle entraîné}
    """
    X_val, y_val = val
    lignes, modeles = [], {}
    for config in configurations:
        X_app, y_app = config.get("app", app)
        modele, scores, t_app, t_score = entrainer_et_scorer(config["modele"], X_app, y_app, {"val": X_val})
        seuil = choisir_seuil_contraint(y_val, scores["val"], precision_min)
        modeles[config["nom"]] = modele
        lignes.append({
            "famille": config["famille"], "nom": config["nom"],
            "pretraitement": config["pretraitement"], "desequilibre": config["desequilibre"],
            "n_app": len(X_app), **{f"{k}_val": v for k, v in evaluer(y_val, scores["val"]).items()},
            "seuil": seuil["seuil"], "precision_val": seuil["precision"], "rappel_val": seuil["rappel"],
            "f1_val": rapport_seuil(y_val, scores["val"], seuil["seuil"])["f1"],
            "alertes_val": seuil["alertes"], "atteint": seuil["atteint"],
            "temps_app_s": t_app, "temps_score_val_s": t_score,
        })
    return pd.DataFrame(lignes), modeles


def ligne_evaluation(y, scores, seuil):
    """
    Rôles : Regrouper l'évaluation du classement et celle des décisions au seuil donné.
    Saisies :
        - y : vraies étiquettes
        - scores : scores orientés
        - seuil : seuil figé
    Résultat : dictionnaire à plat (AP, AUC-ROC, précision, rappel, F1, BA, alertes, FP, FN)
    """
    rapport = rapport_seuil(y, scores, seuil)
    rapport.pop("matrice")
    return {**evaluer(y, scores), **rapport}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def tracer_courbes(y, dict_scores, titre=""):
    """
    Rôles : Tracer les courbes ROC et précision-rappel de plusieurs méthodes.
    Saisies :
        - y : vraies étiquettes
        - dict_scores : dictionnaire {nom de la méthode : scores}
        - titre : titre général
    Résultat : la figure matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for nom, scores in dict_scores.items():
        fpr, tpr, _ = roc_curve(y, scores)
        ax1.plot(fpr, tpr, label=f"{nom} (AUC = {roc_auc_score(y, scores):.3f})")
        precision, rappel, _ = precision_recall_curve(y, scores)
        ax2.plot(rappel, precision, label=f"{nom} (AP = {average_precision_score(y, scores):.3f})")

    ax1.plot([0, 1], [0, 1], "k--", label="Hasard")
    ax1.set(title="Courbe ROC", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
    # Classifieur au hasard : précision = proportion de positifs
    ax2.axhline(np.mean(y), color="k", linestyle="--", label="Hasard")
    ax2.set(title="Courbe précision-rappel", xlabel="Rappel", ylabel="Précision")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.suptitle(titre)
    fig.tight_layout()
    return fig


def metriques_par_seuil(y, scores):
    """
    Rôles : Calculer précision, rappel, F1 et balanced accuracy pour tous les seuils possibles.
    Saisies :
        - y : vraies étiquettes
        - scores : scores orientés (alerte si score >= seuil)
    Résultat : DataFrame indexé par le seuil (seuils produisant au moins une alerte)
    """
    # roc_curve donne, pour chaque seuil, les taux de vrais et de faux positifs
    fpr, tpr, seuils = roc_curve(y, scores, drop_intermediate=False)
    n_pos = np.sum(y)
    vp, fp = tpr * n_pos, fpr * (len(y) - n_pos)
    precision = np.divide(vp, vp + fp, out=np.zeros_like(vp), where=(vp + fp) > 0)
    f1 = np.divide(2 * precision * tpr, precision + tpr, out=np.zeros_like(vp), where=(precision + tpr) > 0)
    table = pd.DataFrame({"precision": precision, "rappel": tpr, "f1": f1,
                          "balanced_accuracy": (tpr + 1 - fpr) / 2, "alertes": vp + fp}, index=seuils)
    # Le premier seuil (infini) ne produit aucune alerte
    return table.iloc[1:]


def tracer_metriques_seuil(y, scores, titre="", seuils_marques=None):
    """
    Rôles : Tracer précision, rappel, F1 et balanced accuracy en fonction du seuil.
    Saisies :
        - y : vraies étiquettes
        - scores : scores orientés
        - titre : titre de la figure
        - seuils_marques : dictionnaire {nom : seuil} à repérer par une ligne verticale
    Résultat : la figure matplotlib
    """
    table = metriques_par_seuil(y, scores)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for cle, nom in [("precision", "Précision"), ("rappel", "Rappel"), ("f1", "F1"),
                     ("balanced_accuracy", "Balanced accuracy")]:
        ax.plot(table.index, table[cle], label=nom)
    for nom, t in (seuils_marques or {}).items():
        ax.axvline(t, linestyle="--", color="grey")
        ax.text(t, 0.02, nom, rotation=90, fontsize=8)
    ax.set(title=titre, xlabel="Seuil sur le score", ylabel="Valeur")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Partie 3 : HBOS (Histogram-Based Outlier Score)
# ---------------------------------------------------------------------------

class HBOS(BaseEstimator):
    """
    Rôles : Détecteur HBOS (Goldstein et Dengel, 2012). Chaque variable est résumée
            par un histogramme ; le score d'une observation est la somme, sur les
            variables, de -log(hauteur normalisée de sa case). Hypothèse : variables
            indépendantes (le score est un log de produit de densités marginales).
    Saisies :
        - n_bins : nombre de cases par histogramme (largeur fixe)
        - alpha : lissage ajouté aux hauteurs, évite log(0) pour les cases vides
        - contamination : proportion attendue d'anomalies (seuil par défaut seulement)
    """

    def __init__(self, n_bins=10, alpha=0.1, contamination=0.1):
        self.n_bins = n_bins
        self.alpha = alpha
        self.contamination = contamination

    def fit(self, X, y=None):
        """
        Rôles : Construire un histogramme par variable (y est ignoré).
        Saisies :
            - X : données d'apprentissage
        Résultat : le détecteur entraîné
        """
        X = np.asarray(X, dtype=float)
        self.bords_ = []
        self.hauteurs_ = []
        for j in range(X.shape[1]):
            hauteurs, bords = np.histogram(X[:, j], bins=self.n_bins)
            # Hauteurs normalisées : la case la plus remplie vaut 1
            self.hauteurs_.append(hauteurs / hauteurs.max())
            self.bords_.append(bords)
        # Même convention que sklearn : score_samples faible = atypique
        self.offset_ = np.quantile(self.score_samples(X), self.contamination)
        return self

    def contributions(self, X):
        """
        Rôles : Calculer la contribution de chaque variable au score HBOS.
        Saisies :
            - X : données à scorer
        Résultat : tableau (n, d) de -log(hauteur + alpha) ; élevé = valeur rare pour la variable
        """
        X = np.asarray(X, dtype=float)
        contrib = np.zeros(X.shape)
        for j, (bords, hauteurs) in enumerate(zip(self.bords_, self.hauteurs_)):
            case = np.searchsorted(bords, X[:, j], side="right") - 1
            # Le bord droit appartient à la dernière case, comme dans np.histogram
            case[X[:, j] == bords[-1]] = self.n_bins - 1
            # Valeur hors de la plage d'apprentissage : densité nulle
            hors_plage = (case < 0) | (case >= self.n_bins)
            h = np.where(hors_plage, 0.0, hauteurs[np.clip(case, 0, self.n_bins - 1)])
            contrib[:, j] = -np.log(h + self.alpha)
        return contrib

    def score_samples(self, X):
        """
        Rôles : Calculer le score HBOS au signe de sklearn (plus faible = plus atypique).
        Saisies :
            - X : données à scorer
        Résultat : tableau numpy de -HBOS
        """
        return -self.contributions(X).sum(axis=1)


# ---------------------------------------------------------------------------
# Partie 5 : anomalies synthétiques
# ---------------------------------------------------------------------------

def generer_anomalies_uniformes(X_ref, n, graine, marge=0.2):
    """
    Rôles : Générer des anomalies synthétiques uniformes dans la boîte englobante élargie.
    Saisies :
        - X_ref : données normales servant à définir la boîte
        - n : nombre d'anomalies
        - graine : graine aléatoire
        - marge : élargissement relatif de la boîte de chaque côté
    Résultat : tableau numpy (n, d)
    """
    X_ref = np.asarray(X_ref, dtype=float)
    mini, maxi = X_ref.min(axis=0), X_ref.max(axis=0)
    etendue = maxi - mini
    rng = np.random.default_rng(graine)
    return rng.uniform(mini - marge * etendue, maxi + marge * etendue, size=(n, X_ref.shape[1]))
