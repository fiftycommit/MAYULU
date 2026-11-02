"""Calculs et restitution du TP, séparés par expérience.

Les modèles et seuils sont choisis sur validation avant tout calcul de test.
Les fonctions d'affichage ne réentraînent aucun modèle et ne changent aucun seuil.
"""

import json
import platform
import time
from pathlib import Path

import imblearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import xgboost
from IPython.display import HTML, display
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

import utils
from modeles_complementaires import configurations_ensembles, configurations_supervisees

GRAINE = 42
RACINE = Path(__file__).resolve().parent
CLASSIQUES = ["Arbre de décision", "Régression logistique"]
NON_SUPERVISES = ["Isolation Forest", "LOF", "HBOS"]


def calculer_mouse(graine=GRAINE):
    """Comparer les outliers, puis préparer un test distinct de nouveautés."""
    X, y = utils.charger_mouse(RACINE / "data/mouse.txt")
    modeles = {"Isolation Forest": IsolationForest(random_state=graine),
               "LOF": LocalOutlierFactor(n_neighbors=20)}
    scores, defauts, temps, lignes, matrices = {}, {}, [], [], {}
    for nom, modele in modeles.items():
        debut = time.perf_counter()
        modele.fit(X)
        temps_app = time.perf_counter() - debut
        debut = time.perf_counter()
        scores[nom] = utils.score_anomalie(modele, X)
        temps.append({"méthode": nom, "temps app. (s)": temps_app,
                      "temps score (s)": time.perf_counter() - debut})
        defauts[nom] = utils.seuil_defaut(modele)
        for regle, seuil in [("défaut", defauts[nom]), ("IQR", utils.seuil_iqr(scores[nom])),
                             ("KMeans", utils.seuil_kmeans(scores[nom], graine))]:
            rapport = utils.rapport_seuil(y, scores[nom], seuil)
            matrices[(nom, regle)] = rapport.pop("matrice")
            lignes.append({"méthode": nom, "règle": regle, "seuil": seuil, **rapport})
    normaux = np.flatnonzero(y == 0)
    app, test_normaux = train_test_split(normaux, test_size=0.3, random_state=graine)
    test = np.r_[test_normaux, np.flatnonzero(y == 1)]
    X_app, X_test, y_test = X.iloc[app], X.iloc[test], y.iloc[test]
    lof, sorties, t_app, t_score = utils.entrainer_et_scorer(
        LocalOutlierFactor(n_neighbors=20, novelty=True), X_app, None, {"test": X_test})
    s_test = sorties["test"]
    seuils = {"défaut": utils.seuil_defaut(lof),
              "IQR (apprentissage)": utils.seuil_iqr(-lof.negative_outlier_factor_)}
    nouveautes, matrices_nov = [], {}
    for regle, seuil in seuils.items():
        rapport = utils.rapport_seuil(y_test, s_test, seuil)
        matrices_nov[regle] = rapport.pop("matrice")
        nouveautes.append({"règle": regle, "seuil": seuil, **rapport,
                          **utils.evaluer(y_test, s_test)})
    xx, yy = np.meshgrid(np.linspace(0, 1, 250), np.linspace(0.05, 1, 250))
    grille = pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})
    zz = utils.score_anomalie(lof, grille).reshape(xx.shape)
    return {"X": X, "y": y, "scores": scores, "defauts": defauts,
            "seuils": pd.DataFrame(lignes), "matrices": matrices, "temps": pd.DataFrame(temps),
            "classement": pd.DataFrame({n: utils.evaluer(y, s) for n, s in scores.items()}).T,
            "app": X_app, "test": (X_test, y_test), "modele_nouveautes": lof,
            "scores_nouveautes": s_test, "seuils_nouveautes": seuils,
            "nouveautes": pd.DataFrame(nouveautes), "matrices_nouveautes": matrices_nov,
            "temps_nouveautes": {"app": t_app, "score": t_score}, "grille": (xx, yy, zz)}


def choisir_configurations(parts, configurations, precision_min, marge=0.0):
    """Ajuster uniquement l'apprentissage et figer les choix sur validation.

    Les seuils sont cherchés avec precision_min + marge sur la validation ; l'objectif
    publié reste precision_min, évalué sur le test."""
    resultats, modeles = utils.comparer_configurations(
        configurations, parts["app"], parts["val"], precision_min + marge)
    retenues = utils.selectionner(resultats).groupby("famille", sort=False).head(1).set_index("famille")
    return {"parts": parts, "resultats_validation": resultats, "retenues": retenues,
            "modeles": modeles, "precision_min": precision_min, "marge_validation": marge}


def calculer_creditcard(graine=GRAINE):
    """Choisir les configurations des familles imposées, sans consulter le test."""
    X, y, temps_transactions = utils.charger_creditcard(RACINE / "data/creditcard.csv")
    parts = utils.decouper_donnees(X, y, graine)
    X_app, y_app = parts["app"]
    X_sous, y_sous = utils.sous_echantillon_stratifie(X_app, y_app, 30_000, graine)
    configs = configurations_supervisees(utils.pretraitement_creditcard, graine)
    configs += configurations_ensembles(utils.pretraitement_creditcard, y_app, graine)
    for arbres in [100, 300]:
        for taille in [256, 2048]:
            configs.append({"famille": "Isolation Forest",
                            "nom": f"Isolation Forest | n_estimators={arbres}, max_samples={taille}",
                            "pretraitement": "log_amount", "desequilibre": "aucun (non supervisé)",
                            "modele": Pipeline(utils.pretraitement_creditcard(False) + [("modele", IsolationForest(
                                n_estimators=arbres, max_samples=taille, random_state=graine, n_jobs=2))])})
    for voisins in [10, 20, 50, 100]:
        configs.append({"famille": "LOF", "nom": f"LOF | n_neighbors={voisins}",
                        "pretraitement": "log_amount + standard", "desequilibre": "aucun (non supervisé)",
                        "app": (X_sous, y_sous),
                        "modele": Pipeline(utils.pretraitement_creditcard(True) + [("modele", LocalOutlierFactor(
                            n_neighbors=voisins, novelty=True, n_jobs=2))])})
    for cases in [10, 50, 200]:
        configs.append({"famille": "HBOS", "nom": f"HBOS | n_bins={cases}, alpha=0.1",
                        "pretraitement": "log_amount", "desequilibre": "aucun (non supervisé)",
                        "modele": Pipeline(utils.pretraitement_creditcard(False) + [("modele", utils.HBOS(
                            n_bins=cases, alpha=0.1))])})
    experience = choisir_configurations(parts, configs, 0.80, marge=0.05)
    experience["informations"] = {"observations": len(X), "positifs": int(y.sum()),
                                  "app_lof": len(X_sous), "positifs_app_lof": int(y_sous.sum()),
                                  "time_min": float(temps_transactions.min()),
                                  "time_max": float(temps_transactions.max())}
    # Effectifs réels des pipelines avant/après rééchantillonnage classique.
    effectifs = []
    for config in configs:
        entree, cible = config.get("app", parts["app"])
        pipeline = experience["modeles"][config["nom"]]
        apres, positifs = len(entree), int(cible.sum())
        for nom, etape in pipeline.steps[:-1]:
            if hasattr(etape, "sample_indices_"):
                apres = len(etape.sample_indices_)
                positifs = int(cible.iloc[etape.sample_indices_].sum())
            elif hasattr(etape, "sampling_strategy_"):
                apres += int(sum(etape.sampling_strategy_.values()))
                positifs += int(etape.sampling_strategy_.get(1, 0))
        effectifs.append({"configuration": config["nom"], "avant": len(entree),
                         "après rééchantillonnage externe": apres, "positifs après": positifs})
    experience["effectifs"] = pd.DataFrame(effectifs)
    return experience


def calculer_ai4i(creditcard, graine=GRAINE):
    """Transférer le meilleur supervisé et IF ; garder le classique de référence."""
    X, y = utils.charger_ai4i(RACINE / "data/ai4i2020.csv")
    parts = utils.decouper_donnees(X, y, graine)
    retenues_cc = creditcard["retenues"]
    supervises = retenues_cc[~retenues_cc.index.isin(NON_SUPERVISES)]
    classique = retenues_cc[retenues_cc.index.isin(CLASSIQUES)].iloc[0]
    choix = [supervises.iloc[0], retenues_cc.loc["Isolation Forest"]]
    if classique.name != supervises.index[0]:
        choix.append(classique)
    configurations = []
    for ligne in choix:
        original = creditcard["modeles"][ligne["nom"]]
        standardiser = "standard" in original.named_steps
        etapes = utils.pretraitement_ai4i(standardiser)
        etapes += [(nom, clone(etape)) for nom, etape in original.steps
                   if nom not in ("log_amount", "standard")]
        configurations.append({"famille": ligne.name, "nom": ligne["nom"] + " (AI4I)",
                               "pretraitement": "one-hot Type" + (" + standard" if standardiser else ""),
                               "desequilibre": ligne["desequilibre"], "modele": Pipeline(etapes)})
    experience = choisir_configurations(parts, configurations, 0.90)
    brut = pd.read_csv(RACINE / "data/ai4i2020.csv")
    modes = brut[["TWF", "HDF", "PWF", "OSF", "RNF"]].any(axis=1).astype(int)
    experience["informations"] = {"observations": len(X), "positifs": int(y.sum()),
                                  "discordances": int(modes.ne(y).sum())}
    return experience


def evaluer_test(experience):
    """Évaluer les modèles et seuils déjà figés ; ne fait aucune sélection."""
    X_test, y_test = experience["parts"]["test"]
    lignes, matrices, scores = [], {}, {}
    for famille, ligne in experience["retenues"].iterrows():
        debut = time.perf_counter()
        s = utils.score_anomalie(experience["modeles"][ligne["nom"]], X_test)
        temps_score = time.perf_counter() - debut
        rapport = utils.rapport_seuil(y_test, s, ligne["seuil"])
        matrices[famille] = rapport.pop("matrice")
        scores[famille] = s
        lignes.append({"famille": famille, "méthode": ligne["nom"],
                       "prétraitement": ligne["pretraitement"], "déséquilibre": ligne["desequilibre"],
                       "n apprentissage fourni": int(ligne["n_app"]), "seuil": ligne["seuil"],
                       **utils.evaluer(y_test, s), **rapport,
                       "objectif validation": "atteint" if ligne["atteint"] else "non atteint",
                       "objectif test": "atteint" if rapport["alertes"] and rapport["precision"] >= experience["precision_min"] else "non atteint",
                       "temps app. (s)": ligne["temps_app_s"], "temps score test (s)": temps_score,
                       "sélectionnée sur validation": famille == experience["retenues"].index[0]})
    experience.update(test=pd.DataFrame(lignes), matrices=matrices, scores_test=scores)


def calculer_robustesse(mouse):
    """Remplacer des points de référence, sans toucher au test réel réservé."""
    X_ref = mouse["app"].to_numpy()
    X_test, y_test = mouse["test"]
    lignes = []
    for taux in [0.0, 0.05, 0.10]:
        for graine in [0, 1, 2]:
            rng = np.random.default_rng(graine)
            n = int(round(taux * len(X_ref)))
            X_cont = X_ref.copy()
            indices = rng.choice(len(X_ref), size=n, replace=False)
            X_cont[indices] = utils.generer_anomalies_uniformes(X_ref, n, graine)
            modeles = {"Isolation Forest": IsolationForest(random_state=graine),
                       "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True)}
            for nom, modele in modeles.items():
                _, sorties, t_app, t_score = utils.entrainer_et_scorer(
                    modele, X_cont, None, {"test": X_test.to_numpy()})
                lignes.append({"méthode": nom, "contamination": taux, "graine": graine,
                               "n injectées": n, "temps app. (s)": t_app, "temps score (s)": t_score,
                               **utils.evaluer(y_test, sorties["test"])})
    return pd.DataFrame(lignes)


def calculer_shap(creditcard, graine=GRAINE):
    """Expliquer six cas de test avec un fond issu de l'apprentissage seul."""
    ligne = creditcard["retenues"].loc["Isolation Forest"]
    pipeline = creditcard["modeles"][ligne["nom"]]
    X_test, y_test = creditcard["parts"]["test"]
    scores = creditcard["scores_test"]["Isolation Forest"]
    alertes = scores >= ligne["seuil"]
    categories = {"fraude détectée": np.flatnonzero((y_test == 1) & alertes),
                  "fausse alerte": np.flatnonzero((y_test == 0) & alertes),
                  "fraude manquée": np.flatnonzero((y_test == 1) & ~alertes)}
    choix, avertissements = [], []
    for categorie, indices in categories.items():
        ordre = indices[np.argsort(scores[indices] if categorie == "fraude manquée" else -scores[indices])]
        choix.extend((categorie, int(i)) for i in ordre[:2])
        if len(ordre) < 2:
            avertissements.append(f"{categorie} : seulement {len(ordre)} cas disponible(s).")
    utilises = {i for _, i in choix}
    for i in np.argsort(-scores):
        if len(choix) >= 6:
            break
        if int(i) not in utilises:
            choix.append(("complément", int(i)))
            utilises.add(int(i))
    fond = pipeline[:-1].transform(creditcard["parts"]["app"][0].sample(50, random_state=graine))
    explicateur = shap.TreeExplainer(pipeline[-1], data=fond)
    X_expl = pipeline[:-1].transform(X_test.iloc[[i for _, i in choix]])
    valeurs = explicateur.shap_values(X_expl)
    taille = pipeline[-1].max_samples_
    c_psi = 2 * (np.log(taille - 1) + np.euler_gamma) - 2 * (taille - 1) / taille
    profondeur = -c_psi * np.log2(-pipeline[-1].score_samples(X_expl))
    base = float(explicateur.expected_value)
    ecart = float(np.abs(base + valeurs.sum(axis=1) - profondeur).max())
    compteurs, noms = {}, []
    for categorie, _ in choix:
        compteurs[categorie] = compteurs.get(categorie, 0) + 1
        noms.append(f"{categorie} #{compteurs[categorie]}")
    table = pd.DataFrame(valeurs, columns=X_test.columns, index=noms)
    return {"contributions": table, "profondeur": profondeur, "scores": scores[[i for _, i in choix]],
            "base": base, "ecart": ecart, "categories": {c: len(i) for c, i in categories.items()},
            "avertissements": avertissements}


def calculer_resultats():
    """Calculer les expériences ; tous les choix précèdent l'évaluation des tests."""
    from experiences_kdd import calculer_kdd

    mouse = calculer_mouse()
    creditcard = calculer_creditcard()
    ai4i = calculer_ai4i(creditcard)
    kdd = calculer_kdd()
    kdd.setdefault("precision_min", 0.80)
    for experience in [creditcard, ai4i, kdd]:
        evaluer_test(experience)
    return {"mouse": mouse, "creditcard": creditcard, "ai4i": ai4i, "kdd": kdd,
            "robustesse": calculer_robustesse(mouse), "shap": calculer_shap(creditcard)}


def tableau_partitions(resultats):
    lignes = []
    for nom in ["creditcard", "ai4i", "kdd"]:
        for partie, (_, y) in resultats[nom]["parts"].items():
            lignes.append({"jeu": nom, "partition": partie, "observations": len(y),
                           "positifs": int(y.sum()), "proportion positive": y.mean()})
    return pd.DataFrame(lignes)


def _afficher_figure(figure):
    display(figure)
    plt.close(figure)


def _afficher_matrices(matrices, titres, labels=("Normal", "Anomalie")):
    colonnes = min(3, len(matrices))
    lignes = (len(matrices) + colonnes - 1) // colonnes
    fig, axes = plt.subplots(lignes, colonnes, figsize=(5 * colonnes, 4 * lignes), squeeze=False)
    for ax, matrice, titre in zip(axes.ravel(), matrices, titres):
        ConfusionMatrixDisplay(matrice, display_labels=list(labels)).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(titre, fontsize=10)
    for ax in axes.ravel()[len(matrices):]:
        ax.set_visible(False)
    fig.tight_layout()
    _afficher_figure(fig)


def afficher_protocole(resultats):
    versions = {"Python": platform.python_version()}
    versions.update({m.__name__: m.__version__ for m in [np, pd, matplotlib, sns, sklearn, imblearn, shap, xgboost]})
    display(pd.DataFrame.from_dict(versions, orient="index", columns=["version"]))
    display(tableau_partitions(resultats).round(5))
    display(pd.DataFrame([
        {"famille": "Arbre / IF / HBOS / ensembles", "Credit Card": "log(1 + Amount)", "KDDCup99": "one-hot catégories"},
        {"famille": "Logistique / LOF", "Credit Card": "log(1 + Amount), standardisation", "KDDCup99": "one-hot et standardisation numérique"},
        {"famille": "Sur-échantillonnage", "Credit Card": "SMOTE après standardisation", "KDDCup99": "RandomOverSampler : catégories préservées"},
    ]))


def afficher_mouse(resultats):
    m = resultats["mouse"]
    X, y = m["X"], m["y"]
    display(pd.DataFrame({"observations": [len(X)], "variables": [X.shape[1]],
                          "valeurs manquantes": [int(X.isna().sum().sum())], "anomalies": [int(y.sum())]}))
    display(m["temps"].round(5))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(X.x1, X.x2, c=np.where(y == 1, "red", "tab:blue"), s=12)
    ax.set(title="Mouse : données (rouge : anomalies de référence)", aspect="equal")
    _afficher_figure(fig)
    for nom, scores in m["scores"].items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        detectes = scores >= m["defauts"][nom]
        axes[0].scatter(X.x1, X.x2, c=np.where(detectes, "red", "lightgrey"), s=12)
        axes[0].scatter(X.x1[y == 1], X.x2[y == 1], facecolors="none", edgecolors="k", s=65)
        axes[0].set(title=f"{nom} : {detectes.sum()} alertes (défaut)", aspect="equal")
        image = axes[1].scatter(X.x1, X.x2, c=scores, cmap="viridis", s=12)
        fig.colorbar(image, ax=axes[1], label="score élevé = suspect")
        axes[1].set(title=f"{nom} : scores", aspect="equal")
        sns.histplot(scores, bins=50, ax=axes[2])
        axes[2].axvline(m["defauts"][nom], color="red", linestyle="--", label="seuil par défaut")
        axes[2].set(title=f"{nom} : histogramme des scores", xlabel="score")
        axes[2].legend()
        fig.tight_layout()
        _afficher_figure(fig)
    display(utils.arrondir(m["seuils"]))
    titres = [f"{nom} — {regle}" for nom, regle in m["matrices"]]
    _afficher_matrices(list(m["matrices"].values()), titres)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (_, ligne) in zip(axes.ravel(), m["seuils"].iterrows()):
        detectes = m["scores"][ligne["méthode"]] >= ligne["seuil"]
        ax.scatter(X.x1, X.x2, c=np.where(detectes, "red", "lightgrey"), s=10)
        ax.scatter(X.x1[y == 1], X.x2[y == 1], facecolors="none", edgecolors="k", s=65)
        ax.set(title=f"{ligne['méthode']} — {ligne['règle']} : F1 {ligne['f1']:.3f}", aspect="equal")
    fig.tight_layout()
    _afficher_figure(fig)
    display(m["classement"].round(4))
    _afficher_figure(utils.tracer_courbes(y, m["scores"], "Mouse : classement dans l’échantillon observé"))


def afficher_nouveautes(resultats):
    m = resultats["mouse"]
    X_test, y_test = m["test"]
    display(utils.arrondir(m["nouveautes"]))
    _afficher_matrices(list(m["matrices_nouveautes"].values()),
                       [f"LOF nouveautés — {regle}" for regle in m["matrices_nouveautes"]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(m["app"].x1, m["app"].x2, s=10, label="apprentissage normal (343)")
    axes[0].scatter(X_test.x1, X_test.x2, c=np.where(y_test == 1, "red", "orange"), s=18,
                    marker="^", label="test réservé (157)")
    axes[0].set(title="Mouse : apprentissage et test distincts", aspect="equal")
    axes[0].legend(fontsize=8)
    xx, yy, zz = m["grille"]
    axes[1].contourf(xx, yy, np.log(zz), levels=20, cmap="Blues_r", alpha=0.6)
    axes[1].contour(xx, yy, zz, levels=[m["seuils_nouveautes"]["défaut"]], colors="k")
    detectes = m["scores_nouveautes"] >= m["seuils_nouveautes"]["défaut"]
    axes[1].scatter(X_test.x1, X_test.x2, c=np.where(detectes, "red", "green"), s=18)
    axes[1].scatter(X_test.x1[y_test == 1], X_test.x2[y_test == 1], facecolors="none", edgecolors="k", s=80)
    axes[1].set(title="LOF : décisions sur le test (seuil par défaut)", aspect="equal")
    fig.tight_layout()
    _afficher_figure(fig)
    _afficher_figure(utils.tracer_courbes(y_test, {"LOF nouveautés": m["scores_nouveautes"]}, "Mouse : test réservé"))


def afficher_comparaison(resultats, jeu, familles=None):
    """Restituer seulement le test ; la colonne de sélection vient de validation."""
    e = resultats[jeu]
    table = e["test"]
    if familles is not None:
        table = table[table.famille.isin(familles)]
    # Séparer identité et métriques évite les tableaux illisibles de 20 colonnes.
    identite = table[["famille", "méthode", "prétraitement", "déséquilibre", "n apprentissage fourni",
                      "sélectionnée sur validation"]].copy()
    identite["sélectionnée sur validation"] = identite["sélectionnée sur validation"].map({True: "oui", False: "non"})
    with pd.option_context("display.max_colwidth", None):
        display(identite)
        metriques = utils.arrondir(table.drop(columns=["méthode", "prétraitement", "déséquilibre",
                                                       "n apprentissage fourni", "sélectionnée sur validation"]))
        display(metriques.rename(columns={"ap": "AP", "auc_roc": "AUC-ROC", "precision": "précision",
                                         "f1": "F1", "balanced_accuracy": "balanced accuracy",
                                         "fausses_alertes": "fausses alertes", "manquees": "manqués"}))
    titres = [f"{famille} — {jeu}" for famille in table.famille]
    _afficher_matrices([e["matrices"][famille] for famille in table.famille], titres)
    scores = {famille: e["scores_test"][famille] for famille in table.famille}
    _afficher_figure(utils.tracer_courbes(e["parts"]["test"][1], scores, f"{jeu} : test, modèles et seuils figés"))


def afficher_hbos(resultats):
    afficher_comparaison(resultats, "creditcard", NON_SUPERVISES)
    e = resultats["creditcard"]
    pipeline = e["modeles"][e["retenues"].loc["HBOS", "nom"]]
    X, y = e["parts"]["test"]
    contributions = pd.DataFrame(pipeline[-1].contributions(pipeline[:-1].transform(X)), columns=X.columns, index=X.index)
    alertes = e["scores_test"]["HBOS"] >= e["retenues"].loc["HBOS", "seuil"]
    groupes = {"fraudes détectées": (y == 1) & alertes, "fraudes manquées": (y == 1) & ~alertes,
               "transactions normales": y == 0}
    profils = pd.DataFrame({nom: contributions[masque].mean() for nom, masque in groupes.items()})
    display(profils.sort_values("fraudes détectées", ascending=False).head(8).round(3))


def afficher_robustesse(resultats):
    table = resultats["robustesse"]
    stats = table.groupby(["méthode", "contamination"])[["ap", "auc_roc"]].agg(["mean", "std"])
    display(stats.round(4))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nom, groupe in table.groupby("méthode"):
        resume = groupe.groupby("contamination").ap.agg(["mean", "std"])
        ax.errorbar(resume.index * 100, resume["mean"], yerr=resume["std"], marker="o", capsize=4, label=nom)
    ax.set(title="Mouse : robustesse sur le test réel inchangé", xlabel="points synthétiques injectés (%)",
           ylabel="AP : moyenne ± écart-type, trois graines")
    ax.legend()
    fig.tight_layout()
    _afficher_figure(fig)


def afficher_shap(resultats):
    s = resultats["shap"]
    display(pd.DataFrame.from_dict(s["categories"], orient="index", columns=["cas disponibles sur test"]))
    for texte in s["avertissements"]:
        display(HTML(f"<p>{texte}</p>"))
    print(f"Écart d’additivité SHAP / profondeur : {s['ecart']:.3g}")
    display(pd.DataFrame({"h(x)": s["profondeur"], "score orienté": s["scores"]}, index=s["contributions"].index).round(3))
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for i, (ax, (nom, ligne)) in enumerate(zip(axes.ravel(), s["contributions"].iterrows())):
        top = ligne.reindex(ligne.abs().sort_values(ascending=False).index[:8])[::-1]
        ax.barh(top.index, top, color=np.where(top < 0, "red", "tab:blue"))
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_title(f"{nom} : h={s['profondeur'][i]:.2f}, base={s['base']:.2f}", fontsize=10)
        ax.set_xlabel("SHAP sur la profondeur (rouge : vers l’alerte)")
    for ax in axes.ravel()[len(s["contributions"]):]:
        ax.set_visible(False)
    fig.tight_layout()
    _afficher_figure(fig)


def afficher_resultats(resultats, partie):
    """Appeler une restitution nommée ; aucun calcul d'apprentissage ici."""
    affichages = {"protocole": afficher_protocole, "mouse": afficher_mouse,
                  "nouveautes": afficher_nouveautes, "hbos": afficher_hbos,
                  "robustesse": afficher_robustesse, "shap": afficher_shap}
    if partie in affichages:
        affichages[partie](resultats)
    elif partie in ("creditcard", "ai4i", "kdd"):
        afficher_comparaison(resultats, partie)
    else:
        raise ValueError(f"Partie inconnue : {partie}")


def exporter_resultats(resultats, chemin=RACINE / "rendus/resultats.json"):
    """Exporter les nombres calculés, source commune des documents de rendu."""
    def lignes(table):
        return json.loads(table.to_json(orient="records", double_precision=12))

    export = {"partitions": lignes(tableau_partitions(resultats)),
              "mouse": {"seuils": lignes(resultats["mouse"]["seuils"]),
                        "classement": lignes(resultats["mouse"]["classement"].rename_axis("méthode").reset_index()),
                        "nouveautes": lignes(resultats["mouse"]["nouveautes"])},
              "robustesse": lignes(resultats["robustesse"]),
              "shap": {"ecart": resultats["shap"]["ecart"], "categories": resultats["shap"]["categories"]}}
    for jeu in ["creditcard", "ai4i", "kdd"]:
        e = resultats[jeu]
        export[jeu] = {"validation": lignes(e["resultats_validation"]), "test": lignes(e["test"]),
                       "informations": e["informations"],
                       "matrices": {n: m.tolist() for n, m in e["matrices"].items()}}
        comparaisons, maxima = [], []
        X_val, y_val = e["parts"]["val"]
        for famille, ligne in e["retenues"].iterrows():
            modele = e["modeles"][ligne["nom"]]
            scores = utils.score_anomalie(modele, X_val)
            for regle, seuil in [("défaut", utils.seuil_defaut(modele)), ("retenu", ligne["seuil"])]:
                rapport = utils.rapport_seuil(y_val, scores, seuil)
                rapport.pop("matrice")
                comparaisons.append({"famille": famille, "règle": regle, "seuil": seuil, **rapport})
            metriques = utils.metriques_par_seuil(y_val, scores)
            for critere in ["f1", "balanced_accuracy"]:
                seuil = metriques[critere].idxmax()
                maxima.append({"famille": famille, "critère": critere, "seuil": seuil,
                               **metriques.loc[seuil].to_dict()})
        export[jeu]["comparaison_seuils_validation"] = lignes(pd.DataFrame(comparaisons))
        export[jeu]["maxima_validation"] = lignes(pd.DataFrame(maxima))
    export["creditcard"]["effectifs"] = lignes(resultats["creditcard"]["effectifs"])
    Path(chemin).write_text(json.dumps(export, ensure_ascii=False, indent=2, default=lambda x: x.item()), encoding="utf-8")
    exporter_figures_validation(resultats, Path(chemin).parent / "figures")


def exporter_figures_validation(resultats, dossier):
    """Conserver les analyses de validation dans les documents, pas dans le notebook."""
    dossier = Path(dossier)
    dossier.mkdir(exist_ok=True)
    noms = {"creditcard": "p2_courbes_val.png", "ai4i": "p4_ai4i.png", "kdd": "p6_courbes_val.png"}
    for jeu, fichier in noms.items():
        e = resultats[jeu]
        X_val, y_val = e["parts"]["val"]
        scores = {famille: utils.score_anomalie(e["modeles"][ligne["nom"]], X_val)
                  for famille, ligne in e["retenues"].iterrows()}
        figure = utils.tracer_courbes(y_val, scores, f"{jeu} : validation")
        figure.savefig(dossier / fichier, dpi=130, bbox_inches="tight")
        plt.close(figure)
        if jeu == "creditcard":
            figure = utils.tracer_courbes(y_val, {n: scores[n] for n in NON_SUPERVISES},
                                          "IF, LOF et HBOS : validation")
            figure.savefig(dossier / "p3_non_supervises.png", dpi=130, bbox_inches="tight")
            plt.close(figure)
            seuil = e["retenues"].loc["Isolation Forest", "seuil"]
            modele = e["modeles"][e["retenues"].loc["Isolation Forest", "nom"]]
            figure = utils.tracer_metriques_seuil(
                y_val, scores["Isolation Forest"], "Isolation Forest : validation",
                {"retenu": seuil, "défaut": utils.seuil_defaut(modele)})
            figure.savefig(dossier / "p2_f1_ba_if.png", dpi=130, bbox_inches="tight")
            plt.close(figure)
