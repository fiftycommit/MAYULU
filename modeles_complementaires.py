"""Configurations supervisées fixées avant consultation des jeux de test.

Chaque prétraitement et chaque rééchantillonnage appartient au Pipeline : seuls
les exemples d'apprentissage servent à ajuster les transformations et à produire
les exemples synthétiques. La sélection des modèles et des seuils reste assurée
par les fonctions communes de utils, sur la validation uniquement.
"""

import numpy as np
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def _configuration(famille, nom, pretraitement, standardiser, desequilibre,
                   modele, reechantillonneur=None):
    etapes = list(pretraitement(standardiser))
    description = " + ".join(nom_etape for nom_etape, _ in etapes) or "aucun"
    if reechantillonneur is not None:
        etapes.append(reechantillonneur)
    etapes.append(("modele", modele))
    return {
        "famille": famille,
        "nom": nom,
        "pretraitement": description,
        "desequilibre": desequilibre,
        "modele": Pipeline(etapes),
    }


def configurations_supervisees(pretraitement, graine=42, sampling_strategy=0.1,
                               sur_echantillonnage="SMOTE"):
    """Construire six configurations par famille classique, sans les entraîner.

    Pour chaque famille : les deux valeurs sans correction, les deux valeurs
    avec poids équilibrés, puis sur-échantillonnage et sous-échantillonnage à la valeur forte.
    Les noms des configurations historiques conservées restent inchangés.
    ``pretraitement(standardiser)`` doit fournir des étapes neuves de Pipeline.
    ``sampling_strategy`` vaut 0.1 pour SMOTE sur Credit Card et « auto » sur KDD.
    KDD utilise RandomOverSampler : les copies gardent leurs catégories valides.
    RandomUnderSampler équilibre toujours les deux classes avec « auto ».
    """
    familles = [
        ("Arbre de décision", "max_depth", (4, 8),
         lambda valeur, poids: DecisionTreeClassifier(
             max_depth=valeur, class_weight=poids, random_state=graine), False),
        ("Régression logistique", "C", (0.01, 1.0),
         lambda valeur, poids: LogisticRegression(
             C=valeur, class_weight=poids, max_iter=2000, random_state=graine), True),
    ]
    configurations = []
    for famille, parametre, valeurs, creer, standardiser in familles:
        for valeur in valeurs:
            for correction in ("aucun", "class_weight"):
                poids = "balanced" if correction == "class_weight" else None
                configurations.append(_configuration(
                    famille, f"{famille} | {parametre}={valeur} | {correction}",
                    pretraitement, standardiser, correction, creer(valeur, poids)))
        valeur = valeurs[-1]
        constructeur = {"SMOTE": SMOTE, "RandomOverSampler": RandomOverSampler}[sur_echantillonnage]
        configurations.append(_configuration(
            famille, f"{famille} | {parametre}={valeur} | {sur_echantillonnage}",
            pretraitement, standardiser or sur_echantillonnage == "SMOTE",
            sur_echantillonnage, creer(valeur, None),
            ("sur_echantillonnage", constructeur(sampling_strategy=sampling_strategy,
                                                random_state=graine))))
        configurations.append(_configuration(
            famille, f"{famille} | {parametre}={valeur} | RandomUnderSampler",
            pretraitement, standardiser, "RandomUnderSampler", creer(valeur, None),
            ("sous_echantillonnage", RandomUnderSampler(
                sampling_strategy="auto", random_state=graine))))
    return configurations


def configurations_ensembles(pretraitement, y_app, graine=42):
    """Construire deux réglages a priori par ensemble, sans accès au test.

    EasyEnsemble agrège 5 ou 10 AdaBoost de 50 arbres de profondeur 1, chacun
    appris sur un sous-échantillon équilibré. RUSBoost équilibre les données à
    chaque tour et utilise 50 ou 100 arbres de profondeur 2. XGBoost emploie
    100 arbres de profondeur 3 ou 5 ; son poids positif vaut exclusivement le
    rapport effectif négatifs/positifs de y_app. La classe positive reste 1,
    même si elle est majoritaire (notamment les intrusions de KDDCup99).
    Aucun arrêt anticipé ni jeu d'évaluation externe n'est utilisé.
    """
    from xgboost import XGBClassifier

    y_app = np.asarray(y_app)
    poids_positif = np.count_nonzero(y_app == 0) / np.count_nonzero(y_app == 1)
    configurations = []
    for nombre in (5, 10):
        modele = EasyEnsembleClassifier(
            n_estimators=nombre,
            estimator=AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=graine),
                n_estimators=50, learning_rate=1.0, random_state=graine),
            sampling_strategy="auto", n_jobs=2, random_state=graine)
        configurations.append(_configuration(
            "EasyEnsemble", f"EasyEnsemble | n_estimators={nombre}",
            pretraitement, False, "sous-échantillonnage interne", modele))
    for nombre in (50, 100):
        modele = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, random_state=graine),
            n_estimators=nombre, learning_rate=1.0,
            sampling_strategy="auto", random_state=graine)
        configurations.append(_configuration(
            "RUSBoost", f"RUSBoost | n_estimators={nombre}, max_depth=2",
            pretraitement, False, "sous-échantillonnage interne", modele))
    for profondeur in (3, 5):
        modele = XGBClassifier(
            n_estimators=100, max_depth=profondeur, learning_rate=0.1,
            scale_pos_weight=poids_positif, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=2,
            random_state=graine)
        configurations.append(_configuration(
            "XGBoost", f"XGBoost | n_estimators=100, max_depth={profondeur}",
            pretraitement, False, "scale_pos_weight (apprentissage)", modele))
    return configurations
