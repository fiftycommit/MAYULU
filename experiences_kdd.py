"""Comparaison binaire sur le corpus officiel KDDCup99 à 10 %.

Source : https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
Archive d'origine :
https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
Noms des 41 variables :
https://kdd.ics.uci.edu/databases/kddcup99/kddcup.names

Le fichier local a été obtenu avec sklearn.datasets.fetch_kddcup99(
percent10=True, data_home="data/kddcup99_cache"), puis réenregistré en CSV gzip
sans index ni en-tête, dans l'ordre original. Les chaînes bytes ont été décodées
en ASCII et les valeurs numériques conservées. Les 494 021 lignes originales
sont présentes : ce n'est ni un échantillon synthétique ni une variante SA/SF.
Ce CSV compressé permet les exécutions suivantes entièrement hors connexion.

Intrusion vaut 1 pour toute étiquette autre que « normal. ». Les attaques sont
majoritaires dans le fichier brut, notamment smurf et neptune : une intrusion
n'y est donc pas forcément une observation statistiquement rare. Les répétitions
modifient fortement ce déséquilibre. Les doublons sont traités AVANT le tirage
stratifié et la partition ; aucun même vecteur prédictif ne traverse les jeux.

KDD99 provient d'un environnement militaire simulé historique. La répétition
massive des connexions, la prédominance de certaines attaques et l'ancienneté des
protocoles limitent toute extrapolation à un réseau actuel. Le découpage aléatoire
commun demandé ici ne mesure ni le transfert temporel ni les attaques inédites.
"""

from pathlib import Path

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import utils
from modeles_complementaires import configurations_ensembles, configurations_supervisees


CHEMIN_DONNEES = Path(__file__).resolve().parent / "data" / "kddcup99_10_percent.csv.gz"
SOURCE_KDD = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html"
ARCHIVE_KDD = "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
SHA256_DONNEES = "7e704c56479ba6179ecb14f79bbc25a77ac07247e336f82771f8c3f20602b388"
PRECISION_MIN_KDD = 0.8
MAX_OBSERVATIONS = 60_000
MAX_APPRENTISSAGE = 30_000

COLONNES_KDD = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORIELLES_KDD = ["protocol_type", "service", "flag"]
NUMERIQUES_KDD = [nom for nom in COLONNES_KDD if nom not in CATEGORIELLES_KDD]


def _effectifs(y):
    """Décrire les classes avant partition ou dans l'apprentissage seulement."""
    return {
        "total": len(y),
        "normales": int(y.eq(0).sum()),
        "intrusions": int(y.eq(1).sum()),
        "proportion_intrusions": float(y.mean()),
    }


def charger_kdd(chemin=CHEMIN_DONNEES):
    """Charger le vrai corpus local et dédoublonner les 41 prédicteurs.

    Un groupe prédictif portant les deux cibles binaires est entièrement exclu,
    plutôt que d'attribuer arbitrairement sa cible ou de choisir la majorité.
    Parmi les groupes restants, une seule ligne est conservée : sa cible binaire
    est univoque, même si deux noms d'attaques différents la décrivent. Les conflits
    d'étiquettes détaillées sont eux aussi comptés mais ne changent pas la cible.

    Dans l'archive obtenue : un groupe présente un conflit binaire (deux lignes)
    et deux groupes des noms d'attaque différents (quatre lignes). Après retrait
    des conflits binaires et des répétitions, il reste 145 583 vecteurs, dont
    57 752 intrusions ; la classe positive devient minoritaire. Ces nombres sont
    recalculés ci-dessous, jamais utilisés pour fabriquer ou remplacer les données.
    """
    types = {nom: "float64" for nom in NUMERIQUES_KDD}
    types.update({nom: str for nom in CATEGORIELLES_KDD + ["attaque"]})
    donnees = pd.read_csv(
        chemin, compression="gzip", header=None, names=COLONNES_KDD + ["attaque"],
        dtype=types, encoding="ascii", keep_default_na=False,
    )
    donnees["intrusion"] = donnees["attaque"].ne("normal.").astype("int8")
    effectifs_bruts = _effectifs(donnees["intrusion"])
    attaques_brutes = {nom: int(n) for nom, n in donnees["attaque"].value_counts().items()}

    groupes = donnees.groupby(COLONNES_KDD, sort=False, dropna=False)
    conflits_binaires = groupes["intrusion"].transform("nunique").gt(1)
    conflits_detailles = groupes["attaque"].transform("nunique").gt(1)
    informations = {
        "source": SOURCE_KDD,
        "archive_officielle": ARCHIVE_KDD,
        "recuperation": "sklearn.datasets.fetch_kddcup99(percent10=True), export CSV gzip ASCII",
        "fichier_local": str(Path(chemin).relative_to(Path(__file__).resolve().parent))
        if Path(chemin).is_absolute() and Path(chemin).is_relative_to(Path(__file__).resolve().parent)
        else str(chemin),
        "sha256_fichier_fourni": SHA256_DONNEES,
        "effectifs_bruts": effectifs_bruts,
        "attaques_brutes": attaques_brutes,
        "conflits_binaires_lignes": int(conflits_binaires.sum()),
        "conflits_binaires_groupes": len(
            donnees.loc[conflits_binaires, COLONNES_KDD].drop_duplicates()),
        "conflits_etiquettes_detaillees_lignes": int(conflits_detailles.sum()),
        "conflits_etiquettes_detaillees_groupes": len(
            donnees.loc[conflits_detailles, COLONNES_KDD].drop_duplicates()),
        "regle_dedoublonnage": (
            "Exclure tout groupe prédictif à cible binaire contradictoire, puis garder "
            "un représentant par vecteur à cible binaire univoque, avant toute partition."
        ),
        "cible": "intrusion = 1 si attaque != normal. ; normale = 0, sans inversion",
        "limites": (
            "Corpus historique de connexions simulées, attaques smurf/neptune dominantes "
            "dans le brut et nombreuses répétitions ; le dédoublonnage change la prévalence. "
            "La stratification aléatoire ne teste ni le futur ni des attaques inédites."
        ),
    }
    univoques = donnees.loc[~conflits_binaires]
    uniques = univoques.drop_duplicates(subset=COLONNES_KDD)
    informations["doublons_univoques_retires"] = len(univoques) - len(uniques)
    informations["effectifs_dedoublonnes"] = _effectifs(uniques["intrusion"])
    return uniques[COLONNES_KDD].copy(), uniques["intrusion"].copy(), informations


def pretraitement_kdd(standardiser):
    """Créer des transformations neuves, ajustées seulement dans chaque Pipeline.

    Les trois catégories textuelles sont décodées en ASCII au chargement puis
    encodées en one-hot ; les indicateurs binaires restent numériques. Une
    catégorie absente de l'apprentissage est ignorée. Les 38 variables numériques
    sont standardisées pour LOF et la logistique. Les arbres n'en ont pas besoin.
    RandomOverSampler duplique les vecteurs d'apprentissage sans interpolation :
    les catégories encodées restent valides.
    """
    colonnes = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORIELLES_KDD),
        ("numeriques", StandardScaler() if standardiser else "passthrough", NUMERIQUES_KDD),
    ])
    return [("colonnes_standard" if standardiser else "colonnes", colonnes)]


def calculer_kdd(graine=42):
    """Apprendre les sept familles et figer leurs seuils sur validation seulement.

    Un tirage stratifié fixe d'au plus 60 000 lignes après dédoublonnage borne le
    coût avant la partition commune 60/20/20. Un second tirage fixe, stratifié et
    limité à 30 000 lignes de l'apprentissage, est partagé par TOUTES les familles.
    La validation et le test gardent leur prévalence et ne sont pas rééquilibrés.
    Les poids XGBoost dépendent seulement de cet apprentissage effectif ; les
    stratégies de sur/sous-échantillonnage « auto » suivent la minorité sans renommer la cible.

    IF (100/300 arbres x 256/2048 points) et LOF (10/20/50/100 voisins, novelty=True)
    sont ajustés sur un apprentissage contaminé, sans exploitation de y par leurs
    estimateurs. Leurs hyperparamètres et seuils sont néanmoins choisis avec les
    étiquettes de validation : cette sélection est supervisée. Les modèles retenus
    sont ordonnés selon la règle commune. La précision minimale de 0.8 est un choix
    pédagogique annoncé a priori pour KDD, non une exigence propre à ce jeu dans
    l'énoncé. Ni scores, ni métriques, ni sélection ne consultent le test ici.

    Retour : parts (app/val/test), resultats_validation, retenues (index famille),
    modeles (nom -> Pipeline ajusté), informations. Le code appelant évalue une
    seule fois les retenues sur le test avec leurs seuils déjà figés.
    """
    X, y, informations = charger_kdd()
    X, y = utils.sous_echantillon_stratifie(X, y, n=MAX_OBSERVATIONS, graine=graine)
    informations["effectifs_echantillon"] = _effectifs(y)
    parts = utils.decouper_donnees(X, y, graine=graine)
    app = utils.sous_echantillon_stratifie(
        *parts["app"], n=MAX_APPRENTISSAGE, graine=graine)
    informations.update({
        "graine": graine,
        "limite_echantillon": MAX_OBSERVATIONS,
        "limite_apprentissage": MAX_APPRENTISSAGE,
        "tailles_partitions": {nom: len(part[0]) for nom, part in parts.items()},
        "effectifs_apprentissage_effectif": _effectifs(app[1]),
        "apprentissage_commun": "Même sous-échantillon stratifié pour les sept familles",
        "precision_min": PRECISION_MIN_KDD,
        "choix_precision_min": "0.8 choisi a priori à titre pédagogique, non imposé spécifiquement à KDD",
        "sampling_strategy": "auto : équilibrer la minorité, sans inverser la cible intrusion=1",
        "poids_positif_xgboost": float(app[1].eq(0).sum() / app[1].eq(1).sum()),
        "sur_echantillonnage": "RandomOverSampler : copies de vecteurs, catégories préservées",
    })
    configurations = configurations_supervisees(
        pretraitement_kdd, graine=graine, sampling_strategy="auto",
        sur_echantillonnage="RandomOverSampler")
    configurations += configurations_ensembles(pretraitement_kdd, app[1], graine=graine)
    for n_arbres in (100, 300):
        for max_samples in (256, 2048):
            configurations.append({
                "famille": "Isolation Forest",
                "nom": f"Isolation Forest | n_estimators={n_arbres}, max_samples={max_samples}",
                "pretraitement": "one-hot + numériques non standardisées",
                "desequilibre": "aucun (non supervisé, apprentissage contaminé)",
                "modele": Pipeline(pretraitement_kdd(False) + [("modele", IsolationForest(
                    n_estimators=n_arbres, max_samples=max_samples,
                    random_state=graine, n_jobs=2))]),
            })
    for voisins in (10, 20, 50, 100):
        configurations.append({
            "famille": "LOF",
            "nom": f"LOF | n_neighbors={voisins}",
            "pretraitement": "one-hot + standardisation numérique",
            "desequilibre": "aucun (non supervisé, apprentissage contaminé)",
            "modele": Pipeline(pretraitement_kdd(True) + [("modele", LocalOutlierFactor(
                n_neighbors=voisins, novelty=True, n_jobs=2))]),
        })
    for configuration in configurations:
        configuration["app"] = app
    resultats, modeles = utils.comparer_configurations(
        configurations, app, parts["val"], precision_min=PRECISION_MIN_KDD)
    retenues = utils.selectionner(resultats).groupby("famille", sort=False).head(1)
    return {
        "parts": parts,
        "resultats_validation": resultats,
        "retenues": retenues.set_index("famille"),
        "modeles": modeles,
        "informations": informations,
    }
