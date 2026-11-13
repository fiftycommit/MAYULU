# Détection d’anomalies

Détecter des événements rares (fraudes bancaires, pannes industrielles, intrusions réseau) dans des données où les anomalies descendent jusqu’à 0,17 % des observations.
Comparaison de détecteurs non supervisés (Isolation Forest, LOF, HBOS) et de modèles supervisés avec traitement du déséquilibre, réglés et seuillés sur validation sous contrainte de précision avant évaluation sur test.
Sur le test Credit Card, la méthode retenue retrouve 74 fraudes sur 94 pour 21 fausses alertes.

## Résultats sur les ensembles de test

Les hyperparamètres, la méthode retenue et le seuil de décision sont choisis sur la partition de validation, puis figés avant l’évaluation finale sur le test.

| Jeu de données | Anomalies (test) | Contrainte visée (validation) | Modèle sélectionné sur validation | AP (test) | Précision (test) | Rappel (test) | Fausses alertes (FP) | Anomalies manquées (FN) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| Credit Card (grille classique) | 94 / 55 133 | Précision ≥ 80 % | Régression logistique (`C=1`, `class_weight`) | 0,6798 | 77,5 % | 73,4 % | 20 | 25 |
| Credit Card (comparaison étendue) | 94 / 55 133 | Précision ≥ 80 % | XGBoost (`n_estimators=100`, `max_depth=5`) | 0,7923 | 77,9 % | 78,7 % | 21 | 20 |
| AI4I 2020 (transfert depuis Credit Card) | 68 / 2 000 | Précision ≥ 90 % | XGBoost (transféré, réajusté sur AI4I) | 0,6985 | 80,4 % | 60,3 % | 10 | 27 |
| KDDCup99 (échantillon dédoublonné) | 4 761 / 12 000 | Précision ≥ 80 % | XGBoost (`n_estimators=100`, `max_depth=5`) | 0,9999 | 97,2 % | 99,9 % | 139 | 3 |

Sur Credit Card et AI4I, un seuil choisi pile au niveau de la contrainte sur la validation peut passer sous le seuil sur le test à cause du faible nombre d’anomalies ; les seuils ne sont pas retouchés après consultation du test. Une analyse complémentaire montre qu’imposer une marge sur validation (précision ≥ 85 % sur Credit Card) permet à XGBoost de tenir la contrainte au test (précision 81,1 %, rappel 77,7 %).

## Documents et code

- [`TP_detection_anomalies.ipynb`](TP_detection_anomalies.ipynb) : notebook principal exécuté, structuré par appels aux modules Python avec tableaux, courbes et matrices de confusion.
- [`rendus/synthese.pdf`](rendus/synthese.pdf) ([source Markdown](rendus/synthese.md)) : synthèse de deux pages.
- [`rendus/compte_rendu.md`](rendus/compte_rendu.md) : compte rendu complet des cinq parties et des expériences complémentaires.
- [`rendus/rendu_questions_reponses.md`](rendus/rendu_questions_reponses.md) ([version HTML](rendus/rendu_questions_reponses.html)) : réponses détaillées par question.

```text
data/
  mouse.txt                        # données synthétiques 2D (outliers et nouveautés)
  creditcard.csv                   # transactions bancaires (suivi avec Git LFS)
  ai4i2020.csv                     # maintenance prédictive
  kddcup99_10_percent.csv.gz       # intrusions réseau (corpus officiel à 10 %)
rendus/
  synthese.pdf
  synthese.md
  compte_rendu.md
  rendu_questions_reponses.md
  rendu_questions_reponses.html
  resultats.json                   # métriques exportées par l’exécution
  figures/                         # figures extraites du notebook
TP_detection_anomalies.ipynb
experiences.py                     # expériences sur Mouse, Credit Card, HBOS, AI4I et SHAP
experiences_kdd.py                 # préparation et comparaison sur KDDCup99
modeles_complementaires.py         # grilles supervisées, rééchantillonnage et ensembles
utils.py                           # découpage, prétraitements, scores, seuils et métriques
generer_notebook.py                # exécution complète et construction du notebook
generer_rendus.py                  # génération des documents à partir de resultats.json
requirements.txt
```

## Méthodologie

1. **Partage et fuites de données** : Credit Card et KDDCup99 sont dédoublonnés sur les variables prédictives avant un découpage stratifié 60 % apprentissage / 20 % validation / 20 % test (graine 42). Les transformations (`StandardScaler`, encodage `OneHotEncoder`) et le rééchantillonnage (`SMOTE`, `RandomOverSampler`, `RandomUnderSampler`) sont ajustés exclusivement sur l’apprentissage.
2. **Modèles évalués** :
   - *Non supervisés* : Isolation Forest, Local Outlier Factor (`novelty=True` hors échantillon), et implémentation de **HBOS** (*Histogram-Based Outlier Score*, Goldstein & Dengel, 2012).
   - *Supervisés* : arbre de décision, régression logistique, EasyEnsemble, RUSBoost et XGBoost.
3. **Explicabilité et robustesse** : décomposition SHAP des profondeurs d’isolement d’Isolation Forest sur six transactions de test (fraudes détectées, fausses alertes, fraudes manquées) et étude de la contamination de la référence normale sur Mouse (0 %, 5 %, 10 % sur trois graines).

## Reproduire les expériences

Les jeux de données sont inclus dans `data/`. `creditcard.csv` est stocké via **Git LFS** (`git lfs install && git lfs pull` après clonage).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generer_notebook.py
.venv/bin/python generer_rendus.py
```

- `generer_notebook.py` entraîne les modèles, évalue les partitions, exporte `rendus/resultats.json` et écrit le notebook exécuté.
- `generer_rendus.py` reconstruit les fichiers de `rendus/` et les figures sans réentraîner les modèles.

## Sources des données

- [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB / Kaggle)
- [AI4I 2020 Predictive Maintenance](https://doi.org/10.24432/C5HS5C) (UCI)
- [KDDCup99](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) (UCI KDD Archive)
- Goldstein, M. & Dengel, A. (2012), *HBOS: A Fast Unsupervised Anomaly Detection Algorithm*, KI-2012.
