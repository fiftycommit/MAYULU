# TP — Détection d’anomalies

Le notebook présente les résultats finaux. Les expériences, l’entraînement et le choix des réglages sont dans les modules Python ; les comparaisons de validation et les analyses détaillées sont dans le [Q/R](rendus/rendu_questions_reponses.md).

## Fichiers nécessaires

```text
data/
  mouse.txt
  creditcard.csv                 # suivi avec Git LFS
  ai4i2020.csv
  kddcup99_10_percent.csv.gz       # corpus officiel à 10 %, utilisable hors connexion
rendus/
  resultats.json                 # source commune des nombres publiés
  figures/                       # images du notebook et courbes de validation
  rendu_questions_reponses.md
  rendu_questions_reponses.html
  compte_rendu.md
  synthese.md
  synthese.pdf
TP_detection_anomalies.ipynb      # imports, appels nommés, résultats exécutés
experiences.py                   # orchestration et expériences des cinq parties
experiences_kdd.py               # préparation et comparaison KDDCup99
modeles_complementaires.py       # grilles classiques et ensembles supervisés
utils.py                         # chargement, prétraitements, scores, seuils, métriques
generer_notebook.py               # construction et exécution du notebook
generer_rendus.py                 # documents depuis JSON et notebook, sans entraînement
requirements.txt
```

Les deux générateurs sont à la racine, comme les autres scripts. Le sujet et le cours sont également fournis en PDF à la racine.

Les modules d’expériences importent les fonctions communes de `utils.py` et les configurations de `modeles_complementaires.py`. Le notebook appelle `calculer_resultats()`, puis `afficher_resultats(resultats, nom)` pour `protocole`, `mouse`, `nouveautes`, `creditcard`, `hbos`, `ai4i`, `robustesse`, `shap` et `kdd`. L’appel final à `exporter_resultats()` écrit le JSON et les courbes de validation. Les sorties sont de vrais tableaux et des figures, dont des matrices de confusion légendées par modèle, pas une capture du texte de la console.

Le travail couvre Mouse, les fraudes Credit Card, HBOS, la maintenance AI4I, la robustesse et SHAP, puis le bloc complémentaire non bonus : RandomUnderSampler, EasyEnsemble, RUSBoost, XGBoost et KDDCup99. Seul le bonus autoencodeur n’est pas réalisé.

## Recalculer les expériences

Depuis la racine :

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generer_notebook.py
.venv/bin/python generer_rendus.py
```

Les dépendances fixées comprennent numpy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn, SHAP, Jupyter et XGBoost. `nbformat` et `nbclient` servent au notebook ; `mistune`, installé avec nbconvert dans l’environnement Jupyter, sert au HTML. Aucun téléchargement de données n’est nécessaire si les fichiers de `data/` sont présents.

`generer_notebook.py` effectue les apprentissages et ne remplace le notebook qu’après une exécution réussie. On peut aussi l’ouvrir dans VS Code ou Jupyter, sélectionner `.venv`, redémarrer le noyau et tout exécuter dans l’ordre. Le premier appel calcule les expériences ; les suivants affichent les résultats par rubrique.

`generer_rendus.py` ne recalcule aucun modèle. Il lit `rendus/resultats.json`, extrait les PNG des cellules identifiées par leur appel nommé dans le notebook exécuté, puis reconstruit Q/R Markdown et HTML, compte rendu et synthèse Markdown. Il conserve les questions conceptuelles du Q/R existant et remplace ses réponses chiffrées ; ce fichier fait donc partie des entrées à conserver. Les figures de validation sont produites lors de l’export des expériences. Utiliser un JSON et un notebook issus de la même exécution.

Pour mettre à jour seulement les documents après une exécution déjà terminée :

```bash
.venv/bin/python generer_rendus.py
```

Le script écrit aussi `/tmp/tp1-synthese.html`, avec une mise en page A4 destinée au PDF. Ouvrir ce fichier dans Chromium et l’imprimer vers `rendus/synthese.pdf`, sans en-têtes ni pieds de page du navigateur. Contrôler que le PDF tient sur deux pages : l’impression est séparée de la génération des documents.

## Données locales et protocole

Sources : Mouse est fourni avec le TP ; [Credit Card](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), [AI4I 2020](https://doi.org/10.24432/C5HS5C) et [KDDCup99 officiel](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html). Le fichier KDD local conserve le corpus officiel à 10 %, réexporté en CSV gzip ASCII depuis `sklearn.datasets.fetch_kddcup99(percent10=True)` ; sa provenance, son empreinte et les règles de dédoublonnage sont documentées dans `experiences_kdd.py` et dans le Q/R.

Credit Card dépasse la limite de taille des fichiers classiques GitHub et utilise Git LFS. Pour récupérer le dépôt complet :

```bash
git lfs install
git clone https://github.com/fiftycommit/MAYULU.git
```

Si le dépôt est déjà cloné, lancer `git lfs pull` depuis sa racine. Git LFS doit être installé au préalable (`brew install git-lfs` sur macOS). Sans téléchargement LFS, `creditcard.csv` peut n’être qu’un fichier pointeur, inutilisable par pandas. Les quotas GitHub LFS s’appliquent.

Les chemins sont relatifs au projet. Credit Card et KDD sont dédoublonnés avant le partage stratifié 60/20/20, graine 42. LOF utilise 30 000 lignes d’apprentissage sur Credit Card ; sur KDD, après un tirage de 60 000 vecteurs dédoublonnés, les sept familles partagent le même apprentissage effectif de 30 000 lignes. Validation et test ne sont jamais rééquilibrés. Credit Card utilise SMOTE et RUS ; KDD utilise ROS (copies de lignes, catégories préservées) et RUS.

La sélection maximise le rappel sous précision minimale sur validation, sinon rapporte le meilleur compromis F1. La logistique était le choix historique Credit Card ; XGBoost est le choix de la comparaison étendue. Le meilleur supervisé global et IF sont réajustés sur AI4I avec leurs hyperparamètres conservés ; la logistique est gardée comme référence. Le choix AI4I et les résultats publiés sont lus dans le JSON, sans gagnant figé dans le générateur.

L’extension a été faite après une première lecture du test Credit Card. Ses réglages sont fixés a priori et ses seuils viennent de la validation, mais elle reste exploratoire : le test n’est plus un holdout entièrement vierge. Les échecs de précision et les faibles rappels restent visibles.

Pour lire le Q/R dépliable, utiliser l’aperçu Markdown de VS Code ou le fichier HTML. Conserver `rendus/figures/` avec les documents.
