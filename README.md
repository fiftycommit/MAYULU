# TP — Détection d’anomalies

## Organisation

```text
data/
  mouse.txt
  ai4i2020.csv
  creditcard.csv              # suivi avec Git LFS
rendus/
  compte_rendu.md
  rendu_questions_reponses.md
  rendu_questions_reponses.html
  synthese.md
  synthese.pdf
  figures/                   # images utilisées par les documents
TP_detection_anomalies.ipynb  # expériences et résultats exécutés
utils.py                     # fonctions communes
requirements.txt
```

Le sujet et le cours sont disponibles dans les PDF à la racine.

## Exécuter le TP

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Ouvrir `TP_detection_anomalies.ipynb`, sélectionner l’environnement `.venv`, puis redémarrer le noyau et tout exécuter.

Les trois jeux de données sont inclus dans le projet. `creditcard.csv` dépasse la limite de taille de GitHub
pour les fichiers classiques et est suivi avec **Git LFS**. Pour récupérer une copie complète après publication :

```bash
git lfs install
git clone https://github.com/fiftycommit/MAYULU.git
```

Si le dépôt a déjà été cloné, lancer `git lfs pull` depuis sa racine.
Git LFS doit être installé sur la machine (`brew install git-lfs` sur macOS). Sans lui, le CSV peut être
remplacé par un petit fichier pointeur que pandas ne pourra pas exploiter.
Les téléchargements LFS dépendent également des quotas du compte GitHub.

Les chemins de données sont relatifs à la racine du projet. Les documents sont regroupés dans [`rendus/`](rendus/).
Pour lire le [Q/R dépliable](rendus/rendu_questions_reponses.md), utiliser l’aperçu Markdown de VS Code.
Conserver le sous-dossier `rendus/figures/` avec ces documents pour afficher les images.
