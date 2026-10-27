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

### Repères dans le notebook

Les titres reprennent la hiérarchie du sujet. Chaque sous-question est suivie de ses explications,
de son code et de ses résultats, selon ce qu’elle demande ; les analyses ne sont plus regroupées loin des figures.
Le sommaire du notebook donne accès aux cinq parties.

| Partie | Organisation |
|---|---|
| 1 — Mouse | Questions 1 à 3, avec leurs sous-questions a, b et c selon le sujet |
| 2 — Fraudes | Préparation 1.a–b, comparaison 2.a–d, évaluation 3.a–d |
| 3 — Nouvelle méthode (HBOS) | Sous-questions a à e |
| 4 — Maintenance (AI4I) | Sous-questions a à e |
| 5 — Robustesse et SHAP | Question 1 : contamination ; question 2 : explication des alertes |

Le bilan final précise le périmètre : les expériences supplémentaires du bloc situé après
« Livrables et synthèse » dans le PDF (sous-échantillonnage supervisé, EasyEnsemble, RUSBoost,
XGBoost, KDDCup99) et le bonus autoencodeur ne sont pas réalisés.

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
