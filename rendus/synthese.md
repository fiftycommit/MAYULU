# Synthèse — Détection d’anomalies

M’BEY Max — M2 Data Science

## Protocole et données

Classe positive = anomalie, fraude, panne ou intrusion ; score élevé = suspect. Partitions stratifiées 60/20/20, graine 42. Transformations et rééchantillonnages sur apprentissage seulement. La validation sélectionne configurations et seuils : rappel maximal sous précision ≥ 80 % (fraude), ≥ 90 % (maintenance), puis précision et AP ; à défaut, compromis au meilleur F1 signalé non conforme. Le test conserve ces choix. Le notebook présente les résultats ; calculs dans les modules Python, analyses de validation dans le Q/R.

Credit Card : 275 663 transactions et 473 fraudes après dédoublonnage des prédicteurs avant partage. Time est exclue ; Amount devient log(1 + Amount). Standardisation pour logistique, LOF et SMOTE. AI4I : 10 000 observations, 339 pannes. Identifiants et modes de panne exclus ; cible d’origine conservée malgré 27 discordances avec les modes. Type est encodé en one-hot.

## Mouse, fraude et méthode recherchée

Sur Mouse, IF et LOF classent les anomalies aux premiers rangs. Changer seulement le seuil ne change pas AP/ROC. IQR réduit les fausses alertes d’IF mais peut dégrader LOF ; aucune règle ne domine. En nouveautés, LOF apprend sur 343 normaux et évalue 157 points distincts : au défaut, 11 alertes, 2 fausses alertes, 1 anomalie manquée. Cette référence normale nécessite les étiquettes.

Grille classique : six essais par famille (profondeur 4/8 ou C=0,01/1 × aucun/poids, puis un SMOTE et un RUS au réglage fort). EasyEnsemble : 5/10 AdaBoost de 50 souches ; RUSBoost : 50/100 arbres profondeur 2 ; XGBoost : 100 arbres profondeur 3/5, taux 0,1, poids appris sur l’apprentissage. IF : quatre essais ; LOF : quatre voisinages sur 30 000 lignes. HBOS : trois essais.

| Modèle | AP | Précision | Rappel | FP | FN |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0,7923 | 0,7789 | 0,7872 | 21 | 20 |
| Régression logistique | 0,6798 | 0,7753 | 0,7340 | 20 | 25 |
| EasyEnsemble | 0,6785 | 0,7848 | 0,6596 | 17 | 32 |
| Arbre de décision | 0,6811 | 0,9200 | 0,7340 | 6 | 25 |
| RUSBoost | 0,4018 | 0,4889 | 0,7021 | 69 | 28 |
| LOF | 0,3295 | 0,4677 | 0,6170 | 66 | 36 |
| HBOS | 0,1640 | 0,3065 | 0,2021 | 43 | 75 |
| Isolation Forest | 0,1524 | 0,2000 | 0,1596 | 60 | 79 |

Le choix étendu de validation est « XGBoost | n_estimators=100, max_depth=5 ». La logistique était le choix historique avant les ensembles ; ses métriques restent visibles, sans la présenter comme gagnante globale. Les seuils suivent la règle de l’énoncé (précision ≥ 80 % sur validation) et ne sont jamais retouchés. Variante complémentaire, distincte de la règle de l’énoncé (80 %) : seuils cherchés à 85 % sur validation (marge de 5 points fixée a priori, la validation ne comptant que peu de fraudes). Modèle choisi « XGBoost | n_estimators=100, max_depth=5 » : précision 0,8111, rappel 0,7766 au test, objectif de 80 % atteint. L’extension (RUS, ensembles et KDD) a été réalisée après une première lecture du test Credit Card. Les réglages sont annoncés a priori et les seuils utilisent uniquement la validation, mais cette comparaison reste exploratoire : le test ne peut plus être présenté comme un holdout entièrement vierge. Une confirmation demanderait de nouvelles données.

HBOS somme −log(hauteur normalisée + alpha) sur des histogrammes marginaux ; 10/50/200 cases, alpha=0,1. Décorrélation des composantes ACP ne signifie pas indépendance. Le score est décomposable, mais une combinaison rare de valeurs marginalement ordinaires peut lui échapper. Les coûts mesurés sont ponctuels, à effectifs d’apprentissage différents pour LOF.

## Maintenance et intrusions

AI4I : transfert du meilleur supervisé global Credit Card et d’IF, avec régression logistique de référence. Hyperparamètres conservés, dont le poids positif XGBoost Credit Card ; modèles réajustés et seuils propres à AI4I. Choix de validation : XGBoost. Objectif de précision de 90 % au test : non atteint. La référence (régression logistique) conserve un faible rappel.

| Modèle | AP | Précision | Rappel | FP | FN |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0,6985 | 0,8039 | 0,6029 | 10 | 27 |
| Régression logistique | 0,4087 | 1,0000 | 0,0588 | 0 | 64 |
| Isolation Forest | 0,0919 | 0,5000 | 0,0147 | 1 | 67 |

KDDCup99 : vrai corpus officiel à 10 % local. 494 021 lignes brutes ; 2 lignes à cible binaire contradictoire exclues, puis 145 583 vecteurs uniques, dont 57 752 intrusions. Tirage stratifié 60 000 puis 60/20/20 ; apprentissage effectif commun de 30 000, validation/test 12 000 chacun. Intrusion=1 ; one-hot des catégories, standardisation selon la famille. ROS copie des lignes réelles pour conserver des catégories valides. Précision ≥ 80 % choisie à titre pédagogique. Choix validation : XGBoost.

| Modèle | AP | Précision | Rappel | FP | FN |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0,9999 | 0,9716 | 0,9994 | 139 | 3 |
| RUSBoost | 0,9998 | 0,8410 | 1,0000 | 900 | 0 |
| EasyEnsemble | 0,9995 | 0,8231 | 0,9998 | 1 023 | 1 |
| Régression logistique | 0,9945 | 0,8115 | 0,9992 | 1 105 | 4 |
| Arbre de décision | 0,9970 | 0,8490 | 0,9994 | 846 | 3 |
| LOF | 0,4439 | 0,7820 | 0,1078 | 143 | 4 247 |
| Isolation Forest | 0,5116 | 0,7727 | 0,0607 | 85 | 4 471 |

## Robustesse, explications et limites

Mouse : remplacement de 0/5/10 % de la référence normale par des points uniformes dans une boîte élargie, trois graines, taille et test fixes. Les moyennes et dispersions du Q/R dépendent de ce générateur ; une amélioration d’IF n’est pas une loi générale. SHAP explique le chemin corrigé d’IF, pas une probabilité : contribution négative vers l’alerte. Fond de 50 transactions d’apprentissage, six cas contrastés ; écart maximal de reconstruction 2.58e-07. Ni causalité ni représentativité globale.

Dix anomalies Mouse et peu de positifs fragilisent les seuils. AI4I est synthétique ; KDD est historique et simulé, avec prévalence modifiée par dédoublonnage. Un partage aléatoire ne mesure ni le futur ni des attaques inédites. Les AP dépendent des prévalences et ne se comparent pas directement entre jeux. Régler IF/LOF avec les étiquettes de validation constitue une sélection supervisée. Bloc complémentaire réalisé ; seul le bonus autoencodeur ne l’est pas.

## Sources

Sujet 2026–2027 et cours de H. Elghazel ; Credit Card : [Kaggle/ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) ; AI4I : [UCI, DOI 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C) ; [KDDCup99 officiel](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) ; Goldstein et Dengel (2012), HBOS, et [PyOD](https://pyod.readthedocs.io/en/latest/_modules/pyod/models/hbos.html). Sources des ensembles et SHAP dans le Q/R.
