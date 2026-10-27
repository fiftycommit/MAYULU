# Synthèse — Détection d’anomalies

M’BEY Max — M2 Data Science

## Protocole et qualité des données

J’ai comparé les méthodes dans un notebook exécuté, avec les fonctions communes dans `utils.py`. La classe positive vaut 1 et un score élevé signifie « suspect ». Pour Credit Card et AI4I, les partitions sont stratifiées (60 % apprentissage, 20 % validation, 20 % test), avec une graine fixée. Les transformations et SMOTE sont ajustés sur l’apprentissage seulement. La sélection utilise la validation, jamais les performances du test. Les versions et les temps mesurés figurent dans le notebook.

Credit Card contient 492 fraudes sur 284 807 transactions. Après retrait de Time des prédicteurs, 9 144 lignes répètent les mêmes valeurs, sans contradiction de classe. Une seule copie est conservée avant le découpage : 275 663 transactions et 473 fraudes. Le montant maximal, 25 691,16, est étiqueté normal : une valeur extrême ne doit pas être supprimée automatiquement. Amount est transformée par log(1 + Amount), puis les variables sont standardisées pour la logistique, LOF et SMOTE.

AI4I contient 339 pannes sur 10 000 observations. Les identifiants et les cinq modes de panne sont exclus des entrées. L’audit révèle 27 discordances entre la cible et le OU des modes : 18 lignes avec RNF=1 mais cible=0, et 9 avec cible=1 sans mode renseigné. Je conserve `Machine failure`, imposée par le sujet, et signale cette limite sans reconstruire les étiquettes.

## Résultats et décisions

**Mouse.** IF et LOF classent très bien les anomalies (AP 0,96 et 1,00). Le seuil par défaut d’IF lève pourtant 125 alertes (F1 0,15), contre 21 avec IQR (F1 0,65). Pour LOF, le seuil par défaut retrouve les 10 anomalies (F1 1,00), alors qu’IQR est trop permissif (F1 0,43). KMeans ne domine pas non plus. Changer le seuil ne change pas l’AP ni l’AUC-ROC. En nouveautés, LOF entraîné sur 343 points normaux trouve 9 anomalies sur 10 avec 2 fausses alertes sur le test réservé. Cette référence propre utilise les étiquettes : ce n’est pas le même cadre que la détection d’outliers.

**Fraude.** Pour chaque configuration, je maximise le rappel sous précision ≥ 80 %, avec au moins une alerte, puis départage par précision et AP. Si la contrainte est inaccessible, le compromis retenu maximise le F1, entre seuils et entre configurations. Six configurations sont comparées pour chaque modèle supervisé : deux réglages, chacun sans correction, avec pondération et avec SMOTE. IF utilise quatre configurations ; LOF quatre, sur un sous-échantillon fixe de 30 000 transactions.

| Modèle retenu | AP test | Précision | Rappel | Alertes | FP | FN |
|---|---|---|---|---|---|---|
| Logistique pondérée, C=1 (sélectionnée) | 0,680 | 0,775 | 0,734 | 89 | 20 | 25 |
| Arbre, profondeur 8, sans correction | 0,681 | 0,920 | 0,734 | 75 | 6 | 25 |
| LOF, k=100 | 0,330 | 0,468 | 0,617 | 124 | 66 | 36 |
| IF, 300 arbres, max_samples=2048 | 0,152 | 0,200 | 0,160 | 75 | 60 | 79 |
| HBOS, 200 cases | 0,164 | 0,306 | 0,202 | 62 | 43 | 75 |

Le test contient 55 133 transactions et 94 fraudes. La logistique sélectionnée atteignait exactement 0,80 de précision et de rappel en validation. Elle ne respecte plus la précision exigée au test : seuil inchangé, 69 fraudes trouvées pour 89 enquêtes. IF, LOF et HBOS n’atteignent pas l’objectif en validation. Les AUC-ROC élevées de LOF et IF ne suffisent pas à rendre les alertes fiables ; l’AP révèle mieux cette difficulté. En validation, maximiser la balanced accuracy d’IF produit 2 183 alertes, contre 439 pour la logistique : ces maxima ne sont pas l’objectif métier. La pondération modifie peu l’AP de la logistique, mais la dégrade pour l’arbre.

**Méthode recherchée : HBOS.** Un histogramme est construit par variable et le score additionne −log(hauteur normalisée + alpha), sous une hypothèse d’indépendance. Les composantes d’ACP sont décorrélées, ce qui motive l’essai sans garantir leur indépendance. Trois nombres de cases sont testés (10, 50, 200), avec alpha=0,1. Le meilleur compromis F1 en validation est obtenu à 200 cases (0,340 contre 0,312 à 10 cases). HBOS repère les transactions rares sur plusieurs variables, avec des contributions lisibles, et un faible coût mesuré. En revanche, sa construction univariée ne représente pas les dépendances : une combinaison rare de valeurs marginalement banales peut lui échapper. Cette limite théorique ne prouve pas que toutes les fraudes manquées ont ce profil.

**Maintenance.** Les hyperparamètres de la logistique sélectionnée et d’IF sont conservés, mais les transformations et modèles sont réajustés sur AI4I. Le seuil est choisi sous précision ≥ 90 %. La logistique atteint 11 pannes sur 12 alertes en validation ; au test, elle trouve 4 pannes sans fausse alerte et en manque 64 sur 68. IF atteint la contrainte sur 2 alertes en validation, mais sa précision tombe à 0,50 au test (1 panne trouvée, 1 intervention inutile, 67 pannes manquées). Un arbre essayé à titre indicatif a une meilleure AP de validation (0,59 contre 0,41), mais n’atteint pas la précision exigée (0,811). Il n’est donc pas meilleur selon la règle opérationnelle.

## Robustesse, explicabilité et limites

Sur Mouse, je remplace 0 %, 5 % et 10 % de l’apprentissage normal par des points uniformes dans une boîte élargie, sur trois graines, sans changer le test. LOF reste stable (AP 0,98 à 0,99) ; IF s’améliore (0,78 ± 0,11 à 0,93 ± 0,04). L’élargissement des zones où les coupures sont possibles est une explication plausible, non démontrée isolément. Les conclusions dépendent du générateur ; certains points injectés tombent dans les nuages normaux.

SHAP explique la longueur moyenne de chemin corrigée d’IF, vérifiée numériquement à moins de 1e−6. Une contribution négative pousse vers l’alerte. Les six exemples couvrent deux fraudes détectées, deux fausses alertes et deux fraudes manquées. Les principales contributions négatives se recoupent entre détections et fausses alertes. Les deux fraudes manquées choisies sont les moins suspectes pour IF : leur profil ne représente pas toutes les fraudes manquées. SHAP décrit le modèle, pas une cause de fraude ni le sens métier des variables anonymisées.

Les faibles effectifs positifs rendent les seuils fragiles. Le découpage aléatoire ne mesure pas une utilisation prospective : il faudrait aussi un test temporel. AI4I est synthétique et ses étiquettes présentent les discordances signalées. Les AP des deux jeux ne se comparent pas directement, car leurs proportions positives diffèrent. Enfin, régler un détecteur non supervisé avec les étiquettes de validation constitue une sélection supervisée.

## Sources

Cours de H. Elghazel, notamment diapos 4, 19–24 et 34–43. Goldstein et Dengel (2012), *Histogram-Based Outlier Score (HBOS): A Fast Unsupervised Anomaly Detection Algorithm*, KI-2012. Documentation : https://pyod.readthedocs.io/en/latest/_modules/pyod/models/hbos.html ; AI4I : https://doi.org/10.24432/C5HS5C.
