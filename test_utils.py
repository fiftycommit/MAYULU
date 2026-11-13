"""Régression : un score sur la frontière native n’est pas une anomalie."""

import unittest

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from utils import desaccords, score_anomalie, seuil_defaut


class SeuilDefautTests(unittest.TestCase):
    def test_frontiere_inclusive_conserve_la_decision_native(self):
        rng = np.random.default_rng(42)
        apprentissage = rng.normal(size=(25, 2))
        nouvelles_observations = rng.normal(loc=1, size=(7, 2))
        detecteurs = [IsolationForest(n_estimators=8, random_state=42),
                      LocalOutlierFactor(n_neighbors=5, novelty=True)]
        for detecteur in detecteurs:
            with self.subTest(detecteur=type(detecteur).__name__):
                detecteur.fit(apprentissage)
                # Fixer une frontière exactement sur un nouveau point : sklearn
                # le garde normal car sa comparaison d’anomalie est stricte.
                detecteur.offset_ = detecteur.score_samples(nouvelles_observations)[0]
                attendu = detecteur.predict(nouvelles_observations) == -1
                obtenu = score_anomalie(detecteur, nouvelles_observations) >= seuil_defaut(detecteur)
                self.assertFalse(attendu[0])
                np.testing.assert_array_equal(obtenu, attendu)


class DesaccordsTests(unittest.TestCase):
    def test_compte_les_alertes_propres_et_les_anomalies(self):
        resultat = desaccords([1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 1], ("IF", "LOF"))
        self.assertEqual(resultat, {"communes": 1, "seulement IF": 2, "seulement LOF": 1,
                                    "anomalies seulement IF": 1, "anomalies seulement LOF": 0})


if __name__ == "__main__":
    unittest.main()
