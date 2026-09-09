import unittest

import pandas as pd

from ml_service import train_classifier


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "radius": [1, 2, 1.2, 8, 9, 8.5, 1.5, 7.5, 2.2, 9.2, 1.8, 8.8],
            "texture": [2, 1, 2.2, 9, 8, 8.5, 1.8, 7.8, 2.4, 9.1, 1.5, 8.2],
            "diagnosis": ["B", "B", "B", "M", "M", "M", "B", "M", "B", "M", "B", "M"],
        })

    def test_trains_supported_model(self):
        result = train_classifier(self.frame, "diagnosis", "logistic_regression", True)
        self.assertIn("accuracy", result["metrics"])
        self.assertEqual(result["training_rows"] + result["testing_rows"], 12)

    def test_rejects_unknown_target(self):
        with self.assertRaisesRegex(ValueError, "target column"):
            train_classifier(self.frame, "missing", "knn", True)


if __name__ == "__main__":
    unittest.main()

