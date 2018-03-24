import logging
import multiprocessing
import autosklearn.classification
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import pickle
import os
from auto_ml.settings import MODEL_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

NAME_TO_LOAD_FUNC = {"digits": sklearn.datasets.load_digits,
                     "iris": sklearn.datasets.load_iris,
                     "wine": sklearn.datasets.load_wine,
                     "breast_cancer": sklearn.datasets.load_breast_cancer}


def load_data() -> dict:
    """

    :return: dict of dataset names and (features, target) tuple
    """
    # These are functions to load datasets suitable for classification problems

    return {name: load_func(return_X_y=True)
            for name, load_func in NAME_TO_LOAD_FUNC.items()}


class AutoMLClassifier(autosklearn.classification.AutoSklearnClassifier):
    def save(self, output_file: str):
        pickle.dump(self, output_file)


def main():
    name_to_data = load_data()
    for name, data in name_to_data.items():
        feat, tgt = data
        feat_train, feat_test, tgt_train, tgt_test = \
            sklearn.model_selection.train_test_split(feat, tgt, random_state=1)
        automl = AutoMLClassifier()
        automl.fit(feat_train, tgt_train)
        model_path = os.path.join(MODEL_DIR, "{}.pkl".format(name))
        automl.save(model_path)
        predictions = automl.predict(feat_test)
        logger.info("accuracy score: {}".format(sklearn.metrics.accuracy_score(tgt_test, predictions)))


if __name__ == "__main__":
    main()
