import logging
import multiprocessing
import autosklearn.classification
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import numpy


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


def load_data():
    # These are functions to load datasets suitable for classification problems

    return {name: load_func(return_X_y=True)
            for name, load_func in NAME_TO_LOAD_FUNC.items()}


def train_automl_model(features_train: numpy.ndarray,
                       target_train: numpy.ndarray) -> autosklearn.classification.AutoSklearnClassifier:
    """
    Trains an automl model
    :param features_train: Features
    :param target_train: Target
    :return: automl model
    :rtype: autosklearn.classfication.AutoSklearnClassifier
    """
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(features_train, target_train)
    return automl


def main():
    pass


if __name__ == "__main__":
    main()
