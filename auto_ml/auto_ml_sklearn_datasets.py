import logging
import os
import multiprocessing
import autosklearn.classification
import sklearn.datasets
import sklearn.metrics


def load_data():
    # These are functions to load datasets suitable for classification problems
    name_to_load_func = {"digits": sklearn.datasets.load_digits,
                         "iris": sklearn.datasets.load_iris,
                         "wine": sklearn.datasets.load_wine,
                         "breast_cancer": sklearn.datasets.load_breast_cancer}
    return {name: load_func(return_X_y=True)
            for name, load_func in name_to_load_func.items()}


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


if __name__ == "__main__":
    main()
