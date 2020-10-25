import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

CURR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_random_grid() -> dict:
    # Parameter values to use in Randomized Search CV
    return {
        "n_estimators": [300, 450, 600, 900, 1200],
        "min_samples_split": [3, 5, 9, 15],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"],
        "max_depth": [25, 50, 100],
    }

def get_pipeline() -> Pipeline:
    """
    Get TF-IDF to Random Forest pipeline.
    NOTE: Use of 10 fold cross validation along with basic search of parameter space.
    :return: a sklearn pipeline
    """
    vectorizer = TfidfVectorizer(analyzer="word", stop_words="english", strip_accents="ascii", ngram_range=(1,2))
    classifier = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=get_random_grid(),
        n_iter=10,  # 10 parameter random samples from grid
        cv=10,  # 10 fold cross validation
        verbose=2,
        random_state=42,
        n_jobs=-1)
    pipe = Pipeline([('tfidf', vectorizer), ('random_forest', classifier)])
    return pipe


class SocClassifier:
    """ The SOC Occupation Classifier class. For creation of models and basic analysis."""

    def __init__(self, y_var):
        """Initialize model and fit based on chosen variable for classification."""
        self.y_var = y_var
        self.data = pd.read_csv(os.path.join(CURR_PATH, '../data/modified_data.csv'))
        self.model = get_pipeline()
        self.fit()

    def fit(self):
        """Fit model using pipeline."""
        self.model.fit(self.data['job_title'], self.data[self.y_var])

    def accuracy(self):
        """Return the accuracy of prediction."""
        prediction = self.model.predict(self.data['job_title'])
        return sum(self.data[self.y_var] == prediction) / len(prediction)

    def confusion_matrix(self) -> pd.DataFrame:
        """Return the confusion matrix."""
        prediction = self.model.predict(self.data['job_title'])
        conf_mat = confusion_matrix(self.data[self.y_var], prediction)
        return pd.DataFrame(conf_mat)

    def best_params_(self) -> dict:
        """Return best parameters from RandomizedSearchCV."""
        return self.model['random_forest'].best_params_

    def pickle_pipeline(self, filename):
        """Pickle the model for storage."""
        pickle.dump(self.model, open(filename, 'wb'))
