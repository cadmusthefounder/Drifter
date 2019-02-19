import numpy as np
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, space_eval, STATUS_OK, Trials

class HyperparametersTuner:

    def __init__(self, classifier_class, fixed_hyperparameters, search_space, max_evaluations=40):
        self._classifier_class = classifier_class
        self._fixed_hyperparameters = fixed_hyperparameters
        self._search_space = search_space
        self._max_evaluations = max_evaluations

    def get_best_hyperparameters(self, data, labels):
        self._training_data = data
        self._training_labels = labels .ravel()

        # Try fixed hyperparameters
        classifier = self._classifier_class()
        classifier.set_params(**self._fixed_hyperparameters)
        classifier.fit(self._training_data, self._training_labels)
        predictions = classifier.predict(self._training_data)

        print('\nFixed hyperparameters')
        print('self._training_labels.shape: {}'.format(self._training_labels.shape))
        print('predictions.shape: {}\n'.format(predictions.shape))
        
        fixed_hyperparameters_score = roc_auc_score(self._training_labels, predictions)

        # Find best trial hyperparameters
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self._search_space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=self._max_evaluations
        )
        best_trial_hyperparameters = space_eval(self._search_space, best)
        best_trial_hyperparameters_score = 1 - np.min([x['loss'] for x in trials.results])

        return self._fixed_hyperparameters if fixed_hyperparameters_score > best_trial_hyperparameters_score else best_trial_hyperparameters

    def objective(self, trial_hyperparameters):
        print('\nTrial hyperparameters')
        print('trial_hyperparameters: {}'.format(trial_hyperparameters))

        classifier = self._classifier_class()
        classifier.set_params(**trial_hyperparameters)
        classifier.fit(self._training_data, self._training_labels)
        predictions = classifier.predict(self._training_data)

        print('self._training_labels.shape: {}'.format(self._training_labels.shape))
        print('predictions.shape: {}\n'.format(predictions.shape))
        
        trial_score = roc_auc_score(self._training_labels, predictions)
        return {'loss': (1 - trial_score), 'status': STATUS_OK }