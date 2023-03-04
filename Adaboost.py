from sklearn.base import ClassifierMixin, is_classifier, is_regressor
from sklearn.ensemble import BaseEnsemble
from abc import ABCMeta, abstractmethod
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from numbers import Integral, Real
from sklearn.utils.validation import _check_sample_weight
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import has_fit_parameter
from sklearn.utils import check_random_state, _safe_indexing
from scipy.special import xlogy
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _num_samples
from sklearn.utils.extmath import softmax
from sklearn.metrics import accuracy_score, r2_score


class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "random_state": ["random_state"],
        "base_estimator": [HasMethods(["fit", "predict"]), StrOptions({"deprecated"})],
    }

    @abstractmethod
    def __init__(
            self,
            estimator=None,
            *,
            n_estimators=50,
            estimator_params=tuple(),
            learning_rate=1.0,
            random_state=None,
            base_estimator="deprecated",
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            base_estimator=base_estimator,
        )

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        # Only called to validate X in non-fit methods, therefore reset=False
        return self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            reset=False,
        )

    def fit(self, X, y, sample_weight=None):
        self._validate_params()

        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            y_numeric=is_regressor(self),
        )

        sample_weight = _check_sample_weight(
            sample_weight, X, np.float64, copy=True, only_non_negative=True
        )
        sample_weight /= sample_weight.sum()

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Initialization of the random number instance that will be used to
        # generate a seed at each iteration
        random_state = check_random_state(self.random_state)
        epsilon = np.finfo(sample_weight.dtype).eps

        zero_weight_mask = sample_weight == 0.0
        for iboost in range(self.n_estimators):
            # avoid extremely small sample weight, for details see issue #20320
            sample_weight = np.clip(sample_weight, a_min=epsilon, a_max=None)
            # do not clip sample weights that were exactly zero originally
            sample_weight[zero_weight_mask] = 0.0

            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight, random_state
            )

            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            if not np.isfinite(sample_weight_sum):
                warnings.warn(
                    "Sample weights have reached infinite values,"
                    f" at iteration {iboost}, causing overflow. "
                    "Iterations stopped. Try lowering the learning rate.",
                    stacklevel=2,
                )
                break

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        pass

        def staged_score(self, X, y, sample_weight=None):
            X = self._check_X(X)

            for y_pred in self.staged_predict(X):
                if is_classifier(self):
                    yield accuracy_score(y, y_pred, sample_weight=sample_weight)
                else:
                    yield r2_score(y, y_pred, sample_weight=sample_weight)

        @property
        def feature_importances_(self):
            if self.estimators_ is None or len(self.estimators_) == 0:
                raise ValueError(
                    "Estimator not fitted, call `fit` before `feature_importances_`."
                )

            try:
                norm = self.estimator_weights_.sum()
                return (
                        sum(
                            weight * clf.feature_importances_
                            for weight, clf in zip(self.estimator_weights_, self.estimators_)
                        )
                        / norm
                )

            except AttributeError as e:
                raise AttributeError(
                    "Unable to compute feature importances "
                    "since estimator does not have a "
                    "feature_importances_ attribute"
                ) from e

def _samme_proba(estimator, n_classes, X):
    proba = estimator.predict_proba(X)

            # Displace zero probabilities so the log is defined.
            # Also fix negative elements which may occur with
            # negative sample weights.
    np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
    log_proba = np.log(proba)

    return (n_classes - 1) * (
            log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis]
    )


class AdaBoost(ClassifierMixin, BaseWeightBoosting):
    _parameter_constraints: dict = {
        **BaseWeightBoosting._parameter_constraints,
        "algorithm": [StrOptions({"SAMME", "SAMME.R"})],
    }

    def __init__(
            self,
            estimator=None,
            *,
            n_estimators=50,
            learning_rate=1.0,
            algorithm="SAMME.R",
            random_state=None,
            base_estimator="deprecated",
    ):

        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            base_estimator=base_estimator,
        )

        self.algorithm = algorithm

    def _validate_estimator(self):
        """Check the estimator and set the estimator_ attribute."""
        super()._validate_estimator(default=DecisionTreeClassifier(max_depth=1))

        #  SAMME-R requires predict_proba-enabled base estimators
        if self.algorithm == "SAMME.R":
            if not hasattr(self.estimator_, "predict_proba"):
                raise TypeError(
                    "AdaBoostClassifier with algorithm='SAMME.R' requires "
                    "that the weak learner supports the calculation of class "
                    "probabilities with a predict_proba method.\n"
                    "Please change the base estimator or set "
                    "algorithm='SAMME' instead."
                )
        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(
                f"{self.estimator.__class__.__name__} doesn't support sample_weight."
            )

    def _boost(self, iboost, X, y, sample_weight, random_state):
        if self.algorithm == "SAMME.R":
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        else:  # elif self.algorithm == "SAMME":
            return self._boost_discrete(iboost, X, y, sample_weight, random_state)

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (
                -1.0
                * self.learning_rate
                * ((n_classes - 1.0) / n_classes)
                * xlogy(y_coding, y_predict_proba).sum(axis=1)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))
            )

        return sample_weight, 1.0, estimator_error

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier "
                    "ensemble is worse than random, ensemble "
                    "can not be fit."
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
                np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight = np.exp(
                np.log(sample_weight)
                + estimator_weight * incorrect * (sample_weight > 0)
            )

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def staged_predict(self, X):
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_

        if n_classes == 2:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(pred > 0, axis=0))

        else:
            for pred in self.staged_decision_function(X):
                yield np.array(classes.take(np.argmax(pred, axis=1), axis=0))

    def decision_function(self, X):
        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if self.algorithm == "SAMME.R":
            # The weights are all 1. for SAMME.R
            pred = sum(
                _samme_proba(estimator, n_classes, X) for estimator in self.estimators_
            )
        else:  # self.algorithm == "SAMME"
            pred = sum(
                (estimator.predict(X) == classes).T * w
                for estimator, w in zip(self.estimators_, self.estimator_weights_)
            )

        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred

    def staged_decision_function(self, X):
        check_is_fitted(self)
        X = self._check_X(X)

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        pred = None
        norm = 0.0

        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            norm += weight

            if self.algorithm == "SAMME.R":
                # The weights are all 1. for SAMME.R
                current_pred = _samme_proba(estimator, n_classes, X)
            else:  # elif self.algorithm == "SAMME":
                current_pred = estimator.predict(X)
                current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            if n_classes == 2:
                tmp_pred = np.copy(pred)
                tmp_pred[:, 0] *= -1
                yield (tmp_pred / norm).sum(axis=1)
            else:
                yield pred / norm

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        if n_classes == 2:
            decision = np.vstack([-decision, decision]).T / 2
        else:
            decision /= n_classes - 1
        return softmax(decision, copy=False)

    def predict_proba(self, X):
        check_is_fitted(self)
        n_classes = self.n_classes_

        if n_classes == 1:
            return np.ones((_num_samples(X), 1))

        decision = self.decision_function(X)
        return self._compute_proba_from_decision(decision, n_classes)

    def staged_predict_proba(self, X):
        n_classes = self.n_classes_

        for decision in self.staged_decision_function(X):
            yield self._compute_proba_from_decision(decision, n_classes)

    def predict_log_proba(self, X):
        return np.log(self.predict_proba(X))



