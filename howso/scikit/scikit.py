import inspect
from typing import Any, Dict, List, Optional, Union
import uuid

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score

from howso import engine
from howso.client import AbstractHowsoClient, HowsoPandasClient
from howso.client.exceptions import HowsoApiError, HowsoError, HowsoNotUniqueError
import howso.utilities as utils
from howso.utilities.feature_attributes import infer_feature_attributes

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
FEATURE = 'x'
ACTION = 'y'
DEFAULT_TTL = 43200000
RENAME_RETRIES = 3


class HowsoEstimator(BaseEstimator):
    """
    This class is intended for use within scikit-learn only.

    This Estimator follows scikit-learn's conventions. For access to a wider
    range of Howso capabilities, please use the client specified in the
    howso.client module.

    Parameters
    ----------
    features : dict of str: dict, default None
        The features that will predict the targets(s). Will be generated
        automatically if not specified.

        Example::

            {
                "feature_name": {
                    "parameter1" : "value1",
                    "parameter2" : "value2"
                },
                "length": { "type" : "continuous", "decimal_places": 1 },
                "width": { "type" : "continuous", "significant_digits": 4 },
                "degrees": { "type" : "continuous", "cycle_length": 360 },
                "class": { "type" : "nominal" }
            }

    targets : dict of str: dict, default None
        The target(s) to be predicted. Will be generated automatically if not
        specified.

        Example::

            {
                "`target_name`": {
                    "parameter1" : "value1",
                    "parameter2" : "value2"
                },
                "klass": { "type" : "nominal" }
            }

    client : AbstractHowsoClient, default None
        A subclass of AbstractHowsoClient used to interface with Howso.
    method : str
        One of 'classification' or 'regression'.
    verbose : boolean, default False
        A flag for verbose output.
    debug : boolean, default False
        A flag for debug output.
    ttl : int, in milliseconds
        The maximum time a server should maintain a connection open for a
        trainee when processing requests.
    client_params : dict, default None
        The parameters with which to instantiate the client.
    trainee_params : dict, default None
        The parameters with which to instantiate the trainee.

    Examples
    --------
    >>> import pandas as pd
    >>> from howso.scikit import HowsoClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> # Read in the data.
    >>> df = pd.read_csv('iris.csv')
    >>>
    >>> # Split the dataset into the feature (X) and targets (y) and convert
    >>> # the string targets into integer hashes.
    >>> X = df.drop('class', axis=1).values.astype(float)
    >>> y = df['class'].apply(hash).values.astype(int)
    >>>
    >>> # Split the dataset into an 80/20 train/test set.
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
    >>>
    >>> # Create a classifier.
    >>> howso = HowsoClassifier()
    >>>
    >>> # Fit the training data.
    >>> howso.fit(X_train, y_train)
    >>>
    >>> # Test against the reserved test data.
    >>> score = howso.score(X_test, y_test)
    >>>
    >>> # Print the resulting accuracy.
    >>> print(score)
    0.9666666666666667
    """

    def __init__(self, client: Optional[AbstractHowsoClient] = None,
                 features: Optional[Dict] = None, targets: Dict = None,
                 method: Optional[str] = None, verbose: bool = False,
                 debug: bool = False, ttl: int = DEFAULT_TTL,
                 trainee_params: Optional[Dict] = None,
                 client_params: Optional[Dict] = None):
        """Initialize HowsoEstimator."""
        if method not in [CLASSIFICATION, REGRESSION]:
            raise ValueError(f'Unsupported method {method}')

        if features is None and targets or features and targets is None:
            raise ValueError(
                "This package only supports supervised learning. Please "
                "specify both features and targets. This feature will be "
                "supported in more flexible frameworks in the future. For "
                "access to a wider range of Howso capabilities, please "
                "use the client specified in the howso.client module.")

        self.features = features
        self.targets = targets
        self.conviction_ = None
        self.method = method
        self.feature_names = []
        self.verbose = verbose
        self.debug = debug
        self.ttl = ttl
        self.client = client
        self.trainee_params = trainee_params
        self.client_params = client_params

        if client is None and client_params is None:
            self.client = HowsoPandasClient(verbose=self.verbose, debug=self.debug)
            self.client_params = self._get_client_params()
        elif client_params:
            cls = client_params["class"]
            self.client = cls(**client_params["args"])
            self.client_params = client_params
        else:
            self.client = client
            self.client_params = self._get_client_params()

        if trainee_params:
            self.trainee = trainee_params["args"]

    def __del__(self) -> None:
        """
        Clean up at garbage collection time.

        Returns
        -------
        None
        """
        self.release_resources()

    def release_resources(self):  # noqa: C901
        """
        Release trainee resources created by this estimator.

        If this estimator's trainee is named (self._trainee_name is not None)
        then we'll make an effort to persist the trainee to disk and release
        it's resources. If the data persistence policy forbids this, that call
        will return an error. Upon error, `delete_trainee()` instead.

        NOTE: Errors are handled immediately because this is the instance's
              destructor. There is no further recourse at this point.

        Returns
        -------
        None
        """
        if getattr(self, 'trainee_id', None):
            # If the user named the trainee, they'll want to use it again later
            # so persist the trainee, rather than delete it, if possible.
            try:
                trainee_name = self.trainee.name
            except AttributeError:
                trainee_name = None
            if trainee_name:
                try:
                    # Ensure we have an up-to-date persistence
                    live_trainee = engine.get_trainee(self.trainee_id)
                    if getattr(live_trainee, 'persistence', '') == 'never':
                        raise AssertionError('Trainee is not persistable.')
                    else:
                        self.trainee.release_resources()
                except Exception as e:
                    if isinstance(e, AssertionError):
                        print("The Howso estimator's trainee was not "
                              "permitted to be saved. It will be deleted.")
                    else:
                        print("An error prevented the saving of the Howso "
                              "estimator's trainee. It will be deleted.")
                    try:
                        self.trainee.delete()
                    except Exception:  # noqa: Deliberately broad
                        print("The Howso estimator could not delete "
                              "its trainee.")
                    else:
                        if self.verbose:
                            print("The Howso estimator has successfully "
                                  "deleted its trainee.")
                else:
                    if self.verbose:
                        print(f'The Howso estimator\'s trainee with name '
                              f'"{trainee_name}" and ID "{self.trainee_id}" '
                              f'was successfully released.')
            else:
                # User has no interest in saving this trainee, Just delete the
                # trainee. Be silent unless error or the verbose flag is set.
                try:
                    self.trainee.delete()
                except Exception:  # noqa: Deliberately broad
                    print("The Howso estimator did not successfully "
                          "delete its trainee.")
                else:
                    if getattr(self, 'verbose', False):
                        print("The Howso estimator has successfully "
                              "deleted its trainee.")

        elif getattr(self, 'verbose', False):
            print("The Howso estimator has no trainees to delete.")

    def _get_trainee_params(self) -> Dict:
        """
        Gets the initial parameters of `self.trainee`.

        This allows the trainee to be recreated if the estimator is cloned.
        This code was borrowed from
        `sklearn.base.BaseEstimator._get_param_names`.

        Returns
        -------
        trainee_params: mapping of string to any
            Parameter names mapped to their values.
        """
        trainee_params = {}
        trainee_class = self.trainee.__class__
        init_signature = inspect.signature(trainee_class.__init__)
        parameters = [p.name for p in init_signature.parameters.values()
                      if p.name != "self" and p.kind != p.VAR_KEYWORD]
        trainee_params["args"] = {p: getattr(self.trainee_id, p, None) for p in parameters}
        return trainee_params

    def _get_client_params(self) -> Dict:
        """
        Get the initial parameters of `self.client`.

        This allows the client to be recreated if the estimator is cloned. This
        code was borrowed from `sklearn.base.BaseEstimator._get_param_names`.

        Returns
        -------
        client_params: mapping of string to any
            Parameter names mapped to their values.
        """
        client_params = dict()
        client_class = self.client.__class__
        init_signature = inspect.signature(client_class.__init__)
        parameters = [p.name for p in init_signature.parameters.values()
                      if p.name != "self" and p.kind != p.VAR_KEYWORD]
        client_params["args"] = {p: getattr(self.client, p, None) for p in parameters}
        client_params["class"] = client_class
        return client_params

    @property
    def trainee_id(self) -> Union[str, None]:
        """Return the trainee's ID, if possible."""
        try:
            return self.trainee.id
        except AttributeError:
            return None

    @property
    def trainee_name(self) -> Union[str, None]:
        """Return the trainee name (getter)."""
        return self.trainee.name

    @trainee_name.setter
    def trainee_name(self, name: str = ''):
        """
        Setter for the `trainee_name` property.

        The name must be unique. If it is not, a ValueError is raised.

        Raises
        ------
        HowsoNotUniqueError
            When there is an attempt to set a name that is not unique.
        HowsoError
            On any other issue.
        """
        old_name = self.trainee.name
        self.trainee.name = name
        try:
            self.trainee.update()
        except Exception as exc:  # noqa: Deliberately broad
            self.trainee.name = old_name
            if (
                    isinstance(exc, HowsoApiError) and
                    getattr(exc, 'status', 0) == 409
            ):
                raise HowsoNotUniqueError(
                    f'Unable to set the name of the Howso estimator\'s '
                    f'trainee to: "{name}". Please use a unique name and '
                    f'try again.') from exc
            else:
                raise HowsoError(
                    f'Unable to set the name of the Howso estimator\'s '
                    f'trainee to: "{name}".') from exc
        else:
            if self.verbose:
                print(f'The trainee name was successfully set '
                      f'to "{self.trainee.name}".')

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        This code is taken from the source of `sklearn.base.BaseEstimator` and
        lightly modified to avoid calling the `get_params` method
        of `self.trainee`.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None

            if key == "client":
                # Never recurse into `client`.
                continue
            elif deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
                out[key] = value
        return out

    def fit(self, X, y, analyze=True) -> "HowsoEstimator":
        """
        Fit a model with Howso.

        Parameters
        -----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data
        y : numpy.ndarray, shape (n_samples,)
            Target. Will be cast to X's dtype if necessary
        analyze : bool, default=True
            A flag to not analyze the trainee by default

                - A user may plan to call analyze themselves after fit() to specify parameters

        Returns
        -------
        HowsoEstimator
            self
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        X, y = utils.align_data(X, y)

        if self.features is None:
            if self.verbose:
                print('Generating features x0, x1 ... xn and target y.')
            self._generate_features_and_targets(X)

        # set/update self.feature_names and self.target_names
        self._store_feature_and_target_names()

        # In Howso 'features' can be either predictors or targets.
        if not self.trainee_params:

            self.trainee = engine.Trainee(
                features={**self.features, **self.targets},
                metadata={'scikit-trainee': True},
                client=self.client
            )

            self.trainee_params = self._get_trainee_params()

        self.persistence = self.trainee.persistence

        self._train(X, y)

        if analyze:
            if self.verbose:
                print('Analyzing trainee')
            self.analyze()

        return self

    def partial_fit(self, X, y):
        """
        Add data to an existing Howso model.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data
        y : numpy.ndarray, shape (n_samples,)
            Target. Will be cast to X's dtype if necessary
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        X, y = utils.align_data(X, y)
        self._train(X, y)

    def predict(self, X) -> np.ndarray:
        """
        Make predictions using Howso.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            The predicted values based on the feature values provided.
        """
        X = utils.align_data(X)
        cases = X.tolist()
        cases = utils.replace_nan_with_none(cases)

        results = self.trainee.react(
            contexts=cases,
            action_features=self.target_names,
            context_features=self.feature_names
        )

        # Convert to dictionary, new trainee outputs a pd.DataFrame
        results['action'] = results['action'].to_dict('records')
        results['action'] = utils.replace_none_with_nan(results['action'])
        action_values = pd.DataFrame(results['action']).values
        out = np.array(action_values).astype(float)
        out.shape = (out.shape[0],)
        if np.isnan(np.sum(out)):
            print('Server returned NaN with predictions.')
        return out

    def score(self, X, y) -> float:
        """
        Score Howso.

        For classifiers, accuracy is calculated.
        For regressors, R^2 is calculated.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Test samples.

        y : numpy.ndarray, shape (n_samples) or (n_samples, n_outputs)
            True values for X.

        Returns
        -------
        float
            The mean squared error or accuracy

        """
        X, y = utils.align_data(X, y)
        if self.method == CLASSIFICATION:
            return accuracy_score(y, self.predict(X))
        return r2_score(y, self.predict(X))

    def react_into_features(
        self,
        features=None,
        *,
        distance_contribution=False,
        familiarity_conviction_addition=False,
        familiarity_conviction_removal=False,
        influence_weight_entropy=False,
        p_value_of_addition=False,
        p_value_of_removal=False,
        similarity_conviction=False,
        use_case_weights=None,
        weight_feature=None,
    ) -> None:
        """
        Calculate conviction and other data and stores them into features.

        Parameters
        ----------
        features : list of str
            A list of the feature names to use when calculating conviction.
        distance_contribution : bool or str, default False
            The name of the feature to store distance contribution. If set to
            True the values will be stored to the feature
            'distance_contribution'.
        familiarity_conviction_addition : bool or str, default False
            The name of the feature to store conviction of addition values. If
            set to True the values will be stored to the feature
            'familiarity_conviction_addition'.
        familiarity_conviction_removal : bool or str, default False
            The name of the feature to store conviction of removal values. If
            set to True the values will be stored to the feature
            'familiarity_conviction_removal'.
        influence_weight_entropy : bool or str, default False
            The name of the feature to store influence weight entropy values in.
            If set to True, the values will be stored in the feature
            'influence_weight_entropy'.
        p_value_of_addition : bool or str, default False
            The name of the feature to store p value of addition values. If set
            to True the values will be stored to the feature
            'p_value_of_addition'.
        p_value_of_removal : bool or str, default False
            The name of the feature to store p value of removal values. If set
            to True the values will be stored to the feature
            'p_value_of_removal'.
        similarity_conviction : bool or str, default False
            The name of the feature to store similarity conviction
            values. If set to True the values will be stored to the feature
            'similarity_conviction'.
        use_case_weights : bool, optional
            When True, will scale influence weights by each case's
            `weight_feature` weight. If unspecified, case weights will
            be used if the Trainee has them.
        weight_feature : str, optional
            Name of feature whose values to use as case weights.
            When left unspecified uses the internally managed case weight.
        """
        # If features is not provided, use all features by default.
        if features is None:
            features = self.feature_names

        # Call conviction store on the trainee.
        self.trainee.react_into_features(
            features=features,
            distance_contribution=distance_contribution,
            familiarity_conviction_addition=familiarity_conviction_addition,
            familiarity_conviction_removal=familiarity_conviction_removal,
            influence_weight_entropy=influence_weight_entropy,
            p_value_of_addition=p_value_of_addition,
            p_value_of_removal=p_value_of_removal,
            similarity_conviction=similarity_conviction,
            use_case_weights=use_case_weights,
            weight_feature=weight_feature,
        )

    def describe_prediction(self, X, details=None) -> Dict:
        """
        Describe a prediction in detail.

        Parameters
        ----------
        X : numpy.ndarray
            Feature values.

        details: dict, default None
            (Optional) If details are specified, the response will
            contain the requested explanation data along with the reaction.
            Below are the valid keys and data types for the different audit
            details. Omitted keys, values set to None, or False values for
            Booleans will not be included in the audit data returned.

            - boundary_cases : bool, optional
                If True, outputs an automatically determined (when
                'num_boundary_cases' is not specified) relevant number of
                boundary cases. Uses both context and action features of the
                reacted case to determine the counterfactual boundary based on
                action features, which maximize the dissimilarity of action
                features while maximizing the similarity of context features.
                If action features aren't specified, uses familiarity conviction
                to determine the boundary instead.
            - boundary_cases_familiarity_convictions : bool, optional
                If True, outputs familiarity conviction of addition for each of
                the boundary cases.
            - case_contributions_full : bool, optional
                If true outputs each influential case's differences between the
                predicted action feature value and the predicted action feature
                value if each individual case were not included. Uses only the
                context features of the reacted case to determine that area.
                Uses full calculations, which uses leave-one-out for cases for
                computations.
            - case_contributions_robust : bool, optional
                If true outputs each influential case's differences between the
                predicted action feature value and the predicted action feature
                value if each individual case were not included. Uses only the
                context features of the reacted case to determine that area.
                Uses robust calculations, which uses uniform sampling from
                the power set of all combinations of cases.
            - case_feature_residuals_full : bool, optional
                If True, outputs feature residuals for all (context and action)
                features for just the specified case. Uses leave-one-out for
                each feature, while using the others to predict the left out
                feature with their corresponding values from this case. Uses
                full calculations, which uses leave-one-out for cases for
                computations.
            - case_feature_residuals_robust : bool, optional
                If True, outputs feature residuals for all (context and action)
                features for just the specified case. Uses leave-one-out for
                each feature, while using the others to predict the left out
                feature with their corresponding values from this case. Uses
                robust calculations, which uses uniform sampling from the power
                set of features as the contexts for predictions.
            - case_mda_robust : bool, optional
                If True, outputs each influential case's mean decrease in
                accuracy of predicting the action feature in the local model
                area, as if each individual case were included versus not
                included. Uses only the context features of the reacted case to
                determine that area. Uses robust calculations, which uses
                uniform sampling from the power set of all combinations of cases.
            - case_mda_full : bool, optional
                If True, outputs each influential case's mean decrease in
                accuracy of predicting the action feature in the local model
                area, as if each individual case were included versus not
                included. Uses only the context features of the reacted case to
                determine that area. Uses full calculations, which uses
                leave-one-out for cases for  computations.
            - categorical_action_probabilities : bool, optional
                If True, outputs probabilities for each class for the action.
                Applicable only to categorical action features.
            - derivation_parameters : bool, optional
                If True, outputs a dictionary of the parameters used in the
                react call. These include k, p, distance_transform,
                feature_weights, feature_deviations, nominal_class_counts,
                and use_irw.

                - k: the number of cases used for the local model.
                - p: the parameter for the Lebesgue space.
                - distance_transform: the distance transform used as an
                  exponent to convert distances to raw influence weights.
                - feature_weights: the weight for each feature used in the
                  distance metric.
                - feature_deviations: the deviation for each feature used in
                  the distance metric.
                - nominal_class_counts: the number of unique values for each
                  nominal feature. This is used in the distance metric.
                - use_irw: a flag indicating if feature weights were
                  derived using inverse residual weighting.
            - distance_contribution : bool, optional
                If True, outputs the distance contribution (expected total
                surprisal contribution) for the reacted case. Uses both context
                and action feature values.
            - distance_ratio : bool, optional
                If True, outputs the ratio of distance (relative surprisal)
                between this reacted case and its nearest case to the minimum
                distance (relative surprisal) in between the closest two cases
                in the local area. All distances are computed using only the
                specified context features.
            - feature_contributions_robust : bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context were not in the
                model for all context features in the local model area Uses
                robust calculations, which uses uniform sampling from the power
                set of features as the contexts for predictions. Directional feature
                contributions are returned under the key
                'directional_feature_contributions_robust'.
            - feature_contributions_full : bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context were not in the
                model for all context features in the local model area. Uses
                full calculations, which uses leave-one-out for cases for
                computations. Directional feature contributions are returned
                under the key 'directional_feature_contributions_full'.
            - case_feature_contributions_robust: bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context feature were not
                in the model for all context features in this case, using only
                the values from this specific case. Uses
                robust calculations, which uses uniform sampling from the power
                set of features as the contexts for predictions.
                Directional case feature contributions are returned under the
                'case_directional_feature_contributions_robust' key.
            - case_feature_contributions_full: bool, optional
                If True outputs each context feature's absolute and directional
                differences between the predicted action feature value and the
                predicted action feature value if each context feature were not
                in the model for all context features in this case, using only
                the values from this specific case. Uses
                full calculations, which uses leave-one-out for cases for
                computations. Directional case feature
                contributions are returned under the
                'case_directional_feature_contributions_full' key.
            - feature_mda_robust : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature given the context.
                Uses only the context features of the reacted case to determine
                that area. Uses robust calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - feature_mda_full : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature given the context.
                Uses only the context features of the reacted case to determine
                that area. Uses full calculations, which uses leave-one-out
                for cases for computations.
            - feature_mda_ex_post_robust : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature as an explanation detail
                given that the specified prediction was already made as
                specified by the action value. Uses both context and action
                features of the reacted case to determine that area. Uses
                robust calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - feature_mda_ex_post_full : bool, optional
                If True, outputs each context feature's mean decrease in
                accuracy of predicting the action feature as an explanation detail
                given that the specified prediction was already made as
                specified by the action value. Uses both context and action
                features of the reacted case to determine that area. Uses
                full calculations, which uses leave-one-out for cases for
                computations.
            - features : list of str, optional
                A list of feature names that specifies for what features will
                per-feature details be computed (residuals, contributions,
                mda, etc.). This should generally preserve compute, but will
                not when computing details robustly. Details will be computed
                for all context and action features if this value is not
                specified.
            - feature_residual_robust : bool, optional
                If True, outputs feature residuals for all (context and action)
                features locally around the prediction. Uses only the context
                features of the reacted case to determine that area. Uses robust
                calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - feature_residuals_full : bool, optional
                If True, outputs feature residuals for all (context and action)
                features locally around the prediction. Uses only the context
                features of the reacted case to determine that area. Uses
                full calculations, which uses leave-one-out for cases for computations.
            - hypothetical_values : dict, optional
                A dictionary of feature name to feature value. If specified,
                shows how a prediction could change in a what-if scenario where
                the influential cases' context feature values are replaced with
                the specified values.  Iterates over all influential cases,
                predicting the action features each one using the updated
                hypothetical values. Outputs the predicted arithmetic over the
                influential cases for each action feature.
            - influential_cases : bool, optional
                If True, outputs the most influential cases and their influence
                weights based on the surprisal of each case relative to the
                context being predicted among the cases. Uses only the context
                features of the reacted case.
            - influential_cases_familiarity_convictions :  bool, optional
                If True, outputs familiarity conviction of addition for each of
                the influential cases.
            - influential_cases_raw_weights : bool, optional
                If True, outputs the surprisal for each of the influential
                cases.
            - case_feature_residual_convictions_robust : bool, optional
                If True, outputs this case's feature residual convictions for
                the region around the prediction. Uses only the context
                features of the reacted case to determine that region.
                Computed as: region feature residual divided by case feature
                residual. Uses robust calculations, which uses uniform sampling
                from the power set of features as the contexts for predictions.
            - case_feature_residual_convictions_full : bool, optional
                If True, outputs this case's feature residual convictions for
                the region around the prediction. Uses only the context
                features of the reacted case to determine that region.
                Computed as: region feature residual divided by case feature
                residual. Uses full calculations, which uses leave-one-out
                for cases for computations.
            - most_similar_cases : bool, optional
                If True, outputs an automatically determined (when
                'num_most_similar_cases' is not specified) relevant number of
                similar cases, which will first include the influential cases.
                Uses only the context features of the reacted case.
            - num_boundary_cases : int, optional
                Outputs this manually specified number of boundary cases.
            - num_most_similar_cases : int, optional
                Outputs this manually specified number of most similar cases,
                which will first include the influential cases.
            - num_most_similar_case_indices : int, optional
                Outputs this specified number of most similar case indices when
                'distance_ratio' is also set to True.
            - num_robust_influence_samples_per_case : int, optional
                Specifies the number of robust samples to use for each case.
                Applicable only for computing robust feature contributions or
                robust case feature contributions. Defaults to 2000. Higher
                values will take longer but provide more stable results.
            - observational_errors : bool, optional
                If True, outputs observational errors for all features as
                defined in feature attributes.
            - outlying_feature_values : bool, optional
                If True, outputs the reacted case's context feature values that
                are outside the min or max of the corresponding feature values
                of all the cases in the local model area. Uses only the context
                features of the reacted case to determine that area.
            - prediction_stats : bool, optional
                When true outputs feature prediction stats for all (context
                and action) features locally around the prediction. The stats
                returned  are ("r2", "rmse", "spearman_coeff", "precision",
                "recall", "accuracy", "mcc", "confusion_matrix", "missing_value_accuracy").
                Uses only the context features of the reacted case to determine that area.
                Uses full calculations, which uses leave-one-out context features for
                computations.
            - selected_prediction_stats : list, optional
                List of stats to output. When unspecified, returns all except the confusion matrix. Allowed values:

                - all : Returns all the the available prediction stats, including the confusion matrix.
                - accuracy : The number of correct predictions divided by the
                  total number of predictions.
                - confusion_matrix : A sparse map of actual feature value to a map of
                  predicted feature value to counts.
                - mae : Mean absolute error. For continuous features, this is
                  calculated as the mean of absolute values of the difference
                  between the actual and predicted values. For nominal features,
                  this is 1 - the average categorical action probability of each case's
                  correct classes. Categorical action probabilities are the probabilities
                  for each class for the action feature.
                - mda : Mean decrease in accuracy when each feature is dropped
                  from the model, applies to all features.
                - feature_mda_permutation_full : Mean decrease in accuracy that used
                  scrambling of feature values instead of dropping each
                  feature, applies to all features.
                - precision : Precision (positive predictive) value for nominal
                  features only.
                - r2 : The r-squared coefficient of determination, for
                  continuous features only.
                - recall : Recall (sensitivity) value for nominal features only.
                - rmse : Root mean squared error, for continuous features only.
                - spearman_coeff : Spearman's rank correlation coefficient,
                  for continuous features only.
                - mcc : Matthews correlation coefficient, for nominal features only.
            - similarity_conviction : bool, optional
                If True, outputs similarity conviction for the reacted case.
                Uses both context and action feature values as the case values
                for all computations. This is defined as expected (local)
                distance contribution divided by reacted case distance
                contribution.
            - generate_attempts : bool, optional
                If True outputs the number of attempts taken to generate each
                case. Only applicable when 'generate_new_cases' is "always" or
                "attempt".

            >>> details = {'num_most_similar_cases': 5,
            ...            'feature_residuals_full': True}


        Returns
        -------
        dict
            Format of::

                {
                    'action': list of dicts of action_features -> action_values,
                    'details': dict with requested audit data
                }

        """
        if details is None:
            details = {
                'num_boundary_cases': 3,
                'case_feature_residuals_robust': True,
                'feature_mda_robust': True,
                'feature_residuals_robust': True,
                'influential_cases': True,
                'num_most_similar_cases': 3,
            }

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.ndim == 1:
            X = np.reshape(X, (1, X.shape[0]))

        if len(X.shape) > 1:
            assert X.shape[1] == len(self.feature_names), 'Number of feature values do not match number of ' \
                                                          'feature names.'
        elif len(X.shape) == 1:
            assert X.shape[0] == len(self.feature_names), 'Number of feature values do not match number of ' \
                                                          'feature names.'
        else:
            assert False, 'Invalid data dimensions.'

        context = X.tolist()
        context = utils.replace_nan_with_none(list(context))
        if details is None:
            audit_data = self.trainee.react(
                contexts=context,
                action_features=self.target_names,
                context_features=self.feature_names
            )
        else:
            audit_data = self.trainee.react(
                contexts=context,
                action_features=self.target_names,
                context_features=self.feature_names,
                details=details
            )

        # Convert to Dictionary
        audit_data['action'] = audit_data['action'].to_dict('records')
        audit_data['action'] = utils.replace_none_with_nan(audit_data['action'])
        return audit_data

    def get_feature_conviction(self, features=None) -> Dict:
        """
        Gets the conviction of the features in a model.

        Parameters
        ----------
        features : str or list of str
            Features to return conviction values for.

        Returns
        -------
        dict
            A map of feature convictions and contributions.
        """
        ret = self.trainee.get_feature_conviction(
            features=self.feature_names,
            action_features=self.target_names,
        )

        feature_conviction = ret.to_dict()

        if features is not None:
            if isinstance(features, str):
                features = [features]

            filtered = {}
            for k, v in feature_conviction.items():
                v = {fkey: fval for fkey, fval in v.items() if fkey in features}
                filtered[k] = v
            feature_conviction = filtered

        return feature_conviction["familiarity_conviction_addition"]

    def get_case_conviction(self, X, features=None) -> List:
        """
        Return case conviction.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data
        features : str or list of str
            A list of feature names to calculate convictions.

        Returns
        -------
        list
            The conviction of the cases. Ex: [1.0, 3.2, 0.4]
        """
        # restructure X as needed
        if hasattr(X, "tolist"):
            X = X.tolist()

        if type(X) is not list:
            raise HowsoError("Cases must be a list.")
        elif type(X[0]) is not list:
            X = [X]

        new_case_groups = []
        for group in X:
            if type(group[0]) is not list:
                group = [group]
            new_case_groups.append(group)

        if features is None:
            features = self.feature_names
        elif isinstance(features, str):
            features = [features]

        ret = self.trainee.react_group(
            new_case_groups,
            features=features
        )
        return [case['familiarity_conviction_addition'] for case in ret]

    def _set_random_name(self, retries: int = RENAME_RETRIES):
        """
        Helper to randomly set trainee name with retries.

        Raises
        ------
        HowsoNotUniqueError:
            If unable to set trainee name, even after retrying.
        Exception:
            May raise other exceptions.
        """
        last_exception = None
        for _ in range(max(1, retries)):
            try:
                self.trainee_name = f'howso-estimator-{uuid.uuid4()}'
            except HowsoNotUniqueError as exception:
                last_exception = exception
                continue
            else:
                break
        else:
            if last_exception:
                raise last_exception

    def __getstate__(self) -> Dict:
        """
        Returns the state of this object (self.__dict__).

        Additionally saves the model on the Howso server for later loading
        when unpickling.

        If this trainee has not already been named, then this method will set
        a randomly generated one. This is done so that the destructor will
        `release_trainee_resources`, otherwise it will `delete_trainee`.

        Returns
        -------
        dict
            self.__dict__

        Raises
        ------
        HowsoNotUniqueError:
            If unable to set the trainee name w/up to RENAME_RETRIES retries.
        Exception:
            if unable to persist the trainee.
        """
        if not self.trainee.name:
            # This may raise, but let it.
            self._set_random_name()

        self.trainee.persist()
        return self.__dict__

    def __setstate__(self, state: Dict):
        """
        Receives the state of the object when unpickling.

        Explicitly sets the object state. Additionally calls `load` on the
        Howso server to load the model.

        Parameters
        ----------
        state : dict
            The state of the HowsoEstimator.
        """
        for attr in state:
            setattr(self, attr, state[attr])
        self.load(self.trainee_id)

    def delete(self):
        """Delete this trainee from the howso cloud service."""
        if self.trainee_id:
            self.trainee.delete()

    def save(self) -> None:
        """
        Persist the trainee.

        By default model resources are released after a short period of time.
        This method saves the model persistently to allow releasing trainee
        resources while keeping the model available for use later.

        If this trainee has not already been named, then this method will set
        a randomly generated one.

        Raises
        ------
        HowsoNotUniqueError:
            If unable to set the trainee name w/up to RENAME_RETRIES retries.
        Exception:
            if unable to persist the trainee.
        """
        if not self.trainee.name:
            # This may raise, but let it.
            self._set_random_name()

        if self.persistence == 'never':
            raise HowsoError("The Howso estimator's trainee was not "
                             "permitted to be saved.")
        else:
            self.trainee.persist()

    def load(self, trainee_id: str):
        """
        Load a model from the server.

        Parameters
        ----------
        trainee_id : str
            Id of the trainee. (can be obtained from this class).

        """
        self._store_feature_and_target_names()
        self.trainee = self.trainee.get_trainee(trainee_id=trainee_id)
        self.trainee.acquire_resources()

    def _train(self, X: np.ndarray, y: np.ndarray):
        """
        Train a Howso model.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data
        y : numpy.ndarray, shape (n_samples,)
            Target. Will be cast to X's dtype if necessary
        """
        self.trainee.train(cases=build_cases(X, y), features=self.feature_names + self.target_names)

    def _store_feature_and_target_names(self):
        self._store_feature_names()
        self._store_target_names()

    def _store_feature_names(self):
        self.feature_names = []
        for fname in self.features.keys():
            self.feature_names.append(str(fname))

    def _store_target_names(self):
        self.target_names = []
        for tname in self.targets.keys():
            self.target_names.append(str(tname))

    def analyze(self, seed=None, **kwargs):
        """
        Analyze a trainee.

        Parameters
        ----------
        seed : int, optional
            A random seed.
        **kwargs
            Refer to docstring in howso.client.analyze method for complete
            reference of all parameters
        """
        if not kwargs:
            kwargs = dict()

        if seed is not None:
            self.trainee.set_random_seed(seed)

        if kwargs.get('targeted_model', "") == 'targetless':
            # for targetless analyze, override 'action_features'
            # and 'context_features'
            kwargs['action_features'] = []
            kwargs['context_features'] = self.feature_names + self.target_names
        else:
            if 'action_features' not in kwargs:
                kwargs['action_features'] = self.target_names

            if 'context_features' not in kwargs:
                kwargs['context_features'] = self.feature_names

        self.trainee.analyze(**kwargs)

    def partial_unfit(self, precision: str, num_cases: int,
                      criteria: Optional[Dict] = None):
        """
        Remove a training case from a trainee.

        The training case will be completely purged from the model and the
        model will behave as if it had never been trained with this training
        case.

        Parameters
        ----------
        precision : str
            The precision to use when removing the case. Options are 'exact' or
            'similar'.
        num_cases : int
            The number of cases to remove; minimum 1 case must be removed.
        criteria : dict, default None
            The condition map to select the cases to remove that meet all the
            provided conditions. Keys - features, values - one of | null (must
            have the feature) | a value (must match exactly) | an array of two
            values (a range, feature values must be between)
        """
        self.trainee.remove_cases(precision, num_cases, criteria)

    def feature_add(self, feature: str = None,
                    value: Union[float, int, str, None] = None):
        """
        Add a feature to a trainee.

        Parameters
        ----------
        feature : str, optional
            The name of the feature. Will be generated automatically
            if not specified.
        value : int or float or str, optional
            The value to populate the feature with.
        """
        if feature is None:
            feature = self._generate_new_feature_name()
        elif feature in self.feature_names:
            raise HowsoError(f"Feature name '{feature}' already exists in "
                             f"this trainee.")

        self.trainee.add_feature(feature, feature_value=value)

    def feature_remove(self, feature: Optional[str] = None):
        """
        Remove a feature from a trainee.

        Parameters
        ----------
        feature : str, default None
            Optional. The name of the feature to remove. Will quietly do
            nothing if the feature was not found.
        """
        # Question: can a target be removed?
        try:
            feature_index = self.feature_names.index(feature)
        except Exception:  # noqa: Intentionally broad.
            feature_index = None

        if feature_index is not None:
            del self.feature_names[feature_index]
        if feature in self.features:
            del self.features[feature]
            self.trainee.remove_feature(feature)

    def _generate_features_and_targets(self, X: np.ndarray):
        """
        Generate feature and target names.

        Parameters
        ----------
        X : numpy.ndarray
            Feature values ndarray.
        """
        # Use infer_feature_attributes to populate features; only accepts pandas df
        df = pd.DataFrame(data=X)
        if isinstance(df.columns, pd.RangeIndex):
            # Convert numeric columns to string
            # Don't use inplace/copy here to support multiple versions of pandas
            df = df.set_axis([str(i) for i in range(X.shape[1])], axis=1)
        self.features = infer_feature_attributes(df)
        targets = {}
        if self.method == CLASSIFICATION:
            targets[ACTION] = {"type": "nominal"}
        else:
            targets[ACTION] = {"type": "continuous"}
        self.targets = targets

    def _generate_new_feature_name(self):
        """Generate a new feature name."""
        name = FEATURE + str(len(self.features))
        self.features[name] = {"type": "continuous"}
        self._store_feature_names()
        return name


def build_cases(X: np.ndarray, y: np.ndarray) -> List:
    """
    Transform the cases from the feature and target ndarrays to a list of case values.

    Parameters
    ----------
    X : numpy.ndarray
        Feature values ndarray.
    y : numpy.ndarray
        Target values ndarray.

    Returns
    -------
    List
        A multi-dimensional List.
    """
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=1)

    dat = np.concatenate([X, y], axis=1).tolist()
    dat = utils.replace_nan_with_none(dat)

    return dat


class HowsoClassifier(HowsoEstimator):
    """
    A HowsoEstimator for classification analysis.

    Parameters
    ----------
    features : dict of str: dict, default None
        The features that will predict the targets(s). Will be generated
        automatically if not specified.

        Example::

            {
                "feature_name": {
                    "parameter1" : "value1",
                    "parameter2" : "value2"
                },
                "length": { "type" : "continuous", "decimal_places": 1 },
                "width": { "type" : "continuous", "significant_digits": 4 },
                "degrees": { "type" : "continuous", "cycle_length": 360 },
                "class": { "type" : "nominal" }
            }

    targets : dict of str: dict, default None
        The target(s) to be predicted. Will be generated automatically if not
        specified.

        Example::

            {
                "target_name": {
                    "parameter1" : "value1",
                    "parameter2" : "value2"
                },
                "klass": { "type" : "nominal" }
            }

    client : AbstractHowsoClient, default None
        A subclass of AbstractHowsoClient used to interface with Howso.
    verbose : boolean, default False
        A flag for verbose output.
    debug : boolean, default False
        A flag for debug output.
    ttl : int, in milliseconds
        The maximum time a server should maintain a connection open for a
        trainee when processing requests.
    client_params : dict, default None
        The parameters with which to instantiate the client.
    trainee_params : dict, default None
        The parameters with which to instantiate the client. Intended for use by `HowsoEstimator.get_params`.
    """

    def __init__(self, client: Optional[AbstractHowsoClient] = None,
                 features: Optional[Dict] = None,
                 targets: Optional[Dict] = None,
                 verbose: bool = False,
                 debug: bool = False, ttl: int = DEFAULT_TTL,
                 client_params: Optional[Dict] = None,
                 trainee_params: Optional[Dict] = None):
        """Initialize HowsoClassifier."""
        super(HowsoClassifier, self).__init__(client=client,
                                              features=features,
                                              targets=targets,
                                              verbose=verbose, debug=debug,
                                              ttl=ttl,
                                              method=CLASSIFICATION,
                                              client_params=client_params,
                                              trainee_params=trainee_params)
        self.classes_ = np.empty((0,), dtype=str)

    def fit(self, X: np.ndarray, y: np.ndarray, analyze: bool = True):
        """
        Fit a model with Howso.

        Parameters
        -----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data
        y : numpy.ndarray, shape (n_samples,)
            Target. Will be cast to X's dtype if necessary
        analyze : bool, default=True
            (Optional) If trainee should be analyzed.

                - a user may plan to call analyze themselves after fit() to specify parameters

        Returns
        -------
        HowsoEstimator
            self
        """
        HowsoEstimator.fit(self, X, y, analyze)
        # To keep fit as an idempotent operation, clear out the classes variable before populating them.
        self.classes_ = np.empty((0,), dtype=str)
        self._populate_classes(y)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Adds data to an existing Howso model.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data
        y : numpy.ndarray, shape (n_samples,)
            Target. Will be cast to X's dtype if necessary
        """
        HowsoEstimator.partial_fit(self, X, y)
        self._populate_classes(y)

    def load(self, trainee_id: str):
        """
        Load the trainee and re-populates the `classes_` variable.

        This is based on the available classes in the loaded trainee.

        Parameters
        ----------
        trainee_id : str
            The id of the trainee.
        """
        HowsoEstimator.load(self, trainee_id)
        self.classes_ = np.empty((0,), dtype=str)
        cases = self.trainee.get_cases()
        for case in cases.cases:
            if str(case[-1]) in self.classes_:
                self.classes_ = np.append(self.classes_, [case[-1]])
        self.classes_ = np.sort(self.classes_, axis=None)

    def _populate_classes(self, y: np.ndarray):
        """
        Populates the self.classes_ variable.

        This is used in classifiers to get the list of available classes.

        Parameters
        ----------
        y : numpy.ndarray, shape (n_samples,)
            A numpy array with the target values.
        """
        for v in list(y):
            if str(v) not in self.classes_:
                self.classes_ = np.append(self.classes_, v)
        self.classes_ = np.sort(self.classes_, axis=None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the label
        of classes.

        For a multi_class problem, if multi_class is set to be “multinomial”
        the softmax function is used to find the predicted probability of each
        class. Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function
        and normalize these values across all the classes.

        NOTE: Only works with single target models at this time.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Data

        Returns
        -------
        numpy.ndarray, shape (n_samples, n_classes)
            The probabilities of the classes for the given prediction.
        """
        if len(self.targets) != 1:
            raise AttributeError("predict_proba is only implemented for single-target models")
        target = self.target_names[0]
        result = self.describe_prediction(X, details={"categorical_action_probabilities": True})
        proba = []
        for exp in result['details']['categorical_action_probabilities']:
            sub_probas = []
            for clss in self.classes_:
                sub_probas += [0 if clss not in exp[target] else
                               exp[target][clss]]
            proba += [sub_probas]
        ret = np.array(proba, dtype=float)
        return ret


class HowsoRegressor(HowsoEstimator):
    """
    A HowsoEstimator for regression analysis.

    Parameters
    ----------
    features : dict of str: dict, default None
        The features that will predict the targets(s). Will be generated
        automatically if not specified.

        Example::

            {
                "feature_name": {
                    "parameter1" : "value1",
                    "parameter2" : "value2"
                },
                "length": { "type" : "continuous", "decimal_places": 1 },
                "width": { "type" : "continuous", "significant_digits": 4 },
                "degrees": { "type" : "continuous", "cycle_length": 360 },
                "class": { "type" : "nominal" }
            }

    targets : dict of str: dict, default None
        The target(s) to be predicted. Will be generated automatically if not
        specified.

        Example::

            {
                "target_name": {
                    "parameter1" : "value1",
                    "parameter2" : "value2"
                },
                "klass": { "type" : "nominal" }
            }

    client : AbstractHowsoClient, default None
        A subclass of AbstractHowsoClient used to interface with Howso.
    verbose : boolean, default False
        A flag for verbose output.
    debug : boolean, default False
        A flag for debug output.
    ttl : int, in milliseconds
        The maximum time a server should maintain a connection open for a
        trainee when processing requests.
    client_params : dict, default None
        The parameters with which to instantiate the client.
    trainee_params : dict, default None
        The parameters with which to instantiate the client. Intended for use
        by `HowsoEstimator.get_params`.
    """

    def __init__(self, client=None, features: Optional[Dict] = None,
                 targets: Optional[Dict] = None,
                 verbose: bool = False,
                 debug: bool = False,
                 ttl: int = DEFAULT_TTL,
                 client_params: Optional[Dict] = None,
                 trainee_params: Optional[Dict] = None):
        """Initialize a HowsoRegressor."""
        super(HowsoRegressor, self).__init__(client=client,
                                             features=features,
                                             targets=targets,
                                             verbose=verbose, debug=debug,
                                             ttl=ttl, method=REGRESSION,
                                             client_params=client_params,
                                             trainee_params=trainee_params)
