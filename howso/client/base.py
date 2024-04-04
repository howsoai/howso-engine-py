from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

from howso.utilities.reaction import Reaction
from pandas import DataFrame, Index

if TYPE_CHECKING:
    from .configuration import HowsoConfiguration
    from howso.openapi.models import (
        Cases,
        Metrics,
        TraineeInformation,
    )


class AbstractHowsoClient(ABC):
    """The base definition of the Howso client interface."""

    configuration: "HowsoConfiguration"

    @property
    @abstractmethod
    def trainee_cache(self):
        """Return the trainee cache."""

    @property
    @abstractmethod
    def active_session(self):
        """Return the active session."""

    @property
    @abstractmethod
    def train_initial_batch_size(self) -> int:
        """The default number of cases in the first train batch."""

    @property
    @abstractmethod
    def react_initial_batch_size(self) -> int:
        """The default number of cases in the first react batch."""

    @abstractmethod
    def get_version(self):
        """Get Howso version."""

    @abstractmethod
    def create_trainee(
        self, trainee, *,
        library_type=None,
        max_wait_time=None,
        overwrite_trainee=False,
        resources=None
    ):
        """Create a trainee on the Howso service."""

    @abstractmethod
    def update_trainee(self, trainee):
        """Update an existing trainee in the Howso service."""

    @abstractmethod
    def get_trainee(self, trainee_id):
        """Get an existing trainee from the Howso service."""

    @abstractmethod
    def get_trainee_information(self, trainee_id) -> "TraineeInformation":
        """Get information about the trainee."""

    @abstractmethod
    def get_trainee_metrics(self, trainee_id) -> "Metrics":
        """Get metric information for a trainee."""

    @abstractmethod
    def get_trainees(self, search_terms=None) -> List:
        """Return a list of all accessible trainees."""

    @abstractmethod
    def delete_trainee(self, trainee_id, file_path=None):
        """Delete a trainee in the Howso service."""

    @abstractmethod
    def copy_trainee(
        self, trainee_id, new_trainee_name=None, *,
        library_type=None,
        resources=None,
    ):
        """Copy a trainee in the Howso service."""

    @abstractmethod
    def copy_subtrainee(
        self, trainee_id, new_trainee_name, *,
        target_name_path=None, target_id=None,
        source_name_path=None, source_id=None
    ):
        """Copy a subtrainee in trainee's hierarchy."""

    @abstractmethod
    def acquire_trainee_resources(self, trainee_id, *, max_wait_time=None):
        """Acquire resources for a trainee in the Howso service."""

    @abstractmethod
    def release_trainee_resources(self, trainee_id):
        """Release a trainee's resources from the Howso service."""

    @abstractmethod
    def persist_trainee(self, trainee_id):
        """Persist a trainee in the Howso service."""

    @abstractmethod
    def set_random_seed(self, trainee_id, seed):
        """Set the random seed for the trainee."""

    @abstractmethod
    def train(
        self, trainee_id, cases, features=None, *,
        accumulate_weight_feature=None,
        batch_size=None,
        derived_features=None,
        initial_batch_size=None,
        input_is_substituted=False,
        progress_callback=None,
        series=None,
        train_weights_only=False,
        validate=True,
    ):
        """Train a trainee with sessions containing training cases."""

    @abstractmethod
    def impute(self, trainee_id, features=None, features_to_impute=None,
               batch_size=1):
        """Impute the missing values for the specified features_to_impute."""

    @abstractmethod
    def remove_cases(self, trainee_id, num_cases, *,
                     case_indices=None,
                     condition=None, condition_session=None,
                     distribute_weight_feature=None, precision=None,
                     preserve_session_data=False) -> int:
        """Remove training cases from a trainee."""

    @abstractmethod
    def move_cases(self, trainee_id, num_cases, *,
                   case_indices=None,
                   condition=None, condition_session=None,
                   precision=None, preserve_session_data=False,
                   target_id=None, source_id=None,
                   source_name_path=None, target_name_path=None) -> int:
        """Move training cases from one trainee to another in the hierarchy."""

    @abstractmethod
    def edit_cases(self, trainee_id, feature_values, *, case_indices=None,
                   condition=None, condition_session=None, features=None,
                   num_cases=None, precision=None) -> int:
        """Edit feature values for the specified cases."""

    @abstractmethod
    def remove_series_store(self, trainee_id, series=None):
        """Clear stored series from trainee."""

    @abstractmethod
    def append_to_series_store(
        self,
        trainee_id,
        series,
        contexts,
        *,
        context_features=None
    ):
        """Append the specified contexts to a series store."""

    @abstractmethod
    def set_substitute_feature_values(self, trainee_id, substitution_value_map):
        """Set a substitution map for use in extended nominal generation."""

    @abstractmethod
    def get_substitute_feature_values(self, trainee_id, clear_on_get=True) -> Dict:
        """Get a substitution map for use in extended nominal generation."""

    @abstractmethod
    def set_feature_attributes(self, trainee_id, feature_attributes):
        """Set feature attributes for a trainee."""

    @abstractmethod
    def get_feature_attributes(self, trainee_id):
        """Get a dict of feature attributes."""

    @abstractmethod
    def get_sessions(self, search_terms=None):
        """Get list of all accessible sessions."""

    @abstractmethod
    def get_session(self, session_id):
        """Get session details."""

    @abstractmethod
    def update_session(self, session_id, *, metadata=None):
        """Update a session."""

    @abstractmethod
    def begin_session(self, name='default', metadata=None):
        """Begin a new session."""

    @abstractmethod
    def get_trainee_sessions(self, trainee_id) -> List[Dict[str, str]]:
        """Get the session ids of a trainee."""

    @abstractmethod
    def delete_trainee_session(self, trainee_id, session):
        """Delete a session from a trainee."""

    @abstractmethod
    def get_trainee_session_indices(self, trainee_id, session) -> Union[Index, List[int]]:
        """Get list of all session indices for a specified session."""

    @abstractmethod
    def get_trainee_session_training_indices(self, trainee_id, session) -> Union[Index, List[int]]:
        """Get list of all session training indices for a specified session."""

    @abstractmethod
    def get_hierarchy(self, trainee_id) -> Dict:
        """Output the hierarchy for a trainee."""

    @abstractmethod
    def rename_subtrainee(
        self,
        trainee_id,
        new_name,
        *,
        child_name_path=None,
        child_id=None
    ) -> None:
        """Renames a contained child trainee in the hierarchy."""

    @abstractmethod
    def get_feature_residuals(
        self, trainee_id, *,
        action_feature=None,
        robust=None,
        robust_hyperparameters=None,
        weight_feature=None,
    ):
        """Get cached feature residuals."""

    @abstractmethod
    def get_prediction_stats(
        self, trainee_id, *,
        action_feature=None,
        condition=None,
        num_cases=None,
        num_robust_influence_samples_per_case=None,
        precision=None,
        robust=None,
        robust_hyperparameters=None,
        stats=None,
        weight_feature=None,
    ) -> Union["DataFrame", Dict]:
        """Get cached feature prediction stats."""

    @abstractmethod
    def get_marginal_stats(
        self, trainee_id, *,
        condition=None,
        num_cases=None,
        precision=None,
        weight_feature=None,
    ) -> Union["DataFrame", Dict]:
        """Get marginal stats for all features."""

    @abstractmethod
    def react_series(
        self, trainee_id, *,
        action_features=None,
        actions=None,
        batch_size=None,
        case_indices=None,
        contexts=None,
        context_features=None,
        continue_series=False,
        continue_series_features=None,
        continue_series_values=None,
        derived_action_features=None,
        derived_context_features=None,
        desired_conviction=None,
        details=None,
        exclude_novel_nominals_from_uniqueness_check=False,
        feature_bounds_map=None,
        final_time_steps=None,
        generate_new_cases="no",
        init_time_steps=None,
        initial_batch_size=None,
        initial_features=None,
        initial_values=None,
        input_is_substituted=False,
        leave_case_out=None,
        max_series_lengths=None,
        new_case_threshold="min",
        num_series_to_generate=1,
        ordered_by_specified_features=False,
        output_new_series_ids=True,
        preserve_feature_values=None,
        progress_callback=None,
        series_context_features=None,
        series_context_values=None,
        series_id_tracking="fixed",
        series_stop_maps=None,
        series_index=None,
        substitute_output=True,
        suppress_warning=False,
        use_case_weights=False,
        use_regional_model_residuals=True,
        weight_feature=None
    ) -> Reaction:
        """React in a series until a stop condition is met."""

    @abstractmethod
    def react_into_features(
        self, trainee_id, *,
        distance_contribution: Union[bool, str] = False,
        familiarity_conviction_addition: Union[bool, str] = False,
        familiarity_conviction_removal: Union[bool, str] = False,
        features=None,
        influence_weight_entropy: Union[bool, str] = False,
        p_value_of_addition: Union[bool, str] = False,
        p_value_of_removal: Union[bool, str] = False,
        similarity_conviction: Union[bool, str] = False,
        use_case_weights: Union[bool, str] = False,
        weight_feature=None
    ):
        """Calculate conviction and other data for the specified feature(s)."""

    @abstractmethod
    def react_into_trainee(
        self, trainee_id, *,
        action_feature=None,
        context_features=None,
        contributions=None,
        contributions_robust=None,
        hyperparameter_param_path=None,
        mda=None,
        mda_permutation=None,
        mda_robust=None,
        mda_robust_permutation=None,
        num_robust_influence_samples=None,
        num_robust_residual_samples=None,
        num_robust_influence_samples_per_case=None,
        num_samples=None,
        residuals=None,
        residuals_robust=None,
        sample_model_fraction=None,
        sub_model_size=None,
        use_case_weights=False,
        weight_feature=None,
    ):
        """Compute and cache specified feature interpretations."""

    @abstractmethod
    def react_group(
        self, trainee_id, new_cases, *,
        distance_contributions=False,
        familiarity_conviction_addition=True,
        familiarity_conviction_removal=False,
        features=None,
        kl_divergence_addition=False,
        kl_divergence_removal=False,
        p_value_of_addition=False,
        p_value_of_removal=False,
        use_case_weights=False,
        weight_feature=None
    ) -> Union["DataFrame", Dict]:
        """Compute specified data for a **set** of cases."""

    @abstractmethod
    def react(
        self, trainee_id, *,
        action_features=None,
        actions=None,
        allow_nulls=False,
        batch_size=None,
        case_indices=None,
        contexts=None,
        context_features=None,
        derived_action_features=None,
        derived_context_features=None,
        desired_conviction=None,
        details=None,
        exclude_novel_nominals_from_uniqueness_check=False,
        feature_bounds_map=None,
        generate_new_cases="no",
        initial_batch_size=None,
        input_is_substituted=False,
        into_series_store=None,
        leave_case_out=None,
        new_case_threshold="min",
        num_cases_to_generate=1,
        ordered_by_specified_features=False,
        post_process_features=None,
        post_process_values=None,
        preserve_feature_values=None,
        progress_callback=None,
        substitute_output=True,
        suppress_warning=False,
        use_case_weights=False,
        use_regional_model_residuals=True,
        weight_feature=None
    ) -> Reaction:
        """Send a `react` to the Howso engine."""

    @abstractmethod
    def evaluate(self, trainee_id, features_to_code_map, *, aggregation_code=None) -> Dict:
        """Evaluate custom code on case values within the trainee."""

    @abstractmethod
    def analyze(
        self,
        trainee_id,
        context_features=None,
        action_features=None,
        *,
        bypass_calculate_feature_residuals=None,
        bypass_calculate_feature_weights=None,
        bypass_hyperparameter_analysis=None,
        dt_values=None,
        inverse_residuals_as_weights=None,
        k_folds=None,
        k_values=None,
        num_analysis_samples=None,
        num_samples=None,
        analysis_sub_model_size=None,
        analyze_level=None,
        p_values=None,
        targeted_model=None,
        use_case_weights=None,
        use_deviations=None,
        weight_feature=None,
        **kwargs
    ):
        """Analyzes a trainee."""

    @abstractmethod
    def auto_analyze(self, trainee_id):
        """Auto-analyze the trainee model."""

    @abstractmethod
    def set_auto_ablation_params(
        self,
        trainee_id,
        auto_ablation_enabled=False,
        *,
        auto_ablation_weight_feature=".case_weight",
        conviction_lower_threshold=None,
        conviction_upper_threshold=None,
        exact_prediction_features=None,
        influence_weight_entropy_threshold=0.6,
        minimum_model_size=1_000,
        relative_prediction_threshold_map=None,
        residual_prediction_features=None,
        tolerance_prediction_threshold_map=None,
        **kwargs
    ):
        """Set trainee parameters for auto ablation."""

    @abstractmethod
    def get_auto_ablation_params(
        self,
        trainee_id
    ):
        """Get trainee parameters for auto ablation set by :meth:`set_auto_ablation_params`."""

    @abstractmethod
    def set_auto_analyze_params(
        self,
        trainee_id,
        auto_analyze_enabled=False,
        analyze_threshold=None,
        *,
        auto_analyze_limit_size=None,
        analyze_growth_factor=None,
        **kwargs
    ):
        """Set trainee parameters for auto analysis."""

    @abstractmethod
    def get_cases(self, trainee_id, session=None, case_indices=None,
                  indicate_imputed=False, features=None, condition=None,
                  num_cases=None, precision=None) -> Union["Cases", "DataFrame"]:
        """Retrieve cases from a trainee."""

    @abstractmethod
    def get_extreme_cases(
        self,
        trainee_id,
        num,
        sort_feature,
        features: Optional[Iterable[str]] = None
    ) -> Union["Cases", "DataFrame"]:
        """Get the extreme cases of a trainee for the given feature(s)."""

    @abstractmethod
    def get_num_training_cases(self, trainee_id) -> int:
        """Return the number of trained cases in the model."""

    @abstractmethod
    def get_feature_conviction(
        self,
        trainee_id,
        *,

        familiarity_conviction_addition: Union[bool, str] = True,
        familiarity_conviction_removal: Union[bool, str] = False,
        use_case_weights: bool = False,
        features=None,
        action_features=None,
        weight_feature=None
    ) -> Union[Dict, "DataFrame"]:
        """Get familiarity conviction for features in the model."""

    @abstractmethod
    def add_feature(self, trainee_id, feature, feature_value=None, *,
                    condition=None, condition_session=None,
                    feature_attributes=None, overwrite=False):
        """Add a feature to a trainee's model."""

    @abstractmethod
    def remove_feature(self, trainee_id, feature, *, condition=None,
                       condition_session=None):
        """Remove a feature from a trainee."""

    @abstractmethod
    def get_feature_mda(
        self, trainee_id, action_feature, *,
        permutation=None,
        robust=None,
        weight_feature=None,
    ) -> "DataFrame":
        """Get cached feature Mean Decrease In Accuracy (MDA)."""

    @abstractmethod
    def get_feature_contributions(
        self, trainee_id, action_feature, *,
        robust=None,
        directional=False,
        weight_feature=None,
    ) -> "DataFrame":
        """Get cached feature contributions."""

    @abstractmethod
    def get_pairwise_distances(self, trainee_id, features=None, *,
                               action_feature=None, from_case_indices=None,
                               from_values=None, to_case_indices=None,
                               to_values=None, use_case_weights=False,
                               weight_feature=None) -> List[float]:
        """Compute pairwise distances between specified cases."""

    @abstractmethod
    def get_distances(self, trainee_id, features=None, *,
                      action_feature=None, case_indices=None,
                      feature_values=None, use_case_weights=False,
                      weight_feature=None) -> Dict:
        """Compute distances matrix for specified cases."""

    @abstractmethod
    def get_params(self, trainee_id, *, action_feature=None,
                   context_features=None, mode=None, weight_feature=None) -> Dict[str, Any]:
        """Get parameters used by the system."""

    @abstractmethod
    def set_params(self, trainee_id, params):
        """Set specific hyperparameters in the trainee."""

    def _resolve_trainee_id(self, trainee_id, *args, **kwargs):
        """Resolve trainee identifier."""
        return trainee_id
