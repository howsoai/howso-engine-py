# These keys were deprecated on 18-FEB-2025
_RENAMED_DETAIL_KEYS = {
    "case_contributions_full": "case_full_prediction_contributions",
    "case_contributions_robust": "case_robust_prediction_contributions",
    "case_feature_contributions_full": "feature_full_prediction_contributions_for_case",
    "case_feature_contributions_robust": "feature_robust_prediction_contributions_for_case",
    "case_feature_residual_convictions_full": "feature_full_residual_convictions_for_case",
    "case_feature_residuals_full": "feature_full_residuals_for_case",
    "case_feature_residuals_robust": "feature_robust_residuals_for_case",
    "case_mda_full": "case_full_accuracy_contributions",
    "case_mda_robust": "case_robust_accuracy_contributions",
    "directional_feature_contributions_full": "feature_full_directional_prediction_contributions",
    "directional_feature_contributions_robust": "feature_robust_directional_prediction_contributions",
    "feature_contributions_full": "feature_full_prediction_contributions",
    "feature_contributions_robust": "feature_robust_prediction_contributions",
    "feature_mda_ex_post_full": "feature_full_accuracy_contributions_ex_post",
    "feature_mda_ex_post_robust": "feature_robust_accuracy_contributions_ex_post",
    "feature_mda_full": "feature_full_accuracy_contributions",
    "feature_mda_robust": "feature_robust_accuracy_contributions",
    "feature_mda_permutation_full": "feature_full_accuracy_contributions_permutation",
    "feature_mda_permutation_robust": "feature_robust_accuracy_contributions_permutation",
    "feature_residuals_full": "feature_full_residuals",
    "feature_residuals_robust": "feature_robust_residuals",
}

_RENAMED_DETAIL_KEYS_EXTRA = {
    "case_feature_contributions_full": {
        "new_key": "feature_full_prediction_contributions_for_case",
        "additional_keys": {
            "case_directional_feature_contributions_full": "feature_full_directional_prediction_contributions_for_case",
        },
    },
    "case_feature_contributions_robust": {
        "new_key": "feature_robust_prediction_contributions_for_case",
        "additional_keys": {
            "case_directional_feature_contributions_robust": "feature_robust_directional_prediction_contributions_for_case",
        },
    },
    "feature_contributions_full": {
        "new_key": "feature_full_prediction_contributions",
        "additional_keys": {
            "directional_feature_contributions_full": "feature_full_directional_prediction_contributions",
        },
    },
    "feature_contributions_robust": {
        "new_key": "feature_robust_prediction_contributions",
        "additional_keys": {
            "directional_feature_contributions_robust": "feature_robust_directional_prediction_contributions",
        },
    },
}
