"""Unit tests for IFASuggestion and IFASuggestionCollector."""
import datetime
import json

import numpy as np
import pytest
from rich.console import Console

from howso.utilities.feature_attributes.suggestions import (
    FanoutFeaturesSuggestion,
    IFASuggestionCollector,
    normalize_fanout_feature_map,
    PRVSuggestion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fanout(keys_to_cols: dict) -> FanoutFeaturesSuggestion:
    return FanoutFeaturesSuggestion(keys_to_cols)


def make_prv(config: dict) -> PRVSuggestion:
    return PRVSuggestion(prvc=config, values_ranking=[], user_set_max_distilled_cases=True)


# ---------------------------------------------------------------------------
# IFASuggestionCollector.append
# ---------------------------------------------------------------------------

class TestCollectorAppend:
    def test_append_new_suggestion(self):
        collector = IFASuggestionCollector()
        collector.append(make_fanout({"key_a": ["col1"]}))
        assert "fanout_features" in collector.suggestions

    def test_append_duplicate_merges_into_existing(self):
        """Second append with the same name merges into the existing suggestion, keeping both."""
        collector = IFASuggestionCollector()
        collector.append(make_fanout({"key_a": ["col1"]}))
        collector.append(make_fanout({"key_b": ["col2"]}))

        fof_map = collector.fanout_features.get_fanout_feature_map()
        assert "key_a" in fof_map, "first append's data must survive"
        assert "key_b" in fof_map, "second append's data must be merged in"

    def test_append_duplicate_does_not_overwrite(self):
        """Appending a duplicate must not replace the existing suggestion wholesale."""
        collector = IFASuggestionCollector()
        collector.append(make_fanout({"key_a": ["col1"], "key_b": ["col2"]}))
        # Append a suggestion that only knows about key_b.
        collector.append(make_fanout({"key_b": ["col2"]}))

        fof_map = collector.fanout_features.get_fanout_feature_map()
        assert "key_a" in fof_map, "key_a must not be lost after a duplicate append"


# ---------------------------------------------------------------------------
# IFASuggestionCollector.merge
# ---------------------------------------------------------------------------

class TestCollectorMerge:
    def test_merge_into_empty_adds_suggestions(self):
        """Merging into an empty collector must populate it."""
        source = IFASuggestionCollector()
        source.append(make_fanout({"key_a": ["col1"]}))

        target = IFASuggestionCollector()
        target.merge(source)

        assert "fanout_features" in target.suggestions
        assert "key_a" in target.fanout_features.get_fanout_feature_map()

    def test_merge_new_suggestion_type_is_added(self):
        """A suggestion type absent from the target must be added, not silently dropped."""
        target = IFASuggestionCollector()
        target.append(make_fanout({"key_a": ["col1"]}))

        other = IFASuggestionCollector()
        other.append(make_prv({"feat_x": {"protected_values_multipliers": [], "unprotected_multiplier": 1.0}}))

        target.merge(other)

        assert "fanout_features" in target.suggestions
        assert "preserve_rare_values" in target.suggestions

    def test_merge_overlapping_suggestion_combines_data(self):
        """Merging two collectors with the same suggestion type must combine their data."""
        c1 = IFASuggestionCollector()
        c1.append(make_fanout({"key_a": ["col1"]}))

        c2 = IFASuggestionCollector()
        c2.append(make_fanout({"key_b": ["col2"]}))

        c1.merge(c2)

        fof_map = c1.fanout_features.get_fanout_feature_map()
        assert "key_a" in fof_map
        assert "key_b" in fof_map

    def test_merge_does_not_mutate_source(self):
        source = IFASuggestionCollector()
        source.append(make_fanout({"key_a": ["col1"]}))

        target = IFASuggestionCollector()
        target.merge(source)

        assert "key_a" in source.fanout_features.get_fanout_feature_map()


# ---------------------------------------------------------------------------
# PRVSuggestion.merge
# ---------------------------------------------------------------------------

class TestPRVSuggestionMerge:
    def test_merge_non_overlapping_features(self):
        prv1 = make_prv({"feat_a": {"protected_values_multipliers": [{"value": "rare", "multiplier": 2.0}],
                                    "unprotected_multiplier": 0.9}})
        prv2 = make_prv({"feat_b": {"protected_values_multipliers": [{"value": "uncommon", "multiplier": 3.0}],
                                    "unprotected_multiplier": 0.8}})
        prv1.merge(prv2)

        config = prv1.get_config()
        assert "feat_a" in config
        assert "feat_b" in config

    def test_merge_identical_feature_config_is_allowed(self):
        cfg = {"protected_values_multipliers": [{"value": "rare", "multiplier": 2.0}],
               "unprotected_multiplier": 0.9}
        prv1 = make_prv({"feat_a": cfg})
        prv2 = make_prv({"feat_a": cfg})
        prv1.merge(prv2)  # must not raise

    def test_merge_conflicting_feature_raises(self):
        prv1 = make_prv({"feat_a": {"protected_values_multipliers": [{"value": "rare", "multiplier": 2.0}],
                                    "unprotected_multiplier": 0.9}})
        prv2 = make_prv({"feat_a": {"protected_values_multipliers": [{"value": "rare", "multiplier": 5.0}],
                                    "unprotected_multiplier": 0.5}})
        with pytest.raises(ValueError, match="differing configurations"):
            prv1.merge(prv2)


# ---------------------------------------------------------------------------
# FanoutFeaturesSuggestion.merge
# ---------------------------------------------------------------------------

class TestFanoutSuggestionMerge:
    def test_merge_combines_keys(self):
        fof1 = make_fanout({"key_a": ["col1", "col2"]})
        fof2 = make_fanout({"key_b": ["col3"]})
        fof1.merge(fof2)

        result = fof1.get_fanout_feature_map()
        assert "key_a" in result
        assert "key_b" in result

    def test_merge_duplicate_key_unions_columns(self):
        """Overlapping keys produce the union of their fanout column lists."""
        fof1 = make_fanout({"key_a": ["col1"]})
        fof2 = make_fanout({"key_a": ["col2", "col3"]})
        fof1.merge(fof2)

        result = fof1.get_fanout_feature_map()["key_a"]
        assert set(result) == {"col1", "col2", "col3"}

    def test_merge_duplicate_key_no_duplicates_in_union(self):
        """Columns already present in self are not duplicated after merge."""
        fof1 = make_fanout({"key_a": ["col1", "col2"]})
        fof2 = make_fanout({"key_a": ["col2", "col3"]})
        fof1.merge(fof2)

        result = fof1.get_fanout_feature_map()["key_a"]
        assert result.count("col2") == 1


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _prv_config(**features_to_num_values: int) -> dict:
    return {
        feature: {
            "protected_values_multipliers": [{"value": i, "multiplier": 2.0} for i in range(num)],
            "unprotected_multiplier": 0.9,
        }
        for feature, num in features_to_num_values.items()
    }


class TestSuggestionSummary:

    def test_fanout_summary_counts_features_and_keys(self):
        suggestion = make_fanout({"key_a": ["c1", "c2", "c3"], ("k1", "k2"): ["c4"]})
        assert suggestion.summary == "Found 4 fan-out features across 2 columns"

    def test_fanout_summary_singular(self):
        assert make_fanout({"key_a": ["c1"]}).summary == "Found 1 fan-out feature across 1 column"

    def test_prv_summary_counts_values_and_features(self):
        suggestion = make_prv(_prv_config(a=3, b=1))
        assert suggestion.summary == (
            "Found 4 rare values across 2 columns whose signal may be lost during data distillation workflows"
        )

    def test_prv_summary_singular(self):
        assert make_prv(_prv_config(a=1)).summary == (
            "Found 1 rare value across 1 column whose signal may be lost during data distillation workflows"
        )


class TestCollectorSummary:

    def test_empty_collector_has_no_summary_lines(self):
        assert IFASuggestionCollector().summary_lines() == []

    def test_empty_collector_prints_nothing(self, capsys):
        IFASuggestionCollector().print_summary()
        assert capsys.readouterr().out == ""

    def test_summary_lines_have_header_one_finding_per_suggestion_and_footer(self):
        collector = IFASuggestionCollector([make_fanout({"key_a": ["c1"]}), make_prv(_prv_config(a=2))])
        lines = collector.summary_lines()
        assert len(lines) == 4
        assert "Feature Attributes Summary" in lines[0]
        assert lines[1].endswith(make_fanout({"key_a": ["c1"]}).summary)
        assert lines[2].endswith(make_prv(_prv_config(a=2)).summary)
        assert "`your_attributes_object.suggestions`" in lines[3]

    def test_summary_footer_names_the_attributes_object(self):
        collector = IFASuggestionCollector([make_fanout({"key_a": ["c1"]})])
        assert "`features.suggestions`" in collector.summary_lines(attributes_name="features")[-1]

    def test_print_summary_renders_to_stdout(self, capsys):
        collector = IFASuggestionCollector([make_fanout({"key_a": ["c1", "c2"]})])
        collector.print_summary()
        out = capsys.readouterr().out
        assert out.splitlines() == [
            "Feature Attributes Summary",
            "  \u00b7 Found 2 fan-out features across 1 column",
            "  Inspect `your_attributes_object.suggestions` for details and how to apply them.",
        ]

    def test_print_summary_does_not_hard_wrap_long_lines(self, capsys):
        """A narrow console must not break a finding across lines; the terminal wraps it instead."""
        collector = IFASuggestionCollector([make_prv(_prv_config(a=2, b=2))])
        collector.print_summary(console=Console(width=20, force_jupyter=False))
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 3
        assert lines[1].endswith("signal may be lost during data distillation workflows")

    def test_print_summary_is_not_a_warning(self, recwarn):
        IFASuggestionCollector([make_fanout({"key_a": ["c1"]})]).print_summary(console=Console(file=None))
        assert not recwarn.list


class TestSuggestionToDict:

    def test_fanout_to_dict(self):
        suggestion = make_fanout({"key_a": ["c1", "c2"], ("k1", "k2"): ["c3"]})
        result = suggestion.to_dict()
        groups = [
            {"key_features": ["key_a"], "fanout_features": ["c1", "c2"]},
            {"key_features": ["k1", "k2"], "fanout_features": ["c3"]},
        ]
        assert result == {
            "name": "fanout_features",
            "summary": suggestion.summary,
            "description": suggestion.description,
            "can_apply": True,
            "caveats": [],
            "details": {"num_key_features": 2, "num_fanout_features": 3, "groups": groups},
            "parameters": {"fanout_feature_map": groups},
        }

    def test_prv_to_dict_with_user_set_max_distilled_cases(self):
        config = _prv_config(a=2, b=1)
        ranking = [{"feature": "a", "value": "v0", "count": 40}]
        result = PRVSuggestion(config, ranking, user_set_max_distilled_cases=True).to_dict()
        assert result["name"] == "preserve_rare_values"
        assert result["can_apply"] is True
        assert result["caveats"] == []
        assert result["details"] == {"num_values": 3, "num_features": 2, "top_values": ranking}
        assert result["parameters"] == {
            "preserve_rare_values_config": config,
            "preserve_rare_values_map": {"a": [0, 1], "b": [0]},
        }

    def test_prv_to_dict_with_default_max_distilled_cases_reports_caveat_without_warning(self, recwarn):
        result = PRVSuggestion(_prv_config(a=1), [], user_set_max_distilled_cases=False).to_dict()
        assert not recwarn.list
        assert result["can_apply"] is False
        assert [c["code"] for c in result["caveats"]] == ["default_max_distilled_cases"]
        assert "max_distilled_cases" in result["caveats"][0]["message"]
        # The config is still offered so it can be edited and passed back
        assert result["parameters"]["preserve_rare_values_map"] == {"a": [0]}

    def test_prv_caveat_message_matches_warning(self):
        suggestion = PRVSuggestion(_prv_config(a=1), [], user_set_max_distilled_cases=False)
        with pytest.warns(UserWarning) as record:
            suggestion.get_config()
        assert str(record[0].message) == suggestion.caveats[0]["message"]


class TestPRVRankingMerge:

    def test_merge_combines_rankings_by_count(self):
        first = PRVSuggestion(_prv_config(a=1), [{"feature": "a", "value": "v0", "count": 10}], True)
        second = PRVSuggestion(
            {"b": _prv_config(b=1)["b"]},
            [{"feature": "b", "value": "v0", "count": 30}],
            True,
        )
        first.merge(second)
        assert [(c["feature"], c["count"]) for c in first.details["top_values"]] == [("b", 30), ("a", 10)]

    def test_merge_keeps_top_five_without_duplicates(self):
        ranking = [{"feature": "a", "value": f"v{i}", "count": i} for i in range(4)]
        first = PRVSuggestion(_prv_config(a=4), list(ranking), True)
        other_ranking = [{"feature": "b", "value": f"v{i}", "count": 10 + i} for i in range(3)]
        second = PRVSuggestion({"b": _prv_config(b=3)["b"]}, ranking[:1] + other_ranking, True)
        first.merge(second)
        top = first.details["top_values"]
        assert [c["count"] for c in top] == [12, 11, 10, 3, 2]


class TestCollectorToDict:

    def test_empty_collector(self):
        assert IFASuggestionCollector().to_dict() == {"schema_version": 1, "suggestions": []}

    def test_suggestions_in_insertion_order(self):
        collector = IFASuggestionCollector([make_prv(_prv_config(a=1)), make_fanout({"key_a": ["c1"]})])
        names = [s["name"] for s in collector.to_dict()["suggestions"]]
        assert names == ["preserve_rare_values", "fanout_features"]

    def test_to_json_handles_numpy_and_datetime_values(self):
        config = {"a": {
            "protected_values_multipliers": [
                {"value": np.int64(3), "multiplier": np.float64(1.5)},
                {"value": datetime.date(2026, 1, 2), "multiplier": 2.0},
            ],
            "unprotected_multiplier": 0.9,
        }}
        ranking = [{"feature": "a", "value": np.int64(3), "count": np.int64(12)}]
        collector = IFASuggestionCollector([PRVSuggestion(config, ranking, True)])
        payload = json.loads(collector.to_json())
        prv = payload["suggestions"][0]
        assert prv["parameters"]["preserve_rare_values_map"] == {"a": [3, "2026-01-02"]}
        assert prv["details"]["top_values"] == [{"feature": "a", "value": 3, "count": 12}]

    def test_to_json_passes_kwargs(self):
        collector = IFASuggestionCollector([make_fanout({"key_a": ["c1"]})])
        assert "\n  " in collector.to_json(indent=2)

    def test_to_dict_does_not_print(self, capsys):
        IFASuggestionCollector([make_fanout({"key_a": ["c1"]})]).to_dict()
        assert capsys.readouterr().out == ""


class TestNormalizeFanoutFeatureMap:

    def test_mapping_passes_through(self):
        fof_map = {"key_a": ["c1"], ("k1", "k2"): ["c2"]}
        assert normalize_fanout_feature_map(fof_map) == fof_map

    def test_list_form(self):
        groups = [
            {"key_features": ["key_a"], "fanout_features": ["c1"]},
            {"key_features": ["k1", "k2"], "fanout_features": ["c2", "c3"]},
        ]
        assert normalize_fanout_feature_map(groups) == {"key_a": ["c1"], ("k1", "k2"): ["c2", "c3"]}

    def test_list_form_round_trips_through_json(self):
        fof_map = {"key_a": ["c1", "c2"], ("k1", "k2"): ["c3"]}
        payload = json.loads(IFASuggestionCollector([make_fanout(fof_map)]).to_json())
        groups = payload["suggestions"][0]["parameters"]["fanout_feature_map"]
        assert normalize_fanout_feature_map(groups) == fof_map

    def test_list_form_merges_repeated_keys(self):
        groups = [
            {"key_features": ["key_a"], "fanout_features": ["c1"]},
            {"key_features": "key_a", "fanout_features": ["c1", "c2"]},
        ]
        assert normalize_fanout_feature_map(groups) == {"key_a": ["c1", "c2"]}

    @pytest.mark.parametrize("value, error", [
        ("key_a", TypeError),
        (5, TypeError),
        ([{"key_features": ["key_a"]}], ValueError),
        ([{"key_features": [], "fanout_features": ["c1"]}], ValueError),
        (["key_a"], ValueError),
    ])
    def test_invalid_input(self, value, error):
        with pytest.raises(error):
            normalize_fanout_feature_map(value)


# ---------------------------------------------------------------------------
# Rendered output
# ---------------------------------------------------------------------------

def _rendering_prv_config(**features_to_num_values: int) -> dict:
    return {
        feature: {
            "protected_values_multipliers": [
                {"value": f"{feature}_v{i}", "multiplier": 2.0} for i in range(num)
            ],
            "unprotected_multiplier": 0.9,
        }
        for feature, num in features_to_num_values.items()
    }


_RENDERING_RANKING = [
    {"feature": "color", "value": "teal", "count": 812},
    {"feature": "size", "value": None, "count": 400},
    {"feature": "color", "value": "mauve", "count": 95},
]

RENDERING_CASES = {
    "fanout_single_and_tuple_keys": lambda: FanoutFeaturesSuggestion({
        "order_id": ["ship_date", "region", "carrier", "warehouse", "zone"],
        ("store_id", "day"): ["weather"],
    }),
    "fanout_many_keys": lambda: FanoutFeaturesSuggestion({f"key_{i}": [f"col_{i}"] for i in range(6)}),
    "prv_user_set_mdc": lambda: PRVSuggestion(
        _rendering_prv_config(color=2, size=1), list(_RENDERING_RANKING), user_set_max_distilled_cases=True),
    "prv_default_mdc": lambda: PRVSuggestion(
        _rendering_prv_config(color=2, size=1), list(_RENDERING_RANKING), user_set_max_distilled_cases=False),
}


# Trailing whitespace is omitted; rendered lines are compared with it stripped.
EXPECTED_REPR = {
    "fanout_single_and_tuple_keys": """\
Fan-out Features

We have detected 2 key(s) that should be considered as fan-out features. Fan-out features are columns that have repeated
values across multiple rows based on a single observation. Informing the Howso Engine of fan-out features via your
feature attributes will help it measure uncertainty more accurately.

        To read more about fan-out features, please see:
\thttps://docs.howso.com/en/latest/user_guide/advanced_capabilities/fanout_features.html

Examples In Your Data:
----------------------
  - Columns `ship_date`, `region`, `carrier`, and 2 more have repeated values derived from observations in `order_id`
  - Columns `weather` have repeated values derived from observations in `('store_id', 'day')`


                                              Summary of Available Options
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Action                     ┃ Details                                    ┃ Relevant Code                              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Get a reusable             │ You may provide `fanout_feature_map` as    │ From this suggestion object call:          │
│ `fanout_feature_map`       │ a parameter to                             │ `get_fanout_feature_map()`                 │
│                            │ `infer_feature_attributes` if you wish     │                                            │
│                            │ to adjust the fan-out feature              │                                            │
│                            │ configuration. Our detected fan-out        │                                            │
│                            │ feature configuration may be a good        │                                            │
│                            │ starting point.                            │                                            │
├────────────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────┤
│ Apply suggestion to this   │ Save the suggested candidate               │ Call `apply_suggestion()` on the feature   │
│ feature attributes         │ `fanout_feature_map` to this feature       │ attributes object:                         │
│ object                     │ attributes object.                         │ `apply_suggestion("fanout_features")`      │
└────────────────────────────┴────────────────────────────────────────────┴────────────────────────────────────────────┘
""",
    "fanout_many_keys": """\
Fan-out Features

We have detected 6 key(s) that should be considered as fan-out features. Fan-out features are columns that have repeated
values across multiple rows based on a single observation. Informing the Howso Engine of fan-out features via your
feature attributes will help it measure uncertainty more accurately.

        To read more about fan-out features, please see:
\thttps://docs.howso.com/en/latest/user_guide/advanced_capabilities/fanout_features.html

Examples In Your Data:
----------------------
  - Columns `col_0` have repeated values derived from observations in `key_0`
  - Columns `col_1` have repeated values derived from observations in `key_1`
  - Columns `col_2` have repeated values derived from observations in `key_2`
  - Columns `col_3` have repeated values derived from observations in `key_3`
  - ...and 2 more keys


                                              Summary of Available Options
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Action                     ┃ Details                                    ┃ Relevant Code                              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Get a reusable             │ You may provide `fanout_feature_map` as    │ From this suggestion object call:          │
│ `fanout_feature_map`       │ a parameter to                             │ `get_fanout_feature_map()`                 │
│                            │ `infer_feature_attributes` if you wish     │                                            │
│                            │ to adjust the fan-out feature              │                                            │
│                            │ configuration. Our detected fan-out        │                                            │
│                            │ feature configuration may be a good        │                                            │
│                            │ starting point.                            │                                            │
├────────────────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────┤
│ Apply suggestion to this   │ Save the suggested candidate               │ Call `apply_suggestion()` on the feature   │
│ feature attributes         │ `fanout_feature_map` to this feature       │ attributes object:                         │
│ object                     │ attributes object.                         │ `apply_suggestion("fanout_features")`      │
└────────────────────────────┴────────────────────────────────────────────┴────────────────────────────────────────────┘
""",
    "prv_user_set_mdc": """\
Rare Value Preservation

Here are some values in your data that may be good candidates for Rare Value Preservation:

    - Column name: color, value: teal
    - Column name: size, value: None
    - Column name: color, value: mauve

In total, we identified 3 values that may be lost during data distillation.

During data distillation workflows, nominal values with weak but detectable signals may be filtered out. To account for
this, you may provide to `infer_feature_attributes` a `preserve_rare_values_map` detailing rare values to protect
automatically, or a full `preserve_rare_values_config` with fine-grained case weight adjustments. Additionally, you may
apply our suggested configuration for all detected possible rare values to this feature attributes object. Applying Rare
Value Preservation may increase the influence of rare values on the aggregate signal of the dataset. This is the
intended effect to help preserve the signal of rare values that would otherwise be lost during distillation.

                                              Summary of Available Options
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Action                        ┃ Details                                  ┃ Relevant Code                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Apply suggestion to this      │ Save the suggested candidate             │ Call `apply_suggestion()` on the feature  │
│ feature attributes            │ `preserve_rare_values_config` to this    │ attributes object:                        │
│ object                        │ feature attributes object.               │ `apply_suggestion("preserve_rare_values"… │
├───────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────┤
│ Get a reusable                │ You may provide a pre-computed           │ From this suggestion object call:         │
│ `preserve_rare_values_config` │ `preserve_rare_values_config` as a       │ `get_config()`                            │
│                               │ parameter to `infer_feature_attributes`  │                                           │
│                               │ if you wish to make adjustments to the   │                                           │
│                               │ case weight multipliers.                 │                                           │
├───────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────┤
│ Edit the preserved rare       │ The rare values to be preserved can be   │ From this suggestion object call:         │
│ values with a                 │ detailed via the                         │ `get_values_map()`                        │
│ `preserve_rare_values_map`    │ `preserve_rare_values_map` parameter to  │                                           │
│                               │ `infer_feature_attributes`. A good       │                                           │
│                               │ starting point may be the "full" map of  │                                           │
│                               │ all candidate values. All case weight    │                                           │
│                               │ multipliers will be automatically        │                                           │
│                               │ configured for the provided values.      │                                           │
└───────────────────────────────┴──────────────────────────────────────────┴───────────────────────────────────────────┘
""",
    "prv_default_mdc": """\
Rare Value Preservation

Here are some values in your data that may be good candidates for Rare Value Preservation:

    - Column name: color, value: teal
    - Column name: size, value: None
    - Column name: color, value: mauve

During data distillation workflows, nominal values with weak but detectable signals may be filtered out. To account for
this, you may provide to `infer_feature_attributes` a `preserve_rare_values_map` detailing rare values to protect
automatically, or a full `preserve_rare_values_config` with fine-grained case weight adjustments. Additionally, you may
apply our suggested configuration for all detected possible rare values to this feature attributes object. Applying Rare
Value Preservation may increase the influence of rare values on the aggregate signal of the dataset. This is the
intended effect to help preserve the signal of rare values that would otherwise be lost during distillation.

                                              Summary of Available Options
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Action                        ┃ Details                                  ┃ Relevant Code                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Get a reusable                │ You may provide a pre-computed           │ From this suggestion object call:         │
│ `preserve_rare_values_config` │ `preserve_rare_values_config` as a       │ `get_config()`                            │
│                               │ parameter to `infer_feature_attributes`  │                                           │
│                               │ if you wish to make adjustments to the   │                                           │
│                               │ case weight multipliers.                 │                                           │
├───────────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────────────┤
│ Edit the preserved rare       │ The rare values to be preserved can be   │ From this suggestion object call:         │
│ values with a                 │ detailed via the                         │ `get_values_map()`                        │
│ `preserve_rare_values_map`    │ `preserve_rare_values_map` parameter to  │                                           │
│                               │ `infer_feature_attributes`. A good       │                                           │
│                               │ starting point may be the "full" map of  │                                           │
│                               │ all candidate values. All case weight    │                                           │
│                               │ multipliers will be automatically        │                                           │
│                               │ configured for the provided values.      │                                           │
└───────────────────────────────┴──────────────────────────────────────────┴───────────────────────────────────────────┘
""",
}

_PRV_SUMMARY = "Found 3 rare values across 2 columns whose signal may be lost during data distillation workflows"

EXPECTED_SUMMARY = {
    "fanout_single_and_tuple_keys": "Found 6 fan-out features across 2 columns",
    "fanout_many_keys": "Found 6 fan-out features across 6 columns",
    "prv_user_set_mdc": _PRV_SUMMARY,
    "prv_default_mdc": _PRV_SUMMARY,
}


@pytest.mark.parametrize("case", RENDERING_CASES)
def test_repr_matches_expected(case):
    """The console rendering of each suggestion matches the expected text line for line."""
    rendered = [line.rstrip() for line in repr(RENDERING_CASES[case]()).splitlines()]
    assert rendered == EXPECTED_REPR[case].splitlines()


@pytest.mark.parametrize("case", RENDERING_CASES)
def test_summary_matches_expected(case):
    """The one-line summary of each suggestion matches the expected text exactly."""
    assert RENDERING_CASES[case]().summary == EXPECTED_SUMMARY[case]
