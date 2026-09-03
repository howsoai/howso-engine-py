"""Unit tests for IFASuggestion and IFASuggestionCollector."""
import pytest
from rich.console import Console

from howso.utilities.feature_attributes.suggestions import (
    FanoutFeaturesSuggestion,
    IFASuggestionCollector,
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
        assert suggestion.summary == "Found 4 fan-out features across 2 keys"

    def test_fanout_summary_singular(self):
        assert make_fanout({"key_a": ["c1"]}).summary == "Found 1 fan-out feature across 1 key"

    def test_prv_summary_counts_values_and_features(self):
        suggestion = make_prv(_prv_config(a=3, b=1))
        assert suggestion.summary == (
            "Found 4 rare values across 2 features that may be lost during data distillation"
        )

    def test_prv_summary_singular(self):
        assert make_prv(_prv_config(a=1)).summary == (
            "Found 1 rare value across 1 feature that may be lost during data distillation"
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
        assert "Feature attributes summary" in lines[0]
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
            "Feature attributes summary",
            "  \u00b7 Found 2 fan-out features across 1 key",
            "  Inspect `your_attributes_object.suggestions` for details and how to apply them.",
        ]

    def test_print_summary_does_not_hard_wrap_long_lines(self, capsys):
        """A narrow console must not break a finding across lines; the terminal wraps it instead."""
        collector = IFASuggestionCollector([make_prv(_prv_config(a=2, b=2))])
        collector.print_summary(console=Console(width=20, force_jupyter=False))
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 3
        assert lines[1].endswith("that may be lost during data distillation")

    def test_print_summary_is_not_a_warning(self, recwarn):
        IFASuggestionCollector([make_fanout({"key_a": ["c1"]})]).print_summary(console=Console(file=None))
        assert not recwarn.list
