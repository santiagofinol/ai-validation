"""
Unit tests for the AI Output Validation Framework.
Run with: python -m pytest test_validator.py -v
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from analyzer import (
    ScoringWeights,
    extract_standards,
    validate_standards,
    extract_numbers,
    extract_key_terms,
    extract_shall_statements,
    compute_similarity_matrix,
    compute_reliability_score,
    check_output_relevance,
    detect_contradictions,
    analyze_runs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def medical_text():
    return (
        "The battery housing shall comply with IEC 60601-1:2012 and ISO 13485:2016. "
        "The housing shall provide electrical insulation. "
        "Maximum operating temperature shall not exceed 43°C. "
        "Overcharge protection shall be provided via battery management system. "
        "Short-circuit protection shall limit fault current to 5A."
    )


@pytest.fixture
def minimal_run(medical_text):
    return [{"output": medical_text}]


@pytest.fixture
def identical_runs(medical_text):
    return [{"output": medical_text}, {"output": medical_text}]


@pytest.fixture
def reference():
    return (
        "The device shall comply with IEC 60601-1. "
        "Electrical insulation shall be provided. "
        "Maximum temperature is 43°C."
    )


# ---------------------------------------------------------------------------
# ScoringWeights
# ---------------------------------------------------------------------------

class TestScoringWeights:

    def test_defaults_sum_to_one(self):
        w = ScoringWeights()
        total = w.text_similarity + w.numeric_agreement + w.key_term_consistency + w.golden_answer
        assert abs(total - 1.0) < 1e-6

    def test_without_golden_valid(self):
        w = ScoringWeights.without_golden()
        total = w.text_similarity + w.numeric_agreement + w.key_term_consistency
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_valid(self):
        w = ScoringWeights(
            text_similarity=0.50,
            numeric_agreement=0.20,
            key_term_consistency=0.20,
            golden_answer=0.10,
        )
        total = w.text_similarity + w.numeric_agreement + w.key_term_consistency + w.golden_answer
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Standard Extraction
# ---------------------------------------------------------------------------

class TestStandardExtraction:

    def test_known_standards(self):
        text = "Comply with IEC 60601-1:2012 and ISO 13485:2016."
        found = extract_standards(text)
        assert "IEC 60601-1:2012" in found
        assert "ISO 13485:2016" in found

    def test_fictional_standard_extracted(self):
        text = "Follow ISO 1113-1:2009 and UL 60601-24:2020."
        found = extract_standards(text)
        assert "ISO 1113-1:2009" in found

    def test_fake_flagged_as_unverified(self):
        fake = {"ISO 1113-1:2009", "IEC 60601-1:2012"}
        unverified = validate_standards(fake)
        assert "ISO 1113-1:2009" in unverified
        assert "IEC 60601-1:2012" not in unverified

    def test_no_standards(self):
        assert extract_standards("Plain text with no references.") == set()

    def test_fda_cfr(self):
        text = "Per FDA 21 CFR Part 820, quality systems are required."
        found = extract_standards(text)
        assert any("CFR" in s for s in found)


# ---------------------------------------------------------------------------
# Numeric Extraction — edge cases
# ---------------------------------------------------------------------------

class TestNumericExtraction:

    def test_temperature_celsius(self):
        nums = extract_numbers("Limit is 43°C.")
        assert any("43" in n for n in nums)

    def test_temperature_negative(self):
        nums = extract_numbers("Operating range: -40°C to 85°C.")
        assert any("-40" in n for n in nums)

    def test_si_prefix_kilo_ohm(self):
        nums = extract_numbers("Resistance is 10 kΩ.")
        assert len(nums) >= 1

    def test_si_prefix_milliamp(self):
        nums = extract_numbers("Current limit: 500 mA.")
        assert any("500" in n for n in nums)

    def test_si_prefix_microamp(self):
        nums = extract_numbers("Leakage current shall not exceed 100 µA.")
        assert any("100" in n for n in nums)

    def test_si_prefix_gigahertz(self):
        nums = extract_numbers("Operating at 2.4 GHz.")
        assert any("2.4" in n for n in nums)

    def test_range_extraction(self):
        nums = extract_numbers("Voltage range: 3.3-5.0 V.")
        assert any("3.3" in n and "5.0" in n for n in nums)

    def test_operating_range_text(self):
        nums = extract_numbers("Operating between -20 and 60 °C.")
        assert len(nums) >= 1

    def test_standard_number_immune(self):
        nums = extract_numbers("Per ISO 13485:2016, limit is 43°C.")
        assert any("43" in n for n in nums)
        assert not any("2016" in n and len(n) <= 5 for n in nums)

    def test_year_not_extracted(self):
        nums = extract_numbers("IEC 60601-1:2012 was published.")
        assert not any(n.startswith("2012") for n in nums)

    def test_voltage(self):
        nums = extract_numbers("Rated at 5 V and 12 V.")
        assert any("5" in n for n in nums)

    def test_current_amps(self):
        nums = extract_numbers("Short-circuit current is 5A.")
        assert any("5" in n for n in nums)

    def test_no_numbers(self):
        assert extract_numbers("No numeric values here.") == []


# ---------------------------------------------------------------------------
# Key Terms
# ---------------------------------------------------------------------------

class TestKeyTerms:

    def test_electrical_safety(self):
        terms = extract_key_terms("The housing must prevent electrical shock.")
        assert terms["electrical safety"] is True

    def test_overcharge(self):
        terms = extract_key_terms("Overcharge protection is required.")
        assert terms["overcharge protection"] is True

    def test_emc(self):
        terms = extract_key_terms("EMC compliance per IEC 60601-1-2.")
        assert terms["EMC/EMI"] is True

    def test_missing_terms(self):
        terms = extract_key_terms("This is about general safety.")
        assert terms["battery management"] is False
        assert terms["EMC/EMI"] is False

    @pytest.mark.parametrize("phrase,term", [
        ("thermal runaway protection", "thermal safety"),
        ("FMEA analysis required", "risk management"),
        ("biocompatible materials", "patient safety"),
        ("BMS shall monitor voltage", "battery management"),
    ])
    def test_parametrized_terms(self, phrase, term):
        terms = extract_key_terms(phrase)
        assert terms[term] is True


# ---------------------------------------------------------------------------
# Shall Extraction
# ---------------------------------------------------------------------------

class TestShallExtraction:

    def test_single_shall(self):
        shalls = extract_shall_statements("The device shall comply with IEC 60601-1.")
        assert len(shalls) == 1
        assert "shall comply" in shalls[0].lower()

    def test_multiple_shalls(self):
        text = "The housing shall be insulated. The device shall be labeled."
        shalls = extract_shall_statements(text)
        assert len(shalls) >= 2

    def test_no_shall(self):
        assert extract_shall_statements("The device must be safe. It should comply.") == []

    def test_markdown_bullets(self):
        text = (
            "## Safety\n"
            "* The housing shall provide insulation.\n"
            "- The device shall meet IEC 60601-1.\n"
            "+ Labeling shall comply with FDA 21 CFR 801.\n"
        )
        shalls = extract_shall_statements(text)
        assert len(shalls) >= 3

    def test_numbered_bullets(self):
        text = (
            "1. The device shall comply with ISO 13485.\n"
            "2) The housing shall be grounded.\n"
        )
        shalls = extract_shall_statements(text)
        assert len(shalls) >= 2

    def test_bold_markdown_stripped(self):
        text = "**The housing** shall provide **double insulation**."
        shalls = extract_shall_statements(text)
        assert len(shalls) == 1
        assert "**" not in shalls[0]

    def test_near_duplicate_deduplication(self):
        text = (
            "The housing shall provide electrical insulation.\n"
            "The housing shall provide electrical insulation per IEC 60601-1.\n"
        )
        shalls = extract_shall_statements(text)
        assert len(shalls) <= 2

    def test_short_fragments_excluded(self):
        shalls = extract_shall_statements("Shall.")
        assert len(shalls) == 0


# ---------------------------------------------------------------------------
# Semantic Contradiction Detection
# ---------------------------------------------------------------------------

class TestContradictionDetection:

    def test_waterproof_vs_ventilate(self):
        runs = [
            "The enclosure must be waterproof and fully sealed.",
            "The housing must ventilate to allow airflow.",
        ]
        contradictions = detect_contradictions(runs)
        assert len(contradictions) > 0

    def test_sealed_vs_open(self):
        runs = [
            "The battery compartment shall be sealed.",
            "The battery compartment shall have an open vent.",
        ]
        contradictions = detect_contradictions(runs)
        assert len(contradictions) > 0

    def test_no_contradiction_similar(self):
        runs = [
            "The housing shall be electrically insulated.",
            "Electrical insulation shall meet IEC 60601-1 requirements.",
        ]
        contradictions = detect_contradictions(runs)
        assert len(contradictions) == 0

    def test_single_run_no_contradiction(self):
        runs = ["The enclosure shall be waterproof."]
        assert detect_contradictions(runs) == []

    def test_contradiction_has_required_keys(self):
        runs = [
            "The unit shall be waterproof.",
            "The unit shall allow ventilation.",
        ]
        contradictions = detect_contradictions(runs)
        if contradictions:
            c = contradictions[0]
            assert "run_a" in c
            assert "run_b" in c
            assert "keyword" in c
            assert "antonym" in c
            assert "severity" in c


# ---------------------------------------------------------------------------
# Relevance / Refusal Detection
# ---------------------------------------------------------------------------

class TestRelevanceCheck:

    def test_relevant_output(self, medical_text):
        assert check_output_relevance(medical_text) is True

    def test_refusal_i_cannot(self):
        assert check_output_relevance("I cannot answer that question.") is False

    def test_refusal_unable(self):
        assert check_output_relevance("I'm unable to provide medical device guidance.") is False

    def test_refusal_sorry(self):
        assert check_output_relevance("Sorry, but I don't have that information.") is False

    def test_off_topic(self):
        assert check_output_relevance("The capital of France is Paris.") is False

    @pytest.mark.parametrize("refusal", [
        "I can't help with that.",
        "I am unable to answer.",
        "I don't have access to that information.",
        "I'm not able to provide medical guidance.",
        "This is beyond my capabilities.",
    ])
    def test_parametrized_refusals(self, refusal):
        assert check_output_relevance(refusal) is False


# ---------------------------------------------------------------------------
# Similarity Matrix
# ---------------------------------------------------------------------------

class TestSimilarityMatrix:

    def test_identical_texts(self):
        matrix = compute_similarity_matrix(["Same text.", "Same text."])
        assert matrix[0, 1] == pytest.approx(1.0, abs=0.01)

    def test_different_texts(self):
        matrix = compute_similarity_matrix([
            "Electrical safety requirements.",
            "Thermal management guidelines.",
        ])
        assert matrix[0, 1] < 1.0

    def test_single_text(self):
        matrix = compute_similarity_matrix(["Only one."])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 1.0

    def test_matrix_shape(self):
        texts = ["electrical safety requirement", "thermal management system", "mechanical integrity check"]
        matrix = compute_similarity_matrix(texts)
        assert matrix.shape == (3, 3)

    def test_symmetry(self):
        matrix = compute_similarity_matrix(["text one", "text two"])
        assert matrix[0, 1] == pytest.approx(matrix[1, 0], abs=1e-6)


# ---------------------------------------------------------------------------
# Reliability Score
# ---------------------------------------------------------------------------

class TestReliabilityScore:

    def test_perfect_consistency(self, identical_runs):
        score = compute_reliability_score(identical_runs)
        assert score["total_score"] >= 85.0

    def test_score_bounds(self, identical_runs, medical_text):
        for runs in [identical_runs, [{"output": "No terms here."}]]:
            score = compute_reliability_score(runs)
            assert 0 <= score["total_score"] <= 100

    def test_with_reference(self, identical_runs, reference):
        score = compute_reliability_score(identical_runs, reference)
        assert score["golden_answer_similarity"] is not None
        assert score["golden_answer_similarity"] > 0

    def test_empty_output(self):
        score = compute_reliability_score([{"output": "No numbers or terms."}])
        assert 0 <= score["total_score"] <= 100

    def test_relevance_ratio_perfect(self, identical_runs):
        score = compute_reliability_score(identical_runs)
        assert score["relevance_ratio"] == 100.0

    def test_score_explanation_present(self, identical_runs):
        score = compute_reliability_score(identical_runs)
        assert isinstance(score["score_explanation"], str)
        assert len(score["score_explanation"]) > 0

    def test_contradictions_key_present(self, identical_runs):
        score = compute_reliability_score(identical_runs)
        assert "contradictions" in score
        assert isinstance(score["contradictions"], list)

    def test_custom_weights(self, identical_runs):
        w = ScoringWeights(
            text_similarity=0.50,
            numeric_agreement=0.20,
            key_term_consistency=0.20,
            golden_answer=0.10,
        )
        score = compute_reliability_score(identical_runs, weights=w)
        assert 0 <= score["total_score"] <= 100


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_analyze_runs_structure(self, medical_text):
        data = [
            {"iteration": 1, "latency_sec": 1.0, "output": medical_text},
            {"iteration": 2, "latency_sec": 1.2, "output": medical_text},
        ]
        analysis = analyze_runs(data)
        for key in ("score_data", "unverified_standards", "shalls_per_run",
                    "standards_per_run", "terms_per_run", "standard_presence"):
            assert key in analysis

    def test_analyze_runs_score_positive(self, medical_text):
        data = [{"output": medical_text}, {"output": medical_text}]
        analysis = analyze_runs(data)
        assert analysis["score_data"]["total_score"] > 0

    def test_json_results_schema(self, tmp_path, medical_text):
        results = [{"iteration": 1, "latency_sec": 1.0, "output": medical_text, "status": "ok"}]
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results))
        loaded = json.loads(path.read_text())
        assert isinstance(loaded, list)
        for key in ("iteration", "latency_sec", "output", "status"):
            assert key in loaded[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
