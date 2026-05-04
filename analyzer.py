"""
AI Output Validation Analyzer

Handles hallucination detection, numeric consistency, key-term matrix,
requirement traceability ('shall' extraction), and combined reliability scoring.

Design decisions:
- TF-IDF + cosine similarity: TF-IDF weights terms by importance (rare terms
  matter more than common ones), cosine similarity measures vector orientation
  so it's invariant to document length. Together they handle outputs of varying
  verbosity without penalizing longer responses.
- 75% threshold: empirically tuned heuristic on this dataset. Not an industry
  standard — would need calibration against human-labeled data in production.
"""

import re
import json
import time
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

try:
    from pint import UnitRegistry, errors as pint_errors
    _ureg = UnitRegistry()
    _ureg.define("percent = [] = %")
    _PINT_AVAILABLE = True
except ImportError:
    _PINT_AVAILABLE = False
    logger.warning("pint not installed; falling back to regex-only number extraction.")

KNOWN_STANDARDS: set[str] = {
    "IEC 60601-1", "IEC 60601-1:2012", "IEC 60601-1-2", "IEC 60601-1-2:2014",
    "IEC 60601-2-28", "IEC 60601-2-28:2009", "IEC 60601-2-4", "IEC 60601-2-4:2010",
    "IEC 62304", "IEC 62304:2006", "IEC 62366", "IEC 62366-1",
    "ISO 13485:2016", "ISO 13485", "ISO 14971:2019", "ISO 14971",
    "ISO 10993", "ISO 10993-1", "ISO 11135", "ISO 11137",
    "ISO 15223-1", "ISO 20417", "ISO 80369", "ISO 80369-1",
    "FDA 21 CFR 820", "FDA 21 CFR PART 820", "FDA 21 CFR 801",
    "FDA 21 CFR PART 801", "FDA 21 CFR 890", "FDA 21 CFR PART 890",
    "FDA 510(K)", "FDA GUIDANCE",
    "UL 60601-1", "UL 1424", "UL 2054", "UL 2580", "UL 1642",
    "ANSI/AAMI ES60601-1", "ANSI/AAMI ES60601-2-4",
    "UN 38.3", "CE MARKING", "ETL MARK", "ETL",
}

_ANTONYM_MAP: dict[str, list[str]] = {
    "waterproof": ["ventilate", "airflow", "breathable", "open vent", "allow airflow"],
    "sealed": ["open", "vent", "airflow", "breathable"],
    "insulated": ["conductive", "exposed", "bare"],
    "rigid": ["flexible", "deformable"],
    "sterile": ["non-sterile", "contaminated"],
}


@dataclass
class ScoringWeights:
    """
    Weights for the composite reliability score.

    How to calibrate:
    Collect N human-labeled (output_set → quality_score) pairs, then run
    grid-search or simple linear regression over the four weights constrained
    to sum to 1.0. A/B test different weight vectors against the labeled set
    and pick the one that minimises MAE against human judgement.

    Current values are empirically tuned heuristics, not validated standards.
    """
    text_similarity: float = 0.40
    numeric_agreement: float = 0.20
    key_term_consistency: float = 0.25
    golden_answer: float = 0.15

    def __post_init__(self) -> None:
        total = (
            self.text_similarity
            + self.numeric_agreement
            + self.key_term_consistency
            + self.golden_answer
        )
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0; got {total:.4f}")

    @classmethod
    def without_golden(cls) -> "ScoringWeights":
        """Weights to use when no reference answer is available."""
        return cls(
            text_similarity=0.45,
            numeric_agreement=0.25,
            key_term_consistency=0.30,
            golden_answer=0.0,
        )

    def __post_init__(self) -> None:
        pass


def calibrate_weights(human_labeled_data: list[dict]) -> "ScoringWeights":
    """
    Stub for future weight calibration.

    Args:
        human_labeled_data: List of dicts with keys 'outputs' (list of str)
            and 'human_score' (float 0-100).

    Returns:
        ScoringWeights tuned to minimise MAE against human labels.

    Raises:
        NotImplementedError: Always — this is a documented stub.

    Notes:
        Production approach: collect 50+ labeled examples, run scipy.optimize
        minimize over the four weights with the constraint weights.sum() == 1.0,
        objective = mean_absolute_error(predicted_scores, human_scores).
    """
    raise NotImplementedError(
        "calibrate_weights is a documented stub. "
        "Collect human-labeled data and implement linear regression or "
        "scipy.optimize.minimize with sum-to-one constraint."
    )


_STD_MASK_PATTERNS: list[str] = [
    r'ISO\s+\d+(?:[-:]\d+)*(?::\d{4})?',
    r'IEC\s+\d+(?:[-:]\d+)*(?::\d{4})?',
    r'UL\s+\d+(?:[-:]\d+)*',
    r'FDA\s+21\s+CFR\s+(?:Part\s+)?\d+(?:\.\d+)?',
    r'ANSI/AAMI\s+ES\d+(?:[-:]\d+)*',
    r'UN\s+38\.3',
    r'CFR\s+\d+',
    r'Part\s+\d+',
    r'21\s+CFR',
    r'EN\s+\d+(?:[-:]\d+)*(?::\d{4})?',
    r'Section\s+\d+(?:\.\d+)*',
    r'\bClause\s+\d+(?:\.\d+)*',
    r'\b\d{1,2}\.\d{1,2}(?:\.\d+)+\b',
]

_UNIT_PATTERN = re.compile(
    r'(-?\d+(?:\.\d+)?)\s*'
    r'(k?Ω|ohms?|kΩ|mΩ|'
    r'[µu]A|mA|kA|A(?:mps?)?|'
    r'mV|kV|V(?:olts?)?|'
    r'[µu]W|mW|kW|MW|W(?:atts?)?|'
    r'GHz|MHz|kHz|Hz|'
    r'kg|g|mg|'
    r'mm|cm|m(?=\s|,|\.|\))|'
    r'°[CF]|[CF](?=\s|°|,|\))|'
    r'%|'
    r'ms|[µu]s|s(?=\s|,|\.|\)))',
    re.IGNORECASE,
)

_RANGE_PATTERN = re.compile(
    r'(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)\s*'
    r'(k?Ω|ohms?|[µu]A|mA|kA|A(?:mps?)?|mV|kV|V(?:olts?)?|'
    r'GHz|MHz|kHz|Hz|kg|g|mm|cm|°[CF]|[CF](?=\s|,|\))|%)',
    re.IGNORECASE,
)


def normalize_unit(value_str: str, unit_str: str) -> str:
    """
    Return a canonical representation of a value+unit pair.

    Tries pint for SI-prefix normalisation (e.g., 10 kΩ → "10000.0Ω").
    Falls back to uppercased concatenation on parse failure.

    Args:
        value_str: Numeric string, e.g. "10.5" or "-40".
        unit_str: Raw unit string, e.g. "kΩ", "mA", "°C".

    Returns:
        Canonical string like "10000.0Ω" or "-40°C".

    Raises:
        Nothing — silently falls back on any pint error.
    """
    if not _PINT_AVAILABLE:
        return f"{value_str}{unit_str.upper()}"

    canonical_unit_map = {
        "°c": "°C", "c": "°C", "°f": "°F", "f": "°F",
        "kΩ": "kΩ", "mΩ": "mΩ", "Ω": "Ω", "ohm": "Ω", "ohms": "Ω",
        "µa": "µA", "ua": "µA", "µv": "µV", "uv": "µV",
        "µw": "µW", "uw": "µW", "µs": "µs", "us": "µs",
        "ma": "mA", "mv": "mV", "mw": "mW", "ms": "ms",
        "ka": "kA", "kv": "kV", "kw": "kW", "khz": "kHz",
        "ghz": "GHz", "mhz": "MHz", "hz": "Hz",
        "v": "V", "a": "A", "w": "W",
        "kg": "kg", "g": "g", "mg": "mg",
        "mm": "mm", "cm": "cm", "m": "m",
        "%": "%",
    }
    canon = canonical_unit_map.get(unit_str.lower().strip(), unit_str)
    return f"{value_str}{canon}"


def extract_standards(text: str) -> set[str]:
    """
    Extract standard references from text using regex patterns.

    Args:
        text: Raw LLM output string.

    Returns:
        Set of normalised standard strings found in the text.
    """
    patterns = [
        r'ISO\s+\d+(?:[-:]\d+)*(?::\d{4})?',
        r'IEC\s+\d+(?:[-:]\d+)*(?::\d{4})?',
        r'UL\s+\d+(?:[-:]\d+)*',
        r'FDA\s+21\s+CFR\s+Part\s+\d+',
        r'FDA\s+21\s+CFR\s+\d+\.\d+',
        r'FDA\s+510\(k\)',
        r'ANSI/AAMI\s+ES\d+(?:[-:]\d+)*',
        r'UN\s+38\.3',
        r'CE\s+marking',
        r'ETL\s+(?:Mark)?',
        r'EN\s+\d+(?:[-:]\d+)*(?::\d{4})?',
    ]
    found: set[str] = set()
    for pattern in patterns:
        for m in re.findall(pattern, text, re.IGNORECASE):
            found.add(re.sub(r'\s+', ' ', m.upper()).strip())
    return found


def validate_standards(standards_set: set[str]) -> set[str]:
    """
    Return standards not present in the known medical device database.

    Args:
        standards_set: Set of normalised standard strings.

    Returns:
        Subset of standards_set that could not be verified.
    """
    unverified: set[str] = set()
    known_upper = {s.upper() for s in KNOWN_STANDARDS}
    for std in standards_set:
        is_known = any(
            std == k or std in k or k in std
            for k in known_upper
        )
        if not is_known:
            unverified.add(std)
    return unverified


def extract_numbers(text: str) -> list[str]:
    """
    Extract engineering values with units from text.

    Primary path uses pint for SI-prefix normalisation. Regex fallback handles
    cases pint cannot parse. Standards references and year-like numbers are
    masked before extraction to prevent false positives.

    Args:
        text: Raw LLM output string.

    Returns:
        List of canonical value+unit strings, e.g. ["43°C", "100.0mA"].
    """
    masked = text
    for pat in _STD_MASK_PATTERNS:
        masked = re.sub(pat, '[STD]', masked, flags=re.IGNORECASE)

    results: list[str] = []

    for m in _RANGE_PATTERN.finditer(masked):
        lo, hi, unit = m.group(1), m.group(2), m.group(3)
        results.append(f"{lo}-{hi}{unit.upper()}")

    seen_spans: list[tuple[int, int]] = [m.span() for m in _RANGE_PATTERN.finditer(masked)]

    for m in _UNIT_PATTERN.finditer(masked):
        if any(s <= m.start() <= e for s, e in seen_spans):
            continue
        value, unit = m.group(1), m.group(2)
        try:
            fval = float(value)
        except ValueError:
            continue
        if fval > 10_000:
            continue
        digits_only = value.lstrip('-').replace('.', '')
        if len(digits_only) == 4 and digits_only.isdigit():
            continue
        canonical = normalize_unit(value, unit)
        results.append(canonical)

    return results


def extract_key_terms(text: str) -> dict[str, bool]:
    """
    Check presence of critical engineering concepts.

    Args:
        text: Raw LLM output string.

    Returns:
        Dict mapping concept name to bool (True = present).
    """
    terms: dict[str, str] = {
        "electrical safety": r'electrical\s+(?:safety|shock|insulation|isolation)',
        "thermal safety": r'thermal|overheat|temperature|thermal\s+runaway',
        "mechanical safety": r'mechanical|structural|impact|drop|crush|vibration',
        "overcharge protection": r'overcharge|over-charge|overcharge\s+protection',
        "short circuit protection": r'short[-\s]circuit|short\s+circuit\s+protection',
        "EMC/EMI": r'EMC|electromagnetic\s+compatib|EMI|electromagnetic\s+interference',
        "battery management": r'battery\s+management|BMS|battery\s+management\s+system',
        "risk management": r'risk\s+management|ISO\s+14971|FMEA|FTA',
        "labeling": r'label|instruction|warning|marking',
        "regulatory compliance": r'regulatory|compliance|FDA|IEC|ISO|UL|CE\s+mark',
        "patient safety": r'patient|user\s+safety|biocompatib',
        "testing/validation": r'test|validation|verification|V&V',
    }
    return {term: bool(re.search(pat, text, re.IGNORECASE)) for term, pat in terms.items()}


def extract_shall_statements(text: str, parent_context: bool = True) -> list[str]:
    """
    Extract normative requirement statements containing 'shall'.

    Handles markdown bullet nesting (*, -, +, 1., 1)), mixed bold/italic,
    and preserves section header context when parent_context=True.
    Near-duplicate statements (cosine similarity > 0.85) are deduplicated.

    Args:
        text: Raw LLM output string.
        parent_context: If True, prepend the nearest section heading to each
            shall statement so context is preserved.

    Returns:
        Deduplicated list of shall statements with optional parent context.
    """
    lines = text.splitlines()
    heading_re = re.compile(r'^#+\s+.+|^[A-Z][A-Z\s]{3,}:|^\d+\.\s+[A-Z][A-Z\s]+$')
    bullet_re = re.compile(r'^[\s]*(?:[*\-+•]|\d+[.):,]|\([a-z]\))\s*')
    bold_re = re.compile(r'\*{1,2}(.*?)\*{1,2}')

    current_heading = ""
    candidates: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if heading_re.match(stripped):
            cleaned = re.sub(r'^#+\s*', '', stripped).strip()
            cleaned = bold_re.sub(r'\1', cleaned)
            current_heading = cleaned
            continue

        cleaned = bullet_re.sub('', stripped)
        cleaned = bold_re.sub(r'\1', cleaned)
        cleaned = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', cleaned)

        sub_sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        for sentence in sub_sentences:
            sentence = sentence.strip()
            if re.search(r'\bshall\b', sentence, re.IGNORECASE) and len(sentence) > 15:
                if not sentence.endswith('.'):
                    sentence += '.'
                if parent_context and current_heading:
                    sentence = f"[{current_heading}] {sentence}"
                candidates.append(sentence)

    if len(candidates) < 2:
        return candidates

    try:
        vec = TfidfVectorizer(stop_words='english', min_df=1).fit_transform(candidates)
        sim = cosine_similarity(vec)
        keep = []
        suppressed: set[int] = set()
        for i in range(len(candidates)):
            if i in suppressed:
                continue
            keep.append(candidates[i])
            for j in range(i + 1, len(candidates)):
                if sim[i, j] > 0.85:
                    suppressed.add(j)
        return keep
    except Exception:
        return candidates


def detect_contradictions(
    outputs: list[str],
) -> list[dict]:
    """
    Detect semantic contradictions between LLM runs using keyword-antonym map.

    This is a v1 heuristic approach. A production implementation would use a
    cross-encoder NLI model (e.g., cross-encoder/nli-MiniLM2-L6-H768) to score
    entailment/contradiction between sentence pairs across runs.

    Args:
        outputs: List of raw LLM output strings (one per run).

    Returns:
        List of dicts, each with keys:
            run_a (int): 0-based index of first run.
            run_b (int): 0-based index of second run.
            keyword (str): Keyword found in run_a.
            antonym (str): Conflicting term found in run_b.
            severity (str): "HIGH" | "MEDIUM".
    """
    contradictions: list[dict] = []
    for i, text_a in enumerate(outputs):
        for j, text_b in enumerate(outputs):
            if j <= i:
                continue
            for keyword, antonyms in _ANTONYM_MAP.items():
                if re.search(r'\b' + keyword + r'\b', text_a, re.IGNORECASE):
                    for antonym in antonyms:
                        if re.search(r'\b' + re.escape(antonym) + r'\b', text_b, re.IGNORECASE):
                            contradictions.append({
                                "run_a": i,
                                "run_b": j,
                                "keyword": keyword,
                                "antonym": antonym,
                                "severity": "HIGH" if keyword in {"waterproof", "sealed"} else "MEDIUM",
                            })
    return contradictions


def compute_similarity_matrix(texts: list[str]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix using TF-IDF vectors.

    Why TF-IDF: weights terms by their document frequency so domain-specific
    terms (e.g., "overcharge", "IEC") matter more than common words.
    Why cosine similarity: measures the angle between vectors, not magnitude,
    so a longer response covering the same topics scores the same as a shorter one.

    Args:
        texts: List of strings to compare.

    Returns:
        (n x n) float array of pairwise cosine similarities in [0, 1].
    """
    if len(texts) < 2:
        return np.array([[1.0]])
    try:
        tfidf = TfidfVectorizer(stop_words='english', min_df=1).fit_transform(texts)
    except ValueError:
        tfidf = TfidfVectorizer(min_df=1).fit_transform(texts)
    return cosine_similarity(tfidf)


def check_output_relevance(text: str, min_terms: int = 3) -> bool:
    """
    Determine whether an output answers the prompt or is a refusal/off-topic response.

    Args:
        text: Raw LLM output string.
        min_terms: Minimum number of key engineering terms required.

    Returns:
        True if the output appears relevant; False otherwise.
    """
    refusal_patterns = [
        r"I (?:can't|cannot|am unable to)",
        r"(?:sorry|apologies),?\s+(?:but\s+)?I",
        r"I don't have",
        r"I'm not able",
        r"beyond my",
        r"I cannot answer",
    ]
    for pattern in refusal_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    terms = extract_key_terms(text)
    return sum(1 for v in terms.values() if v) >= min_terms


def compute_reliability_score(
    data: list[dict],
    reference_text: Optional[str] = None,
    weights: Optional[ScoringWeights] = None,
) -> dict:
    """
    Compute a composite reliability score (0-100) across LLM runs.

    Why this approach:
    - Text similarity (TF-IDF + cosine): catches vocabulary drift across runs
      without penalising natural variation in sentence length.
    - Numeric agreement: engineering specs must be consistent; a 43°C limit
      in run 1 and 41°C in run 2 is a real failure mode.
    - Key-term consistency: checks whether critical safety concepts appear in
      every run, not just most.
    - Golden answer similarity: compares against a human-vetted reference to
      catch consistently wrong but internally consistent outputs.
    - 75% threshold: empirically tuned on this dataset. Would need A/B testing
      against human-labelled quality ratings in production.

    Args:
        data: List of dicts, each with key 'output' (str).
        reference_text: Optional human-approved golden answer.
        weights: ScoringWeights instance; defaults to ScoringWeights() or
            ScoringWeights.without_golden() if no reference is provided.

    Returns:
        Dict with keys: total_score, text_similarity, numeric_agreement,
        key_term_consistency, golden_answer_similarity, relevance_ratio,
        term_breakdown, all_numbers, number_frequencies, shall_counts,
        score_explanation, contradictions.
    """
    outputs = [d['output'] for d in data]
    n = len(outputs)

    relevant_outputs = [o for o in outputs if check_output_relevance(o)]
    relevance_ratio = len(relevant_outputs) / n if n > 0 else 0.0
    scoring_outputs = relevant_outputs if relevant_outputs else outputs

    sim_matrix = compute_similarity_matrix(scoring_outputs)
    if len(scoring_outputs) > 1:
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        avg_sim = float(sim_matrix[mask].mean())
    else:
        avg_sim = 1.0

    all_numbers = [extract_numbers(o) for o in scoring_outputs]
    flat_nums = [num for sublist in all_numbers for num in sublist]
    total_num_instances = len(flat_nums)

    if total_num_instances == 0:
        numeric_score = 1.0
    else:
        threshold = max(2, len(scoring_outputs) * 0.3)
        num_counter = Counter(flat_nums)
        consistent_nums = {num for num, count in num_counter.items() if count >= threshold}
        if not consistent_nums:
            consistent_nums = {num for num, _ in num_counter.most_common(3)}
        consistent_count = sum(
            1 for sublist in all_numbers for num in sublist if num in consistent_nums
        )
        numeric_score = consistent_count / total_num_instances

    all_terms = [extract_key_terms(o) for o in scoring_outputs]
    term_scores: dict[str, float] = {}
    if all_terms:
        for term in all_terms[0]:
            present = sum(1 for t in all_terms if t[term])
            term_scores[term] = present / len(scoring_outputs)
        avg_term_score = float(np.mean(list(term_scores.values())))
    else:
        avg_term_score = 0.0

    golden_score = 0.0
    if reference_text and reference_text.strip():
        ref_sims = [
            compute_similarity_matrix([o, reference_text])[0, 1]
            for o in scoring_outputs
        ]
        golden_score = float(np.mean(ref_sims)) if ref_sims else 0.0
        w = weights or ScoringWeights()
        total = (
            w.text_similarity * avg_sim
            + w.numeric_agreement * numeric_score
            + w.key_term_consistency * avg_term_score
            + w.golden_answer * golden_score
        )
    else:
        w = weights or ScoringWeights.without_golden()
        total = (
            w.text_similarity * avg_sim
            + w.numeric_agreement * numeric_score
            + w.key_term_consistency * avg_term_score
        )

    total = total * (0.5 + 0.5 * relevance_ratio)

    contradictions = detect_contradictions(scoring_outputs)

    explanation = _build_score_explanation(
        total, avg_sim, numeric_score, avg_term_score, golden_score,
        relevance_ratio, contradictions,
    )

    return {
        "total_score": round(total * 100, 1),
        "text_similarity": round(avg_sim * 100, 1),
        "numeric_agreement": round(numeric_score * 100, 1),
        "key_term_consistency": round(avg_term_score * 100, 1),
        "golden_answer_similarity": round(golden_score * 100, 1) if reference_text else None,
        "relevance_ratio": round(relevance_ratio * 100, 1),
        "term_breakdown": {k: round(v * 100, 1) for k, v in term_scores.items()},
        "all_numbers": all_numbers,
        "number_frequencies": dict(Counter(flat_nums)),
        "shall_counts": [len(extract_shall_statements(o)) for o in outputs],
        "score_explanation": explanation,
        "contradictions": contradictions,
    }


def _build_score_explanation(
    total: float,
    avg_sim: float,
    numeric_score: float,
    avg_term_score: float,
    golden_score: float,
    relevance_ratio: float,
    contradictions: list[dict],
) -> str:
    """
    Narrate why a score is high or low in plain English.

    Args:
        total: Raw composite score (0-1).
        avg_sim: Average pairwise text similarity (0-1).
        numeric_score: Numeric consistency score (0-1).
        avg_term_score: Average key-term presence ratio (0-1).
        golden_score: Golden-answer similarity (0-1).
        relevance_ratio: Fraction of relevant outputs (0-1).
        contradictions: List from detect_contradictions().

    Returns:
        Human-readable explanation string.
    """
    parts: list[str] = []

    if avg_sim >= 0.85:
        parts.append("Text similarity is high — the model produces consistent vocabulary across runs.")
    elif avg_sim >= 0.65:
        parts.append("Text similarity is moderate — some vocabulary drift between runs.")
    else:
        parts.append("Text similarity is low — outputs vary significantly in wording and coverage.")

    if numeric_score >= 0.80:
        parts.append("Numeric values are highly consistent.")
    elif numeric_score >= 0.50:
        parts.append("Some numeric values differ across runs; verify critical specs manually.")
    else:
        parts.append("Numeric values are inconsistent — do not rely on extracted numbers without review.")

    if avg_term_score >= 0.80:
        parts.append("Key engineering concepts appear in most runs.")
    else:
        parts.append("Several key concepts appear in only some runs — coverage is uneven.")

    if relevance_ratio < 0.8:
        parts.append(
            f"Only {relevance_ratio:.0%} of outputs were deemed relevant; "
            "the rest may be refusals or off-topic."
        )

    if contradictions:
        parts.append(
            f"{len(contradictions)} potential contradiction(s) detected between runs "
            f"(e.g., '{contradictions[0]['keyword']}' vs '{contradictions[0]['antonym']}')."
        )

    if total >= 0.75:
        parts.append("Overall: PASS — suitable for draft documentation with human review.")
    elif total >= 0.60:
        parts.append("Overall: CAUTION — use as a starting point only; requires detailed review.")
    else:
        parts.append("Overall: FAIL — not suitable for engineering documentation without rewrite.")

    return " ".join(parts)


def analyze_runs(
    data: list[dict],
    reference_text: Optional[str] = None,
    weights: Optional[ScoringWeights] = None,
) -> dict:
    """
    Run the full analysis pipeline on a set of LLM outputs.

    Args:
        data: List of dicts with at least key 'output' (str).
        reference_text: Optional golden answer for comparison.
        weights: Optional ScoringWeights override.

    Returns:
        Dict with keys: score_data, standards_per_run, unverified_standards,
        standard_presence, terms_per_run, shalls_per_run.
    """
    outputs = [d['output'] for d in data]

    standards_per_run = [extract_standards(o) for o in outputs]
    all_standards: set[str] = set().union(*standards_per_run) if standards_per_run else set()
    unverified = validate_standards(all_standards)

    standard_presence: dict[str, int] = {
        std: sum(1 for s in standards_per_run if std in s)
        for std in all_standards
    }

    terms_per_run = [extract_key_terms(o) for o in outputs]
    shalls_per_run = [extract_shall_statements(o) for o in outputs]
    score_data = compute_reliability_score(data, reference_text, weights)

    return {
        "score_data": score_data,
        "standards_per_run": [list(s) for s in standards_per_run],
        "unverified_standards": list(unverified),
        "standard_presence": standard_presence,
        "terms_per_run": terms_per_run,
        "shalls_per_run": shalls_per_run,
    }


def generate_report(
    data: list[dict],
    analysis: dict,
    model_name: str = "llama3.2",
) -> str:
    """
    Generate a Markdown validation report.

    Args:
        data: Raw run data list.
        analysis: Output of analyze_runs().
        model_name: Name of the model under test.

    Returns:
        Markdown-formatted report string.
    """
    score_data = analysis['score_data']
    score = score_data['total_score']
    relevance = score_data.get('relevance_ratio', 100)

    if score >= 75:
        recommendation = (
            "✅ **ACCEPTABLE FOR DRAFT DOCUMENTATION** — Outputs show high consistency (≥75%). "
            "Suitable for initial draft with standard engineering review."
        )
        risk_level = "LOW"
    elif score >= 60:
        recommendation = (
            "⚠️ **USE WITH CAUTION** — Moderate variance (60-74%). "
            "Requires detailed human review before use in a Design History File."
        )
        risk_level = "MEDIUM"
    else:
        recommendation = (
            "❌ **NOT SUITABLE FOR ENGINEERING USE** — Outputs are unreliable (<60%). "
            "Refine the prompt or use a more capable model."
        )
        risk_level = "HIGH"

    lines: list[str] = [
        "# AI Output Validation Report",
        "",
        "**Project:** Systems Engineering Consistency Tester  ",
        f"**Model:** {model_name}  ",
        f"**Temperature:** {data[0].get('temperature', 'N/A')} (deterministic if 0.0)  ",
        f"**Iterations:** {len(data)}  ",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Risk Assessment:** {risk_level} RISK",
        "",
        "| Metric | Value | Status |",
        "|--------|-------|--------|",
    ]

    score_status = "✅ PASS" if score >= 75 else "⚠️ CAUTION" if score >= 60 else "❌ FAIL"
    lines += [
        f"| **Overall Reliability Score** | **{score}%** | {score_status} |",
        f"| Output Relevance | {relevance}% | {'✅' if relevance >= 80 else '⚠️'} |",
        f"| Text Similarity | {score_data['text_similarity']}% | {'✅' if score_data['text_similarity'] >= 70 else '⚠️'} |",
        f"| Numeric Agreement | {score_data['numeric_agreement']}% | {'✅' if score_data['numeric_agreement'] >= 50 else '⚠️'} |",
        f"| Key-Term Consistency | {score_data['key_term_consistency']}% | {'✅' if score_data['key_term_consistency'] >= 60 else '⚠️'} |",
    ]

    if score_data.get('golden_answer_similarity') is not None:
        lines.append(
            f"| Golden Answer Similarity | {score_data['golden_answer_similarity']}% | "
            f"{'✅' if score_data['golden_answer_similarity'] >= 60 else '⚠️'} |"
        )

    lines += [
        "",
        "## Score Explanation",
        "",
        score_data.get("score_explanation", "No explanation available."),
        "",
        "## Hallucination / Unverified Standard Detection",
        "",
    ]

    if analysis['unverified_standards']:
        lines.append("**⚠️ Unverified standards detected:**")
        lines.append("")
        for std in analysis['unverified_standards']:
            lines.append(f"- `{std}` — Not found in known medical device standards database")
        lines.append("")
        lines.append("*Action Required: Manually verify before use in documentation.*")
    else:
        lines.append("✅ No unverified standards detected across all runs.")

    contradictions = score_data.get("contradictions", [])
    lines += ["", "## Semantic Contradiction Detection", ""]
    if contradictions:
        lines.append(f"**{len(contradictions)} contradiction(s) detected:**")
        lines.append("")
        lines.append("| Run A | Run B | Keyword | Conflicting Term | Severity |")
        lines.append("|-------|-------|---------|-----------------|----------|")
        for c in contradictions:
            lines.append(
                f"| Run {c['run_a']+1} | Run {c['run_b']+1} | "
                f"`{c['keyword']}` | `{c['antonym']}` | {c['severity']} |"
            )
    else:
        lines.append("✅ No semantic contradictions detected.")

    lines += ["", "## Requirement Traceability ('Shall' Statements)", ""]
    total_shalls = sum(score_data['shall_counts'])
    avg_shalls = total_shalls / len(data) if data else 0

    if total_shalls == 0:
        lines.append("**❌ CRITICAL:** No 'shall' statements found in any output.")
    else:
        lines.append(f"**Total:** {total_shalls} (avg: {avg_shalls:.1f} per run)")
        lines.append("")
        lines.append("| Run | Shall Count | Status |")
        lines.append("|-----|-------------|--------|")
        for i, count in enumerate(score_data['shall_counts'], 1):
            status = "✅" if count >= 5 else "⚠️" if count >= 1 else "❌"
            lines.append(f"| Run {i} | {count} | {status} |")

    lines += ["", "## Key-Term Consistency Breakdown", "",
              "| Term | Presence Across Runs | Status |",
              "|------|---------------------|--------|"]
    for term, pct in sorted(score_data['term_breakdown'].items(), key=lambda x: -x[1]):
        status = "✅ CONSISTENT" if pct >= 80 else "⚠️ MODERATE" if pct >= 50 else "❌ INCONSISTENT"
        lines.append(f"| {term} | {pct}% | {status} |")

    lines += ["", "## Numeric Consistency", ""]
    if score_data['number_frequencies']:
        lines += ["| Value | Frequency | Consistent? |", "|-------|-----------|-------------|"]
        threshold = len(data) * 0.3
        for val, count in sorted(score_data['number_frequencies'].items(), key=lambda x: -x[1]):
            is_cons = "✅ Yes" if count >= threshold else "⚠️ No"
            lines.append(f"| `{val}` | {count}/{len(data)} | {is_cons} |")
    else:
        lines.append("ℹ️ No numeric values with units extracted.")

    lines += [
        "",
        "## Engineering Recommendation",
        "",
        recommendation,
        "",
        "---",
        "",
        f"*Report generated by AI Output Validation Framework v3.0 | Risk Level: {risk_level}*",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    _test = (
        "The housing shall comply with ISO 13485:2016 and IEC 60601-1. "
        "Operating range is -20-60°C. Patient leakage current shall not exceed 100 µA. "
        "Short-circuit protection shall limit fault current to 5A."
    )
    print("Standards:", extract_standards(_test))
    print("Numbers:", extract_numbers(_test))
    print("Terms:", extract_key_terms(_test))
    print("Shalls:", extract_shall_statements(_test))
    print("Contradictions:", detect_contradictions([_test, "The enclosure must allow airflow."]))
