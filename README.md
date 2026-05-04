# AI Output Validation Framework

A proof of concept tool for testing how consistently a local LLM answers the same engineering prompt.

---

## What & Why

I wanted to see the difference in output when you ask an LLM the same technical question ten times. Is it possible to actually get the same standards referenced? The same numbers? The same requirement structure? the answer is "sometimes, but not always" (could probably be most times with better testing)

I chose medical device requirements as the test domain because safety standards are publicly listed and their language is unambiguous ("shall" statements, specific numeric limits). That made it easy to define what a correct, consistent answer looks like. Im not an expert. 

---

## What I Learned

**Prompt engineering matters more than I expected and is probably the most important.** Changing one sentence in the prompt — explicitly asking for "shall" statements — took detected requirements from zero to sixteen per run. The analysis tool didn't change; the prompt did. Still tried to make the prompt not too complex.

**Regex is brittle for NLP.** I spent a lot of time tweaking regex patterns to extract numbers like `43°C` or `100 µA`. The patterns kept breaking on edge cases: ranges like `3.3–5.0 V`, negative temperatures, SI prefixes like `kΩ` or `µA`. The right fix is a unit-parsing library (pint), not more regex.

**Cosine similarity vs. Euclidean distance.** I learned why cosine similarity is preferred for text: it measures the angle between vectors, so a longer response covering the same topics scores the same as a shorter one. Euclidean distance penalises length, which isn't what you want when comparing documents.

**Heuristic scoring requires calibration.** My scoring weights (40% text similarity, 25% key-term consistency, 20% numeric agreement, 15% golden-answer match) are guesses. I picked them based on what felt right, then adjusted when scores seemed off. In a real system, you'd collect human-labelled examples and fit the weights against those. The 75% pass threshold is similarly a heuristic — not an industry standard.

**Consistency ≠ correctness.** A model can confidently give the same wrong answer ten times. That's why comparing against a human-written reference answer matters — internal consistency alone isn't enough.

---

## Limitations

- Obviously Not production-ready. Don't use this for anything that matters please.
- The standards database is manually curated regex. It'll miss new or niche standards and has no live connection to ISO or IEC.
- Not trained on real medical device expertise. simply used for it being well defined.
- Scoring weights are empirically tuned heuristics, not validated against human judgement.
- No semantic contradiction detection yet
- Local LLM limitations. Llama 3.2 doesn't know medical device regulations well. Scores would improve with a frontier model.

---

## Tech Stack

Python, scikit-learn, Streamlit, Ollama, pint, pytest

---

## How to Run

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai) installed, `llama3.2` model pulled.

```bash
ollama pull llama3.2
pip install -r requirements.txt

python test_engine.py          # run 10 iterations, saves JSON results
streamlit run app.py           # launch dashboard at localhost:8501
python -m pytest test_validator.py -v  # run unit tests
```

---

## Architecture

```
config.yaml → test_engine.py → validation_results.json
                                        ↓
reference_answer.txt → analyzer.py ────┘
                              ↓
                   validation_analysis.json
                              ↓
                           app.py  (Streamlit dashboard)
```

---

## Scoring

| Score | Meaning |
|-------|---------|
| ≥75% | Consistent enough to use as a draft starting point with human review |
| 60–74% | Use with caution; verify everything before relying on it |
| <60% | Too inconsistent; give better prompt or switch models |

These thresholds are just heuristics, not validated standards.

---

## Next Steps

Things to do if i want to keep working:

1. Replace the keyword-antonym contradiction map with a proper NLI model (e.g., `cross-encoder/nli-MiniLM2-L6-H768`) to catch logical contradictions between runs.
2. Calibrate scoring weights against human-labelled examples using `scipy.optimize`.
3. Add a live standards API integration so the database doesn't have to be manually curated.
4. Deploy a public demo (Streamlit Community Cloud config is already included).

---

## Project Structure

```
.
├── analyzer.py              # Core: extraction, scoring, contradiction detection
├── app.py                   # Streamlit dashboard
├── test_engine.py           # Ollama benchmark runner
├── test_validator.py        # pytest unit tests
├── config.yaml              # Prompt, model, iteration settings
├── reference_answer.txt     # Human-written golden answer
├── requirements.txt
├── .streamlit/config.toml   # Deployment config
├── .github/workflows/ci.yml # CI: pytest + lint on push
└── docs/INTERVIEW_GUIDE.md
```
