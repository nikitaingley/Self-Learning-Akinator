# Genie AI — Akinator-Style Intelligent Guessing Game

An Akinator-style guessing game built with Python and Streamlit.  
The AI thinks of questions, eliminates candidates using Bayesian reasoning, and tries to guess what you're thinking of.

---

## Features

- **Bayesian guessing engine** — a probability distribution over all candidates is updated after every answer using Bayes' rule with graded evidence (Yes / Probably / Don't Know / Probably Not / No).
- **Information-gain question selection** — every question is chosen to maximise expected entropy reduction over the current candidate set.
- **Smart guessing threshold** — the AI only guesses when the leading candidate is both sufficiently probable **and** well-separated from the second-best candidate.  No premature guesses after two or three broad answers.
- **Candidate elimination** — wrong guesses are removed and play continues.
- **Group exclusion** — once a question from a mutually-exclusive group is answered "Yes" (e.g. "Is the person from India?"), the engine skips all sibling questions in that group.
- **Self-learning** — correct guesses reinforce the stored probabilities; wrong guesses can be taught with new facts.
- **Developer / debug mode** — see candidate scores, information gain, and elimination reasons in real time.
- **Expanded seed dataset** — 145+ items across all 5 categories with rich attributes.
- **26 automated tests** covering algorithmic correctness, guessing quality, and edge cases.

---

## Quick start

```bash
# 1. Use the included virtual environment (Python 3.14)
source .venv/bin/activate   # macOS / Linux
# or: .venv\Scripts\activate  (Windows)

# 2. Launch
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** On first run, Streamlit may ask for an email address — just press Enter to skip.

---

## Dataset

The game ships with a built-in seed dataset (~145 items).  
To rebuild it (offline, fast):

```bash
python build_dataset.py --seed
```

To build a much larger dataset from Wikidata (requires internet):

```bash
# ~1,500 well-known items per category (a few minutes)
python build_dataset.py --depth quick --force

# ~6,000 items per category (15-30 minutes)
python build_dataset.py --force

# 25,000+ items (hours, huge dataset)
python build_dataset.py --depth deep --force
```

Wikidata queries are cached in `./wikidata_cache/` so interrupted runs resume automatically.

> **Warning:** `--force` wipes the existing database including learned items.

---

## Running tests

```bash
# Human-readable output (no dependencies beyond the venv)
python test_engine.py

# pytest (install once: pip install pytest)
pytest test_engine.py -v
```

All 26 tests pass:

| Suite | Tests |
|-------|-------|
| `TestEngineBasics` | 14 — posterior mechanics, undo, reject, group exclusion, edge cases |
| `TestGuessingQuality` | 7 — does-not-guess-too-early, disambiguation, probably/probably-not handling |
| `TestEdgeCases` | 5 — empty mask, all don't-know, new candidates, large sets, category imbalance |

---

## Architecture

```
app.py             Streamlit UI (screens, callbacks, CSS)
engine.py          Bayesian guessing engine (information-gain question selection)
db.py              SQLite + numpy matrix storage and self-learning
build_dataset.py   Seed dataset and Wikidata pipeline
analytics.py       Analytics dashboard
test_engine.py     Automated test suite
akinator.db        SQLite database (items, questions, game logs)
matrix.npy         uint8 P[item, question] probability matrix
.streamlit/        Streamlit configuration (headless, no usage stats)
```

### How the engine works

1. Every item starts with a prior proportional to `sqrt(popularity)`.
2. `P[item, question]` is `uint8`: `0 = no`, `255 = yes`, `128 = unknown`.
3. For each answer `a ∈ [0, 1]`:  
   `likelihood = a × p_yes + (1−a) × (1−p_yes)`  
   The posterior is multiplied by the likelihood and renormalised.
4. `next_question()` computes expected information gain (entropy drop) for all unasked questions and returns the best one, skipping:
   - already asked questions
   - questions in the same exclusive group as a "yes" answer
5. `should_guess()` returns `True` only when:
   - The top candidate probability ≥ `GUESS_CONF` (0.85), **and**
   - The gap over the 2nd candidate ≥ `CONF_GAP` (0.30), **or**
   - No question has info gain ≥ `MIN_GAIN`, **or**
   - `MAX_QUESTIONS` (30) is reached.

---

## Answer options

| Button | Meaning | Evidence weight |
|--------|---------|----------------|
| ✅ Yes | Definitely true | 1.0 |
| 🤔 Probably | Likely true | 0.75 |
| ❓ Don't know | No information | no update |
| 🤷 Probably not | Likely false | 0.25 |
| ❌ No | Definitely false | 0.0 |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `matrix.npy does not match akinator.db` | Delete both files and run `python build_dataset.py --seed` |
| App shows "Not enough items in that theme" | Switch to "Everything" or rebuild the dataset |
| Streamlit credentials error on startup | Cosmetic warning — the app still runs. Run `streamlit config show` to check settings. |
| Real Akinator mode fails | Requires internet and the `akinator` package to be working. Use "My AI" mode offline. |

---

## File layout

```
.
├── app.py                 # Main Streamlit application
├── engine.py              # Bayesian guessing engine
├── db.py                  # Database and matrix management
├── build_dataset.py       # Dataset pipeline (seed + Wikidata)
├── analytics.py           # Game analytics dashboard
├── test_engine.py         # Automated tests
├── akinator.db            # SQLite (generated)
├── matrix.npy             # Probability matrix (generated)
├── .streamlit/
│   └── config.toml        # Streamlit config (headless, no stats)
├── wikidata_cache/        # SPARQL query cache (generated)
└── .venv/                 # Python virtual environment
```
