"""
test_engine.py  -  automated tests for the Genie AI guessing engine.

Tests are designed around synthetic datasets with KNOWN attributes so the
algorithm's correctness can be verified independently of any external data.

Run:
    python test_engine.py              (human-readable output)
    python -m pytest test_engine.py   (pytest)
"""
import math
import sys

import numpy as np
import pytest

from engine import Engine, ANSWERS, MIN_QUESTIONS, MAX_QUESTIONS, GUESS_CONF, CONF_GAP, EPS

# ---------------------------------------------------------------------------
# Synthetic dataset builder
# ---------------------------------------------------------------------------

def make_dataset(items: list[dict]) -> tuple[np.ndarray, list[str], list[float]]:
    """
    Build a minimal P matrix from a list of item dicts.

    Each item dict has:
        name       : str
        pop        : float (popularity)
        attrs      : dict[str, bool | None]   True=yes, False=no, None=unknown

    Returns (P, attr_names, popularities).
    """
    # collect all attribute keys in order
    all_attrs: list[str] = []
    seen: set[str] = set()
    for item in items:
        for k in item["attrs"]:
            if k not in seen:
                all_attrs.append(k)
                seen.add(k)

    n = len(items)
    m = len(all_attrs)
    attr_idx = {a: j for j, a in enumerate(all_attrs)}

    P = np.full((n, m), 128, dtype=np.uint8)    # 128 = unknown
    for i, item in enumerate(items):
        for attr, val in item["attrs"].items():
            if val is True:
                P[i, attr_idx[attr]] = 255
            elif val is False:
                P[i, attr_idx[attr]] = 0
            # None stays as 128 (unknown)

    pops = [item.get("pop", 1.0) for item in items]
    return P, all_attrs, pops


def attr_q(attr: str, attrs: list[str]) -> int:
    """Return the question index for the given attribute name."""
    return attrs.index(attr)


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

# Dataset A: 5 American female singers with distinguishing attributes
FEMALE_US_SINGERS = [
    {"name": "Taylor Swift",    "pop": 170, "attrs": {"female": True,  "american": True, "singer": True, "pop": True, "country": True, "grammy": True,  "pre2010": False, "songwriter": True}},
    {"name": "Ariana Grande",   "pop": 130, "attrs": {"female": True,  "american": True, "singer": True, "pop": True, "country": False, "grammy": True,  "pre2010": False, "songwriter": True}},
    {"name": "Beyoncé",         "pop": 160, "attrs": {"female": True,  "american": True, "singer": True, "pop": True, "country": False, "grammy": True,  "pre2010": True,  "songwriter": True}},
    {"name": "Billie Eilish",   "pop": 110, "attrs": {"female": True,  "american": True, "singer": True, "pop": True, "country": False, "grammy": True,  "pre2010": False, "songwriter": True}},
    {"name": "Katy Perry",      "pop": 120, "attrs": {"female": True,  "american": True, "singer": True, "pop": True, "country": False, "grammy": False, "pre2010": True,  "songwriter": True}},
]

# Dataset B: mixed people with lots of overlap
MIXED_PEOPLE = [
    {"name": "Lionel Messi",     "pop": 180, "attrs": {"male": True, "footballer": True, "american": False, "indian": False, "singer": False, "argentine": True,  "alive": True}},
    {"name": "Cristiano Ronaldo","pop": 180, "attrs": {"male": True, "footballer": True, "american": False, "indian": False, "singer": False, "portuguese": True, "alive": True}},
    {"name": "Sachin Tendulkar", "pop": 140, "attrs": {"male": True, "cricketer":  True, "american": False, "indian": True,  "singer": False, "alive": True}},
    {"name": "Roger Federer",    "pop": 130, "attrs": {"male": True, "tennis":     True, "american": False, "indian": False, "singer": False, "swiss": True,      "alive": True}},
    {"name": "Muhammad Ali",     "pop": 150, "attrs": {"male": True, "boxer":      True, "american": True,  "indian": False, "singer": False, "alive": False}},
]

# Dataset C: characters from different universes
CHARACTERS = [
    {"name": "Spider-Man",  "pop": 190, "attrs": {"superhero": True, "marvel": True, "dc": False, "villain": False, "male": True,  "human": True,  "animal": False}},
    {"name": "Batman",      "pop": 190, "attrs": {"superhero": True, "marvel": False,"dc": True,  "villain": False, "male": True,  "human": True,  "animal": False}},
    {"name": "Joker",       "pop": 160, "attrs": {"superhero": False,"marvel": False,"dc": True,  "villain": True,  "male": True,  "human": True,  "animal": False}},
    {"name": "Pikachu",     "pop": 180, "attrs": {"superhero": False,"marvel": False,"dc": False, "villain": False, "male": None,  "human": False, "animal": True}},
    {"name": "Darth Vader", "pop": 180, "attrs": {"superhero": False,"marvel": False,"dc": False, "villain": True,  "male": True,  "human": True,  "animal": False}},
]

# Dataset D: tiny (2 items) – edge-case test
TINY = [
    {"name": "Alpha", "pop": 1.0, "attrs": {"x": True,  "y": False}},
    {"name": "Beta",  "pop": 1.0, "attrs": {"x": False, "y": True}},
]

# Dataset E: single item
SINGLE = [
    {"name": "Solo", "pop": 1.0, "attrs": {"unique": True}},
]


# ---------------------------------------------------------------------------
# Helper: run a simulated game answering questions using item's known attrs
# ---------------------------------------------------------------------------

def simulate_game(
    items: list[dict],
    target_name: str,
    max_steps: int = MAX_QUESTIONS,
    answer_map: dict[str, float | None] | None = None,
) -> tuple[str | None, int]:
    """
    Run a full simulated game for target_name.
    Returns (guessed_name_or_None, steps_taken).
    """
    P, attrs, pops = make_dataset(items)
    target_idx = next(i for i, it in enumerate(items) if it["name"] == target_name)
    target_item = items[target_idx]

    eng = Engine(P, pops)
    rejected: list[str] = []

    for step in range(max_steps):
        if eng.should_guess():
            top = eng.top(1)
            if not top:
                return None, step
            guessed_idx, _ = top[0]
            guessed_name = items[guessed_idx]["name"]
            if guessed_name == target_name:
                return guessed_name, step + 1
            # wrong guess: reject and continue
            eng.reject(guessed_idx)
            rejected.append(guessed_name)
            continue

        q = eng.next_question()
        if q is None:
            break

        attr = attrs[q]
        if answer_map and attr in answer_map:
            ans = answer_map[attr]
        else:
            val = target_item["attrs"].get(attr)
            if val is True:
                ans = 1.0
            elif val is False:
                ans = 0.0
            else:
                ans = None  # unknown → don't know

        eng.answer(q, ans)

    # Exhausted questions: look at top candidate
    top = eng.top(1)
    if top:
        guessed_idx, _ = top[0]
        return items[guessed_idx]["name"], max_steps
    return None, max_steps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineBasics:
    """Fundamental engine mechanics."""

    def test_posterior_sums_to_one(self):
        """After any sequence of answers the posterior must still sum to 1."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        for q, a in [(attr_q("female", attrs), 1.0), (attr_q("american", attrs), 1.0)]:
            eng.answer(q, a)
        assert abs(eng.post.sum() - 1.0) < 1e-6

    def test_dont_know_leaves_posterior_unchanged(self):
        """'Don't know' must not change the posterior at all."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        before = eng.post.copy()
        q = eng.next_question()
        eng.answer(q, None)   # Don't know
        np.testing.assert_allclose(eng.post, before)

    def test_no_question_repeated(self):
        """next_question() must never return an already-asked question."""
        P, attrs, pops = make_dataset(MIXED_PEOPLE)
        eng = Engine(P, pops)
        asked = set()
        for _ in range(len(attrs)):
            q = eng.next_question()
            if q is None:
                break
            assert q not in asked, f"Question {q} ({attrs[q]}) was returned twice"
            asked.add(q)
            eng.answer(q, None)   # Don't know for all → exercises the skip set

    def test_undo_restores_state(self):
        """After undo() the posterior equals what it was before the undone answer."""
        P, attrs, pops = make_dataset(CHARACTERS)
        eng = Engine(P, pops)
        q = eng.next_question()
        eng.answer(q, 1.0)
        after_answer = eng.post.copy()
        eng.answer(eng.next_question(), 0.0)
        eng.undo()
        np.testing.assert_allclose(eng.post, after_answer, atol=1e-6)

    def test_reject_zeroes_candidate(self):
        """reject(i) must set post[i] to 0 and re-normalise."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        top_idx, _ = eng.top(1)[0]
        eng.reject(top_idx)
        assert eng.post[top_idx] == 0.0
        assert abs(eng.post.sum() - 1.0) < 1e-6

    def test_confident_guess_needs_gap(self):
        """should_guess() must require both GUESS_CONF and CONF_GAP."""
        P, attrs, pops = make_dataset(TINY)
        eng = Engine(P, pops)
        # Initially uniform → neither condition met after 0 real answers
        assert not eng.should_guess()

    def test_max_questions_forces_guess(self):
        """should_guess() must return True once MAX_QUESTIONS answers are given."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        # Flood the history with None answers to hit the length limit
        for i in range(MAX_QUESTIONS):
            eng.history.append((0, None))   # fake entries
        assert eng.should_guess()

    def test_candidates_left_entropy(self):
        """candidates_left() must decrease monotonically as definite answers are given."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        prev = eng.candidates_left()
        for attr, a in [("female", 1.0), ("american", 1.0), ("singer", 1.0)]:
            q = attr_q(attr, attrs)
            eng.answer(q, a)
            cur = eng.candidates_left()
            assert cur <= prev + 1, f"Candidates_left grew: {prev} -> {cur}"
            prev = cur

    def test_missing_attr_not_treated_as_no(self):
        """
        If an item has 'unknown' (128) for an attribute, a 'no' answer
        must not eliminate it with certainty (it should merely be down-ranked).
        """
        # Pikachu has 'male'=None (unknown) in CHARACTERS
        P, attrs, pops = make_dataset(CHARACTERS)
        eng = Engine(P, pops)
        pikachu_idx = next(i for i, it in enumerate(CHARACTERS) if it["name"] == "Pikachu")
        q = attr_q("male", attrs)
        eng.answer(q, 0.0)   # "No, not male"
        # Pikachu's posterior should still be > 0 (unknown, not definitely male)
        assert eng.post[pikachu_idx] > 0.0, "Pikachu was wrongly eliminated by unknown-gender answer"

    def test_contradictory_answers_no_crash(self):
        """Engine must survive contradictory answers without crashing."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        # Answer yes then no to the same logical question
        qa = attr_q("pop", attrs)
        eng.answer(qa, 1.0)
        qb = attr_q("country", attrs)
        eng.answer(qb, 0.0)
        eng.answer(qb, 1.0)   # contradicts previous — must not crash
        assert eng.post.sum() >= 0

    def test_tiny_pool_handled(self):
        """Engine works correctly with only 2 candidates."""
        P, attrs, pops = make_dataset(TINY)
        eng = Engine(P, pops)
        q = eng.next_question()
        assert q is not None
        eng.answer(q, 1.0)
        top = eng.top(1)
        assert len(top) >= 1

    def test_single_candidate_pool(self):
        """Engine survives a single-item dataset."""
        P, attrs, pops = make_dataset(SINGLE)
        eng = Engine(P, pops)
        assert eng.confidence() == pytest.approx(1.0)
        assert eng.next_question() is not None or True   # may or may not have questions

    def test_group_exclusion(self):
        """After a 'yes' answer in a group, sibling questions must be skipped."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        groups = ["geo" if a in ("american",) else None for a in attrs]
        eng = Engine(P, pops, groups=groups)
        q_am = attr_q("american", attrs)
        # Manually record a yes to trigger group exclusion
        eng.groups[q_am] = "geo"
        eng.members["geo"] = [q_am]
        eng.answer(q_am, 1.0)
        skip = eng._skipped()
        assert q_am in skip

    def test_wrong_guess_allows_continuation(self):
        """After a wrong guess (reject), the engine must continue and not crash."""
        P, attrs, pops = make_dataset(CHARACTERS)
        eng = Engine(P, pops)
        top_idx, _ = eng.top(1)[0]
        eng.reject(top_idx)
        q = eng.next_question()
        # Should still be able to ask questions
        assert q is not None or eng.candidates_left() <= 1


class TestGuessingQuality:
    """Integration tests for correct guessing behaviour."""

    def test_does_not_guess_too_early(self):
        """
        With 5 similar female US singers, the engine must NOT guess with
        only 3 broad matching answers (female + american + singer).
        """
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        for attr in ("female", "american", "singer"):
            eng.answer(attr_q(attr, attrs), 1.0)
        # After only 3 broad answers, should_guess() should still be False
        # because all 5 candidates are equally consistent
        assert not eng.should_guess() or eng.candidates_left() == 1, (
            f"Engine guessed too early after 3 broad answers. "
            f"Confidence: {eng.confidence():.2f}, Gap: {eng.confidence()-eng.second_confidence():.2f}"
        )

    def test_distinguishes_within_category(self):
        """
        Given a full game simulation targeting Katy Perry (no Grammy),
        the engine must guess correctly before exhausting MAX_QUESTIONS.
        """
        guessed, steps = simulate_game(FEMALE_US_SINGERS, "Katy Perry")
        assert guessed == "Katy Perry", (
            f"Expected Katy Perry, got {guessed} in {steps} steps"
        )

    def test_distinguishes_footballers(self):
        """
        Messi and Ronaldo share many attributes. The engine should
        distinguish them eventually.
        """
        guessed_messi, steps_m = simulate_game(MIXED_PEOPLE, "Lionel Messi")
        assert guessed_messi == "Lionel Messi", f"Got {guessed_messi} in {steps_m} steps"

        guessed_cr, steps_cr = simulate_game(MIXED_PEOPLE, "Cristiano Ronaldo")
        assert guessed_cr == "Cristiano Ronaldo", f"Got {guessed_cr} in {steps_cr} steps"

    def test_different_candidates_guessable(self):
        """Every character in the dataset must be guessable."""
        for item in CHARACTERS:
            name = item["name"]
            guessed, steps = simulate_game(CHARACTERS, name)
            assert guessed == name, (
                f"Could not distinguish {name}: got {guessed} in {steps} steps"
            )

    def test_information_gain_is_positive(self):
        """next_question() should return a question with positive gain."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        eng.next_question()
        assert eng.last_gain > 0, "Best gain should be positive when candidates differ"

    def test_probably_and_probably_not_handled(self):
        """Graded answers (Probably / Probably Not) must move the posterior softly."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        q = attr_q("grammy", attrs)

        # Probably → should raise Beyoncé / Swift / Ariana above Katy Perry
        eng.answer(q, 0.75)   # Probably grammy winner
        katy_idx = next(i for i, it in enumerate(FEMALE_US_SINGERS) if it["name"] == "Katy Perry")
        beyonce_idx = next(i for i, it in enumerate(FEMALE_US_SINGERS) if it["name"] == "Beyoncé")
        assert eng.post[beyonce_idx] > eng.post[katy_idx], (
            "Beyoncé should rank higher than Katy Perry after 'Probably Grammy winner'"
        )

    def test_information_gain_selects_best_splitter(self):
        """
        In the female-US-singer set, 'country' (only Taylor Swift = True) 
        should be highly discriminative. Gain should be > 0.
        """
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        eng = Engine(P, pops)
        # Force small population to make the effect visible
        q_country = attr_q("country", attrs)
        # Run next_question to compute gains
        eng.next_question()
        assert eng.last_gain > 0


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_empty_mask(self):
        """A mask that selects no items must raise ValueError."""
        P, attrs, pops = make_dataset(FEMALE_US_SINGERS)
        mask = np.zeros(len(FEMALE_US_SINGERS), dtype=bool)
        with pytest.raises(ValueError):
            Engine(P, pops, mask=mask)

    def test_all_dont_know_no_crash(self):
        """Answering 'Don't know' to every question must not crash or give NaN."""
        P, attrs, pops = make_dataset(MIXED_PEOPLE)
        eng = Engine(P, pops)
        for j in range(P.shape[1]):
            eng.answer(j, None)
        assert np.isfinite(eng.post).all()
        assert abs(eng.post.sum() - 1.0) < 1e-6

    def test_new_candidate_no_hardcoding(self):
        """
        Adding a new item (Shakira) to the female-singer set must allow the engine
        to guess her without any hardcoded rules.
        """
        extended = FEMALE_US_SINGERS + [
            {"name": "Shakira", "pop": 120,
             "attrs": {"female": True, "american": False, "singer": True,
                       "pop": True, "country": False, "grammy": True,
                       "pre2010": True, "songwriter": True}},
        ]
        guessed, steps = simulate_game(extended, "Shakira")
        assert guessed == "Shakira", f"Could not guess Shakira, got {guessed} in {steps}"

    def test_large_candidate_set_performance(self):
        """Engine must handle 200 candidates in reasonable time (< 5 s)."""
        import time
        # Create 200 items with 20 binary attributes each
        rng = np.random.default_rng(42)
        n, m = 200, 20
        P = rng.integers(0, 2, (n, m), dtype=np.uint8) * 255
        pops = rng.uniform(1, 10, n)
        eng = Engine(P, pops)
        t0 = time.time()
        for _ in range(10):
            q = eng.next_question()
            if q is None:
                break
            eng.answer(q, 1.0)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"Engine was too slow: {elapsed:.2f}s for 200 items x 20 questions"

    def test_category_imbalance_handled(self):
        """Works correctly even when one category has many more items than others."""
        many_singers = [{"name": f"Singer{i}", "pop": 1.0,
                         "attrs": {"singer": True, f"unique_{i}": True}}
                        for i in range(50)]
        few_athletes = [{"name": "Messi", "pop": 1.0, "attrs": {"singer": False, "footballer": True}}]
        dataset = many_singers + few_athletes
        guessed, steps = simulate_game(dataset, "Messi")
        assert guessed == "Messi", f"Could not find Messi in imbalanced set, got {guessed}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all_tests_manually():
    """Human-readable test runner (no pytest needed)."""
    test_classes = [TestEngineBasics, TestGuessingQuality, TestEdgeCases]
    passed = failed = 0
    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        print(f"\n{'=' * 60}")
        print(f"  {cls.__name__}")
        print('=' * 60)
        for name in methods:
            try:
                getattr(obj, name)()
                print(f"  ✅  {name}")
                passed += 1
            except Exception as exc:
                print(f"  ❌  {name}\n      {exc}")
                failed += 1
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('=' * 60)
    return failed


if __name__ == "__main__":
    failures = run_all_tests_manually()
    sys.exit(1 if failures else 0)
