"""
engine.py - probabilistic guessing engine (Bayesian, information-gain based).

Architecture
------------
State: a probability vector over ALL candidate items.
Each answer applies Bayes' rule (with graded evidence for Probably/Don't Know).
next_question() picks the question with the highest expected information gain
over the top-K candidates, with per-question penalties for high "don't know" rates
and a group-exclusion rule to skip obviously redundant questions.

Tunable constants
-----------------
EPS            Noise floor: probability that a stored fact is simply wrong.
MIN_QUESTIONS  Never guess before this many meaningful (non-None) answers.
MAX_QUESTIONS  Force a guess / give up after this many.
GUESS_CONF     Guess when the top candidate has at least this posterior.
MIN_GAIN       If no question has this much information gain, stop asking.
CONF_GAP       Top candidate must also be this much larger than #2.

P[item, question] is uint8: 0 = "no", 255 = "yes", 128 = unknown.
"""
import numpy as np

EPS = 0.03             # noise: chance data or player is simply wrong
MIN_QUESTIONS = 5      # never guess before this many answered questions
MAX_QUESTIONS = 30     # force a guess after this many questions
GUESS_CONF = 0.85      # guess once top candidate probability ≥ this
CONF_GAP = 0.30        # top must be this much larger than 2nd to guess
MIN_GAIN = 5e-4        # if best gain < this, no useful question remains → guess

# answer label → float value (None = "don't know": leaves posteriors unchanged)
ANSWERS = {"Yes": 1.0, "Probably": 0.75, "Don't know": None, "Probably not": 0.25, "No": 0.0}


class Engine:
    """
    Bayesian guessing engine with information-gain question selection.

    Parameters
    ----------
    P          : uint8 numpy array, shape (n_items, n_questions)
    popularity : float array length n_items
    mask       : bool array length n_items (restrict to a category/theme)
    groups     : list of n_questions group-labels (str or None)
    penalty    : float array length n_questions, question quality multiplier
    top_k      : only look at top-K candidates for question selection (speed)
    """

    def __init__(self, P, popularity, mask=None, groups=None, penalty=None, top_k=3000):
        self.P = P
        self.top_k = top_k
        m = P.shape[1]

        # question groups: exclusive questions (once one is "Yes", skip the rest)
        self.groups = list(groups) if groups is not None else [None] * m
        self.members: dict[str, list[int]] = {}
        for j, g in enumerate(self.groups):
            if g:
                self.members.setdefault(g, []).append(j)

        # per-question penalty (lowered for high "don't know" rates)
        self.penalty = (np.ones(m, dtype=np.float32)
                        if penalty is None
                        else np.asarray(penalty, dtype=np.float32))

        # base prior: square-root-softened popularity, restricted to mask
        pop = np.maximum(np.asarray(popularity, dtype=np.float64), 1.0) ** 0.5
        if mask is not None:
            pop = pop * np.asarray(mask, dtype=np.float64)
        if pop.sum() <= 0:
            raise ValueError("No candidates match this theme.")
        self.base = pop / pop.sum()

        self.rejected: set[int] = set()
        self._last_debug: dict = {}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.history: list[tuple[int, float | None]] = []   # [(question_idx, answer_value)]
        self.asked: set[int] = set()
        self.guess_after = MIN_QUESTIONS
        self.last_gain = np.inf
        self._rebuild(self.history)

    def _rebuild(self, history: list) -> None:
        """Recompute posterior from scratch (used after undo or reset)."""
        self.post = self.base.copy()
        for r in self.rejected:
            self.post[r] = 0.0
        self._normalise()
        for q, a in history:
            self._apply(q, a)

    def _normalise(self) -> None:
        s = self.post.sum()
        if s > 0:
            self.post /= s

    def _apply(self, q: int, a: float | None) -> None:
        """Apply one answer to the posterior (Bayes update)."""
        if a is None:
            return  # "Don't know" — leave posteriors unchanged
        # p_yes[i] = probability that item i would answer "yes" to question q
        p_yes = EPS + (1.0 - 2.0 * EPS) * (self.P[:, q].astype(np.float32) / 255.0)
        # graded likelihood: answer 'a' is a blend of yes/no evidence
        like = a * p_yes + (1.0 - a) * (1.0 - p_yes)
        self.post = self.post * like
        self._normalise()

    def _skipped(self) -> set[int]:
        """
        Return question indices that should be skipped:
          - already asked
          - sibling questions in a group where a "yes-ish" answer was given
        """
        skip = set(self.asked)
        for q, a in self.history:
            if a is not None and a >= 0.75 and self.groups[q]:
                skip.update(self.members[self.groups[q]])
        return skip

    # ------------------------------------------------------------------ playing
    def answer(self, q: int, a: float | None) -> None:
        """Record one answer and update the posterior."""
        self.asked.add(q)
        self.history.append((q, a))
        self._apply(q, a)

    def undo(self) -> int | None:
        """Back-button: drop the last answer and replay. Returns the question index."""
        if not self.history:
            return None
        q, _ = self.history.pop()
        self.asked = {x for x, _ in self.history}
        self._rebuild(self.history)
        return q

    def reject(self, item: int) -> None:
        """Wrong-guess: zero out this item and continue."""
        self.rejected.add(int(item))
        self.post[int(item)] = 0.0
        self._normalise()
        # require at least 3 more informative answers before next guess
        self.guess_after = len(self.history) + 3

    # ------------------------------------------------------------------ question selection
    def next_question(self) -> int | None:
        """
        Pick the question with the highest expected information gain over the
        top-K candidates (weighted by their posterior probability).

        Returns None when no useful question exists.
        """
        # Work on the top-K candidates only (speed)
        idx = np.argsort(-self.post)[: self.top_k]
        w = self.post[idx]
        s = w.sum()
        if s <= 0 or self.P.shape[1] == 0:
            self.last_gain = 0.0
            self._last_debug = {}
            return None

        w = (w / s).astype(np.float32)                                 # normalised weights
        py = EPS + (1.0 - 2.0 * EPS) * (self.P[idx].astype(np.float32) / 255.0)   # K × M
        pn = 1.0 - py

        p_y = w @ py   # marginal P(answer=yes) for each question
        p_n = 1.0 - p_y

        def _entropy(weights_times_like, total):
            q = weights_times_like / np.maximum(total, 1e-12)
            return -(q * np.log2(np.maximum(q, 1e-12))).sum(axis=0)

        h_if_yes = _entropy(w[:, None] * py, p_y)   # expected H after "yes"
        h_if_no  = _entropy(w[:, None] * pn, p_n)   # expected H after "no"
        h0 = float(-(w * np.log2(np.maximum(w, 1e-12))).sum())

        expected_h = p_y * h_if_yes + p_n * h_if_no
        gain = (h0 - expected_h) * self.penalty   # penalise high-idk questions

        # mask out skipped questions
        skip = self._skipped()
        if skip:
            gain[list(skip)] = -np.inf

        best = int(np.argmax(gain))
        self.last_gain = float(gain[best]) if np.isfinite(gain[best]) else 0.0

        # debug info
        finite_gains = gain[np.isfinite(gain)]
        self._last_debug = {
            "h0": h0,
            "best_gain": self.last_gain,
            "n_skipped": len(skip),
            "n_candidates": int(round(2 ** h0)) if h0 > 0 else 1,
            "top5_questions": (
                sorted(
                    [(int(j), float(gain[j])) for j in range(len(gain)) if np.isfinite(gain[j])],
                    key=lambda x: -x[1]
                )[:5]
            ) if finite_gains.size else [],
        }

        if not np.isfinite(gain[best]):
            return None
        return best

    # ------------------------------------------------------------------ status
    def top(self, k: int = 5) -> list[tuple[int, float]]:
        """Return the (row, probability) for the top-k candidates."""
        idx = np.argsort(-self.post)[:k]
        return [(int(i), float(self.post[i])) for i in idx if self.post[i] > 0]

    def confidence(self) -> float:
        """Probability of the leading candidate (0–1)."""
        return float(self.post.max())

    def second_confidence(self) -> float:
        """Probability of the 2nd-best candidate."""
        if self.post.size < 2:
            return 0.0
        idx = np.argsort(-self.post)
        return float(self.post[idx[1]]) if len(idx) > 1 else 0.0

    def candidates_left(self) -> int:
        """
        Effective number of remaining candidates: 2^H where H is the
        entropy of the posterior (a.k.a. perplexity).
        """
        p = self.post[self.post > 0]
        if p.size == 0:
            return 0
        return int(round(2 ** float(-(p * np.log2(p)).sum())))

    def meaningful_answers(self) -> int:
        """Count of answers that were not None (don't know)."""
        return sum(1 for _, a in self.history if a is not None)

    def should_guess(self) -> bool:
        """
        True when we have enough evidence to commit to a guess.
        Rules (all of the following must hold):
          1. We have asked at least guess_after questions with real answers.
          2. Either:
             a. confidence ≥ GUESS_CONF AND the gap over 2nd place ≥ CONF_GAP
             b. No remaining question has gain > MIN_GAIN
             c. We've hit MAX_QUESTIONS
        """
        n_real = self.meaningful_answers()
        n_total = len(self.history)

        if n_total >= MAX_QUESTIONS:
            return True
        if n_real < self.guess_after:
            return False
        if self.last_gain < MIN_GAIN:
            return True

        conf = self.confidence()
        conf2 = self.second_confidence()
        if conf >= GUESS_CONF and (conf - conf2) >= CONF_GAP:
            return True

        return False

    def debug_info(self) -> dict:
        """Return diagnostic information for the developer panel."""
        top5 = self.top(5)
        return {
            "candidates_left": self.candidates_left(),
            "meaningful_answers": self.meaningful_answers(),
            "last_gain": self.last_gain,
            "top5": top5,
            "confidence": self.confidence(),
            "second_confidence": self.second_confidence(),
            "conf_gap": self.confidence() - self.second_confidence(),
            "questions_asked": len(self.history),
            "guess_after": self.guess_after,
            "rejected": sorted(self.rejected),
            **self._last_debug,
        }