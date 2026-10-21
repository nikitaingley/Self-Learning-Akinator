"""
db.py - storage for Genie AI.

  akinator.db   SQLite: items, questions, game_log, answer_log
  matrix.npy    uint8 matrix P[item, question]; 0 = "no", 255 = "yes", 128 = unknown

Convention: item id = row index + 1, question id = column index + 1.
"""
import os
from contextlib import closing
import sqlite3

import numpy as np

DB_PATH = "akinator.db"
MATRIX_PATH = "matrix.npy"
UNKNOWN = 128

CATEGORIES = ["person", "animal", "place", "character", "thing"]
CATEGORY_LABELS = {"person": "Real person", "animal": "Animal", "place": "Place",
                   "character": "Fictional character", "thing": "Object"}
THEMES = {"Everything": None, "People": "person", "Animals": "animal",
          "Places": "place", "Characters": "character", "Objects": "thing"}
CATEGORY_EMOJI = {"person": "🧑", "animal": "🦁", "place": "🗺️", "character": "🦸", "thing": "📦"}

# How "Is it a living thing?" is answered for each category (fictional = unsure).
LIVING = {"person": 255, "animal": 255, "character": UNKNOWN, "place": 0, "thing": 0}

# Questions in the same group are mutually exclusive: once one is answered "yes",
# the engine skips the others (e.g. after "from India?" it never asks "from Peru?").
EXCLUSIVE = {("person", "country"), ("person", "gender"), ("person", "century"),
             ("person", "decade"), ("person", "alive"), ("person", "birthplace"),
             ("character", "gender"), ("place", "country"), ("place", "continent"),
             ("animal", "group")}


def group_of(attr, cat):
    if attr == "category":
        return "category"
    return f"{cat}:{attr}" if (cat, attr) in EXCLUSIVE else None


SCHEMA = """
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
  description TEXT DEFAULT '', image_url TEXT DEFAULT '', wiki_title TEXT DEFAULT '',
  popularity REAL DEFAULT 1, is_adult INTEGER DEFAULT 0,
  source TEXT DEFAULT 'wikidata', added_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY, text TEXT NOT NULL, category TEXT, attribute TEXT,
  value TEXT, times_asked INTEGER DEFAULT 0, times_idk INTEGER DEFAULT 0,
  source TEXT DEFAULT 'wikidata');
CREATE TABLE IF NOT EXISTS game_log(
  game_id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT, result TEXT,
  questions INTEGER, guesses INTEGER DEFAULT 1, item_id INTEGER,
  played_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS answer_log(
  game_id INTEGER, item_id INTEGER, question_id INTEGER, answer REAL);
"""


def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init_schema():
    with closing(conn()) as c:
        c.executescript(SCHEMA)
        for alter in ("ALTER TABLE questions ADD COLUMN times_idk INTEGER DEFAULT 0",
                      "ALTER TABLE items ADD COLUMN wiki_title TEXT DEFAULT ''"):
            try:                               # older databases: add the new columns
                c.execute(alter)
            except sqlite3.OperationalError:
                pass
        c.commit()


def reset_files():
    for p in (DB_PATH, MATRIX_PATH):
        if os.path.exists(p):
            os.remove(p)


def base_cell(attr, qcat, item_cat):
    """Cell value for the built-in 'category' and 'living' questions."""
    if attr == "category":
        return 255 if qcat == item_cat else 0
    if attr == "living":
        return LIVING[item_cat]
    return None


class Store:
    """Everything the game needs in memory, plus the self-learning writes."""

    def __init__(self):
        init_schema()
        self.load()
        if self.n_items == 0:
            import build_dataset            # first run: create the small built-in dataset
            build_dataset.build_seed()
            self.load()

    # ------------------------------------------------------------ loading
    def load(self):
        with closing(conn()) as c:
            items = c.execute("SELECT * FROM items ORDER BY id").fetchall()
            qs = c.execute("SELECT * FROM questions ORDER BY id").fetchall()
        self.names = [r["name"] for r in items]
        self.cats = np.array([r["category"] for r in items], dtype=object)
        self.desc = [r["description"] or "" for r in items]
        self.img = [r["image_url"] or "" for r in items]
        self.wiki = [r["wiki_title"] or "" for r in items]
        self.adult = np.array([bool(r["is_adult"]) for r in items], dtype=bool)
        self.pop = np.array([r["popularity"] for r in items], dtype=np.float64)
        self.qtext = [r["text"] for r in qs]
        self.qattr = [r["attribute"] for r in qs]
        self.qcat = [r["category"] for r in qs]
        self.qgroup = [group_of(a, c) for a, c in zip(self.qattr, self.qcat)]
        self.q_asked = np.array([r["times_asked"] or 0 for r in qs], dtype=np.float64)
        self.q_idk = np.array([r["times_idk"] or 0 for r in qs], dtype=np.float64)
        self._update_penalty()
        self.n_items, self.n_q = len(items), len(qs)

        if os.path.exists(MATRIX_PATH):
            self.P = np.load(MATRIX_PATH)
        else:
            self.P = np.zeros((0, 0), dtype=np.uint8)
        if self.n_items and self.P.shape != (self.n_items, self.n_q):
            raise RuntimeError(
                f"matrix.npy {self.P.shape} does not match akinator.db "
                f"({self.n_items} items x {self.n_q} questions). "
                "Delete both files and run build_dataset.py again.")
        self._by_name = {}
        for i in np.argsort(self.pop):          # most popular wins on duplicate names
            self._by_name[self.names[i].strip().lower()] = int(i)

    def _update_penalty(self):
        """Self-learning: questions players often can't answer get a lower score."""
        self.qpen = 1.0 - 0.6 * (self.q_idk / (self.q_asked + 5.0))

    def save_matrix(self):
        tmp = MATRIX_PATH + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, self.P)
        os.replace(tmp, MATRIX_PATH)

    # ------------------------------------------------------------ lookups
    def item(self, row):
        return {"row": int(row), "name": self.names[row], "category": self.cats[row],
                "description": self.desc[row], "image_url": self.img[row],
                "wiki_title": self.wiki[row]}

    def find_item(self, name):
        return self._by_name.get(name.strip().lower())

    # ------------------------------------------------------------ logging
    def log_game(self, category, result, n_questions, guesses, item_row, history):
        with closing(conn()) as c:
            gid = c.execute(
                "INSERT INTO game_log(category,result,questions,guesses,item_id) VALUES(?,?,?,?,?)",
                (category, result, n_questions, guesses,
                 None if item_row is None else int(item_row) + 1)).lastrowid
            if item_row is not None:
                c.executemany(
                    "INSERT INTO answer_log VALUES(?,?,?,?)",
                    [(gid, int(item_row) + 1, int(q) + 1, a) for q, a in history if a is not None])
            c.executemany(
                "UPDATE questions SET times_asked = times_asked + 1, times_idk = times_idk + ? WHERE id=?",
                [(1 if a is None else 0, int(q) + 1) for q, a in history])
            c.commit()
        for q, a in history:
            self.q_asked[q] += 1
            if a is None:
                self.q_idk[q] += 1
        self._update_penalty()
        return gid

    # ------------------------------------------------------------ learning
    def reinforce(self, row, history, lr=0.1):
        """Nudge the item's probabilities toward the answers the player gave."""
        for q, a in history:
            if a is None:
                continue
            cur = self.P[row, q] / 255.0
            rate = 0.5 if self.P[row, q] == UNKNOWN else lr    # learn fast where we knew nothing
            self.P[row, q] = int(round(255 * (cur + rate * (a - cur))))
        self.pop[row] += 1
        with closing(conn()) as c:
            c.execute("UPDATE items SET popularity=? WHERE id=?", (float(self.pop[row]), int(row) + 1))
            c.commit()
        self.save_matrix()

    def add_item(self, name, category, description, image_url, history,
                 wrong_row=None, question=None, new_is_yes=True):
        """Teach the AI a brand-new item. Returns its row index."""
        new_pop = float(np.median(self.pop)) if self.n_items else 1.0
        with closing(conn()) as c:
            item_id = c.execute(
                "INSERT INTO items(name,category,description,image_url,popularity,source) "
                "VALUES(?,?,?,?,?,'learned')",
                (name.strip(), category, description.strip(), image_url.strip(), new_pop)).lastrowid
            row = item_id - 1
            assert row == self.n_items, "item ids must stay contiguous"

            vec = np.full(self.n_q, UNKNOWN, dtype=np.uint8)
            for j in range(self.n_q):
                b = base_cell(self.qattr[j], self.qcat[j], category)
                if b is not None:
                    vec[j] = b
            for q, a in history:                      # the player's answers describe the item
                if a is not None:
                    vec[q] = int(round(255 * a))
            self.P = np.vstack([self.P, vec[None, :]]) if self.n_q else vec[None, :]

            if question and question.strip():
                qid = c.execute(
                    "INSERT INTO questions(text,category,attribute,value,source) "
                    "VALUES(?,?,?,?,'learned')", (question.strip(), None, "custom", "")).lastrowid
                col = np.full((self.n_items + 1, 1), UNKNOWN, dtype=np.uint8)
                col[row, 0] = 255 if new_is_yes else 0
                if wrong_row is not None:
                    col[wrong_row, 0] = 0 if new_is_yes else 255
                self.P = np.hstack([self.P, col])
                self.qtext.append(question.strip()); self.qattr.append("custom"); self.qcat.append(None)
                self.qgroup.append(None)
                self.q_asked = np.append(self.q_asked, 0.0); self.q_idk = np.append(self.q_idk, 0.0)
                self._update_penalty()
                self.n_q += 1
                assert qid == self.n_q
            c.commit()

        self.names.append(name.strip()); self.desc.append(description.strip())
        self.img.append(image_url.strip())
        self.wiki.append("")
        self.cats = np.append(self.cats, category)
        self.adult = np.append(self.adult, False)
        self.pop = np.append(self.pop, new_pop)
        self.n_items += 1
        self._by_name[name.strip().lower()] = row
        self.save_matrix()
        return row