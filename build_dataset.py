"""
build_dataset.py - creates akinator.db + matrix.npy

  python build_dataset.py --seed                 tiny built-in dataset, works offline
  python build_dataset.py --depth quick          most famous band per category (a few thousand items)
  python build_dataset.py                        standard Wikidata build (well-known items)
  python build_dataset.py --depth deep           also lesser-known items: tens of thousands, hours
  python build_dataset.py --categories person animal --force

Every SPARQL result is cached in ./wikidata_cache, so an interrupted run resumes where it stopped.
Rebuilding wipes learned items, so it needs --force when a database already exists.
"""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from contextlib import closing

import numpy as np
import pandas as pd

import db

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "GenieAI-student-project/1.0 (student project; contact: you@example.com)",
           "Accept": "application/sparql-results+json"}
CACHE_DIR = "wikidata_cache"
BIG = 10 ** 6

# ----------------------------------------------------------------------------
# Question wording. (category, attribute) -> template. {} is replaced by the value.
# ----------------------------------------------------------------------------
TEMPLATES = {
    ("person", "occupation"): "Is the person a {}?",
    ("person", "gender"): "Is the person {}?",
    ("person", "country"): "Is the person from {}?",
    ("person", "alive"): "Is the person still alive?",
    ("person", "century"): "Was the person born in the {} century?",
    ("person", "decade"): "Was the person born in the {}?",
    ("person", "birthplace"): "Was the person born in {}?",
    ("person", "genre"): "Is the person known for {}?",
    ("person", "instrument"): "Does the person play the {}?",
    ("person", "sport"): "Does the person play {}?",
    ("person", "award"): "Has the person won the {}?",
    ("person", "member"): "Is the person a member of {}?",
    ("animal", "group"): "Is it a {}?",
    ("animal", "eats"): "Does it eat {}?",
    ("animal", "lives"): "Does it live in {}?",
    ("animal", "status"): "Is its conservation status \"{}\"?",
    ("animal", "order"): "Does it belong to the order {}?",
    ("animal", "family"): "Does it belong to the family {}?",
    ("animal", "feature"): "Does it have {}?",
    ("animal", "can"): "Can it {}?",
    ("place", "type"): "Is it a {}?",
    ("place", "country"): "Is it located in {}?",
    ("place", "continent"): "Is it in {}?",
    ("character", "universe"): "Is the character from {}?",
    ("character", "kind"): "Is the character a {}?",
    ("character", "gender"): "Is the character {}?",
    ("character", "work"): "Does the character appear in {}?",
    ("character", "occupation"): "Is the character a {}?",
    ("thing", "kind"): "Is it a {}?",
    ("thing", "material"): "Is it made of {}?",
    ("thing", "use"): "Is it used for {}?",
    ("thing", "feature"): "Does it have {}?",
}
BASE_Q = {"person": "Is it a real person?", "animal": "Is it an animal?",
          "place": "Is it a place?", "character": "Is it a fictional character?",
          "thing": "Is it an object?"}
MAX_Q = {"birthplace": 200, "award": 120, "genre": 80, "member": 100, "sport": 50, "instrument": 40,
         "decade": 30, "occupation": 350, "country": 120, "century": 30, "gender": 4, "alive": 1,
         "universe": 80, "work": 120, "type": 80, "kind": 60}
DEFAULT_MAX_Q = 60
COMPLETE = {("person", "alive")}        # for these, "no fact" really means "no"

# ----------------------------------------------------------------------------
# Wikidata configuration.
#
# The public endpoint cuts queries off after 60 s, so "all humans with many sitelinks" always
# times out. Instead:
#   phase 1  find famous item IDs inside SMALL slices (one occupation, one class, ...)
#   phase 2  fetch descriptions, images and facts for those IDs in batches (anchored by VALUES, fast)
# Wikidata IDs below were written from memory; a wrong ID only gives a slightly different slice.
# ----------------------------------------------------------------------------
THING_ROOTS = ["Q39546", "Q11019", "Q2424752", "Q34379", "Q42889", "Q11460", "Q2095", "Q11422", "Q14745", "Q728"]
THING_ROOTS_V = " ".join(f"wd:{q}" for q in THING_ROOTS)
OCCUPATIONS = [  # singer, actor, politician, footballer, writer, painter, director, musician, composer ...
    "Q177220", "Q33999", "Q82955", "Q937857", "Q36180", "Q1028181", "Q2526255", "Q639669", "Q36834",
    "Q901", "Q169470", "Q170790", "Q4964182", "Q49757", "Q2252262", "Q753110", "Q12299841", "Q3665646",
    "Q10833314", "Q2066131", "Q43845", "Q131524", "Q81096", "Q205375", "Q11631", "Q189290", "Q116",
    "Q593644", "Q864503", "Q42973", "Q1930187", "Q245068", "Q4610556", "Q5716684", "Q947873", "Q28389",
    "Q214917", "Q6625963", "Q1281618", "Q33231", "Q10873124", "Q378622", "Q11338576", "Q13365117",
    "Q10871364", "Q11774891", "Q13474373", "Q17125263", "Q15253558", "Q40348", "Q188094", "Q11900058",
    "Q2405480", "Q3282637", "Q183945", "Q855091", "Q486748", "Q130857", "Q488205"]
ANIMAL_GROUPS = ["Q7377", "Q5113", "Q10811", "Q152", "Q10908", "Q1390"]
PLACE_TYPES = [  # city, capital, country, mountain, river, island, monument, castle, museum ...
    "Q515", "Q1549591", "Q5119", "Q6256", "Q3624078", "Q8502", "Q8072", "Q4022", "Q23397", "Q23442",
    "Q8514", "Q34038", "Q46169", "Q4989906", "Q23413", "Q16560", "Q12518", "Q11303", "Q33506",
    "Q483110", "Q12280", "Q44539", "Q16970", "Q32815", "Q9259", "Q839954", "Q40080", "Q150784",
    "Q165", "Q9430", "Q5107", "Q39594", "Q1248784", "Q3918", "Q2416723"]
CHAR_CLASSES = ["Q95074", "Q15632617", "Q15773347", "Q1114461", "Q15711870", "Q3658341"]

# min sitelinks per depth, and max items kept per category
CAPS = {"quick": 1500, "standard": 6000, "deep": 25000}
CATS = {
    "person": dict(
        slices=[f"?item wdt:P31 wd:Q5; wdt:P106 wd:{q}." for q in OCCUPATIONS],
        min_links={"quick": 120, "standard": 60, "deep": 30},
        attrs=[("occupation", "?item wdt:P106 ?val."),
               ("gender", "?item wdt:P21 ?val."),
               ("country", "?item wdt:P27 ?val."),
               ("alive", 'FILTER NOT EXISTS { ?item wdt:P570 ?d . } BIND("alive" AS ?val)'),
               ("century", "?item wdt:P569 ?dob. BIND(STR(1 + FLOOR((YEAR(?dob) - 1) / 100)) AS ?val)"),
               ("decade", "?item wdt:P569 ?dob. BIND(STR(FLOOR(YEAR(?dob) / 10) * 10) AS ?val)"),
               ("birthplace", "?item wdt:P19 ?val."),
               ("genre", "?item wdt:P136 ?val."),
               ("instrument", "?item wdt:P1303 ?val."),
               ("sport", "?item wdt:P641 ?val."),
               ("award", "?item wdt:P166 ?val."),
               ("member", "?item wdt:P463 ?val.")]),
    "animal": dict(
        slices=[f"?item wdt:P105 wd:Q7432; wdt:P171+ wd:{q}." for q in ANIMAL_GROUPS],
        min_links={"quick": 60, "standard": 35, "deep": 20},
        attrs=[("group", "?item wdt:P171+ ?val. VALUES ?val { %s }" % " ".join(f"wd:{q}" for q in ANIMAL_GROUPS)),
               ("order", "?item wdt:P171+ ?val. ?val wdt:P105 wd:Q36602."),
               ("family", "?item wdt:P171+ ?val. ?val wdt:P105 wd:Q35409."),
               ("eats", "?item wdt:P1034 ?val."),
               ("lives", "?item wdt:P2974 ?val."),
               ("status", "?item wdt:P141 ?val.")]),
    "place": dict(
        slices=[f"?item wdt:P31 wd:{q}." for q in PLACE_TYPES],
        min_links={"quick": 100, "standard": 60, "deep": 35},
        attrs=[("type", "?item wdt:P31 ?val."),
               ("country", "?item wdt:P17 ?val."),
               ("continent", "?item wdt:P17/wdt:P30 ?val.")]),
    "character": dict(
        slices=[f"?item wdt:P31 wd:{q}." for q in CHAR_CLASSES],
        min_links={"quick": 35, "standard": 20, "deep": 10},
        attrs=[("universe", "?item wdt:P1080 ?val."),
               ("kind", "?item wdt:P31 ?val."),
               ("gender", "?item wdt:P21 ?val."),
               ("work", "?item wdt:P1441 ?val."),
               ("occupation", "?item wdt:P106 ?val.")]),
    "thing": dict(
        slices=[f"?item wdt:P279+ wd:{q}." for q in THING_ROOTS],
        min_links={"quick": 35, "standard": 20, "deep": 10},
        attrs=[("kind", "?item wdt:P279+ ?val. VALUES ?val { %s }" % THING_ROOTS_V),
               ("material", "?item wdt:P186 ?val."),
               ("use", "?item wdt:P366 ?val."),
               ("feature", "?item wdt:P527 ?val.")]),
}
ADULT_WORDS = ("porn", "erotic", "adult film", "adult model")
LOWERCASE_ATTRS = ("occupation", "type", "kind", "group", "genre", "instrument", "sport")
BATCH = 150


# ----------------------------------------------------------------------------
# SPARQL helpers
# ----------------------------------------------------------------------------
def sparql(query, retries=2):
    import requests
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, hashlib.sha1(query.encode()).hexdigest() + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    last = "unknown error"
    for attempt in range(retries):
        try:
            r = requests.post(ENDPOINT, data={"query": query}, headers=HEADERS, timeout=80)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 30)))
                last = "rate limited"
                continue
            if r.status_code in (500, 502, 503, 504):
                last = f"server busy or timeout ({r.status_code})"
                time.sleep(8 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()["results"]["bindings"]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            time.sleep(1.0)                  # be polite to the public endpoint
            return data
        except requests.RequestException as e:
            last = str(e)[:120]
            time.sleep(5)
    raise RuntimeError(last)


def q_ids(select, lo):
    return (f"SELECT DISTINCT ?item ?links WHERE {{ {select} ?item wikibase:sitelinks ?links. "
            f"FILTER(?links >= {lo}) }}")


def q_meta(qids):
    vals = " ".join(f"wd:{q}" for q in qids)
    return (f"SELECT ?item ?itemLabel ?itemDescription ?img ?wtitle WHERE {{ VALUES ?item {{ {vals} }} "
            f"OPTIONAL {{ ?item wdt:P18 ?img. }} "
            f"OPTIONAL {{ ?article schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>; schema:name ?wtitle. }} "
            f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}')


def q_facts(cat, qids):
    vals = " ".join(f"wd:{q}" for q in qids)
    parts = " UNION ".join("{ %s BIND(\"%s\" AS ?attr) }" % (pat, attr) for attr, pat in CATS[cat]["attrs"])
    return (f"SELECT ?item ?attr ?val ?valLabel WHERE {{ VALUES ?item {{ {vals} }} {parts} "
            f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}')


def qid(binding):
    return binding["item"]["value"].rsplit("/", 1)[-1]


def is_qid(text):
    return bool(re.fullmatch(r"[QP]\d+", text or ""))


def ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def clean_value(attr, value):
    if is_qid(value):
        return None
    if attr in ("century", "decade"):
        try:
            n = int(float(value))
        except ValueError:
            return None
        if n < 1:
            return None
        return ordinal(n) if attr == "century" else f"{n}s"
    return value.lower() if attr in LOWERCASE_ATTRS else value


def collect_wikidata(categories, depth):
    items, facts = {}, []
    for cat in categories:
        cfg = CATS[cat]
        lo, cap = cfg["min_links"][depth], CAPS[depth]
        found, failed = {}, 0
        print(f"[{cat}] phase 1: finding famous items (sitelinks >= {lo}) in {len(cfg['slices'])} slices")
        for i, sel in enumerate(cfg["slices"], 1):
            try:
                for b in sparql(q_ids(sel, lo)):
                    k = qid(b)
                    found[k] = max(found.get(k, 0), float(b["links"]["value"]))
            except RuntimeError as e:
                failed += 1
                print(f"    slice {i} skipped: {e}")
            if i % 10 == 0:
                print(f"    {i}/{len(cfg['slices'])} slices done, {len(found)} items so far")
        ranked = sorted(found, key=found.get, reverse=True)[:cap]
        print(f"[{cat}] {len(found)} found ({failed} slices failed), keeping {len(ranked)}; phase 2: facts")
        for n in range(0, len(ranked), BATCH):
            chunk = ranked[n:n + BATCH]
            try:
                meta = sparql(q_meta(chunk))
                rows = sparql(q_facts(cat, chunk))
            except RuntimeError as e:
                print(f"    batch {n // BATCH + 1} skipped: {e}")
                continue
            for b in meta:
                k = qid(b)
                label = b.get("itemLabel", {}).get("value", "")
                if not label or is_qid(label):
                    continue
                it = items.setdefault(k, dict(name=label, category=cat, image_url="", wiki_title="", popularity=found[k],
                                              description=b.get("itemDescription", {}).get("value", "")))
                if not it["wiki_title"] and "wtitle" in b:
                    it["wiki_title"] = b["wtitle"]["value"]
                if not it["image_url"] and "img" in b:
                    it["image_url"] = b["img"]["value"].replace("http://", "https://") + "?width=480"
            for b in rows:
                k = qid(b)
                if k not in items:
                    continue
                attr = b["attr"]["value"]
                v = clean_value(attr, b.get("valLabel", b["val"])["value"])
                if v:
                    facts.append((k, cat, attr, v))
            if (n // BATCH) % 10 == 9:
                print(f"    {min(n + BATCH, len(ranked))}/{len(ranked)} items fetched")
    return items, facts


def frames_from_wikidata(items, facts, min_facts=3):
    f = pd.DataFrame(facts, columns=["qid", "category", "attr", "value"]).drop_duplicates()
    counts = f.groupby("qid").size()
    keep = [k for k in items if counts.get(k, 0) >= min_facts
            and "disambiguation" not in items[k]["description"].lower()]
    order = {k: i for i, k in enumerate(keep)}
    rows = [items[k] for k in keep]
    for r in rows:
        r["is_adult"] = 0
    df = pd.DataFrame(rows)
    f = f[f.qid.isin(order)].copy()
    f["row"] = f.qid.map(order)
    adult_rows = f[(f.attr == "occupation") & f.value.str.lower().str.contains("|".join(ADULT_WORDS))].row
    df.loc[adult_rows.unique(), "is_adult"] = 1
    return df, f[["row", "category", "attr", "value"]]


# ----------------------------------------------------------------------------
# Common pipeline: items + facts -> akinator.db + matrix.npy
# ----------------------------------------------------------------------------
def phrase(template, value):
    text = template.format(value)
    if value[:1].lower() in "aeiou" and f" a {value}" in text:      # "a actor" -> "an actor"
        text = text.replace(f" a {value}", f" an {value}")
    return text


def write_dataset(items, facts, min_support=5, source="wikidata"):
    n = len(items)
    cats = items["category"].to_numpy(dtype=object)

    qspecs = [(BASE_Q[c], c, "category", c) for c in db.CATEGORIES]
    qspecs.append(("Is it a living thing?", None, "living", ""))
    counts = facts.groupby(["category", "attr", "value"]).size().reset_index(name="n")
    counts = counts[counts.n >= min_support]
    seen = {q[0] for q in qspecs}
    for (cat, attr), g in counts.groupby(["category", "attr"]):
        tmpl = TEMPLATES.get((cat, attr))
        if tmpl is None:
            continue
        for _, r in g.sort_values("n", ascending=False).head(MAX_Q.get(attr, DEFAULT_MAX_Q)).iterrows():
            text = phrase(tmpl, r.value)
            if text not in seen:
                seen.add(text)
                qspecs.append((text, cat, attr, r.value))
    m = len(qspecs)
    print(f"Writing {n} items x {m} questions ...")

    has, vals = {}, {}
    for (cat, attr), g in facts.groupby(["category", "attr"]):
        h = np.zeros(n, dtype=bool)
        h[g.row.to_numpy()] = True
        has[(cat, attr)] = h
    for (cat, attr, v), g in facts.groupby(["category", "attr", "value"]):
        vals[(cat, attr, v)] = g.row.to_numpy()

    P = np.zeros((n, m), dtype=np.uint8)
    for j, (_, cat, attr, val) in enumerate(qspecs):
        col = np.zeros(n, dtype=np.uint8)
        if attr in ("category", "living"):
            for c in db.CATEGORIES:
                col[cats == c] = db.base_cell(attr, cat, c)
        else:
            if (cat, attr) not in COMPLETE:
                col[(cats == cat) & ~has.get((cat, attr), np.zeros(n, dtype=bool))] = db.UNKNOWN
            col[vals[(cat, attr, val)]] = 255
        P[:, j] = col

    db.reset_files()
    db.init_schema()
    with closing(sqlite3.connect(db.DB_PATH)) as c:
        c.executemany(
            "INSERT INTO items(id,name,category,description,image_url,wiki_title,popularity,is_adult,source) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            [(i + 1, r.name, r.category, r.description, r.image_url, r.wiki_title, r.popularity,
              int(r.is_adult), source)
             for i, r in enumerate(items.itertuples())])
        c.executemany(
            "INSERT INTO questions(id,text,category,attribute,value,source) VALUES(?,?,?,?,?,?)",
            [(j + 1, t, cat, a, v, source) for j, (t, cat, a, v) in enumerate(qspecs)])
        c.commit()
    with open(db.MATRIX_PATH, "wb") as f:
        np.save(f, P)
    print("Done: akinator.db and matrix.npy written.")


# ----------------------------------------------------------------------------
# Built-in seed dataset (offline). Format: name | category | description | popularity | facts
# ----------------------------------------------------------------------------
SEED = """
Albert Einstein|person|German-born physicist, theory of relativity|200|occupation=physicist;gender=male;country=Germany|United States;alive=dead;century=19th
Marie Curie|person|Polish-French physicist and chemist, two Nobel Prizes|150|occupation=physicist|chemist;gender=female;country=Poland|France;alive=dead;century=19th
Leonardo da Vinci|person|Italian Renaissance painter and inventor|200|occupation=painter|inventor;gender=male;country=Italy;alive=dead;century=15th
William Shakespeare|person|English playwright and poet|200|occupation=writer|playwright;gender=male;country=United Kingdom;alive=dead;century=16th
Wolfgang Amadeus Mozart|person|Austrian composer|150|occupation=composer|musician;gender=male;country=Austria;alive=dead;century=18th
Michael Jackson|person|American singer, the King of Pop|180|occupation=singer|dancer;gender=male;country=United States;alive=dead;century=20th
Taylor Swift|person|American singer-songwriter|170|occupation=singer|songwriter;gender=female;country=United States;alive=alive;century=20th
Beyoncé|person|American singer and actress|160|occupation=singer|actor;gender=female;country=United States;alive=alive;century=20th
Elvis Presley|person|American rock and roll singer|160|occupation=singer|actor;gender=male;country=United States;alive=dead;century=20th
Ed Sheeran|person|English singer-songwriter|120|occupation=singer|songwriter;gender=male;country=United Kingdom;alive=alive;century=20th
Rihanna|person|Barbadian singer and businesswoman|130|occupation=singer;gender=female;country=Barbados;alive=alive;century=20th
Lata Mangeshkar|person|Indian playback singer|100|occupation=singer;gender=female;country=India;alive=dead;century=20th
A. R. Rahman|person|Indian composer and singer|100|occupation=composer|singer;gender=male;country=India;alive=alive;century=20th
Lionel Messi|person|Argentine footballer|180|occupation=footballer;gender=male;country=Argentina;alive=alive;century=20th
Cristiano Ronaldo|person|Portuguese footballer|180|occupation=footballer;gender=male;country=Portugal;alive=alive;century=20th
Sachin Tendulkar|person|Indian cricketer|140|occupation=cricketer;gender=male;country=India;alive=alive;century=20th
Virat Kohli|person|Indian cricketer|130|occupation=cricketer;gender=male;country=India;alive=alive;century=20th
Serena Williams|person|American tennis player|130|occupation=tennis player;gender=female;country=United States;alive=alive;century=20th
Usain Bolt|person|Jamaican sprinter|120|occupation=sprinter;gender=male;country=Jamaica;alive=alive;century=20th
Mahatma Gandhi|person|Indian independence leader|180|occupation=politician|lawyer;gender=male;country=India;alive=dead;century=19th
Barack Obama|person|44th president of the United States|180|occupation=politician|lawyer;gender=male;country=United States;alive=alive;century=20th
Abraham Lincoln|person|16th president of the United States|160|occupation=politician|lawyer;gender=male;country=United States;alive=dead;century=19th
Queen Elizabeth II|person|Longest-reigning British monarch|170|occupation=monarch;gender=female;country=United Kingdom;alive=dead;century=20th
Napoleon Bonaparte|person|French emperor and military leader|170|occupation=politician|military leader;gender=male;country=France;alive=dead;century=18th
Cleopatra|person|Last active ruler of Ptolemaic Egypt|150|occupation=monarch;gender=female;country=Egypt;alive=dead;century=1st
Elon Musk|person|Entrepreneur behind Tesla and SpaceX|170|occupation=entrepreneur|engineer;gender=male;country=United States|South Africa;alive=alive;century=20th
Steve Jobs|person|Co-founder of Apple|150|occupation=entrepreneur;gender=male;country=United States;alive=dead;century=20th
Amitabh Bachchan|person|Indian film actor|120|occupation=actor;gender=male;country=India;alive=alive;century=20th
Shah Rukh Khan|person|Indian film actor|130|occupation=actor;gender=male;country=India;alive=alive;century=20th
Leonardo DiCaprio|person|American actor|140|occupation=actor;gender=male;country=United States;alive=alive;century=20th
Stephen Hawking|person|English theoretical physicist|140|occupation=physicist;gender=male;country=United Kingdom;alive=dead;century=20th
A. P. J. Abdul Kalam|person|Indian scientist and 11th president|110|occupation=scientist|politician;gender=male;country=India;alive=dead;century=20th
Selena Gomez|person|American singer and actress|120|occupation=singer|actor;gender=female;country=United States;alive=alive;century=20th;decade=1990s;genre=pop music
Ariana Grande|person|American pop singer|130|occupation=singer|actor;gender=female;country=United States;alive=alive;century=20th;decade=1990s;genre=pop music
Lady Gaga|person|American singer-songwriter and actress|140|occupation=singer|songwriter|actor;gender=female;country=United States;alive=alive;century=20th;decade=1980s;genre=pop music
Katy Perry|person|American pop singer|120|occupation=singer|songwriter;gender=female;country=United States;alive=alive;century=20th;decade=1980s;genre=pop music
Britney Spears|person|American pop singer|130|occupation=singer|dancer;gender=female;country=United States;alive=alive;century=20th;decade=1980s;genre=pop music
Madonna|person|American singer, the Queen of Pop|150|occupation=singer|songwriter|actor;gender=female;country=United States;alive=alive;century=20th;decade=1950s;genre=pop music
Miley Cyrus|person|American singer and actress|110|occupation=singer|actor;gender=female;country=United States;alive=alive;century=20th;decade=1990s;genre=pop music|country music
Billie Eilish|person|American singer-songwriter|110|occupation=singer|songwriter;gender=female;country=United States;alive=alive;century=21st;decade=2000s;genre=pop music
Whitney Houston|person|American singer|120|occupation=singer|actor;gender=female;country=United States;alive=dead;century=20th;decade=1960s;genre=pop music|soul music
Dua Lipa|person|British-Albanian pop singer|100|occupation=singer|songwriter;gender=female;country=United Kingdom;alive=alive;century=20th;decade=1990s;genre=pop music
Adele|person|English singer-songwriter|130|occupation=singer|songwriter;gender=female;country=United Kingdom;alive=alive;century=20th;decade=1980s;genre=pop music|soul music
Shakira|person|Colombian singer and dancer|120|occupation=singer|dancer|songwriter;gender=female;country=Colombia;alive=alive;century=20th;decade=1970s;genre=pop music
Celine Dion|person|Canadian singer|120|occupation=singer;gender=female;country=Canada;alive=alive;century=20th;decade=1960s;genre=pop music
Shreya Ghoshal|person|Indian playback singer|90|occupation=singer;gender=female;country=India;alive=alive;century=20th;decade=1980s;genre=film music
Arijit Singh|person|Indian playback singer|90|occupation=singer;gender=male;country=India;alive=alive;century=20th;decade=1980s;genre=film music
Justin Bieber|person|Canadian pop singer|140|occupation=singer|songwriter;gender=male;country=Canada;alive=alive;century=20th;decade=1990s;genre=pop music
Bruno Mars|person|American singer and producer|110|occupation=singer|songwriter;gender=male;country=United States;alive=alive;century=20th;decade=1980s;genre=pop music|funk
The Weeknd|person|Canadian singer|110|occupation=singer|songwriter;gender=male;country=Canada;alive=alive;century=20th;decade=1990s;genre=r&b
Drake|person|Canadian rapper and singer|130|occupation=rapper|singer;gender=male;country=Canada;alive=alive;century=20th;decade=1980s;genre=hip hop music
Eminem|person|American rapper|140|occupation=rapper|songwriter;gender=male;country=United States;alive=alive;century=20th;decade=1970s;genre=hip hop music
Frank Sinatra|person|American singer and actor|130|occupation=singer|actor;gender=male;country=United States;alive=dead;century=20th;decade=1910s;genre=jazz
Bob Marley|person|Jamaican reggae singer|130|occupation=singer|songwriter;gender=male;country=Jamaica;alive=dead;century=20th;decade=1940s;genre=reggae
Tom Hanks|person|American actor|130|occupation=actor|film producer;gender=male;country=United States;alive=alive;century=20th;decade=1950s
Scarlett Johansson|person|American actress|120|occupation=actor;gender=female;country=United States;alive=alive;century=20th;decade=1980s
Priyanka Chopra|person|Indian actress and producer|100|occupation=actor|singer;gender=female;country=India;alive=alive;century=20th;decade=1980s
Deepika Padukone|person|Indian actress|100|occupation=actor;gender=female;country=India;alive=alive;century=20th;decade=1980s
Emma Watson|person|British actress|110|occupation=actor;gender=female;country=United Kingdom;alive=alive;century=20th;decade=1990s
Dwayne Johnson|person|American actor and former wrestler|120|occupation=actor|wrestler;gender=male;country=United States;alive=alive;century=20th;decade=1970s
Lion|animal|Big cat that lives in prides|150|group=mammal;eats=meat;lives=Africa|savanna;feature=a mane|fur;can=run fast
Tiger|animal|Largest big cat, with stripes|150|group=mammal;eats=meat;lives=Asia|forests;feature=stripes|fur;can=swim
Elephant|animal|Largest land animal|150|group=mammal;eats=plants;lives=Africa|Asia|savanna;feature=a trunk|tusks
Giraffe|animal|Tallest animal|120|group=mammal;eats=plants;lives=Africa|savanna;feature=a long neck|spots
Zebra|animal|African horse relative with stripes|110|group=mammal;eats=plants;lives=Africa|savanna;feature=stripes|hooves;can=run fast
Dog|animal|Domesticated companion animal|180|group=mammal;eats=meat|plants;lives=people's homes;feature=fur|a tail;can=be kept as a pet
Cat|animal|Small domesticated feline|180|group=mammal;eats=meat;lives=people's homes;feature=fur|whiskers;can=be kept as a pet|climb trees
Horse|animal|Large domesticated mammal used for riding|130|group=mammal;eats=plants;lives=farms;feature=hooves|a mane;can=run fast
Cow|animal|Farm animal that gives milk|110|group=mammal;eats=plants;lives=farms;feature=horns|hooves
Monkey|animal|Playful climbing primate|110|group=mammal;eats=plants|fruit;lives=forests;feature=a tail;can=climb trees
Panda|animal|Black and white bear that eats bamboo|130|group=mammal;eats=plants;lives=Asia|forests;feature=fur;can=climb trees
Kangaroo|animal|Hopping marsupial|110|group=mammal;eats=plants;lives=Australia;feature=a pouch|a tail;can=hop
Dolphin|animal|Intelligent marine mammal|120|group=mammal;eats=meat;lives=the ocean;can=swim|jump
Eagle|animal|Large bird of prey|110|group=bird;eats=meat;lives=mountains|forests;feature=wings|feathers;can=fly
Penguin|animal|Flightless bird of the south|120|group=bird;eats=meat;lives=Antarctica|the ocean;feature=wings|feathers;can=swim
Shark|animal|Large predatory fish|120|group=fish;eats=meat;lives=the ocean;feature=fins|sharp teeth;can=swim
Crocodile|animal|Large reptile of rivers and swamps|110|group=reptile;eats=meat;lives=Africa|Australia|rivers;feature=sharp teeth|scales;can=swim
Snake|animal|Legless reptile|120|group=reptile;eats=meat;lives=everywhere;feature=scales;can=climb trees
Frog|animal|Jumping amphibian|100|group=amphibian;eats=insects;lives=ponds|rivers;can=swim|jump
Butterfly|animal|Colourful flying insect|110|group=insect;eats=plants;lives=gardens;feature=wings;can=fly
Taj Mahal|place|White marble mausoleum in Agra|200|type=monument|mausoleum;country=India;continent=Asia
Eiffel Tower|place|Iron tower in Paris|200|type=monument|tower;country=France;continent=Europe
Great Wall of China|place|Ancient wall across northern China|190|type=monument;country=China;continent=Asia
Statue of Liberty|place|Statue on Liberty Island, New York|180|type=monument|statue;country=United States;continent=North America
Colosseum|place|Ancient amphitheatre in Rome|170|type=monument|amphitheatre;country=Italy;continent=Europe
Pyramids of Giza|place|Ancient pyramids near Cairo|180|type=monument|pyramid;country=Egypt;continent=Africa
Machu Picchu|place|Inca citadel in the Andes|150|type=monument|ruins;country=Peru;continent=South America
Big Ben|place|Clock tower at Westminster|150|type=monument|tower;country=United Kingdom;continent=Europe
Sydney Opera House|place|Performing arts centre in Sydney|150|type=building;country=Australia;continent=Oceania
Burj Khalifa|place|Tallest building in the world|150|type=building|skyscraper;country=United Arab Emirates;continent=Asia
Gateway of India|place|Arch monument in Mumbai|110|type=monument;country=India;continent=Asia
Mount Everest|place|Highest mountain on Earth|190|type=mountain;country=Nepal|China;continent=Asia
Mount Fuji|place|Iconic volcano in Japan|150|type=mountain|volcano;country=Japan;continent=Asia
Amazon River|place|Largest river by volume|150|type=river;country=Brazil|Peru;continent=South America
Niagara Falls|place|Waterfalls on the US-Canada border|150|type=waterfall;country=United States|Canada;continent=North America
Sahara|place|Largest hot desert|140|type=desert;country=Algeria|Egypt|Libya;continent=Africa
Paris|place|Capital of France|200|type=city|capital;country=France;continent=Europe
New York City|place|Largest city in the United States|200|type=city;country=United States;continent=North America
Tokyo|place|Capital of Japan|190|type=city|capital;country=Japan;continent=Asia
Mumbai|place|Large Indian port city|150|type=city;country=India;continent=Asia
Mickey Mouse|character|Cartoon mouse mascot of Disney|200|universe=Disney;kind=animal|mouse;work=cartoons|movies
Winnie the Pooh|character|Honey-loving teddy bear|150|universe=Disney;kind=animal|bear;work=books|cartoons
Elsa|character|Snow queen from Frozen|140|universe=Disney;kind=princess|queen;gender=female;work=movies
Simba|character|Lion cub from The Lion King|130|universe=Disney;kind=animal|lion;work=movies
Harry Potter|character|Boy wizard with a lightning scar|200|universe=Harry Potter;kind=wizard|hero;gender=male;work=books|movies
Voldemort|character|Dark wizard from Harry Potter|150|universe=Harry Potter;kind=wizard|villain;gender=male;work=books|movies
Spider-Man|character|Wall-crawling superhero|190|universe=Marvel;kind=superhero|hero;gender=male;work=comic books|movies
Iron Man|character|Armoured billionaire superhero|170|universe=Marvel;kind=superhero|hero;gender=male;work=comic books|movies
Batman|character|Dark Knight of Gotham|190|universe=DC Comics;kind=superhero|hero;gender=male;work=comic books|movies
Joker|character|Batman's clown villain|160|universe=DC Comics;kind=villain;gender=male;work=comic books|movies
Superman|character|Man of Steel|190|universe=DC Comics;kind=superhero|hero|alien;gender=male;work=comic books|movies
Darth Vader|character|Sith lord in Star Wars|180|universe=Star Wars;kind=villain;gender=male;work=movies
Luke Skywalker|character|Jedi hero of Star Wars|150|universe=Star Wars;kind=hero|jedi;gender=male;work=movies
Yoda|character|Wise green Jedi master|150|universe=Star Wars;kind=hero|jedi|alien;gender=male;work=movies
Mario|character|Plumber mascot of Nintendo|190|universe=Nintendo;kind=hero|plumber;gender=male;work=video games
Pikachu|character|Electric Pokémon|180|universe=Pokémon;kind=creature|animal;work=video games|anime
Sherlock Holmes|character|Consulting detective of Baker Street|160|universe=Sherlock Holmes;kind=detective|hero;gender=male;work=books|movies
Doraemon|character|Robot cat from the future|130|universe=Doraemon;kind=robot;work=anime|cartoons
Naruto Uzumaki|character|Ninja who wants to be Hokage|140|universe=Naruto;kind=ninja|hero;gender=male;work=anime
Goku|character|Saiyan hero of Dragon Ball|150|universe=Dragon Ball;kind=hero|alien;gender=male;work=anime
Shrek|character|Green ogre from the swamp|140|universe=DreamWorks;kind=ogre;gender=male;work=movies
Chhota Bheem|character|Indian cartoon hero|100|universe=Chhota Bheem;kind=hero;gender=male;work=cartoons
Smartphone|thing|Pocket computer with a touchscreen|200|kind=electronic device;material=glass|metal;use=communication;feature=a screen|a battery|a camera
Laptop|thing|Portable computer|180|kind=electronic device;material=plastic|metal;use=working;feature=a screen|a keyboard|a battery
Television|thing|Screen for watching shows|170|kind=electronic device;material=plastic|glass;use=entertainment;feature=a screen|a remote
Camera|thing|Device for taking photographs|130|kind=electronic device;material=plastic|glass;use=taking photos;feature=a lens|a battery
Wristwatch|thing|Timepiece worn on the wrist|120|kind=accessory;material=metal|glass;use=telling time;feature=a strap
Refrigerator|thing|Keeps food cold|130|kind=kitchen appliance;material=metal;use=cooking|storing food;feature=a door|a motor
Bicycle|thing|Two-wheeled pedal vehicle|160|kind=vehicle;material=metal;use=travel;feature=wheels|pedals|a handle
Car|thing|Four-wheeled road vehicle|180|kind=vehicle;material=metal|glass;use=travel;feature=wheels|an engine|a steering wheel
Guitar|thing|Stringed musical instrument|150|kind=musical instrument;material=wood;use=music;feature=strings
Piano|thing|Keyboard musical instrument|150|kind=musical instrument;material=wood;use=music;feature=keys|strings
Violin|thing|Bowed string instrument|120|kind=musical instrument;material=wood;use=music;feature=strings
Chair|thing|Seat with a back|140|kind=furniture;material=wood|metal;use=sitting;feature=legs
Sofa|thing|Soft seat for several people|110|kind=furniture;material=fabric|wood;use=sitting|sleeping;feature=cushions
Bed|thing|Furniture for sleeping|130|kind=furniture;material=wood|fabric;use=sleeping;feature=a mattress
Pen|thing|Writing instrument with ink|140|kind=stationery;material=plastic;use=writing;feature=ink
Book|thing|Bound pages to read|160|kind=stationery;material=paper;use=reading;feature=pages
Scissors|thing|Two blades for cutting|110|kind=tool;material=metal|plastic;use=cutting;feature=blades|handles
Hammer|thing|Tool for driving nails|110|kind=tool;material=metal|wood;use=building;feature=a handle
Umbrella|thing|Shelter from rain|110|kind=accessory;material=fabric|metal;use=staying dry;feature=a handle
Football|thing|Ball for kicking|130|kind=sports equipment;material=leather|plastic;use=playing;feature=air
Cricket Bat|thing|Wooden bat for cricket|100|kind=sports equipment;material=wood;use=playing;feature=a handle
Pizza|thing|Baked flatbread with cheese|150|kind=food;use=eating;feature=cheese|a crust
Toothbrush|thing|Cleans your teeth|110|kind=hygiene item;material=plastic;use=cleaning;feature=bristles|a handle
"""


# extra facts for the original seed singers (name -> "attr=v|v;...")
SEED_EXTRA = {
    "Taylor Swift": "decade=1980s;genre=pop music|country music", "Beyoncé": "decade=1980s;genre=pop music|r&b",
    "Rihanna": "decade=1980s;genre=pop music|r&b", "Ed Sheeran": "decade=1990s;genre=pop music|folk music",
    "Michael Jackson": "decade=1950s;genre=pop music|r&b", "Elvis Presley": "decade=1930s;genre=rock and roll",
    "Lata Mangeshkar": "decade=1920s;genre=film music", "A. R. Rahman": "decade=1960s;genre=film music",
    "Albert Einstein": "decade=1870s", "Marie Curie": "decade=1860s", "Mahatma Gandhi": "decade=1860s",
    "Lionel Messi": "decade=1980s", "Cristiano Ronaldo": "decade=1980s", "Sachin Tendulkar": "decade=1970s",
    "Virat Kohli": "decade=1980s", "Barack Obama": "decade=1960s", "Elon Musk": "decade=1970s",
    "Shah Rukh Khan": "decade=1960s", "Amitabh Bachchan": "decade=1940s", "Leonardo DiCaprio": "decade=1970s",
    "Serena Williams": "decade=1980s", "Usain Bolt": "decade=1980s",
}


# Wikipedia article titles for seed items whose plain name would hit the wrong page
SEED_TITLES = {"Panda": "Giant panda", "Cow": "Cattle", "Joker": "Joker (character)",
               "Elsa": "Elsa (Frozen)", "Shrek": "Shrek (character)", "Harry Potter": "Harry Potter (character)",
               "Cleopatra": "Cleopatra VII", "Drake": "Drake (musician)", "Mario": "Mario"}


def seed_frames():
    rows, facts = [], []
    for line in SEED.strip().splitlines():
        name, cat, desc, pop, fact_str = line.split("|", 4)
        if name in SEED_EXTRA:
            fact_str += ";" + SEED_EXTRA[name]
        i = len(rows)
        rows.append(dict(name=name, category=cat, description=desc, image_url="",
                         wiki_title=SEED_TITLES.get(name, name),
                         popularity=float(pop), is_adult=0))
        for part in fact_str.split(";"):
            attr, vals = part.split("=", 1)
            for v in vals.split("|"):
                facts.append((i, cat, attr, "alive" if (attr == "alive" and v == "alive") else v))
    f = pd.DataFrame(facts, columns=["row", "category", "attr", "value"])
    f = f[~((f.attr == "alive") & (f.value == "dead"))]      # dead = no "alive" fact
    return pd.DataFrame(rows), f


def build_seed():
    items, facts = seed_frames()
    write_dataset(items, facts, min_support=2, source="seed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true", help="small built-in dataset, no internet needed")
    ap.add_argument("--quick", action="store_true", help="same as --depth quick")
    ap.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard",
                    help="quick = most famous only; standard = famous + well known; "
                         "deep = also lesser-known items (much bigger, takes hours)")
    ap.add_argument("--categories", nargs="+", choices=list(CATS), default=list(CATS))
    ap.add_argument("--force", action="store_true", help="overwrite an existing database")
    args = ap.parse_args()
    if os.path.exists(db.DB_PATH) and not args.force:
        sys.exit(f"{db.DB_PATH} already exists. Re-run with --force to rebuild (learned items are lost).")
    if args.seed:
        return build_seed()
    items, facts = collect_wikidata(args.categories, "quick" if args.quick else args.depth)
    df, f = frames_from_wikidata(items, facts)
    print(f"{len(df)} items with enough facts")
    if len(df) < 200:
        sys.exit("Too few items came back (Wikidata was probably busy). Your existing database was NOT "
                 "changed. Wait a few minutes and run the same command again: finished queries are cached.")
    write_dataset(df, f, min_support=5)


if __name__ == "__main__":
    main()