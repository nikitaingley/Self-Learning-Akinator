"""app.py - Genie AI: an Akinator-style guessing game.   Run:  streamlit run app.py"""
import html
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import analytics
import db
from engine import ANSWERS, MAX_QUESTIONS, Engine

st.set_page_config(page_title="Genie AI", page_icon="🧞", layout="centered",
                   menu_items={"Get help": None, "Report a bug": None, "About": None})

# Top-right ⋮ menu: keep only "Settings" (where the Light/Dark theme choice lives).
# Rerun / Print / Record / Clear cache / Deploy are hidden.
st.markdown("""
<style>
[data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stStatusWidget"]{display:none !important;}
</style>
""", unsafe_allow_html=True)
components.html("""
<script>
try {
const doc = window.parent.document;
const KEEP = ["settings"];
function trim() {
  const menu = doc.querySelector('[data-testid="stMainMenuList"]');
  if (!menu) return;
  menu.querySelectorAll('li').forEach(li => {
    const t = (li.innerText || "").trim().toLowerCase();
    li.style.display = KEEP.some(k => t.startsWith(k)) ? "" : "none";
  });
  menu.querySelectorAll('hr, [role="separator"], [data-testid="stDivider"]').forEach(d => d.style.display = "none");
}
new MutationObserver(trim).observe(doc.body, {childList: true, subtree: true});
trim();
} catch (e) {}
</script>
""", height=0)

# ──────────────────────────────────────────────────────── CSS / design tokens
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,300&display=swap');

:root{
  --ink:#09071a;--dusk:#120f2e;--mid:#1c1850;--lantern:#ffb547;
  --ember:#ff8a4c;--rose:#e0577a;--violet:#9d78ff;--cyan:#5beafe;
  --paper:#f5ead3;--paper-ink:#1e1535;--mist:#bdb5e8;--soft:#7a6eaa;
  --glow-gold: 0 0 18px rgba(255,181,71,.45);
}

/* ── base ── */
html,body,.stApp,[data-testid="stAppViewContainer"]{
  font-family:'DM Sans',sans-serif;
  color:var(--mist);
}
.stApp{
  background:
    radial-gradient(ellipse 110% 55% at 50% 0%,  #2e2060 0%, transparent 65%),
    radial-gradient(ellipse 80%  50% at 80% 90%,  #1a0a40 0%, transparent 60%),
    var(--ink);
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:var(--dusk);border-right:1px solid #24205a;}
.block-container{max-width:780px;padding-top:1.8rem;}
h1,h2,h3,.serif{font-family:'Cinzel',serif!important;letter-spacing:.02em;}

/* ── hero ── */
.hero{text-align:center;margin:.4rem 0 1.4rem;position:relative;}
.hero svg{width:140px;height:140px;filter:drop-shadow(0 0 28px rgba(157,120,255,.5));
  animation:bob 5s ease-in-out infinite;}
.hero h1{font-family:'Cinzel',serif;font-size:2.8rem;font-weight:900;
  margin:.15rem 0 0;
  background:linear-gradient(135deg,var(--lantern),var(--violet));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero .tagline{color:var(--mist);font-size:1rem;margin:.2rem 0 0;font-style:italic;}
@keyframes bob{0%,100%{transform:translateY(0) rotate(-1deg)}50%{transform:translateY(-9px) rotate(1deg)}}
@media(prefers-reduced-motion:reduce){.hero svg{animation:none}}

/* ── question scroll ── */
.scroll{
  background:linear-gradient(135deg,rgba(40,30,90,.92) 0%,rgba(30,20,70,.92) 100%);
  border:1px solid rgba(157,120,255,.3);
  border-top:4px solid var(--lantern);
  border-radius:4px 4px 20px 20px;
  padding:1.5rem 1.8rem 1.8rem;
  box-shadow:0 24px 60px rgba(0,0,0,.5),0 0 0 1px rgba(255,181,71,.08);
  margin:.5rem 0 1rem;position:relative;overflow:hidden;
}
.scroll::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--lantern),transparent);}
.scroll .step{font-size:.85rem;color:var(--soft);margin-bottom:.4rem;letter-spacing:.06em;text-transform:uppercase;}
.scroll .q{font-family:'Cinzel',serif;font-weight:700;font-size:1.7rem;line-height:1.3;
  color:var(--paper);}

/* ── progress bar ── */
.lantern{margin:.2rem 0 .8rem;}
.lantern .label{display:flex;justify-content:space-between;color:var(--soft);
  font-size:.82rem;margin-bottom:.3rem;letter-spacing:.04em;}
.lantern .track{height:8px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden;}
.lantern .fill{height:100%;border-radius:99px;
  background:linear-gradient(90deg,var(--violet),var(--ember),var(--lantern));
  box-shadow:0 0 12px var(--lantern);transition:width .5s ease;}

/* ── guess card ── */
.guess{
  display:flex;gap:1.2rem;align-items:flex-start;
  background:linear-gradient(135deg,rgba(40,30,90,.95),rgba(25,18,60,.95));
  border:1px solid rgba(157,120,255,.35);border-radius:20px;
  padding:1.3rem;box-shadow:0 28px 70px rgba(0,0,0,.55);margin:.5rem 0 1rem;
}
.guess .pic{flex:0 0 140px;height:185px;border-radius:14px;
  background:linear-gradient(135deg,#2a1f6a,#1a1245);
  display:flex;align-items:center;justify-content:center;font-size:4rem;
  overflow:hidden;border:1px solid rgba(255,181,71,.2);}
.guess .pic img{width:100%;height:100%;object-fit:cover;border-radius:14px;}
.guess .info h2{margin:.1rem 0 .35rem;font-size:1.75rem;color:var(--paper);}
.guess .cat{display:inline-block;background:rgba(157,120,255,.18);color:var(--violet);
  border:1px solid rgba(157,120,255,.4);border-radius:99px;
  padding:.1rem .8rem;font-size:.82rem;margin-bottom:.5rem;}
.guess .desc{margin:0;color:var(--soft);font-size:.93rem;line-height:1.5;}
@media(max-width:560px){.guess{flex-direction:column}.guess .pic{flex-basis:auto;width:100%;height:220px}}

/* ── chip row ── */
.chips{margin:.2rem 0 .4rem;}
.chips span{display:inline-block;background:rgba(255,181,71,.1);color:var(--lantern);
  border:1px solid rgba(255,181,71,.3);border-radius:99px;
  padding:.15rem .75rem;margin:.15rem .2rem .15rem 0;font-size:.83rem;}

/* ── category card grid ── */
.cat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin:1rem 0;}
.cat-card{
  background:rgba(30,22,80,.8);border:1px solid rgba(157,120,255,.25);
  border-radius:14px;padding:1rem .8rem;text-align:center;cursor:pointer;
  transition:transform .15s,border-color .15s,box-shadow .15s;
}
.cat-card:hover{transform:translateY(-3px);border-color:var(--lantern);
  box-shadow:0 8px 24px rgba(255,181,71,.2);}
.cat-card .icon{font-size:2rem;margin-bottom:.3rem;}
.cat-card .label{font-size:.85rem;font-weight:500;color:var(--paper);}
.cat-card.selected{border-color:var(--lantern);background:rgba(255,181,71,.1);
  box-shadow:var(--glow-gold);}

/* category buttons styled as cards */
[class*="st-key-cat_"] button{
  padding:1rem .8rem;min-height:6.5rem;border-radius:14px;white-space:pre-line;
  background:rgba(30,22,80,.8);border:1px solid rgba(157,120,255,.25);
}
[class*="st-key-cat_"] button[kind="primary"],
[class*="st-key-cat_"] button[data-testid="stBaseButton-primary"]{
  border-color:var(--lantern);background:rgba(255,181,71,.1);box-shadow:var(--glow-gold);
}

/* ── answer buttons ── */
div.stButton>button,div.stFormSubmitButton>button{
  width:100%;border-radius:12px;border:1px solid rgba(157,120,255,.35);
  background:rgba(30,22,80,.9);color:var(--paper);font-weight:600;
  padding:.65rem .4rem;transition:transform .12s,box-shadow .12s,border-color .12s;
  letter-spacing:.02em;
}
div.stButton>button:hover,div.stFormSubmitButton>button:hover{
  border-color:var(--lantern);color:#fff;
  transform:translateY(-2px);box-shadow:0 6px 20px rgba(255,181,71,.28);
}
.st-key-a_yes button{
  background:linear-gradient(135deg,#2a4a1a,#3a6a22);border-color:#5a9a30;color:#d0f0a0;
}
.st-key-a_yes button:hover{background:linear-gradient(135deg,#3a6a22,#4a8a2a);}
.st-key-a_prob button{border-color:rgba(255,181,71,.6);color:var(--lantern);}
.st-key-a_idk button{color:var(--soft);}
.st-key-a_probnot button{border-color:rgba(224,87,122,.5);color:var(--rose);}
.st-key-a_no button{background:linear-gradient(135deg,#4a1020,#6a1530);border-color:#a02040;color:#f0a0b0;}
.st-key-a_no button:hover{background:linear-gradient(135deg,#6a1530,#8a1a40);}
.st-key-go button,.st-key-ok button{
  background:linear-gradient(135deg,var(--ember),var(--lantern));
  color:#1a0800;border:none;font-weight:700;box-shadow:0 4px 16px rgba(255,181,71,.3);
}
.st-key-go button:hover,.st-key-ok button:hover{
  box-shadow:0 6px 24px rgba(255,181,71,.5);transform:translateY(-2px);
}
.st-key-bad button{background:linear-gradient(135deg,#a9335a,var(--rose));color:#fff;border:none;}

/* ── victory ── */
.victory{text-align:center;padding:1.5rem;
  background:linear-gradient(135deg,rgba(40,30,90,.9),rgba(20,15,55,.9));
  border:1px solid rgba(255,181,71,.4);border-radius:20px;
  box-shadow:0 0 40px rgba(255,181,71,.2);}
.victory .emojiwin{font-size:4rem;animation:pop .4s cubic-bezier(.175,.885,.32,1.275);}
@keyframes pop{0%{transform:scale(0)}100%{transform:scale(1)}}
.victory h2{font-family:'Cinzel',serif;font-size:2rem;
  background:linear-gradient(135deg,var(--lantern),var(--ember));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:.5rem 0;}
.victory p{color:var(--mist);margin:0;}

/* ── debug panel ── */
.debug-panel{background:rgba(0,0,0,.5);border:1px solid rgba(157,120,255,.2);
  border-radius:10px;padding:1rem;margin-top:.5rem;font-size:.82rem;font-family:monospace;}
.debug-panel h4{margin:0 0 .5rem;color:var(--violet);font-family:monospace;font-size:.9rem;}
.debug-kv{display:grid;grid-template-columns:auto 1fr;gap:.15rem .8rem;margin-bottom:.5rem;}
.debug-kv .k{color:var(--soft);}
.debug-kv .v{color:var(--cyan);}

/* ── misc ── */
[data-testid="stMetric"]{background:rgba(30,22,80,.7);border-radius:12px;
  padding:.8rem 1rem;border:1px solid rgba(157,120,255,.2);}
.stRadio [data-testid="stMarkdownContainer"] p{color:var(--mist);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────── Genie SVG
GENIE_SVG = """
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" aria-label="Genie">
<defs>
  <radialGradient id="g1" cx="50%" cy="38%" r="60%">
    <stop offset="0" stop-color="#b09aff"/>
    <stop offset="1" stop-color="#5237c8"/>
  </radialGradient>
  <radialGradient id="g2" cx="50%" cy="50%" r="50%">
    <stop offset="0" stop-color="#ffcf60"/>
    <stop offset="1" stop-color="#e07820"/>
  </radialGradient>
  <filter id="glow">
    <feGaussianBlur stdDeviation="3" result="c"/>
    <feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
<!-- smoke tail -->
<path d="M60 200c0-28 20-38 40-38s40 10 40 38z" fill="#2a1a7a" opacity=".8"/>
<path d="M72 178c8-16 48-16 56 0" fill="none" stroke="url(#g2)" stroke-width="5" stroke-linecap="round"/>
<!-- body / head -->
<circle cx="100" cy="100" r="48" fill="url(#g1)" filter="url(#glow)"/>
<!-- turban -->
<path d="M50 86c5-36 38-54 54-54s47 18 50 54c-30-16-74-16-104 0z" fill="url(#g2)"/>
<path d="M58 80c26-12 58-12 84 0" fill="none" stroke="#ff8a2c" stroke-width="5" stroke-linecap="round"/>
<!-- jewel -->
<path d="M100 26l9 15-9 11-9-11z" fill="#e0577a" filter="url(#glow)"/>
<!-- eyes -->
<ellipse cx="83" cy="106" rx="7" ry="9" fill="white"/>
<ellipse cx="117" cy="106" rx="7" ry="9" fill="white"/>
<circle cx="85"  cy="108" r="4" fill="#1b1245"/>
<circle cx="119" cy="108" r="4" fill="#1b1245"/>
<circle cx="87"  cy="106" r="1.5" fill="white"/>
<circle cx="121" cy="106" r="1.5" fill="white"/>
<!-- smile -->
<path d="M84 128c10 10 22 10 32 0" fill="none" stroke="white" stroke-width="4" stroke-linecap="round"/>
<!-- arms -->
<path d="M58 126c-18 9-24 24-14 35" fill="none" stroke="#8060e8" stroke-width="11" stroke-linecap="round"/>
<path d="M142 126c18 9 24 24 14 35" fill="none" stroke="#8060e8" stroke-width="11" stroke-linecap="round"/>
</svg>
"""

ANSWER_BUTTONS = [
    ("a_yes",     "✅  Yes",          1.0),
    ("a_prob",    "🤔  Probably",     0.75),
    ("a_idk",     "❓  Don't know",   None),
    ("a_probnot", "🤷  Probably not", 0.25),
    ("a_no",      "❌  No",           0.0),
]

CATEGORY_ICONS = {
    "Everything": "🌟", "People": "🧑", "Animals": "🦁",
    "Places": "🗺️", "Characters": "🦸", "Objects": "📦",
}


# ──────────────────────────────────────────────────────── state & storage
@st.cache_resource(show_spinner="The genie is awakening…")
def get_store():
    return db.Store()


store = get_store()
ss = st.session_state

_defaults = dict(
    screen="start",
    engine=None,
    cur_q=None,
    guess_row=None,
    wrong=0,
    last_wrong=None,
    error="",
    result=None,
    theme="Everything",
    debug_mode=False,
)
for k, v in _defaults.items():
    ss.setdefault(k, v)


# ──────────────────────────────────────────────────────── safe wrapper
def safe(fn):
    def wrapper(*a, **kw):
        try:
            fn(*a, **kw)
        except Exception as e:          # noqa: BLE001
            ss.error = f"Something went wrong: {e}"
            ss.screen = "start"
    return wrapper


# ──────────────────────────────────────────────────────── callbacks
@safe
def on_start():
    ss.error, ss.wrong, ss.last_wrong, ss.result = "", 0, None, None
    cat = db.THEMES[ss.theme]
    mask = np.ones(store.n_items, dtype=bool)
    if cat:
        mask &= store.cats == cat
    if mask.sum() < 2:
        ss.error = "Not enough items in that theme yet. Pick another one."
        return
    ss.engine = Engine(store.P, store.pop, mask, groups=store.qgroup, penalty=store.qpen)
    ss.cur_q = ss.engine.next_question()
    ss.screen = "ask"


def _advance():
    eng = ss.engine
    q = eng.next_question()
    top = eng.top(1)
    if not top:
        ss.screen = "learn"
    elif q is None or eng.should_guess():
        ss.guess_row = top[0][0]
        ss.screen = "guess"
    else:
        ss.cur_q = q


@safe
def on_answer(value):
    ss.engine.answer(ss.cur_q, value)
    _advance()


@safe
def on_back():
    q = ss.engine.undo()
    if q is not None:
        ss.cur_q = q


@safe
def on_correct():
    row, eng = ss.guess_row, ss.engine
    store.reinforce(row, eng.history)
    store.log_game(store.cats[row], "correct", len(eng.history), ss.wrong + 1, row, eng.history)
    ss.result = ("correct", store.names[row])
    ss.screen = "done"


@safe
def on_wrong():
    eng = ss.engine
    ss.wrong += 1
    ss.last_wrong = ss.guess_row
    eng.reject(ss.guess_row)
    q = eng.next_question()
    if ss.wrong >= 5 or q is None or not eng.top(1):
        ss.screen = "learn"
    else:
        ss.cur_q = q
        ss.screen = "ask"


def on_giveup():
    ss.last_wrong = ss.guess_row
    ss.wrong += 1
    ss.screen = "learn"


def on_restart():
    ss.screen = "start"
    ss.engine = None


# ──────────────────────────────────────────────────────── UI helpers
def _lantern(conf: float, left: str, right: str):
    st.markdown(
        f'<div class="lantern"><div class="label"><span>{html.escape(left)}</span>'
        f'<span>{html.escape(right)}</span></div>'
        f'<div class="track"><div class="fill" style="width:{max(2, conf * 100):.0f}%"></div>'
        f'</div></div>',
        unsafe_allow_html=True)


def _scroll(step_text: str, question: str):
    st.markdown(
        f'<div class="scroll"><div class="step">{html.escape(step_text)}</div>'
        f'<div class="q">{html.escape(question)}</div></div>',
        unsafe_allow_html=True)


def _guess_card(name: str, desc: str, cat: str, img: str):
    pic = (f'<img src="{html.escape(img)}" alt="" onerror="this.remove()">' if img else "")
    emoji = "" if img else db.CATEGORY_EMOJI.get(cat, "🧞")
    label = db.CATEGORY_LABELS.get(cat, "")
    chip = f'<span class="cat">{html.escape(label)}</span><br>' if label else ""
    st.markdown(
        f'<div class="guess"><div class="pic">{pic}{emoji}</div>'
        f'<div class="info"><h2>{html.escape(name)}</h2>{chip}'
        f'<p class="desc">{html.escape(desc or "")}</p></div></div>',
        unsafe_allow_html=True)


def _answer_row(callback, back_cb, can_back: bool):
    cols = st.columns(5)
    for col, (key, label, value) in zip(cols, ANSWER_BUTTONS):
        col.button(label, key=key, on_click=callback, args=(value,))
    b1, b2 = st.columns(2)
    b1.button("⬅  Back", key="back", on_click=back_cb, disabled=not can_back)
    b2.button("🔄  Start over", key="restart", on_click=on_restart)


def _debug_panel(eng: Engine):
    info = eng.debug_info()
    st.markdown(
        '<div class="debug-panel"><h4>🔬 Debug — what the genie thinks</h4>'
        '<div class="debug-kv">'
        f'<span class="k">Candidates (effective):</span><span class="v">{info["candidates_left"]}</span>'
        f'<span class="k">Meaningful answers:</span><span class="v">{info["meaningful_answers"]}</span>'
        f'<span class="k">Last info gain:</span><span class="v">{info["last_gain"]:.4f}</span>'
        f'<span class="k">Top confidence:</span><span class="v">{info["confidence"]*100:.1f}%</span>'
        f'<span class="k">2nd confidence:</span><span class="v">{info["second_confidence"]*100:.1f}%</span>'
        f'<span class="k">Conf gap:</span><span class="v">{info["conf_gap"]*100:.1f}%</span>'
        f'<span class="k">Will guess after N answers:</span><span class="v">{info["guess_after"]}</span>'
        f'<span class="k">Rejected items:</span><span class="v">{info["rejected"]}</span>'
        '</div></div>',
        unsafe_allow_html=True)
    if info["top5"]:
        st.markdown("**Top candidates:**")
        for row, p in info["top5"]:
            label = f"{store.names[row]}  ({p*100:.1f}%)"
            st.progress(min(1.0, p), text=label)


# ──────────────────────────────────────────────────────── screens
def screen_start():
    st.markdown(
        f'<div class="hero">{GENIE_SVG}'
        '<h1>Genie AI</h1>'
        '<p class="tagline">Think of anyone or anything — I will read your mind.</p></div>',
        unsafe_allow_html=True)

    if ss.error:
        st.error(ss.error)

    # ── category picker
    st.markdown("#### Choose a category")
    cols = st.columns(3)
    for i, (theme_name, icon) in enumerate(CATEGORY_ICONS.items()):
        selected = ss.theme == theme_name
        if cols[i % 3].button(f"{icon}\n\n{theme_name}", key=f"cat_{theme_name}",
                              type="primary" if selected else "secondary",
                              use_container_width=True, help=f"Play with {theme_name}"):
            ss.theme = theme_name
            st.rerun()

    st.divider()

    st.toggle("🔬 Developer / debug mode", key="debug_mode",
              help="Shows candidate scores and question analysis during the game.")

    st.button("✨  Start the game", key="go", on_click=on_start)
    st.caption(f"🧞 My AI knows **{store.n_items:,}** items and **{store.n_q:,}** questions "
               f"— and learns from every game.")


def screen_ask():
    eng = ss.engine
    n = len(eng.history)
    pct = eng.confidence()
    _lantern(pct,
             f"Question {n + 1} / {MAX_QUESTIONS}",
             f"~{eng.candidates_left():,} candidates")
    _scroll(f"Question {n + 1}", store.qtext[ss.cur_q])
    _answer_row(on_answer, on_back, n > 0)

    if ss.debug_mode:
        with st.expander("🔬 Genie's thoughts", expanded=True):
            _debug_panel(eng)


def screen_guess():
    eng, row = ss.engine, ss.guess_row
    it = store.item(row)

    st.markdown(
        f'<div class="victory">'
        f'<div class="emojiwin">🔮</div>'
        f'<h2>I think I know…</h2>'
        f'<p>After {len(eng.history)} questions, my best guess is:</p></div>',
        unsafe_allow_html=True)
    st.markdown("")

    _guess_card(it["name"], it["description"], it["category"], it["image_url"])

    # show other plausible candidates as chips
    others = [store.names[r] for r, _ in eng.top(6) if r != row][:5]
    if others:
        st.markdown(
            '<div class="chips">Also possible: '
            + "".join(f'<span>{html.escape(o)}</span>' for o in others)
            + '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.button("✅  Yes, that's it!", key="ok", on_click=on_correct)
    col2.button("❌  No, keep guessing", key="bad", on_click=on_wrong)
    st.button("🏳️  I give up — teach the genie", key="giveup", on_click=on_giveup)

    if ss.debug_mode:
        with st.expander("🔬 Genie's confidence scores"):
            _debug_panel(eng)


def screen_learn():
    eng = ss.engine
    wrong_name = store.names[ss.last_wrong] if ss.last_wrong is not None else "my guess"
    default_cat = store.cats[ss.last_wrong] if ss.last_wrong is not None else "person"

    st.markdown(
        '<div class="victory" style="border-color:rgba(224,87,122,.4)">'
        '<div class="emojiwin">🤔</div>'
        '<h2>You beat me!</h2>'
        '<p>Help me learn so I can do better next time.</p>'
        '</div>', unsafe_allow_html=True)
    st.markdown("")
    st.caption("What you tell me is saved and used in future games.")

    with st.form("learn"):
        name = st.text_input("What were you thinking of?")
        cat = st.selectbox("Category", db.CATEGORIES,
                           index=db.CATEGORIES.index(default_cat),
                           format_func=lambda c: db.CATEGORY_LABELS[c])
        desc = st.text_input("Short description (optional)")
        img = st.text_input("Image URL (optional)")
        q = st.text_input(f"A yes/no question that separates it from '{wrong_name}' (optional)")
        new_yes = st.radio("For your item, the answer to that question is",
                           ["Yes", "No"], horizontal=True) == "Yes"
        submitted = st.form_submit_button("📚  Teach the genie")

    if submitted:
        if not name.strip():
            st.warning("Tell me what you were thinking of first.")
            return
        existing = store.find_item(name)
        if existing is not None:
            store.reinforce(existing, eng.history)
            row, note = existing, f"'{name.strip()}' was already in my memory — I've strengthened it."
        else:
            row = store.add_item(name, cat, desc, img, eng.history,
                                 wrong_row=ss.last_wrong, question=q, new_is_yes=new_yes)
            note = f"Learned '{name.strip()}'. I now know {store.n_items:,} items."
        store.log_game(cat, "wrong", len(eng.history), ss.wrong, row, eng.history)
        ss.result = ("learned", note)
        ss.screen = "done"
        st.rerun()

    st.button("🔄  Start over", key="restart", on_click=on_restart)


def screen_done():
    kind, text = ss.result or ("", "")

    if kind == "correct":
        st.markdown(
            f'<div class="victory">'
            f'<div class="emojiwin">🎉</div>'
            f'<h2>Got it: {html.escape(text)}!</h2>'
            f'<p>My knowledge has been strengthened. Want to play again?</p>'
            f'</div>', unsafe_allow_html=True)
    elif kind == "learned":
        st.markdown(
            '<div class="victory" style="border-color:rgba(91,234,254,.4)">'
            '<div class="emojiwin">📚</div>'
            '<h2>Thanks, I\'m smarter now!</h2>'
            f'<p>{html.escape(text)}</p>'
            '</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="victory"><div class="emojiwin">🧞</div>'
            '<h2>Round finished!</h2></div>',
            unsafe_allow_html=True)

    st.markdown("")
    st.button("🔮  Play again", key="go", on_click=on_restart)


# ──────────────────────────────────────────────────────── sidebar + router
page = st.sidebar.radio("", ["🎮  Play", "📊  Analytics", "⚙️  Settings"], key="page")
st.sidebar.markdown("---")
st.sidebar.caption(f"**{store.n_items:,}** items · **{store.n_q:,}** questions")
if ss.screen != "start":
    st.sidebar.button("🔄  Restart game", on_click=on_restart, key="sb_restart")

if "📊" in page:
    analytics.show(store)
elif "⚙️" in page:
    # ── Settings screen
    st.markdown("## ⚙️  Settings")
    st.toggle("🔬 Developer / debug mode", key="debug_mode",
              help="Shows candidate scores, info gains, and elimination reasons during gameplay.")
    st.divider()
    st.markdown("### Dataset")
    st.caption(f"Items: **{store.n_items:,}**  |  Questions: **{store.n_q:,}**")
    st.info("To rebuild the dataset with more items, run:\n```\npython build_dataset.py --seed\n```\n"
            "or for Wikidata:\n```\npython build_dataset.py --force\n```")
    st.divider()
    st.markdown("### About Genie AI")
    st.markdown(
        "Genie AI uses a **Bayesian guessing engine** with **information-gain question selection**.\n\n"
        "- Every answer updates a probability distribution over all candidates.\n"
        "- The next question is chosen to maximise the expected reduction in uncertainty (entropy).\n"
        "- A guess is only made when the top candidate is sufficiently more probable than all others.\n"
        "- The AI learns from every game, reinforcing correct guesses and adding new items.\n\n"
        "No decision trees, no hardcoded category rules."
    )
else:
    # ── Game router
    screens = {
        "start": screen_start,
        "ask": screen_ask,
        "guess": screen_guess,
        "learn": screen_learn,
        "done": screen_done,
    }
    screens.get(ss.screen, screen_start)()