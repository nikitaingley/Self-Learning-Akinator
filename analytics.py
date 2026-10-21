"""analytics.py - the Analytics page."""
from contextlib import closing

import pandas as pd
import plotly.express as px
import streamlit as st

import db

GOLD, ROSE, MIST = "#ffb547", "#e0577a", "#9d93d6"
RESULT_COLORS = {"correct": GOLD, "wrong": ROSE}


def _style(fig):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def show(store):
    with closing(db.conn()) as c:
        games = pd.read_sql("SELECT * FROM game_log ORDER BY game_id", c)
        learned = pd.read_sql(
            "SELECT name, category, added_at FROM items WHERE source='learned' ORDER BY id", c)
        found = pd.read_sql(
            "SELECT i.name, i.category, COUNT(*) AS times_found FROM game_log g "
            "JOIN items i ON i.id = g.item_id WHERE g.result='correct' "
            "GROUP BY i.id ORDER BY times_found DESC LIMIT 10", c)

    st.markdown("## Analytics")
    a, b, c3, d = st.columns(4)
    a.metric("Items known", f"{store.n_items:,}")
    b.metric("Questions", f"{store.n_q:,}")
    c3.metric("Learned from players", len(learned))
    d.metric("Games played", len(games))

    if games.empty:
        st.info("Play a few rounds, then come back to see how the genie is doing.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Guessing success rate")
            counts = games["result"].value_counts().reset_index()
            counts.columns = ["result", "games"]
            fig = px.pie(counts, names="result", values="games", hole=0.55,
                         color="result", color_discrete_map=RESULT_COLORS)
            st.plotly_chart(_style(fig), use_container_width=True)
        with right:
            st.markdown("#### Accuracy by category")
            acc = (games.assign(ok=games["result"] == "correct")
                   .groupby("category")["ok"].mean().mul(100).round(1).reset_index())
            acc.columns = ["category", "accuracy_%"]
            acc = acc.sort_values("accuracy_%")
            fig = px.bar(acc, x="category", y="accuracy_%", range_y=[0, 100],
                         color_discrete_sequence=[GOLD])
            st.plotly_chart(_style(fig), use_container_width=True)
            st.caption(f"Hardest category: {acc.iloc[0]['category']}")

        st.markdown("#### Questions needed per game")
        g = games.reset_index(drop=True)
        g["game"] = g.index + 1
        g["average of last 5"] = g["questions"].rolling(5, min_periods=1).mean()
        fig = px.line(g, x="game", y=["questions", "average of last 5"],
                      color_discrete_sequence=[MIST, GOLD])
        fig.update_layout(legend_title_text="")
        st.plotly_chart(_style(fig), use_container_width=True)

    if not found.empty:
        st.markdown("#### Most often guessed correctly")
        st.dataframe(found, hide_index=True, use_container_width=True)

    st.markdown("#### Taught by players")
    if learned.empty:
        st.caption("Nothing yet. When the genie guesses wrong, teach it and it appears here.")
    else:
        learned = learned.copy()
        learned["total learned"] = range(1, len(learned) + 1)
        st.dataframe(learned, hide_index=True, use_container_width=True)
        st.line_chart(learned.set_index("added_at")["total learned"])