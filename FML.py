import streamlit as st
import numpy as np
import random

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LearnAI - Learning Path Recommendation",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# TOPICS
# =========================================================

TOPICS = [
    "Python",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Machine Learning",
    "Deep Learning"
]

# =========================================================
# Q-LEARNING PARAMETERS
# =========================================================

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2
EPISODES = 1000

# Q Table
Q = np.zeros((3, 6))


# =========================================================
# GET PERFORMANCE STATE
# =========================================================

def get_state(score):

    if score < 50:
        return 0

    elif score < 75:
        return 1

    else:
        return 2


# =========================================================
# REWARD
# =========================================================

def get_reward(score):

    if score < 50:
        return 10

    elif score < 75:
        return 5

    else:
        return 2


# =========================================================
# TRAIN Q-LEARNING
# =========================================================

for episode in range(EPISODES):

    state = random.randint(0, 2)

    if random.random() < EPSILON:

        action = random.randint(0, 5)

    else:

        action = np.argmax(Q[state])

    if state == 0:

        score = random.randint(20, 49)

    elif state == 1:

        score = random.randint(50, 74)

    else:

        score = random.randint(75, 95)

    reward = get_reward(score)

    next_state = get_state(score)

    Q[state, action] = Q[state, action] + LEARNING_RATE * (
        reward
        + DISCOUNT_FACTOR * np.max(Q[next_state])
        - Q[state, action]
    )


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend(scores):

    average = sum(scores.values()) / len(scores)

    state = get_state(average)

    if state == 0:

        performance = "Low Performance"

    elif state == 1:

        performance = "Medium Performance"

    else:

        performance = "High Performance"

    # Weak topics first
    learning_path = sorted(
        scores,
        key=scores.get
    )

    return {
        "average": round(average, 2),
        "performance": performance,
        "recommended_topics": learning_path[:3],
        "learning_path": learning_path
    }


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #312e81, transparent 35%),
        radial-gradient(circle at bottom right, #581c87, transparent 35%),
        #070b18;
    color: white;
}

.main-title {
    text-align: center;
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        rgba(79,70,229,0.45),
        rgba(124,58,237,0.35)
    );
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 30px;
}

.main-title h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.main-title p {
    color: #cbd5e1;
    font-size: 17px;
}

.card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 20px;
}

.result-card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        rgba(16,185,129,0.15),
        rgba(59,130,246,0.12)
    );
    border: 1px solid rgba(52,211,153,0.25);
}

.big-number {
    font-size: 42px;
    font-weight: bold;
    color: #c4b5fd;
}

.topic-card {
    padding: 18px;
    margin: 10px 0;
    border-radius: 15px;
    background: #111827;
    border: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🧠 LearnAI")

    st.markdown("---")

    st.markdown("### 👤 Student Profile")

    student_name = st.text_input(
        "Student Name",
        "Laxmi Mankar"
    )

    roll_no = st.text_input(
        "Roll No.",
        "CM24092"
    )

    target_goal = st.selectbox(
        "Target Goal",
        [
            "AI & Data Science",
            "Machine Learning",
            "Data Analytics",
            "Python Development"
        ]
    )

    st.markdown("---")

    st.markdown("### ⚙️ RL Agent Settings")

    learning_rate = st.slider(
        "Learning Rate (α)",
        0.01,
        1.0,
        LEARNING_RATE
    )

    discount_factor = st.slider(
        "Discount Factor (γ)",
        0.01,
        1.0,
        DISCOUNT_FACTOR
    )

    st.markdown("---")

    st.caption("Model: Q-Learning")
    st.caption("Training Episodes: 1000")
    st.caption("Actions: 6 Topics")


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">

<h1>🧠 Learning Path Recommendation System</h1>

<p>
PBL Project | Personalized Learning using
Reinforcement Learning (Q-Learning)
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# MAIN COLUMNS
# =========================================================

left, right = st.columns([1.25, 1])


# =========================================================
# LEFT SIDE
# =========================================================

with left:

    st.markdown("## 📊 Student Skill Analysis")

    st.markdown("""
    <div class="card">
    Enter your current performance score for each topic.
    The RL agent will analyze your knowledge state and
    generate a personalized learning path.
    </div>
    """, unsafe_allow_html=True)

    scores = {}

    col1, col2 = st.columns(2)

    for i, topic in enumerate(TOPICS):

        with col1 if i % 2 == 0 else col2:

            scores[topic] = st.number_input(
                topic,
                min_value=0,
                max_value=100,
                value=50,
                step=1
            )


    generate = st.button(
        "🚀 Generate Personalized Learning Path",
        use_container_width=True
    )


# =========================================================
# RIGHT SIDE
# =========================================================

with right:

    st.markdown("## 🤖 RL Agent Analytics")

    st.markdown("""
    <div class="card">

    <h3>Q-Learning Model</h3>

    <p>
    The agent observes the student's performance state,
    evaluates possible learning actions and uses rewards
    to learn which topic should be recommended.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.metric(
        "Training Episodes",
        EPISODES
    )

    st.metric(
        "Learning Rate (α)",
        f"{learning_rate:.2f}"
    )

    st.metric(
        "Discount Factor (γ)",
        f"{discount_factor:.2f}"
    )


# =========================================================
# RECOMMENDATION
# =========================================================

if generate:

    result = recommend(scores)

    average = result["average"]
    performance = result["performance"]

    # -----------------------------------------------------
    # Analytics
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown("## 🎯 Student Performance Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Average Score",
            f"{average}%"
        )

    with c2:
        st.metric(
            "Performance",
            performance
        )

    with c3:
        st.metric(
            "RL State",
            result["performance"].replace(
                " Performance", ""
            )
        )

    with c4:
        st.metric(
            "Topics",
            len(TOPICS)
        )


    # -----------------------------------------------------
    # RL Recommendation
    # -----------------------------------------------------

    st.markdown("## 🎯 RL Agent Recommendation")

    recommended = result["recommended_topics"]

    st.markdown(f"""
    <div class="result-card">

    <h3>🤖 Recommended Next Topic</h3>

    <h2>🚀 {recommended[0]}</h2>

    <p>
    Based on the student's current performance,
    this topic has the highest learning priority.
    </p>

    </div>
    """, unsafe_allow_html=True)


    # -----------------------------------------------------
    # TOP 3
    # -----------------------------------------------------

    st.markdown("## ⭐ Top Recommended Topics")

    r1, r2, r3 = st.columns(3)

    for col, topic in zip(
        [r1, r2, r3],
        recommended
    ):

        with col:

            st.markdown(f"""
            <div class="topic-card">

            <h3>🎯 {topic}</h3>

            <p>
            Current Score:
            <b>{scores[topic]}%</b>
            </p>

            </div>
            """, unsafe_allow_html=True)


    # -----------------------------------------------------
    # COMPLETE LEARNING PATH
    # -----------------------------------------------------

    st.markdown("## 🛣️ Personalized Learning Path")

    for i, topic in enumerate(
        result["learning_path"],
        1
    ):

        score = scores[topic]

        st.markdown(f"""
        <div class="topic-card">

        <h3>
        #{i} &nbsp; {topic}
        </h3>

        <p>
        Current Performance: <b>{score}%</b>
        </p>

        </div>
        """, unsafe_allow_html=True)


    # -----------------------------------------------------
    # Q TABLE
    # -----------------------------------------------------

    st.markdown("## 🧠 Q-Table Matrix")

    st.markdown("""
    <div class="card">

    <p>
    <b>Rows:</b> Current Knowledge State
    &nbsp;&nbsp;&nbsp;
    <b>Columns:</b> Learning Actions / Topics
    </p>

    </div>
    """, unsafe_allow_html=True)

    qtable_data = Q.copy()

    import pandas as pd

    qtable_df = pd.DataFrame(
        qtable_data,
        index=[
            "Low",
            "Medium",
            "High"
        ],
        columns=TOPICS
    )

    st.dataframe(
        qtable_df.round(2),
        use_container_width=True
    )


    # -----------------------------------------------------
    # TOPIC PERFORMANCE TABLE
    # -----------------------------------------------------

    st.markdown("## 📋 Student Performance Table")

    performance_df = pd.DataFrame({
        "Topic": TOPICS,
        "Score": [
            scores[topic]
            for topic in TOPICS
        ]
    })

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#94a3b8; padding:20px;">

<b>LearnAI</b><br>

Learning Path Recommendation using Reinforcement Learning<br>

PBL Project • Q-Learning

</div>
""", unsafe_allow_html=True)