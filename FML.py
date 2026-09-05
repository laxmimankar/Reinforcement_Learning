from flask import Flask, request, jsonify
import numpy as np
import random

app = Flask(__name__)

# =========================================================
# PROJECT CONFIGURATION
# =========================================================

TOPICS = [
    "Python",
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Machine Learning",
    "Deep Learning"
]

STATES = ["Low", "Medium", "High"]

LEARNING_RATE = 0.10
DISCOUNT_FACTOR = 0.90
EPSILON = 0.20
EPISODES = 1000

# Q-Table
Q = np.zeros((len(STATES), len(TOPICS)))


# =========================================================
# RL FUNCTIONS
# =========================================================

def get_state(score):
    """
    Convert student performance into RL state.

    0 = Low
    1 = Medium
    2 = High
    """

    if score < 50:
        return 0
    elif score < 75:
        return 1
    return 2


def get_reward(score):
    """
    Reward is based on learning need.

    Lower score = higher learning priority.
    """

    if score < 50:
        return 10

    elif score < 75:
        return 5

    return 2


# =========================================================
# Q-LEARNING TRAINING
# =========================================================

def train_q_learning():

    global Q

    Q = np.zeros((len(STATES), len(TOPICS)))

    for episode in range(EPISODES):

        # Random current knowledge state
        state = random.randint(0, 2)

        # Epsilon-Greedy action selection
        if random.random() < EPSILON:

            action = random.randint(0, len(TOPICS) - 1)

        else:

            action = np.argmax(Q[state])

        # Simulated performance
        if state == 0:
            score = random.randint(20, 49)

        elif state == 1:
            score = random.randint(50, 74)

        else:
            score = random.randint(75, 95)

        # Reward
        reward = get_reward(score)

        # Next state
        next_state = get_state(score)

        # Q-Learning update
        old_value = Q[state, action]

        best_future_value = np.max(Q[next_state])

        new_value = old_value + LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR * best_future_value
            - old_value
        )

        Q[state, action] = new_value


train_q_learning()


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

def generate_learning_path(scores):

    average = np.mean(list(scores.values()))

    current_state = get_state(average)

    performance = STATES[current_state]

    # -----------------------------------------------------
    # Calculate weakness priority
    # -----------------------------------------------------

    weakness = {}

    for topic, score in scores.items():

        # Higher value = more learning need
        weakness[topic] = 100 - score

    # -----------------------------------------------------
    # RL Q-values
    # -----------------------------------------------------

    q_values = Q[current_state]

    # Normalize Q values
    if np.max(q_values) > 0:

        q_normalized = (
            q_values / np.max(q_values)
        )

    else:

        q_normalized = np.zeros(len(TOPICS))

    # -----------------------------------------------------
    # Combined recommendation score
    # -----------------------------------------------------

    recommendation_score = {}

    for i, topic in enumerate(TOPICS):

        weakness_score = weakness[topic]

        rl_score = q_normalized[i] * 100

        # 70% student weakness
        # 30% learned RL policy
        final_score = (
            0.70 * weakness_score
            + 0.30 * rl_score
        )

        recommendation_score[topic] = final_score

    # Sort according to final recommendation score
    learning_path = sorted(
        TOPICS,
        key=lambda x: recommendation_score[x],
        reverse=True
    )

    # Top 3
    recommended = learning_path[:3]

    return {
        "average": round(float(average), 2),
        "state": current_state,
        "performance": performance,
        "learning_path": learning_path,
        "recommended": recommended,
        "q_values": {
            TOPICS[i]: round(float(q_values[i]), 3)
            for i in range(len(TOPICS))
        },
        "priority_scores": {
            topic: round(
                float(recommendation_score[topic]), 2
            )
            for topic in TOPICS
        }
    }


# =========================================================
# CSS
# =========================================================

CSS = """

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background:
        radial-gradient(circle at top left, #312e81, transparent 35%),
        radial-gradient(circle at bottom right, #581c87, transparent 35%),
        #070b18;
    color: #f8fafc;
}

.container {
    width: 92%;
    max-width: 1150px;
    margin: auto;
    padding: 35px 0;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 35px;
}

.logo {
    font-size: 24px;
    font-weight: 800;
}

.logo span {
    color: #a78bfa;
}

.badge {
    padding: 8px 15px;
    border-radius: 20px;
    background: rgba(139,92,246,0.18);
    border: 1px solid rgba(167,139,250,0.35);
    color: #c4b5fd;
    font-size: 13px;
}

.hero {
    text-align: center;
    padding: 65px 25px;
    border-radius: 30px;
    background:
        linear-gradient(
            135deg,
            rgba(79,70,229,0.30),
            rgba(124,58,237,0.22)
        );
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 25px 80px rgba(0,0,0,0.35);
}

.hero-icon {
    font-size: 65px;
}

.hero h1 {
    font-size: 48px;
    margin: 15px 0;
}

.hero h1 span {
    color: #a78bfa;
}

.hero p {
    color: #cbd5e1;
    font-size: 18px;
    line-height: 1.7;
}

.btn {
    display: inline-block;
    margin-top: 25px;
    padding: 15px 30px;
    border-radius: 14px;
    text-decoration: none;
    color: white;
    font-weight: 700;
    background: linear-gradient(135deg,#6366f1,#a855f7);
    box-shadow: 0 10px 30px rgba(124,58,237,0.35);
}

.section-title {
    margin: 35px 0 20px;
    font-size: 26px;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 18px;
}

.card {
    background: rgba(15,23,42,0.80);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 20px;
}

.card h3 {
    margin-top: 0;
}

.muted {
    color: #94a3b8;
}

.form-card {
    max-width: 800px;
    margin: auto;
    background: rgba(15,23,42,0.90);
    padding: 35px;
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.08);
}

.input-grid {
    display: grid;
    grid-template-columns: repeat(2,1fr);
    gap: 18px;
}

label {
    display: block;
    margin-bottom: 8px;
    color: #cbd5e1;
    font-weight: 600;
}

input {
    width: 100%;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #334155;
    background: #0f172a;
    color: white;
    font-size: 15px;
}

input:focus {
    outline: none;
    border-color: #8b5cf6;
}

.submit {
    width: 100%;
    margin-top: 25px;
    padding: 16px;
    border: none;
    border-radius: 14px;
    color: white;
    font-weight: 800;
    font-size: 16px;
    background: linear-gradient(135deg,#6366f1,#a855f7);
    cursor: pointer;
}

.stats {
    display: grid;
    grid-template-columns: repeat(4,1fr);
    gap: 15px;
    margin: 25px 0;
}

.stat {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 22px;
}

.stat-value {
    font-size: 30px;
    font-weight: 800;
    color: #c4b5fd;
}

.stat-label {
    margin-top: 6px;
    color: #94a3b8;
    font-size: 13px;
}

.recommendation {
    padding: 25px;
    border-radius: 22px;
    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.14),
            rgba(59,130,246,0.12)
        );
    border: 1px solid rgba(52,211,153,0.25);
}

.path-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px;
    margin: 10px 0;
    border-radius: 15px;
    background: #111827;
    border: 1px solid #1e293b;
}

.rank {
    font-size: 20px;
    font-weight: 800;
    color: #a78bfa;
}

.score {
    color: #94a3b8;
}

.progress-bg {
    height: 9px;
    background: #1e293b;
    border-radius: 10px;
    margin-top: 10px;
    overflow: hidden;
}

.progress {
    height: 100%;
    background: linear-gradient(90deg,#6366f1,#c084fc);
    border-radius: 10px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}

th, td {
    padding: 14px;
    text-align: left;
    border-bottom: 1px solid #1e293b;
}

th {
    color: #c4b5fd;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 45px;
    padding-bottom: 20px;
}

@media(max-width:700px) {

    .hero h1 {
        font-size: 34px;
    }

    .card-grid,
    .stats,
    .input-grid {
        grid-template-columns: 1fr;
    }

}

</style>

"""


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return CSS + """

    <div class="container">

        <div class="navbar">

            <div class="logo">
                🧠 Learn<span>AI</span>
            </div>

            <div class="badge">
                REINFORCEMENT LEARNING
            </div>

        </div>


        <div class="hero">

            <div class="hero-icon">🎓</div>

            <h1>
                Learning Path <span>Recommendation</span>
            </h1>

            <p>
                A personalized learning recommendation system
                powered by Reinforcement Learning and Q-Learning.
            </p>

            <a class="btn" href="/dashboard">
                Start Learning →
            </a>

        </div>


        <h2 class="section-title">
            How the System Works
        </h2>


        <div class="card-grid">

            <div class="card">
                <h3>📊 01. Analyze</h3>
                <p class="muted">
                    Student performance scores are used
                    to identify the current knowledge state.
                </p>
            </div>

            <div class="card">
                <h3>🤖 02. Learn</h3>
                <p class="muted">
                    The Q-Learning agent evaluates possible
                    learning actions using Q-values.
                </p>
            </div>

            <div class="card">
                <h3>🎯 03. Recommend</h3>
                <p class="muted">
                    The system generates a personalized
                    learning path based on weakness and RL policy.
                </p>
            </div>

        </div>


        <div class="footer">
            PBL Project • Learning Path Recommendation using Reinforcement Learning
        </div>

    </div>

    """


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    inputs = ""

    for topic in TOPICS:

        inputs += f"""

        <div>

            <label>{topic}</label>

            <input
                type="number"
                name="{topic}"
                min="0"
                max="100"
                placeholder="Enter score (0-100)"
                required
            >

        </div>

        """

    return CSS + f"""

    <div class="container">

        <div class="navbar">

            <div class="logo">
                🧠 Learn<span>AI</span>
            </div>

            <a class="badge"
               href="/">
               ← Home
            </a>

        </div>


        <div class="form-card">

            <h1>📊 Student Skill Analysis</h1>

            <p class="muted">
                Enter your current performance scores.
                The RL agent will generate your personalized
                learning path.
            </p>


            <form action="/recommend" method="POST">

                <div class="input-grid">

                    {inputs}

                </div>


                <button class="submit" type="submit">

                    🚀 Generate Personalized Learning Path

                </button>

            </form>

        </div>

    </div>

    """


# =========================================================
# RESULT
# =========================================================

@app.route("/recommend", methods=["POST"])
def recommendation():

    scores = {}

    for topic in TOPICS:

        value = request.form.get(topic, 0)

        try:
            value = float(value)

        except:
            value = 0

        scores[topic] = max(
            0,
            min(100, value)
        )


    result = generate_learning_path(scores)


    # -----------------------------------------------------
    # Topic cards
    # -----------------------------------------------------

    topic_html = ""

    for i, topic in enumerate(
        result["learning_path"],
        1
    ):

        score = scores[topic]

        topic_html += f"""

        <div class="path-item">

            <div>

                <span class="rank">
                    #{i}
                </span>

                &nbsp;&nbsp;

                <b>{topic}</b>

                <div class="progress-bg">

                    <div
                        class="progress"
                        style="width:{score}%">
                    </div>

                </div>

            </div>

            <div class="score">
                {score:.0f}%
            </div>

        </div>

        """


    # -----------------------------------------------------
    # Q-table
    # -----------------------------------------------------

    qtable_html = ""

    for i, state in enumerate(STATES):

        qtable_html += "<tr>"

        qtable_html += f"<td><b>{state}</b></td>"

        for j in range(len(TOPICS)):

            qtable_html += (
                f"<td>{Q[i,j]:.2f}</td>"
            )

        qtable_html += "</tr>"


    # -----------------------------------------------------
    # Recommended topics
    # -----------------------------------------------------

    recommended_html = ""

    for topic in result["recommended"]:

        recommended_html += f"""

        <div class="card">

            <h3>🎯 {topic}</h3>

            <p class="muted">
                Current Score:
                <b>{scores[topic]:.0f}%</b>
            </p>

            <p class="muted">
                Priority Score:
                <b>
                    {result["priority_scores"][topic]}
                </b>
            </p>

        </div>

        """


    return CSS + f"""

    <div class="container">

        <div class="navbar">

            <div class="logo">
                🧠 Learn<span>AI</span>
            </div>

            <a class="badge"
               href="/dashboard">
               ← Analyze Again
            </a>

        </div>


        <h1>
            🎯 Personalized Learning Dashboard
        </h1>


        <div class="stats">

            <div class="stat">

                <div class="stat-value">
                    {result["average"]}%
                </div>

                <div class="stat-label">
                    Average Performance
                </div>

            </div>


            <div class="stat">

                <div class="stat-value">
                    {result["performance"]}
                </div>

                <div class="stat-label">
                    Knowledge State
                </div>

            </div>


            <div class="stat">

                <div class="stat-value">
                    {result["state"]}
                </div>

                <div class="stat-label">
                    RL State
                </div>

            </div>


            <div class="stat">

                <div class="stat-value">
                    {EPISODES}
                </div>

                <div class="stat-label">
                    Training Episodes
                </div>

            </div>

        </div>


        <div class="recommendation">

            <h2>
                🤖 RL Agent Recommendation
            </h2>

            <p class="muted">

                Based on the student's current state,
                performance weakness and learned Q-values,
                the agent recommends:

            </p>

            <h2>
                🚀 {result["recommended"][0]}
            </h2>

            <p>
                This topic has the highest current
                learning priority.
            </p>

        </div>


        <h2 class="section-title">
            ⭐ Top Recommended Topics
        </h2>


        <div class="card-grid">

            {recommended_html}

        </div>


        <h2 class="section-title">
            🛣️ Complete Learning Path
        </h2>


        {topic_html}


        <h2 class="section-title">
            🧠 Q-Table — State × Action Values
        </h2>


        <div class="card">

            <p class="muted">

                Rows represent the current knowledge state.
                Columns represent possible learning actions.

            </p>


            <div style="overflow-x:auto;">

                <table>

                    <tr>

                        <th>State</th>

                        {''.join(
                            f'<th>{topic}</th>'
                            for topic in TOPICS
                        )}

                    </tr>

                    {qtable_html}

                </table>

            </div>

        </div>


        <h2 class="section-title">
            ⚙️ RL Configuration
        </h2>


        <div class="card-grid">

            <div class="card">

                <h3>Learning Rate α</h3>

                <p class="stat-value">
                    {LEARNING_RATE}
                </p>

            </div>

            <div class="card">

                <h3>Discount Factor γ</h3>

                <p class="stat-value">
                    {DISCOUNT_FACTOR}
                </p>

            </div>

            <div class="card">

                <h3>Exploration ε</h3>

                <p class="stat-value">
                    {EPSILON}
                </p>

            </div>

        </div>


        <div class="footer">

            LearnAI • PBL Project
            <br>
            Learning Path Recommendation using Reinforcement Learning

        </div>

    </div>

    """


# =========================================================
# API — Q TABLE
# =========================================================

@app.route("/api/qtable")
def qtable():

    return jsonify({

        "states": STATES,

        "topics": TOPICS,

        "learning_rate": LEARNING_RATE,

        "discount_factor": DISCOUNT_FACTOR,

        "epsilon": EPSILON,

        "episodes": EPISODES,

        "q_table": Q.tolist()

    })


# =========================================================
# API — PROJECT INFO
# =========================================================

@app.route("/api/project")
def project():

    return jsonify({

        "project":
            "Learning Path Recommendation using Reinforcement Learning",

        "algorithm":
            "Q-Learning",

        "topics":
            TOPICS,

        "states":
            STATES,

        "episodes":
            EPISODES,

        "status":
            "TRAINED"

    })


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("          LEARNAI - RL LEARNING PATH")
    print("=" * 60)
    print(" Model            : Q-Learning")
    print(" States           : Low / Medium / High")
    print(" Actions          : 6 Learning Topics")
    print(" Learning Rate    :", LEARNING_RATE)
    print(" Discount Factor  :", DISCOUNT_FACTOR)
    print(" Exploration       :", EPSILON)
    print(" Training Episodes :", EPISODES)
    print(" Status            : TRAINED")
    print("=" * 60)
    print(" Open: http://127.0.0.1:5000")
    print("=" * 60)
    print("\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )