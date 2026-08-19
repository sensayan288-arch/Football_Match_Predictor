from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("football_data.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# =========================
# LOAD MODEL
# =========================

with open("football_winner_model.pkl", "rb") as file:
    model = pickle.load(file)


# =========================
# RECENT MATCHES
# =========================

def get_recent_matches(team, current_date, n=10):

    past_matches = df[
        (
            (df["Home Teams"] == team) |
            (df["Away Teams"] == team)
        )
        &
        (df["Date"] < current_date)
    ].sort_values("Date").tail(n)

    return past_matches


# =========================
# RECENT FORM
# =========================

def get_recent_form(team, current_date, n=10):

    past_matches = get_recent_matches(
        team,
        current_date,
        n
    )

    if len(past_matches) == 0:
        return 0.5

    wins = (
        past_matches["Winner"] == team
    ).sum()

    return wins / len(past_matches)


# =========================
# GOAL DIFFERENCE
# =========================

def get_goal_difference(team, current_date, n=10):

    past_matches = get_recent_matches(
        team,
        current_date,
        n
    )

    if len(past_matches) == 0:
        return 0.0

    goal_diffs = []

    for _, match in past_matches.iterrows():

        if match["Home Teams"] == team:

            goal_diff = (
                match["Home Goals"] -
                match["Away Goals"]
            )

        else:

            goal_diff = (
                match["Away Goals"] -
                match["Home Goals"]
            )

        goal_diffs.append(goal_diff)

    return sum(goal_diffs) / len(goal_diffs)


# =========================
# HEAD-TO-HEAD
# =========================

def get_head_to_head(
    home_team,
    away_team,
    current_date
):

    h2h_matches = df[
        (
            (
                (df["Home Teams"] == home_team) &
                (df["Away Teams"] == away_team)
            )
            |
            (
                (df["Home Teams"] == away_team) &
                (df["Away Teams"] == home_team)
            )
        )
        &
        (df["Date"] < current_date)
    ].sort_values("Date")

    if len(h2h_matches) == 0:
        return 0.5

    home_team_wins = (
        h2h_matches["Winner"] == home_team
    ).sum()

    return home_team_wins / len(h2h_matches)




def calculate_features(
    home_team,
    away_team
):

    # Use the latest available date
    current_date = (
        df["Date"].max()
        + pd.Timedelta(days=1)
    )

    # Recent form
    home_form = get_recent_form(
        home_team,
        current_date
    )

    away_form = get_recent_form(
        away_team,
        current_date
    )

    # Head-to-head
    h2h_home_win_rate = get_head_to_head(
        home_team,
        away_team,
        current_date
    )

    # Form difference
    form_difference = (
        home_form - away_form
    )

    # Goal difference
    home_goal_diff = get_goal_difference(
        home_team,
        current_date
    )

    away_goal_diff = get_goal_difference(
        away_team,
        current_date
    )

    return [
        home_form,
        away_form,
        h2h_home_win_rate,
        form_difference,
        home_goal_diff,
        away_goal_diff
    ]


@app.route("/", methods=["GET", "POST"])
def home():

    # Get all teams
    teams = sorted(
        set(df["Home Teams"].dropna())
        |
        set(df["Away Teams"].dropna())
    )

    prediction = None
    probability = None

    home_team = None
    away_team = None

    winner = None
    loser = None

    

    if request.method == "POST":

        home_team = request.form["home_team"]

        away_team = request.form["away_team"]

        # Same team check
        if home_team == away_team:

            prediction = (
                "Please select two different teams."
            )

        else:

            # Calculate engineered features
            feature_values = calculate_features(
                home_team,
                away_team
            )

            # Create DataFrame
            input_data = pd.DataFrame(
                [feature_values],
                columns=[
                    "Home_Recent_Form",
                    "Away_Recent_Form",
                    "Head_to_Head_HomeWinRate",
                    "Form_Difference",
                    "Home_Goal_Diff",
                    "Away_Goal_Diff"
                ]
            )

            # Make prediction
            result = model.predict(
                input_data
            )[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(
                input_data
            )[0]

            probability = round(
                max(probabilities) * 100,
                2
            )

            # Determine winner and loser
            if result == 1:

                winner = home_team
                loser = away_team

            else:

                winner = away_team
                loser = home_team

            prediction = f"{winner} Wins"

    

    return render_template(
        "index.html",
        teams=teams,
        prediction=prediction,
        probability=probability,
        winner=winner,
        loser=loser,
        home_team=home_team,
        away_team=away_team
    )




if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
