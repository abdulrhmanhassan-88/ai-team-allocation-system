data = [
    [316, 10, 9, 10, 9, 9, 8, 9, 6, 88],
    [320, 10, 9, 9, 9, 10, 8, 6, 7, 84],
    [317, 10, 9, 9, 9, 10, 7, 7, 5, 83],
    [376, 10, 10, 10, 10, 10, 7, 8, 6, 83],
    [354, 9, 9, 9, 9, 10, 7, 6, 5, 82],
    [310, 10, 10, 4, 10, 10, 8, 4, 6, 80],
    [341, 10, 9, 9, 9, 10, 7, 6, 4, 80],
    [375, 10, 9, 9, 9, 9, 8, 2, 6, 79],
    [356, 9, 10, 10, 10, 10, 8, 0, 9, 78],
    [389, 10, 10, 8, 10, 10, 8, 3, 5, 78],
    [304, 10, 9, 9, 9, 10, 4, 5, 5, 77],
    [344, 10, 9, 10, 9, 10, 9, 3, 6, 76],
    [359, 10, 10, 10, 9, 10, 9, 2, 2, 73],
    [390, 8, 8, 6, 7, 9, 6, 5, 7, 72],
    [334, 10, 9, 9, 9, 10, 7, 5, 0, 69],
    [337, 10, 6, 6, 7, 7, 6, 4, 6, 68],
    [322, 10, 8, 6, 8, 3, 3, 3, 9, 67],
    [325, 10, 10, 9, 8, 10, 2, 3, 5, 66],
    [364, 7, 6, 6, 8, 7, 7, 6, 7, 66],
    [395, 9, 9, 9, 8, 8, 5, 5, 5, 66],
    [384, 9, 8, 10, 8, 10, 3, 3, 5, 63],
    [311, 10, 10, 9, 9, 9, 7, 2, 0, 62],
    [318, 10, 9, 10, 9, 8, 2, 4, 4, 61],
    [319, 9, 9, 9, 8, 9, 7, 2, 2, 61],
    [398, 10, 9, 10, 8, 9, 6, 0, 0, 61],
    [340, 8, 2, 9, 10, 9, 5, 4, 5, 60],
    [383, 9, 10, 10, 9, 9, 3, 3, 1, 59],
    [365, 10, 9, 9, 9, 10, 6, 0, 1, 57],
    [400, 5, 8, 8, 8, 6, 2, 6, 4, 57],
    [300, 9, 7, 7, 5, 6, 3, 5, 7, 56]
]

import random
import math
import copy

# ========= Data is already in the variable data =========
NUM_TEAMS = 6
TEAM_SIZE = 5
INITIAL_TEMP = 1000
FINAL_TEMP = 0.1
COOLING_RATE = 0.995
MAX_ITER = 5000
NUM_FEATURES = 9# Number of variables to use after ID

def validate(data):
    if len(data) != NUM_TEAMS * TEAM_SIZE:
        raise ValueError(f"Expected {NUM_TEAMS*TEAM_SIZE} rows.")
    row_len = len(data[0])
    for row in data:
        if len(row) != row_len:
            raise ValueError("Rows not equal length.")

def row_features(row):
    # Take first 9 variables after ID (index 1 to *)
    return [float(x) for x in row[1:1+NUM_FEATURES]]

def compute_team_sums(teams, features):
    team_sums = []
    for team in teams:
        sums = [0] * len(features[0])
        for i in team:
            for j, val in enumerate(features[i]):
                sums[j] += val
        team_sums.append(sums)
    return team_sums

def objective(team_sums):
    cols = len(team_sums[0])
    teams_count = len(team_sums)
    total = 0
    for c in range(cols):
        col_vals = [team_sums[t][c] for t in range(teams_count)]
        avg = sum(col_vals) / teams_count
        total += sum((x - avg)**2 for x in col_vals)
    return total

def random_swap(teams):
    new_teams = copy.deepcopy(teams)
    t1, t2 = random.sample(range(len(teams)), 2)
    if not new_teams[t1] or not new_teams[t2]:
        return new_teams
    i1 = random.randrange(len(new_teams[t1]))
    i2 = random.randrange(len(new_teams[t2]))
    new_teams[t1][i1], new_teams[t2][i2] = new_teams[t2][i2], new_teams[t1][i1]
    return new_teams

def make_initial_teams(n, team_size):
    indices = list(range(n))
    random.shuffle(indices)
    return [indices[i*team_size:(i+1)*team_size] for i in range(NUM_TEAMS)]

def simulated_annealing(data):
    validate(data)
    features = [row_features(r) for r in data]
    current = make_initial_teams(len(data), TEAM_SIZE)
    best = copy.deepcopy(current)
    temp = INITIAL_TEMP

    best_score = objective(compute_team_sums(current, features))
    current_score = best_score

    while temp > FINAL_TEMP:
        for _ in range(MAX_ITER):
            new = random_swap(current)
            new_score = objective(compute_team_sums(new, features))
            delta = new_score - current_score

            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = new
                current_score = new_score
                if new_score < best_score:
                    best = copy.deepcopy(new)
                    best_score = new_score
        temp *= COOLING_RATE

    return best, compute_team_sums(best, features)

# ======= Run the code =======
if __name__ == "__main__":
    # Data is already in the variable data
    best_teams, team_sums = simulated_annealing(data)

    print(f"\n===== Best team distribution (using first {NUM_FEATURES} variables) =====")
    for t_idx, team in enumerate(best_teams, start=1):
        ids = [int(data[i][0]) for i in team]
        print(f"Team {t_idx}: {ids}")

    print(f"\n===== Sum of first {NUM_FEATURES} variables for each team =====")
    for t_idx, sums in enumerate(team_sums, start=1):
        print(f"Team {t_idx}: {[round(x, 3) for x in sums]}")

    print(f"\n===== Total sum of first {NUM_FEATURES} variables for each team =====")
    overall_totals = [sum(team_sums[t]) for t in range(NUM_TEAMS)]
    for t_idx, tot in enumerate(overall_totals, start=1):
        print(f"Team {t_idx}: {round(tot, 3)}")

    print(f"\nMax Total: {round(max(overall_totals), 3)}")
    print(f"Min Total: {round(min(overall_totals), 3)}")
    print(f"Difference: {round(max(overall_totals) - min(overall_totals), 3)}")
