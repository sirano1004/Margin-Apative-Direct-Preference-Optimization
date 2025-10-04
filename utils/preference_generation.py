import numpy as np

def determine_preference(row, scale = 1.0, rng=np.random.default_rng(42)):
    score1 = row['first_response_score'] + rng.gumbel(0,1) * scale
    score2 = row['second_response_score'] + rng.gumbel(0,1) * scale

    return '1 > 2' if score1 >= score2 else '2 > 1'
