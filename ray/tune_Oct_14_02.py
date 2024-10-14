from ray import train, tune

import os

# Set TMPDIR to a shorter path
os.environ["TMPDIR"] = "C:/temp"

# Verify that the directory exists or create it
os.makedirs("C:/temp", exist_ok=True)

# def objective(config):  # ①
#     score = config["a"] ** 2 + config["b"]
#     return {"score": score}


# search_space = {  # ②
#     "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
#     "b": tune.choice([1, 2, 3]),
# }

# tuner = tune.Tuner(objective, param_space=search_space)  # ③

# results = tuner.fit()
# print(results.get_best_result(metric="score", mode="min").config)

