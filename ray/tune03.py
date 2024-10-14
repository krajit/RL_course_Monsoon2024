from ray import train, tune


def objective(config):  # 
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


search_space = {  # 
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

import os
# Set a valid directory path for Ray logs (replace with your desired path)
#os.environ["TUNE_RESULT_DIR"] = "C:/temp/ray_results"


storage_dir = os.path.abspath("./run5")
os.environ["TMPDIR"] = storage_dir

from ray.air import RunConfig
tuner = tune.Tuner(
    objective, 
    param_space=search_space,
    #   run_config=tune.RunConfig(
    #     local_dir=storage_dir,  # Set a local directory for your results
    #     verbose=1,  # Adjust verbosity as needed
    #     # disable tensorboard logging
    #     log_to_file=False,
    # )
        run_config=RunConfig(
        storage_path="C:/temp/ray_results",  # Shorter, custom directory
        verbose=1,
        log_to_file=False,  # Disable TensorBoard logging if not needed
    )
    )  # ③

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)