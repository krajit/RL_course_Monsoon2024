from ray import tune
from ray.air import RunConfig
import os

# Set a valid directory path for Ray logs (replace with your desired path)
os.environ["TUNE_RESULT_DIR"] = "C:/temp/ray_results"

tuner = tune.Tuner(
    objective,
    param_space={
        "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
        "b": tune.choice([1, 2, 3]),
    },
    run_config=RunConfig(
        local_dir="C:/temp/ray_results",  # Shorter, custom directory
        verbose=1,
        log_to_file=False,  # Disable TensorBoard logging if not needed
    )
)

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)
