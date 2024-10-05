import optuna

def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-20, 0, 20])
    return x**2 + y

if __name__ == "__main__":
    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="quadratic-simple"
    )
    study.optimize(objective, n_trials=1000)
    print(f"Best value: {study.best_value} (params: {study.best_params})")