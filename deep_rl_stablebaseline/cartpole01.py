import gym
from stable_baselines3 import A2C
import optuna
import numpy as np

# Initialize the environment with render mode
def create_env():
    return gym.make("CartPole-v1", render_mode="human")

# Objective function for Optuna
def optimize_a2c(trial):
    env = create_env()
    gamma = trial.suggest_float("gamma", 0.85, 0.99)
    learning_rate = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    model = A2C(
        "MlpPolicy", 
        env, 
        gamma=gamma, 
        learning_rate=learning_rate, 
        verbose=0
    )
    model.learn(total_timesteps=10000)
    mean_reward = evaluate_model(model, env)
    env.close()
    return mean_reward

# Evaluate the model
def evaluate_model(model, env, n_episodes=5):
    all_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        all_rewards.append(total_reward)
    return np.mean(all_rewards)

# Visualize the best model
def visualize_best_model(model, env):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
    env.close()

# Optuna study setup
if __name__ == "__main__":
    study_name = "cartpole_a2c_study"  # Consistent study name
    storage = "sqlite:///optuna_study.db"

    # Create or load the study to avoid duplication issues
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )

    # Optimize using Optuna
    study.optimize(optimize_a2c, n_trials=10)

    # Log the best hyperparameters
    print("Best hyperparameters:", study.best_params)

    # Load the best model and visualize it
    env = create_env()
    best_params = study.best_params
    best_model = A2C(
        "MlpPolicy", 
        env, 
        gamma=best_params["gamma"], 
        learning_rate=best_params["lr"], 
        verbose=1
    )
    best_model.learn(total_timesteps=10000)
    visualize_best_model(best_model, env)
