import gym
import optuna
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Define a global variable to keep track of the best model performance
best_mean_reward = -float('inf')

# Define the Optuna objective function for tuning
def objective(trial):
    global best_mean_reward

    # Suggest hyperparameters
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)

    # Create the environment
    env = make_vec_env("CartPole-v1", n_envs=1)

    # Set up the A2C model
    model = A2C('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma, verbose=0)

    # Train the model
    model.learn(total_timesteps=10000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    # If the current model performs better, save it
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        model.save("best_a2c_cartpole_model")
        print(f"New best model saved with mean reward: {mean_reward}")

    return mean_reward

# Running the Optuna optimization
if __name__ == "__main__":
    study = optuna.create_study(
        direction='maximize',
        storage="sqlite:///db4.sqlite3",  # Specify the storage URL here.
        study_name="cartpole A2C 03"
        )
    study.optimize(objective, n_trials=10)

    # Get the best hyperparameters from the study
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    def visualize_best_model():
        # Create the environment with the correct render mode
        env = gym.make("CartPole-v1", render_mode="human")

        # Load the best model saved during the trials
        best_model = A2C.load("best_a2c_cartpole_model")

        # Reset the environment and extract the observation from the tuple
        obs, _ = env.reset()

        # Run the environment using the best model for 1000 timesteps
        for _ in range(1000):
            env.render()  # Render the environment

            # Predict the next action using the model
            action, _states = best_model.predict(obs)

            # Take a step in the environment
            obs, reward, done, _, _ = env.step(action)  # Handle the tuple returned from step()

            # If the episode is done, reset the environment
            if done:
                obs, _ = env.reset()  # Extract the observation again after reset

        env.close()  # Close the rendering window

    

    # Visualize the best model
    visualize_best_model()
