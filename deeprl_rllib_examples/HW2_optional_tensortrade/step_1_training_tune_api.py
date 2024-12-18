import yfinance as yf
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.env.default import create
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.env.default.rewards import SimpleProfit
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
import tensortrade.env.default as default
import pandas as pd
import numpy as np


# 1. Define Instruments
INR = Instrument("INR", 2, "Indian Rupees")         # Base currency with 2 decimal places
RELIANCE = Instrument("RELIANCE", 8, 'Reliance Ind Ltd')  # Stock with 2 decimal places

# 2. Fetch Historical Data for Reliance
data = yf.download('RELIANCE.NS', start='2022-01-01', end='2023-01-01').dropna()
data.to_csv('datafile.csv')



#data = pd.read_csv('datafile.csv')

# 3. Create Data Streams
price_stream = Stream.source(np.array(data['Close']).reshape(-1), dtype="float").rename("INR-RELIANCE")
#volume_stream = Stream.source(np.array(data['Volume']).reshape(-1), dtype="float").rename("RELIANCE_VOLUME")


# 4. Set up DataFeed
feed = DataFeed([price_stream,
                 # volume_stream 
                 ])
#feed.compile()

# 5. Define Wallets and Portfolio
exchange = Exchange("yfinance", service=execute_order)( price_stream )

cash = Wallet(exchange, 100000 * INR)  # 100,000 INR
asset = Wallet(exchange, 0 * RELIANCE)  # Initially no stock

portfolio = Portfolio(INR, [cash, asset])

reward_scheme = default.rewards.PBR(price=price_stream)
#reward_scheme = default.rewards.SimpleProfit()


action_scheme = default.actions.BSH(
    cash=cash,
    asset=asset
).attach(reward_scheme)
#action_scheme = default.actions.SimpleOrders()


# 6. Create the TensorTrade Environment
env = create(
    feed=feed,
    portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    window_size=20,
    max_allowed_loss=0.6
)

# 7. Test the Environment
obs = env.reset()
done = False
truncated = False

# while not done:
#     action = env.action_space.sample()  # Take a random action
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"Action: {action}, Reward: {reward}")



# portfolio.ledger.as_frame().to_csv("ledger.csv")
# print("done")

import ray
import numpy as np
import pandas as pd

from ray import tune
from ray.tune.registry import register_env

def create_env(config):
    return env 

register_env("TradingEnv", create_env)


from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

def create_env(config):
    return env

register_env("TradingEnv", create_env)

config = PPOConfig().environment("TradingEnv").framework("torch").training(model={"fcnet_hiddens": [64, 64]})
algo = config.build()

for _ in range(50):
    print(algo.train())


algo.evaluate()

checkpoint_dir = "ppo_trading_model_reliance"
save_result = algo.save(checkpoint_dir)

# algo.evaluate()  # 4. and evaluate it.

# # Save the trained model to a directory
# checkpoint_dir = "ppo_trading_model_reliance"
# save_result = algo.save(checkpoint_dir)


# steps to load and run trained model
# import os
# checkpoint_dir = os.path.join("C:\\Users\\ajit.kumar\\Documents\\GitHub\\tensortrade_learning\\ppo_trading_model_reliance")

# algo.restore(checkpoint_dir)
# print("restoration done")


# # Test inference
# obs, _ = env.reset()
# done, truncated = False, False

# while not done and not truncated:
#     action = algo.compute_single_action(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"Action: {action}, Reward: {reward}")