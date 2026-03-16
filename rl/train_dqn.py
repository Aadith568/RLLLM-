from stable_baselines3 import DQN
from config import TOTAL_TIMESTEPS


def train_agent(env):

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=5000,
        learning_starts=10,
        batch_size=32,
        gamma=0.95,
        device="cuda"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    return model
