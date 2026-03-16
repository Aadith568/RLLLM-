import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from models.bilstm import BiLSTM, device
from config import *


class HyperparamEnv(gym.Env):
    def __init__(self, x_train, y_train, x_test, y_test, vocab_size, advisor=None):
        super().__init__()

        # Optional Phi-3 LLM advisor for action guidance
        self.advisor = advisor
        self.last_llm_hint = 0  # last action suggested by LLM (0-7, or -1=no hint)

        self.x_train = torch.tensor(x_train[:TRAIN_SUBSET], dtype=torch.long)
        self.y_train = torch.tensor(y_train[:TRAIN_SUBSET], dtype=torch.float32)

        self.x_test = torch.tensor(x_test[:TEST_SUBSET], dtype=torch.long)
        self.y_test = torch.tensor(y_test[:TEST_SUBSET], dtype=torch.float32)

        self.vocab_size = vocab_size

        self.action_space = gym.spaces.Discrete(6)  # 4 LR/Dropout + 2 batch size

        # obs = [prev_acc, lr, dropout, batch/MAX_BATCH, llm_hint/5]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(5,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.lr = 0.001
        # units is removed from state, using fixed config value
        self.dropout = 0.3
        self.batch_size = 128       # default batch size (index 2 in BATCH_SIZES)
        self.prev_acc = 0
        self.steps = 0
        self.last_llm_hint = 0

        # best seen so far (persists across episode resets)
        if not hasattr(self, 'best_acc'):
            self.best_acc = 0.0
            self.best_params = {
                'lr': self.lr,
                'dropout': self.dropout, 'batch_size': self.batch_size
            }

        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize llm_hint: treat -1 (no suggestion) as 0
        hint_norm = max(self.last_llm_hint, 0) / 5.0
        return np.array([
            self.prev_acc,
            self.lr,
            self.dropout,
            self.batch_size / MAX_BATCH_SIZE,
            hint_norm
        ], dtype=np.float32)

    def step(self, action):

        # --- LLM soft guidance ---
        if self.advisor is not None:
            current_state = (self.prev_acc, self.lr, self.dropout, self.batch_size)
            llm_action = self.advisor.suggest_action(current_state)
            self.last_llm_hint = llm_action

            if llm_action != -1 and random.random() < LLM_GUIDANCE_PROB:
                print(f"[LLM] Guidance applied (overriding DQN action {action} -> {llm_action})")
                action = llm_action
            else:
                print(f"[LLM] DQN action retained: {action}")

        # --- Apply chosen action ---
        if action == 0:
            self.lr = round(min(self.lr + 0.001, MAX_LR), 3)
        elif action == 1:
            self.lr = round(max(self.lr - 0.001, MIN_LR), 3)
        elif action == 2:
            self.dropout = min(self.dropout + 0.05, MAX_DROPOUT)
        elif action == 3:
            self.dropout = max(self.dropout - 0.05, MIN_DROPOUT)
        elif action == 4:
            # Move to the next larger batch size in the discrete list
            idx = BATCH_SIZES.index(self.batch_size) if self.batch_size in BATCH_SIZES else 2
            self.batch_size = BATCH_SIZES[min(idx + 1, len(BATCH_SIZES) - 1)]
        elif action == 5:
            # Move to the next smaller batch size in the discrete list
            idx = BATCH_SIZES.index(self.batch_size) if self.batch_size in BATCH_SIZES else 2
            self.batch_size = BATCH_SIZES[max(idx - 1, 0)]

        acc = self.train_model()

        # Track the best hyperparameters seen across all steps
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_params = {
                'lr': self.lr,
                'dropout': self.dropout, 'batch_size': self.batch_size
            }
            print(f"[BEST] New best acc={acc:.4f}  lr={self.lr}  "
                  f"dropout={self.dropout:.2f}  batch={self.batch_size}")

        reward = (acc - self.prev_acc) * 100
        self.prev_acc = acc

        self.steps += 1
        done = self.steps >= MAX_STEPS_PER_EPISODE

        return self._get_obs(), reward, done, False, {}

    def train_model(self):

        model = BiLSTM(self.vocab_size, 128, UNITS, self.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        batch_size = self.batch_size  # controlled by RL agent (actions 6 & 7)

        dataset_size = self.x_train.size(0)

        model.train()

        for _ in range(2):  # small epochs for RL

            for i in range(0, dataset_size, batch_size):

                batch_x = self.x_train[i:i+batch_size].to(device)
                batch_y = self.y_train[i:i+batch_size].to(device)

                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # ---------- Evaluation ----------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, self.x_test.size(0), batch_size):

                batch_x = self.x_test[i:i+batch_size].to(device)
                batch_y = self.y_test[i:i+batch_size].to(device)

                outputs = model(batch_x).squeeze()
                preds = (outputs > 0.5).float()

                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        acc = correct / total

        return acc

    def get_best_params(self):
        """Return the hyperparameters that achieved the highest validation accuracy."""
        return {
            'lr':         self.best_params['lr'],
            'dropout':    self.best_params['dropout'],
            'batch_size': self.best_params['batch_size'],
            'best_acc':   self.best_acc
        }

