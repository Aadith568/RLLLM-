from llama_cpp import Llama
import random


# Keyword -> action index mapping (mirrors HyperparamEnv action space)
# 0: lr up  1: lr down  2: units up  3: units down  4: dropout up  5: dropout down
_KEYWORD_MAP = [
    (["increase lr", "higher learning rate", "larger learning rate", "raise lr"], 0),
    (["decrease lr", "lower learning rate", "smaller learning rate", "reduce lr"], 1),
    (["more units", "increase units", "larger model", "more hidden", "add units"], 2),
    (["fewer units", "decrease units", "smaller model", "less hidden", "reduce units"], 3),
    (["more dropout", "increase dropout", "higher dropout", "add dropout"], 4),
    (["less dropout", "decrease dropout", "lower dropout", "reduce dropout"], 5),
]


class PhiAdvisor:
    def __init__(self, model_path, n_gpu_layers=0):
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=n_gpu_layers
        )

    def suggest(self, state):
        """Return a natural-language hyperparameter suggestion for the given state."""
        val_acc, lr, units, dropout = state

        prompt = (
            "<|system|>\n"
            "You are an expert ML hyperparameter tuning advisor. "
            "Give ONE short, direct improvement suggestion in one sentence.\n"
            "<|end|>\n"
            "
        )