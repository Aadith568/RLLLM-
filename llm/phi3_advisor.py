from llama_cpp import Llama
import random

_KEYWORD_MAP = [
    (["increase lr", "higher learning rate", "larger learning rate", "raise lr"], 0),
    (["decrease lr", "lower learning rate", "smaller learning rate", "reduce lr"], 1),
    (["more dropout", "increase dropout", "higher dropout", "add dropout"], 2),
    (["less dropout", "decrease dropout", "lower dropout", "reduce dropout"], 3),
    (["larger batch", "increase batch", "bigger batch", "larger batch size", "more samples per"], 4),
    (["smaller batch", "decrease batch", "reduce batch", "smaller batch size", "fewer samples per"], 5),
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
        val_acc, lr, dropout, batch_size = state  # 4-element state + 1 (val_acc)
        sys_tag  = chr(60) + chr(124) + "system"    + chr(124) + chr(62)
        end_tag  = chr(60) + chr(124) + "end"       + chr(124) + chr(62)
        user_tag = chr(60) + chr(124) + "user"      + chr(124) + chr(62)
        asst_tag = chr(60) + chr(124) + "assistant" + chr(124) + chr(62)
        prompt = (
            sys_tag + "\n"
            "You are an expert ML hyperparameter tuning advisor. "
            "Give ONE short, direct improvement suggestion.\n"
            + end_tag + "\n"
            + user_tag + "\n"
            + f"Validation Accuracy: {val_acc:.4f}\n"
            + f"Learning Rate: {lr}\n"
            + f"Dropout: {dropout}\n"
            + f"Batch Size: {batch_size}\n"
            + end_tag + "\n"
            + asst_tag + "\n"
        )
        output = self.model(prompt, max_tokens=60, temperature=0.7)
        return output["choices"][0]["text"].strip()

    def suggest_action(self, state):
        text = self.suggest(state).lower()
        print(f"[LLM] Suggestion: {text!r}")
        for keywords, action_id in _KEYWORD_MAP:
            if any(kw in text for kw in keywords):
                print(f"[LLM] Mapped to action: {action_id}")
                return action_id
        print("[LLM] No keyword match, returning -1 (DQN fallback)")
        return -1
