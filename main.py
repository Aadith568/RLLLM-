from stable_baselines3 import DQN
from env.hyperparam_env import HyperparamEnv
from data.imdb_loader import load_imdb_csv
from llm.phi3_advisor import PhiAdvisor
from config import TOTAL_TIMESTEPS, LLM_MODEL_PATH, LLM_N_GPU_LAYERS
import train_final


def main():

    x_train, y_train, x_test, y_test, vocab_size = load_imdb_csv()

    # Load the Phi-3 LLM advisor once (heavy: ~2.4 GB model)
    print("[LLM] Loading Phi-3 advisor...")
    advisor = PhiAdvisor(model_path=LLM_MODEL_PATH, n_gpu_layers=LLM_N_GPU_LAYERS)
    print("[LLM] Phi-3 advisor ready.")

    env = HyperparamEnv(x_train, y_train, x_test, y_test, vocab_size, advisor=advisor)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        device="cuda"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save("dqn_hyperparam_agent")

    # ── Report best hyperparameters found during RL search ─────────────────
    best = env.get_best_params()
    print("\n" + "="*50)
    print("  BEST HYPERPARAMETERS FOUND")
    print("="*50)
    print(f"  Validation Accuracy : {best['best_acc']:.4f}")
    print(f"  Learning Rate       : {best['lr']}")
    print(f"  Dropout             : {best['dropout']:.2f}")
    print(f"  Batch Size          : {best['batch_size']}")
    print("="*50)

    print("\nTraining completed. Agent saved to dqn_hyperparam_agent.zip")

    # ── Final 50-epoch training with best hyperparameters ─────────────────
    # Trains the improved BiLSTM properly and generates all metric plots.
    final_acc = train_final.run(
        best_params=best,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        vocab_size=vocab_size,
    )

    print(f"\n[FINAL] Best Validation Accuracy after 50 epochs: {final_acc:.4f}")

    import train_final_no_early_stop
    
    # ── Final 50-epoch training (WITHOUT early stopping) ──────────────────
    # print("\n[INFO] Starting full 50-epoch training without early stopping...")
    # acc_no_early = train_final_no_early_stop.run(
    #     best_params=best,
    #     x_train=x_train,
    #     y_train=y_train,
    #     x_test=x_test,
    #     y_test=y_test,
    #     vocab_size=vocab_size,
    # )
    # print(f"\n[FINAL NO-EARLY] Best Validation Accuracy: {acc_no_early:.4f}")

    return best, final_acc
if __name__ == "__main__":
    main()
