TOTAL_TIMESTEPS = 10
MAX_STEPS_PER_EPISODE = 100

MIN_LR = 0.001
MAX_LR = 0.010

UNITS = 100

MIN_DROPOUT = 0.1
MAX_DROPOUT = 0.5

# Batch size: discrete choices the agent steps through
BATCH_SIZES = [64, 128, 256, 512]
MIN_BATCH_SIZE = BATCH_SIZES[0]
MAX_BATCH_SIZE = BATCH_SIZES[-1]

TRAIN_SUBSET = None   # None = use full 80% training split
TEST_SUBSET  = None   # None = use full 20% test split

# --- LLM Advisor ---
LLM_MODEL_PATH = "llm/Phi-3-mini-4k-instruct-q4.gguf"
LLM_GUIDANCE_PROB = 0.3   # probability that LLM suggestion overrides DQN action
LLM_N_GPU_LAYERS = 0      # set to -1 to fully offload Phi-3 to GPU

# --- Final Training ---
FINAL_EPOCHS = 50         # epochs for the dedicated post-RL final training run
