# RL-LLM Hyperparameter Optimization for Text Classification (BiLSTM)

This project introduces a novel approach to hyperparameter optimization by combining **Reinforcement Learning (Deep Q-Network)** with a **Large Language Model (Phi-3 Advisor)** to automatically tune a PyTorch-based **Bidirectional LSTM (BiLSTM)** for sentiment analysis on the IMDb dataset.

## Features

- **RL-Based Tuning**: A custom Gym environment (`HyperparamEnv`) where a DQN agent interacts to find optimal hyperparameters (learning rate, dropout rate, batch size) dynamically.
- **LLM Guidance**: Integrates a localized Phi-3 LLM (`llama.cpp` compatible) to probabilistically guide the DQN agent, providing heuristic hints for parameter adjustments based on current states.
- **BiLSTM Architecture**: Robust PyTorch implementation of a Bidirectional LSTM for text classification tasks.
- **Automated Final Training**: Once the RL search completes, the system automatically trains final models utilizing the best-found hyperparameters and outputs comprehensive evaluation metrics.
- **Performance Evaluation**: Generates accuracy, precision, recall, F1-scores, training/validation loss curves, and confusion matrices.

## Project Structure

- `data/imdb_loader.py`: Handles IMDb dataset downloading, preprocessing, sequence padding, and vocabulary indexing.
- `models/bilstm.py`: Contains the PyTorch `nn.Module` definition for the Bidirectional LSTM model.
- `env/hyperparam_env.py`: Custom OpenAI Gym environment designed for the reinforcement learning agent to interact and evaluate hyperparameter configurations.
- `llm/phi3_advisor.py`: The LLM advisor module that loads a local Phi-3 GGUF model and queries it for heuristic tuning hints.
- `config.py`: Global configuration file managing RL timesteps, RL search spaces, LLM paths, and final training parameters.
- `main.py`: The main entry point script that orchestrates the entire pipeline: loading data, setting up the advisor and environment, training the DQN agent, and executing the final text classification models.
- `train_final.py`: Triggers the final full-scale PyTorch training runs with early stopping enabled.
- `train_final_no_early_stop.py`: Triggers the final full-scale PyTorch training runs for a fixed mathematical duration (without early stopping).
- `IMPLEMENTATION_AND_RESULTS.md`: Detailed documentation explaining the project's methodologies, pseudo-code algorithms, architectural integration steps, and empirical comparative results against standard baselines.

## Prerequisites

- Python 3.11+
- PyTorch (with CUDA support strongly recommended)
- `stable-baselines3`
- `gymnasium`
- `llama-cpp-python` (for the LLM advisor)
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`

*Note: Ensure you have obtained and downloaded standard Phi-3 LLM weights (e.g., `Phi-3-mini-4k-instruct-q4.gguf`) and placed them within the `llm/` directory as specified in `config.py`, or alternatively update the `LLM_MODEL_PATH` variable to reflect your custom file.*

## Usage

To start the hyperparameter optimization loop and subsequent final model training runs, simply execute:

```bash
python main.py
```

### Pipeline Overview
1. **Data Loading**: Prepares the IMDb training and test sequence splits.
2. **LLM Initialization**: Loads the Phi-3 advisor weight structures directly into memory/GPU.
3. **RL Optimization**: The Deep Q-Network rapidly steps sequentially across the hyperparameter boundary configurations (`TOTAL_TIMESTEPS` defined in `config.py`), seamlessly utilizing heuristic LLM guidance. It simultaneously saves the resulting agent locally as `dqn_hyperparam_agent.zip`.
4. **Final Model Run (Early Stopping)**: Automatically utilizes the absolute best parameters resolved from the Gym environment to fully train the BiLSTM from scratch until validation accuracy improvements plateau, subsequently saving metric analytics and graphical plots into the `outputs/` folder.
5. **Final Model Run (Fixed Epochs)**: Instantiates a secondary complete training phase bypassing early-stopping mechanisms, designed to directly measure maximum epoch boundaries, appending output statistics into the `output_without_early/` framework folder.
##  PSEUDO CODE

## Phase 1: Environment Definition and Initialization
**Input**: Preprocessed IMDb training/validation splits, token dimensions.
**Output**: Initialized `HyperparamEnv` with LLM integration.
1. Load dataset partitions (`x_train`, `y_train`, `x_test`, `y_test`) and `vocab_size`.
2. Initialize `PhiAdvisor` using a localized Phi-3 LLM weight via LlamaCpp.
3. Define the RL observation space: `[prev_acc, lr, dropout, batch_norm, llm_hint_norm]`.
4. Setup valid discrete action bounds:
   - Expand or contract base Learning Rate.
   - Adjust Dropout probability.
   - Increment or Decrement Batch Size.

## Phase 2: RL-LLM Based Hyperparameter Optimization
**Input**: Instantiated `HyperparamEnv` and standard DQN.
**Output**: Optimized hyperparameters.
1. Initialize DQN Agent (Neural network initialized via MlpPolicy).
2. For each episode step:
   a. Query Phi-3 LLM based on current state parameters to yield `llm_action` hint.
   b. Select environment `action` probabilistically—epsilon-greedy from DQN or overriding with LLM guidance if active.
   c. Instanciate PyTorch BiLSTM and train using small mini-epochs evaluating parameters.
   d. Evaluate small-scale validation accuracy.
   e. Compute `reward` as the positive differential improvement in validation accuracy.
   f. Store transitions in replay memory and update Q-network.
   g. Track running highest performing parameters.
3. Conclude search and extract the absolute max parameter set configurations on convergence.

## Phase 3: Final Model Training
**Input**: Optimized hyperparameters (`lr`, `dropout`, `batch_size`).
**Output**: Fine-tuned PyTorch Model providing heightened inference metrics.
1. Compile mapping using identical BiLSTM topology loaded heavily with optimized values.
2. Formally execute epochs scaling fully through standard training boundaries (optionally bypassing or using early-stopping validation).
3. Evaluate metrics explicitly mapping Test subsets.
4. Extricate evaluation criteria saving confusion matrices, loss profiles, and exact F1 scores directly to `outputs` logs.

##  SYSTEM INTEGRATION
The overall system merges classic textual manipulation, heuristic agent structures, and robust PyTorch compilation iteratively:
1. **Data Preprocessing Module**: Implements core logic wrapping text padding algorithms and index tokenization static mapping native within `imdb_loader.py`.
2. **LLM Heuristic Module**: Intervenes dynamically processing abstract hyperparameter adjustments through local `phi3_advisor.py` to prompt discrete guidance paths.
3. **RL Optimization Environment**: Defines dynamic Gym frameworks structuring rewards connecting PyTorch mini-trainers sequentially inside `hyperparam_env.py`.
4. **Training Integration**: Fetches highest observed variables triggering conclusive full-scale PyTorch training natively handled inside `train_final.py`.
5. **Evaluation Module**: Constructs the final mathematical pipeline parsing precision statistics sequentially saving comprehensive Matplotlib diagnostic visualizations against the testing domain.

##  CODING IMPLEMENTATION
The project logic relies heavily on clean Python PyTorch bindings organized via modular design. Critical scripts included:
- `data/imdb_loader.py` - Performs dataset parsing, sequence padding, and vocabulary indexing vector conversions.
- `models/bilstm.py` - Establishes PyTorch `nn.Module` classes strictly defining the mathematical Bidirectional LSTM structure and device routing.
- `env/hyperparam_env.py` - Defines custom OpenAI Gym instance where discrete RL optimization actions interface natively running PyTorch training instances to allocate rewards.
- `llm/phi3_advisor.py` - Interfaces explicit `llama.cpp` local parameters constructing prompts resolving directly towards valid agent hints.
- `main.py` - Houses orchestration loops triggering environment assembly, deploying the DQN instances, retrieving ultimate variables, and initiating concluding outputs.
- `train_final.py` / `train_final_no_early_stop.py` - Manages concluding large epoch structures utilizing resulting optimal parameters generating absolute metric graphics.

## PROPOSED RESULTS
## Configuration Options

You can iteratively tweak key variables universally in `config.py`:
- `TOTAL_TIMESTEPS`: Total number of interaction sequence steps the agent negotiates with the optimization environment.
- `LLM_GUIDANCE_PROB`: Abstract decimal Probability (e.g., `0.3`) dictating whether the DQN directly enforces an LLM override sequence.
- `BATCH_SIZES`, `MIN_LR`, `MAX_LR`, `MIN_DROPOUT`, `MAX_DROPOUT`: Bounds strictly defining the environment state spaces.
- `FINAL_EPOCHS`: Default maximal numerical threshold limit constraining terminal training.

## Output
**Table : Final Results Comparison**

| Model | Accuracy | Precision | Recall | F-1 Score |
| :--- | :---: | :---: | :---: | :---: |
| Baseline BiLSTM | ~82.0% | ~79.0% | ~87.0% | ~82.0% |
| RL-LLM Optimized BiLSTM | ~87.3% | ~85.6% | ~88.4% | ~87.0% |


Following execution completion, comprehensive reporting visuals can be located natively:
- `outputs/` mapped with resultant `loss_curve.png` and `confusion_matrix.png`.
- Formal `metrics.txt` evaluations detailing explicit validation scores directly comparing Baseline parameters.
