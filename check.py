import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==================================
# GPU Configuration
# ==================================
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU is enabled:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("❌ No GPU detected. Running on CPU.")

# ==================================
# 1. Hyperparameters
# ==================================
LEARNING_RATE = 0.0012
HIDDEN_UNITS = 128
DROPOUT_RATE = 0.30
BATCH_SIZE = 64
EPOCHS = 50
MAX_VOCAB = 20000
MAX_LENGTH = 200

# ==================================
# 2. Load Dataset
# ==================================
print("Loading dataset...")
data = pd.read_csv("data/cleaned_dataset.csv")

print("Unique sentiment values BEFORE conversion:")
print(data["sentiment"].unique())

texts = data["review"].astype(str)

labels = data["sentiment"].astype(str).str.strip().str.lower()

labels = labels.replace({
    "positive": 1,
    "negative": 0,
    "pos": 1,
    "neg": 0
})

labels = pd.to_numeric(labels, errors="coerce")

valid_rows = labels.notna()
texts = texts[valid_rows]
labels = labels[valid_rows]

labels = labels.astype(int)

print("Unique sentiment values AFTER conversion:")
print(labels.unique())
print("Dataset cleaned successfully!")

# ==================================
# 3. Train-Test Split (80/20)
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# ==================================
# 4. Tokenization & Padding
# ==================================
print("Tokenizing text...")

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')

# ==================================
# 5. Build BiLSTM Model
# ==================================
print("Building model...")

model = Sequential([
    Embedding(input_dim=MAX_VOCAB, output_dim=128),
    Bidirectional(LSTM(HIDDEN_UNITS)),
    Dropout(DROPOUT_RATE),
    Dense(64, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])

model.build(input_shape=(None, MAX_LENGTH))

# ==================================
# 6. Compile Model
# ==================================
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

model.summary()

# ==================================
# 7. Train Model
# ==================================
print("Starting training...")

history = model.fit(
    X_train_pad,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# ==================================
# 8. Evaluate Model
# ==================================
print("Evaluating model...")

y_pred = model.predict(X_test_pad)
y_pred = (y_pred > 0.5).astype(int)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==================================
# 9. Save Model
# ==================================
model.save("bilstm_imdb_model.h5")
print("Model saved successfully!")