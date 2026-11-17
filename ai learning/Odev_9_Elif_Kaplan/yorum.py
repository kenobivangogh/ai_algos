import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout

comments_to_predict = [
    "Buna bayıldım, harika",  
    "Beklediğimden daha iyi",
    "Fena değil iş görüyor",  
    "Pek kaliteli değil tavsiye etmem", 
    "Rezalet berbat kesinlike almayın", 
]

# Veri setinin konumu
data_dir = "yorumlar" 
train_dir = os.path.join(data_dir, "train")
validation_dir = os.path.join(data_dir, "validation")
test_dir = os.path.join(data_dir, "test")

def load_data(directory):
    texts, labels = [], []
    for label_type in ["pozitif", "negatif"]:
        dir_name = os.path.join(directory, label_type)
        label = 1 if label_type == "pozitif" else 0
        for fname in os.listdir(dir_name):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_name, fname), encoding="utf-8") as f:
                    texts.append(f.read())
                labels.append(label)
    return texts, labels

train_texts, train_labels = load_data(train_dir)
validation_texts, validation_labels = load_data(validation_dir)
test_texts, test_labels = load_data(test_dir)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

max_length = 100  # Maksimum cümle uzunluğu
padding_type = "post"
trunc_type = "post"

train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
validation_sequences = tokenizer.texts_to_sequences(validation_texts)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train_labels = np.array(train_labels)
validation_labels = np.array(validation_labels)
test_labels = np.array(test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    train_padded,
    train_labels,
    epochs=10,
    validation_data=(validation_padded, validation_labels),
    verbose=2
)

test_loss, test_accuracy = model.evaluate(test_padded, test_labels, verbose=2)
print(f"Test Doğruluğu: {test_accuracy:.4f}")

for comment in comments_to_predict:
    sequence = tokenizer.texts_to_sequences([comment])
    padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded)
    sentiment = "Pozitif" if prediction[0] > 0.5 else "Negatif"
    print(f"Cümle: \"{comment}\" - Tahmin: {sentiment} ({prediction[0][0]:.4f})")