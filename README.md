# Sentiment-Analysis-on-Social-Media-Posts
#### Step 1: Data Collection and Preprocessing
You can use the Sentiment140 dataset for this project, which includes tweets labeled with sentiment.

python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset_path = 'path_to_your_dataset/sentiment140.csv'
data = pd.read_csv(dataset_path, encoding='ISO-8859-1', header=None)
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Preprocess the text data
data['text'] = data['text'].str.lower().str.replace('[^a-zA-Z0-9\s]', '', regex=True)
data['target'] = data['target'].replace(4, 1)  # Replace 4 (positive) with 1

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(data['target'])
X = data['text']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=20000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Build the LSTM model
model = Sequential()
model.add(Embedding(20000, 128, input_length=max_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_pad, y_test)
print('Test accuracy:', test_acc)


#### Step 2: Model Evaluation and Visualization
python
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
