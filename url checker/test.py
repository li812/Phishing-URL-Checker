# Load Libraries
import pandas as pd
import numpy as np
from string import printable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf  
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence

df = pd.read_csv('dataset/dataset.csv')

url_int_tokens = [[printable.index(x) + 1 for x in url if x in printable] for url in df.url]
max_len = 75
X = sequence.pad_sequences(url_int_tokens, maxlen=max_len)
target = np.array(df.isMalicious)

X_train, X_test, target_train, target_test = train_test_split(X, target, test_size=0.2, random_state=42)

def lstm_conv(max_len=75, emb_dim=32, max_vocab_len=100, lstm_output_size=32, W_reg=tf.keras.regularizers.l2(1e-4)):
    main_input = Input(shape=(max_len,), dtype=tf.int32, name='main_input')
    emb = Embedding(input_dim=max_vocab_len, output_dim=emb_dim, embeddings_regularizer=W_reg)(main_input)
    emb = Dropout(0.25)(emb)
    conv = Conv1D(filters=256, kernel_size=5, padding='same')(emb)
    conv = tf.keras.layers.ELU()(conv)
    conv = MaxPooling1D(pool_size=4)(conv)
    conv = Dropout(0.5)(conv)
    conv = Conv1D(filters=256, kernel_size=6, padding='same')(emb)
    conv = tf.keras.layers.ELU()(conv)
    conv = MaxPooling1D(pool_size=4)(conv)
    conv = Dropout(0.5)(conv)

    conv = Conv1D(filters=256, kernel_size=7, padding='same')(emb)
    conv = tf.keras.layers.ELU()(conv)
    conv = MaxPooling1D(pool_size=4)(conv)
    conv = Dropout(0.5)(conv)

    lstm = LSTM(lstm_output_size)(conv)
    lstm = Dropout(0.5)(lstm)

    output = Dense(1, activation='sigmoid', name='output')(lstm)

    model = Model(inputs=[main_input], outputs=[output])
    adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Fit the model
epochs = 20
batch_size = 32
model = lstm_conv()
model.fit(X_train, target_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, target_test, verbose=1)
print('\nFinal Cross-Validation Accuracy:', accuracy, '\n')
