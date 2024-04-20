import keras
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense

losses = []


def custom_scaler(y):
    max_val = y.max()
    min_val = y.min()
    return (y-min_val)/max_val


# Custom rmse
def rmse(y_pred, y_true):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


def custom_error(y_true, y_pred):
    true_min = y_true[:, 0]
    true_max = y_true[:, 1]

    # Calculate error based on closest target value
    error = tf.where(y_pred < true_min,
                     rmse(y_pred, true_min),  # RMSE between pred and true_min
                     tf.where(y_pred > true_max,
                              rmse(y_pred, true_max),  # RMSE between pred and true_max
                              tf.constant(0.0)))  # No error if pred is between true_min and true_max

    return error


def create_model(num_layers, nodes):

    model = keras.Sequential()
    model.add(keras.Input(shape=(6000,)))
    if num_layers == 1:
        model.add(Dense(nodes, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=custom_error)
    elif num_layers > 1:
        if type(nodes) is list:
            for i in range(num_layers):
                model.add(Dense(nodes[i], activation='relu'))
                i += 1
        elif type(nodes) is int:
            for i in range(num_layers):
                model.add(Dense(nodes, activation='relu'))
                i += 1
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=custom_error)
    return model


def preprocess(filename):
    # Import CSV
    df = pd.read_csv(filename, delimiter='\t')

    # Tockenization and Vectorization with tf-idf
    # max-df,min-df for words less import due to them being too common or rarely used words
    # max features= 1000 to keep from the words left only the 1000 words with the higher tf-idf values

    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=6000)
    X = vectorizer.fit_transform(df['text'].to_list())

    # Vocabulary Generated from tf-idf vectorizer
    dicts = vectorizer.vocabulary_

    # Min-Max Scaling
    # Transform x to array to use MinMax
    X_array = X.toarray()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_array)

    # Min-Max Scaling for output
    y = df[['date_min', 'date_max']].values
    y_scaled = custom_scaler(y)

    # Create training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # 5-fold cross-validation with training and validation sets
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test, dicts, kf


def train_model(X_train, y_train, kf, num_layers, nodes, epochs):
    model = create_model(num_layers, nodes)

    # Actions for each of the 5 folds
    for idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        # Set for each iteration Training and Validation Sets using cross-validation indexes
        X_it_train = X_train[train_idx]
        X_it_val = X_train[val_idx]
        y_it_train = y_train[train_idx]
        y_it_val = y_train[val_idx]

        reg = model.fit(x=X_it_train,  y=y_it_train, validation_data=(X_it_val, y_it_val), epochs=epochs, batch_size=32)
        losses.append(reg.history['loss'])

X_train, X_test, y_train, y_test, dicts, kf = preprocess('iphi2802.csv')

train_model(X_train, y_train, kf, 1, 125, 50)

