import keras
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

total_losses = []
total_val_losses = []
mul2_losses = []
mul2_val_losses = []
mul3_losses = []
mul3_val_losses = []


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


def create_model(num_layers, nodes, lr, momentum):
    # Callback to use for early stopping
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model = keras.Sequential()
    model.add(keras.Input(shape=(8000,)))
    if num_layers == 1:
        model.add(Dense(nodes, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=momentum), loss=custom_error)
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
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=momentum), loss=custom_error)
    return model, callback


def preprocess(filename):
    # Import CSV
    df = pd.read_csv(filename, delimiter='\t')

    # Tockenization and Vectorization with tf-idf
    # max-df,min-df for words less import due to them being too common or rarely used words
    # max features= 8000 to keep the 8000 words with the higher tf-idf values

    vectorizer = TfidfVectorizer(max_features=8000)

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

    # 5-fold cross-validation with training and validation sets
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    return X_train, X_test, y_train, y_test, dicts, kf


def train_model(X_train, y_train, kf, num_layers, nodes, lr, momentum, epochs, early_s):
    train_loss = []
    val_loss = []

    model, callback = create_model(num_layers, nodes, lr, momentum)

    # Actions for each of the 5 folds
    for idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        # Set for each iteration Training and Validation Sets using cross-validation indexes
        X_it_train = X_train[train_idx]
        X_it_val = X_train[val_idx]
        y_it_train = y_train[train_idx]
        y_it_val = y_train[val_idx]
        if early_s:
            reg = model.fit(x=X_it_train,  y=y_it_train, validation_data=(X_it_val, y_it_val), epochs=epochs, batch_size=32, callbacks=[callback])
        else:
            reg = model.fit(x=X_it_train, y=y_it_train, validation_data=(X_it_val, y_it_val), epochs=epochs,
                            batch_size=32)
        train_loss.append(reg.history['loss'])
        val_loss.append(reg.history['val_loss'])
    mean_train = np.mean(train_loss, axis=0)
    mean_val = np.mean(val_loss, axis=0)
    t_loss = np.mean(mean_train)
    v_loss = np.mean(mean_val)

    # np.array(train_loss), np.array(val_loss),
    return mean_train, mean_val, t_loss, v_loss


def quest_d():
    X_train, X_test, y_train, y_test, dicts, kf = preprocess('iphi2802.csv')

    train1_mean_train, train1_mean_val, train1_loss, train1_val_loss,  = train_model(X_train, y_train, kf, 1, 125, 0.001, 0.6, 50, 0)
    total_losses.append(train1_mean_train)
    total_val_losses.append(train1_mean_val)
    print("train1_l:", train1_loss, "train1_vl:", train1_val_loss)

    train2_mean_train, train2_mean_val, train2_loss, train2_val_loss = train_model(X_train, y_train, kf, 1, 300, 0.001, 0.6, 50, 0)
    total_losses.append(train2_mean_train)
    total_val_losses.append(train2_mean_val)
    print("train2_l:", train2_loss, "train2_vl:", train2_val_loss)

    train3_mean_train, train3_mean_val, train3_loss, train3_val_loss = train_model(X_train, y_train, kf, 1, 600, 0.001, 0.6, 50, 0)
    total_losses.append(train3_mean_train)
    total_val_losses.append(train3_mean_val)
    print("train3_l:", train3_loss, "train3_vl:", train3_val_loss)

    plt.figure(figsize=(12, 4))

    for i, (loss, v_loss) in enumerate(zip(total_losses, total_val_losses)):
        plt.plot(loss, label='Mean Train Loss Network {}'.format(i + 1))
        plt.plot(v_loss, linestyle='--', label='Mean Val Loss Network {}'.format(i + 1))
    plt.legend(fontsize=6)
    plt.title('Mean Loss Over Epochs')
    plt.show()


def quest_e():

    X_train, X_test, y_train, y_test, dicts, kf = preprocess('iphi2802.csv')

    mult2_1_mean_train, mult2_1_mean_val, mult2_1_loss, mult2_2_val_loss,  = train_model(X_train, y_train, kf, 2, [200, 50], 0.001, 0.6, 50, 0)
    mul2_losses.append(mult2_1_mean_train)
    mul2_val_losses.append(mult2_1_mean_val)
    print("mult2_1_loss:", mult2_1_loss, "mult2_1_val_loss:", mult2_2_val_loss)

    mult2_2_mean_train, mult2_2_mean_val, mult2_2_loss, mult2_2_val_loss,  = train_model(X_train, y_train, kf, 2, [400, 100], 0.001, 0.6, 50, 0)
    mul2_losses.append(mult2_2_mean_train)
    mul2_val_losses.append(mult2_2_mean_val)
    print("mult2_2_loss:", mult2_2_loss, "mult2_2_val_loss:", mult2_2_val_loss)

    mult2_3_mean_train, mult2_3_mean_val, mult2_3_loss, mult2_3_val_loss,  = train_model(X_train, y_train, kf, 2, [600, 200], 0.001, 0.6, 50, 0)
    mul2_losses.append(mult2_3_mean_train)
    mul2_val_losses.append(mult2_3_mean_val)
    print("mult2_3_loss:", mult2_3_loss, "mult2_3_val_loss:", mult2_3_val_loss)

    mult3_1_mean_train, mult3_1_mean_val, mult3_1_loss, mult3_1_val_loss = train_model(X_train, y_train, kf, 3, [400, 100, 100], 0.001, 0.6, 50, 0)
    mul3_losses.append(mult3_1_mean_train)
    mul3_val_losses.append(mult3_1_mean_val)
    print("mult3_1_loss:", mult3_1_loss, "mult3_1_val_loss:", mult3_1_val_loss)

    mult3_2_mean_train, mult3_2_mean_val, mult3_2_loss, mult3_2_val_loss = train_model(X_train, y_train, kf, 3, [600, 300, 150], 0.001, 0.6, 50, 0)
    mul3_val_losses.append(mult3_2_mean_val)
    print("mult3_2_loss:", mult3_2_loss, "mult3_2_val_loss:", mult3_2_val_loss)

    mult3_3_mean_train, mult3_3_mean_val, mult3_3_loss, mult3_3_val_loss = train_model(X_train, y_train, kf, 3, [800, 400, 200], 0.001, 0.6, 50, 0)
    mul3_losses.append(mult3_3_mean_train)
    mul3_val_losses.append(mult3_3_mean_val)
    print("mult3_3_loss:", mult3_3_loss, "mult3_3_val_loss:", mult3_3_val_loss)

    mult3_4_mean_train, mult3_4_mean_val, mult3_4_loss, mult3_4_val_loss = train_model(X_train, y_train, kf, 3, [800, 800, 800], 0.001, 0.6, 50, 0)
    mul3_losses.append(mult3_4_mean_train)
    mul3_val_losses.append(mult3_4_mean_val)
    print("mult3_4_loss:", mult3_4_loss, "mult3_4_val_loss:", mult3_4_val_loss)

    mult3_5_mean_train, mult3_5_mean_val, mult3_5_loss, mult3_5_val_loss = train_model(X_train, y_train, kf, 3, [600, 700, 800],0.001, 0.6, 50, 0)
    mul3_val_losses.append(mult3_5_mean_val)
    print("mult3_5_loss:", mult3_5_loss, "mult3_5_val_loss:", mult3_5_val_loss)

    plt.figure(figsize=(12, 4))
    for i, (loss, v_loss) in enumerate(zip(mul2_losses, mul2_val_losses)):
        plt.plot(loss, label='Mean Train Loss 2 Level Network {}'.format(i + 1))
        plt.plot(v_loss, linestyle='--', label='Mean Val Loss 2 Level Network {}'.format(i + 1))
    plt.legend(fontsize=6)
    plt.title('Mean Loss Over Epochs')
    plt.show()

    plt.figure(figsize=(12, 4))
    for i, (loss, v_loss) in enumerate(zip(mul3_losses, mul3_val_losses)):
        plt.plot(loss, label='Mean Train Loss 3 Level Network {}'.format(i + 1))
        plt.plot(v_loss, linestyle='--', label='Mean Val Loss 3 Level Network {}'.format(i + 1))
    plt.legend(fontsize=6)
    plt.title('Mean Loss Over Epochs')
    plt.show()


def quest_st():
    X_train, X_test, y_train, y_test, dicts, kf = preprocess('iphi2802.csv')
    mean_train, mean_val, loss, val_loss = train_model(X_train, y_train, kf, 3,
                                                                   [600, 700, 800], 0.001, 0.6, 30, 1)
    print("Early Stopping Training Loss:", loss, "Early Stopping Validation Loss:", val_loss)


def quest_3():
    X_train, X_test, y_train, y_test, dicts, kf = preprocess('iphi2802.csv')
    t1_mean_train, t1_mean_val, t1_loss, t1_val_loss = train_model(X_train, y_train, kf, 3,
                                                                                       [600, 700, 800], 0.001, 0.2, 30, 0)
    print("t1_loss:", t1_loss, "t1_val_loss:", t1_val_loss)

    t2_mean_train, t2_mean_val, t2_loss, t2_val_loss = train_model(X_train, y_train, kf, 3,
                                                                   [600, 700, 800], 0.001, 0.6, 30, 0)
    print("t2_loss:", t2_loss, "t2_val_loss:", t2_val_loss)

    t3_mean_train, t3_mean_val, t3_loss, t3_val_loss = train_model(X_train, y_train, kf, 3,
                                                                   [600, 700, 800], 0.05, 0.6, 30, 0)
    print("t3_loss:", t3_loss, "t3_val_loss:", t3_val_loss)

    t4_mean_train, t4_mean_val, t4_loss, t4_val_loss = train_model(X_train, y_train, kf, 3,
                                                                   [600, 700, 800], 0.1, 0.6, 30, 0)
    print("t4_loss:", t4_loss, "t4_val_loss:", t4_val_loss)


'''
#Uncomment to execute
X_train, X_test, y_train, y_test, dicts, kf = preprocess('iphi2802.csv')
#print(X_train, X_test, y_train, y_test, dicts, kf)

mean_train, mean_val, loss, val_loss,  = train_model(X_train, y_train, kf, 1, 300, 0.001, 0.6, 30, 0)
print(mean_train, mean_val, loss, val_loss)
'''

# quest_d()
# quest_e()
# quest_st()
# quest_3()
