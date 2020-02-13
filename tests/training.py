from littlenn.model import Sequential
from littlenn.layers import Dense, Dropout
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer

X, y = make_classification(1000, 30, 27, 1, random_state=42)
#X, y = load_breast_cancer(return_X_y=True)
X /= np.max(X)
N = X.shape[1]
T = int(X.shape[0] * 0.9)

model = Sequential(input_size=N)
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(keep_prob=0.8))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(keep_prob=0.8))
model.add(Dense(1, activation='sigmoid'))

def binary_loss(y_true, y_pred):
    return np.mean(- (y_true * np.log(y_pred + 1e-12) + (1-y_true) * np.log(1-y_pred + 1e-12)))

def binary_loss_deriv(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1. - y_pred) + 1e-12)

def batch_generator(X, y, batch_size):
    random_indices = np.random.choice(range(0, X.shape[0]), size=X.shape[0], replace=False)
    arr = X[random_indices]
    lab = y[random_indices]
    for i in range(0, arr.shape[0], batch_size):
        yield arr[i:i+batch_size].T, lab[i:i+batch_size].T

X, y = X.reshape(X.shape[0], N), y.reshape(X.shape[0], 1)

X_val, y_val = X[T:].T, y[T:].T
X, y = X[:T], y[:T]

epochs = 500
batch_size = 64
lr = 1e-3

for epoch in range(epochs):
    rolling_loss = 0.0
    ind = 0
    for batch, batch_labels in batch_generator(X, y, batch_size):
        ind += 1
        y_pred = model(batch)
        #print(y_pred[:10])
        dprev = binary_loss_deriv(batch_labels, y_pred)
        loss = binary_loss(batch_labels, y_pred)
        model._backprop(dprev, lr)
        rolling_loss += loss
    rolling_loss /= ind    
    if (epoch + 1) % 10 == 0:
        print('EPOCH %d | Loss : %.5f' %  (epoch + 1, rolling_loss), end='\t|\t')
        val_pred = model(X_val, training=False)
        val_pred[val_pred >= 0.5] = 1
        val_pred[val_pred < 0.5] = 0
        accuracy = np.sum(val_pred == y_val) / y_val.size
        print('Validation accuracy : ', accuracy)