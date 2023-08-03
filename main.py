import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse # to parse arguments in the command line

def init_params(cell_L1, cell_L2):
    W1 = np.random.rand(cell_L1, 784) - 0.5
    b1 = np.random.rand(cell_L1, 1) - 0.5
    W2 = np.random.rand(cell_L2, cell_L1) - 0.5  # Augmenter le nombre de neurones de 10 à 20 (par exemple)
    b2 = np.random.rand(cell_L2, 1) - 0.5  # Assurez-vous d'avoir le même nombre de neurones pour b2
    W3 = np.random.rand(10, cell_L2) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    rounded_values = [np.round(x, 3)[0] for x in A3]
    print(rounded_values)
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations, cell_L1, cell_L2, init_params: None or list):
    if init_params is None:
        W1, b1, W2, b2, W3, b3 = init_params(cell_L1, cell_L2)
    else:
        W1, b1, W2, b2, W3, b3 = init_params
        
    accuracy = 0
    epochs = iterations
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            print("Epochs: ", i)
            predictions = get_predictions(A3)
            accuracy = get_accuracy(predictions, Y)
            print("Accuracy: ", accuracy)
            
    save_model(W1, b1, W2, b2, W3, b3, accuracy, epochs)
    return W1, b1, W2, b2, W3, b3

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def save_model(W1, b1, W2, b2, W3, b3, accuracy, epochs):
    write_file = open("./weights/model.txt", "w")
    write_file.write(str(accuracy) + "\n")
    write_file.write(str(epochs))
    write_file.close()

    np.save("./weights/W1.npy", W1)
    np.save("./weights/b1.npy", b1)
    np.save("./weights/W2.npy", W2)
    np.save("./weights/b2.npy", b2)
    np.save("./weights/W3.npy", W3)
    np.save("./weights/b3.npy", b3)

def load_model():
    W1 = np.load("./weights/W1.npy")
    b1 = np.load("./weights/b1.npy")
    W2 = np.load("./weights/W2.npy")
    b2 = np.load("./weights/b2.npy")
    W3 = np.load("./weights/W3.npy")
    b3 = np.load("./weights/b3.npy")
    return W1, b1, W2, b2, W3, b3

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
    
if __name__ == "__main__":
    ## load args
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the model", action="store_true", default=True)
    parser.add_argument("--test", help="test the model", action="store_true", default=False)
    parser.add_argument("--resume", help="resume training (must be used with --train flag)", action="store_true", default=False)
    
    args = parser.parse_args()
    
    data = pd.read_csv('./datasets/train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    _,m_train = X_train.shape

    print("X_train shape: " + str(X_train.shape))

    if args.test:
        W1, b1, W2, b2, W3, b3 = load_model()
    elif args.train:
        if args.resume:
            W1, b1, W2, b2, W3, b3 = load_model()
            W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.1, 1000, 64, 32, [W1, b1, W2, b2, W3, b3])
        else:
            W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.1, 1000, 64, 32)
    
        
    for i in range(15):
        test_prediction(i, W1, b1, W2, b2, W3, b3)