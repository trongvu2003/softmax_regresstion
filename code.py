import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("./winequality-red.csv")

X = df.drop("quality", axis=1).values
y = df["quality"].values

# Chuyển quality (3–8) → (0–5)
classes = np.unique(y)
class_to_idx = {c:i for i,c in enumerate(classes)}
y_idx = np.array([class_to_idx[i] for i in y])

# One-hot
def one_hot(y, num_class):
    m = len(y)
    oh = np.zeros((m, num_class))
    oh[np.arange(m), y] = 1
    return oh

Y = one_hot(y_idx, len(classes))



X_train, X_test, y_train, y_test = train_test_split(
    X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# one-hot cho train
Y_train = one_hot(y_train, len(classes))



def softmax(z):
    z = z - np.max(z)  # chống tràn số
    exp = np.exp(z)
    return exp / np.sum(exp)

def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat + 1e-9))


def train_softmax(X, Y, lr=0.01, epochs=1000):
    m, n = X.shape
    k = Y.shape[1]

    # thêm bias
    Xb = np.hstack([np.ones((m,1)), X])

    # khởi tạo theta
    theta = np.random.randn(k, n+1) * 0.01

    losses = []

    for epoch in range(epochs):
        total_loss = 0

        for i in range(m):
            x = Xb[i].reshape(-1,1)      # (n+1,1)
            y = Y[i].reshape(-1,1)       # (k,1)

            z = theta @ x
            y_hat = softmax(z)

            loss = cross_entropy(y, y_hat)
            total_loss += loss

            # gradient (slide): (y_hat - y) * x^T
            grad = (y_hat - y) @ x.T
            theta -= lr * grad

        losses.append(total_loss/m)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss = {losses[-1]:.4f}")

    return theta, losses


#train
theta, losses = train_softmax(X_train, Y_train, lr=0.01, epochs=1000)

def predict(X, theta):
    m = X.shape[0]
    Xb = np.hstack([np.ones((m,1)), X])
    preds = []

    for i in range(m):
        z = theta @ Xb[i].reshape(-1,1)
        y_hat = softmax(z)
        preds.append(np.argmax(y_hat))

    return np.array(preds)

y_pred = predict(X_test, theta)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
