from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X, y   = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=1)
model.fit(X_train, y_train)

preds    = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Digit recognition accuracy: {accuracy*100:.1f}%")
