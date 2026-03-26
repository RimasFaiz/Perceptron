import numpy as np
import pandas as pd

def load_data(filename):
    try:
        # Read CSV with no header, assuming columns: f1, f2, f3, f4, label
        df = pd.read_csv(filename, header=None, names=['f1', 'f2', 'f3', 'f4', 'label'])

        # Convert features to numpy array
        features = df[['f1', 'f2', 'f3', 'f4']].values
        # Convert labels to numpy array
        labels = df['label'].values

        return features, labels
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return np.array([]), np.array([])

def filter_data(X, y, class1, class2):
    mask = (y == class1) | (y == class2)
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Convert labels to 1 and -1
    y_binary = np.where(y_filtered == class1, 1, -1)

    return X_filtered, y_binary

class Perceptron:
    def __init__(self, num_features, iterations=20):
        # Initialize weights to zero (including bias)
        # Weights vector size is num_features + 1 (for bias)
        self.weights = np.zeros(num_features + 1)
        self.iterations = iterations

    def train(self, X, y):
        # Add bias term (column of 1s) to X
        X_bias = np.c_[np.ones(X.shape[0]), X]

        for _ in range(self.iterations):
            for i in range(X_bias.shape[0]):
                xi = X_bias[i]
                target = y[i]

                # Activation: w . x
                activation = np.dot(self.weights, xi)

                # Update rule: if y * activation <= 0 (misclassification)
                if target * activation <= 0:
                    self.weights += target * xi

    def predict(self, X):
        X_bias = np.c_[np.ones(X.shape[0]), X]
        activations = np.dot(X_bias, self.weights)
        return np.where(activations >= 0, 1, -1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y) * 100

def get_most_discriminative_feature(weights):
    # Exclude bias weight
    feature_weights = weights[1:]

    # Feature names for the Iris dataset
    feature_names = [
        "Feature 1 (sepal length)",
        "Feature 2 (sepal width)",
        "Feature 3 (petal length)",
        "Feature 4 (petal width)"
    ]

    # Find index of max absolute weight
    max_idx = np.argmax(np.abs(feature_weights))
    return feature_names[max_idx], feature_weights[max_idx]

def main():
    print("="*40)
    print("BINARY PERCEPTRON CLASSIFICATION")
    print("="*40)

    # File paths
    train_file = "attached_assets/train.data"
    test_file = "attached_assets/test.data"

    print(f"Loading training data from: {train_file}")
    print(f"Loading testing data from: {test_file}")

    X_train_all, y_train_all = load_data(train_file)
    X_test_all, y_test_all = load_data(test_file)

    class_pairs = [
        ("class-1", "class-2"),
        ("class-2", "class-3"),
        ("class-1", "class-3")
    ]

    results = []

    print("\nTraining perceptrons (20 iterations).")
    print("-" * 40)

    for c1, c2 in class_pairs:
        print(f"\n  - Classifier: {c1} vs {c2} -")

        # Prepare data
        X_train, y_train = filter_data(X_train_all, y_train_all, c1, c2)
        X_test, y_test = filter_data(X_test_all, y_test_all, c1, c2)

        # Train
        p = Perceptron(num_features=X_train.shape[1], iterations=20)
        p.train(X_train, y_train)

        # Evaluate
        train_acc = p.accuracy(X_train, y_train)
        test_acc = p.accuracy(X_test, y_test)

        # Discriminative feature
        best_feature, best_weight = get_most_discriminative_feature(p.weights)

        print(f"Train Accuracy: {train_acc:.2f}%")
        print(f"Test Accuracy:  {test_acc:.2f}%")
        print(f"Most Discriminative Feature: {best_feature} (weight: {best_weight:.4f})")

        results.append({
            'pair': f"{c1} vs {c2}",
            'test_acc': test_acc
        })

    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)

    # Find most difficult pair (lowest test accuracy)
    min_acc_result = min(results, key=lambda x: x['test_acc'])

    print(f"Most Difficult Pair to Separate: {min_acc_result['pair']}")
    print(f"Reason: Lowest test accuracy ({min_acc_result['test_acc']:.2f}%) indicates these classes are not linearly separable.")
   

if __name__ == "__main__":
    main()
