import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Define the Boolean function
def boolean_function(x1, x2, x3, x4):
    return (not x1) and x2 and (x3 or x4)

# Generate the truth table
def generate_truth_table():
    inputs = []
    outputs = []
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            for x3 in [0, 1]:
                for x4 in [0, 1]:
                    inputs.append([x1, x2, x3, x4])
                    outputs.append(boolean_function(x1, x2, x3, x4))
    return np.array(inputs), np.array(outputs)

# Train the NN and collect metrics
def train_nn_with_metrics(inputs, outputs, epochs=100, learning_rate=0.1):
    num_inputs = inputs.shape[1]
    weights = np.random.uniform(-1, 1, num_inputs)  # Initialize weights
    bias = np.random.uniform(-1, 1)                # Initialize bias

    epoch_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for epoch in range(epochs):
        predictions = []
        total_error = 0

        for i in range(len(inputs)):
            x = inputs[i]
            y_true = outputs[i]

            # Calculate net input
            net = np.dot(weights, x) + bias
            y_pred = 1 if net >= 0.5 else 0  # Threshold activation
            predictions.append(y_pred)

            # Calculate error
            error = y_true - y_pred
            total_error += abs(error)

            # Update weights and bias
            weights += learning_rate * error * x
            bias += learning_rate * error

        # Calculate metrics
        accuracy = accuracy_score(outputs, predictions)
        precision = precision_score(outputs, predictions)
        recall = recall_score(outputs, predictions)
        f1 = f1_score(outputs, predictions)

        epoch_metrics['accuracy'].append(accuracy)
        epoch_metrics['precision'].append(precision)
        epoch_metrics['recall'].append(recall)
        epoch_metrics['f1'].append(f1)

        # Stop if no error
        if total_error == 0:
            break

    return weights, bias, epoch_metrics

# Main program
inputs, outputs = generate_truth_table()
weights, bias, metrics = train_nn_with_metrics(inputs, outputs)

# Plot metrics
plt.figure(figsize=(10, 6))
plt.plot(metrics['accuracy'], label='Accuracy', marker='o')
plt.plot(metrics['precision'], label='Precision', marker='s')
plt.plot(metrics['recall'], label='Recall', marker='^')
plt.plot(metrics['f1'], label='F1-score', marker='d')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Performance Metrics over Training Epochs')
plt.legend()
plt.grid()
plt.savefig("performance_metrics.png")
plt.show()

# Display final weights and metrics
print("Trained Weights:", weights)
print("Trained Bias:", bias)
print("Final Metrics:", {k: v[-1] for k, v in metrics.items()})
