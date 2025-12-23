import numpy as np

class MLP:
    """
    Multilayer Perceptron (MLP).
    Updated for Multi-Class Classification (3 Classes: Normal, Anomaly, Storm).
    
    Architecture: Input -> Hidden (Sigmoid) -> Output (Softmax)
    """
    def __init__(self, input_size, hidden_size, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights (Xavier/Glorot Initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def softmax(self, x):
        """
        Stable Softmax function for multi-class probability.
        """
        # Shift x for numerical stability (subtract max)
        # Handle 1D or 2D arrays
        if x.ndim == 2:
            shift_x = x - np.max(x, axis=1, keepdims=True)
            exps = np.exp(shift_x)
            return exps / np.sum(exps, axis=1, keepdims=True)
        else:
            shift_x = x - np.max(x)
            exps = np.exp(shift_x)
            return exps / np.sum(exps)
    
    def forward(self, X):
        """
        Forward pass.
        Returns probabilities for each class.
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Output activation: Softmax for multi-class
        self.a2 = self.softmax(self.z2) 
        return self.a2
    
    def get_weights_flat(self):
        """
        Flattens all weights/biases for PSO.
        """
        return np.concatenate([
            self.W1.flatten(), 
            self.b1.flatten(), 
            self.W2.flatten(), 
            self.b2.flatten()
        ])
    
    def set_weights_flat(self, flat_weights):
        """
        Reconstructs weights from PSO particle.
        """
        w1_end = self.input_size * self.hidden_size
        b1_end = w1_end + self.hidden_size
        w2_end = b1_end + (self.hidden_size * self.output_size)
        
        self.W1 = flat_weights[:w1_end].reshape(self.input_size, self.hidden_size)
        self.b1 = flat_weights[w1_end:b1_end].reshape(1, self.hidden_size)
        self.W2 = flat_weights[b1_end:w2_end].reshape(self.hidden_size, self.output_size)
        self.b2 = flat_weights[w2_end:].reshape(1, self.output_size)

    def compute_loss(self, y_true_indices, y_pred_probs):
        """
        Categorical Cross Entropy Loss.
        Args:
            y_true_indices: Array of true class indices (0, 1, 2)
            y_pred_probs: Array of predicted probabilities (N, 3)
        """
        m = y_true_indices.shape[0]
        epsilon = 1e-15
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1. - epsilon)
        
        # Extract the probability of the true class for each sample
        # We use advanced indexing to pick the prob corresponding to the true label
        correct_confidences = y_pred_probs[range(m), y_true_indices.astype(int)]
        
        loss = -np.sum(np.log(correct_confidences)) / m
        return loss

    def predict(self, X):
        """ Return class indices """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
