import numpy as np

class Autoencoder:
    """
    NumPy-based Autoencoder for Anomaly Detection / Feature Extraction.
    Architecture: Input -> Hidden (Latent) -> Output (Reconstruction)
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Encoder Weights
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Decoder Weights
        self.W2 = np.random.randn(hidden_size, input_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, input_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        """
        Returns:
            reconstruction: Standard output [N, input_size]
            latent: Compressed representation [N, hidden_size]
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.latent = self.sigmoid(self.z1) # Latent Representation
        
        self.z2 = np.dot(self.latent, self.W2) + self.b2
        self.reconstruction = self.z2 # Linear activation
        
        return self.reconstruction, self.latent

    def compute_loss(self, X, reconstruction):
        """ Mean Squared Error (MSE) """
        return np.mean((X - reconstruction) ** 2)

    def train(self, X, epochs=50, lr=0.01):
        """ Trains with simple Gradient Descent """
        loss_history = []
        m = X.shape[0]
        
        for i in range(epochs):
            # Forward
            reconstruction, latent = self.forward(X)
            
            # Loss
            loss = self.compute_loss(X, reconstruction)
            loss_history.append(loss)
            
            # Backward
            error = reconstruction - X
            d_z2 = (2/m) * error 
            
            d_W2 = np.dot(latent.T, d_z2)
            d_b2 = np.sum(d_z2, axis=0, keepdims=True)
            
            d_hidden = np.dot(d_z2, self.W2.T) * self.sigmoid_derivative(latent)
            
            d_W1 = np.dot(X.T, d_hidden)
            d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
            
            # Update
            self.W1 -= lr * d_W1
            self.b1 -= lr * d_b1
            self.W2 -= lr * d_W2
            self.b2 -= lr * d_b2
            
        return loss_history
    
    def get_features(self, X):
        """
        Returns Hybrid Feature Set:
        1. Latent Representation vectors
        2. Reconstruction Error (MSE) per sample
        """
        reconstruction, latent = self.forward(X)
        
        # Calculate MSE per row (sample)
        mse_per_sample = np.mean((X - reconstruction) ** 2, axis=1).reshape(-1, 1)
        
        # Stack latent attributes and the error metric
        return np.hstack([latent, mse_per_sample])
