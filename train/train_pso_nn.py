import numpy as np
import time
from models.neural_network import MLP
from models.adaptive_pso import AdaptivePSO

def train_with_backprop(X_train, y_train, input_size, hidden_size, epochs=100, lr=0.01):
    """
    Trains MLP using standard Backpropagation.
    """
    print(f"Starting Backpropagation Training (Epochs={epochs})...")
    start_time = time.time()
    
    model = MLP(input_size, hidden_size)
    loss_history = model.train_backprop(X_train, y_train, epochs, lr)
    
    end_time = time.time()
    print(f"BP Training Complete. Time: {end_time - start_time:.2f}s")
    
    return model, loss_history

def train_with_pso(X_train, y_train, input_size, hidden_size, particle_count=30, iterations=50):
    """
    Trains MLP using Adaptive Particle Swarm Optimization.
    """
    print(f"Starting Adaptive PSO Training (Particles={particle_count}, Iterations={iterations})...")
    start_time = time.time()
    
    # 1. Initialize Model to define structure
    temp_model = MLP(input_size, hidden_size)
    initial_weights = temp_model.get_weights_flat()
    dimensions = len(initial_weights)
    
    # 2. Define Fitness Function
    # PSO minimizes this function. We want to minimize NN Loss.
    def fitness_function(weights):
        temp_model.set_weights_flat(weights)
        preds = temp_model.forward(X_train)
        return temp_model.compute_loss(y_train, preds)
    
    # 3. Run PSO
    optimizer = AdaptivePSO(num_particles=particle_count, dimensions=dimensions, bounds=(-2, 2))
    best_weights, loss_history = optimizer.optimize(fitness_function, max_iter=iterations, verbose=True)
    
    # 4. Set final weights to a new model instance
    final_model = MLP(input_size, hidden_size)
    final_model.set_weights_flat(best_weights)
    
    end_time = time.time()
    print(f"PSO Training Complete. Time: {end_time - start_time:.2f}s")
    
    return final_model, loss_history
