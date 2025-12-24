import numpy as np

class AdaptivePSO:
    """
    Adaptive Particle Swarm Optimization (APSO).
    Optimizes a target function (Neural Network Loss).
    
    Adaptation Strategy (Time-Varying):
    - Inertia Weight (w): Linearly decreases (Exploration -> Exploitation)
    - c1 (Cognitive): High -> Low (Explore individual history early)
    - c2 (Social): Low -> High (Converge to swarm best late)
    """
    def __init__(self, num_particles, dimensions, bounds=(-1, 1)):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.min_bound, self.max_bound = bounds
        
        # Initialize Particles
        self.positions = np.random.uniform(self.min_bound, self.max_bound, (num_particles, dimensions))
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
        
        # Personal Best
        self.p_best_positions = self.positions.copy()
        self.p_best_scores = np.full(num_particles, float('inf'))
        
        # Global Best
        self.g_best_position = np.zeros(dimensions)
        self.g_best_score = float('inf')
        
    def optimize(self, fitness_function, max_iter=100, verbose=False):
        """
        Runs the PSO optimization loop.
        
        Args:
            fitness_function: Function that takes (1D weight array) and returns Loss (float).
            max_iter: Number of iterations.
            
        Returns:
            g_best_position: Best weights found.
            history: List of best loss per iteration.
        """
        history = []
        
        # Parameters for Adaptation (TVAC: Time-Varying Acceleration Coefficients)
        w_max, w_min = 0.9, 0.4
        c1_start, c1_end = 2.5, 0.5
        c2_start, c2_end = 0.5, 2.5
        
        for it in range(max_iter):
            # 1. Update Adaptive Parameters
            frac = it / max_iter
            w = w_max - frac * (w_max - w_min)
            c1 = c1_start + frac * (c1_end - c1_start)
            c2 = c2_start + frac * (c2_end - c2_start)
            
            # 2. Evaluate Fitness for all particles
            for i in range(self.num_particles):
                score = fitness_function(self.positions[i])
                
                # Update Personal Best
                if score < self.p_best_scores[i]:
                    self.p_best_scores[i] = score
                    self.p_best_positions[i] = self.positions[i].copy()
                    
                # Update Global Best
                if score < self.g_best_score:
                    self.g_best_score = score
                    self.g_best_position = self.positions[i].copy()
            
            history.append(self.g_best_score)
            
            if verbose and (it % 10 == 0):
                print(f"Iteration {it}/{max_iter}, Best Loss: {self.g_best_score:.6f} | w={w:.2f}, c1={c1:.2f}, c2={c2:.2f}")

            # Early Stopping Check
            if it > 10:
                if self.g_best_score < (min(history[:-10]) - 1e-4):
                    # Improved
                    pass
                else:
                    # No improvement for 10 iterations
                    # Check if strictly flat
                    if history[-1] >= history[-10]:
                        if verbose: print(f"⏹️ Early Stopping at iteration {it} (No improvement for 10 iters)")
                        break

            # 3. Update Velocity and Position
            r1 = np.random.rand(self.num_particles, self.dimensions)
            r2 = np.random.rand(self.num_particles, self.dimensions)
            
            # Velocity Update Formula
            # v = w*v + c1*r1*(pBest - x) + c2*r2*(gBest - x)
            cognitive = c1 * r1 * (self.p_best_positions - self.positions)
            social = c2 * r2 * (self.g_best_position - self.positions)
            
            self.velocities = w * self.velocities + cognitive + social
            
            # Clamp Velocity (Optional but recommended)
            # v_max = (max_bound - min_bound) * 0.2
            # self.velocities = np.clip(self.velocities, -v_max, v_max)
            
            # Position Update
            self.positions = self.positions + self.velocities
            
            # Clamp Position (Keep weights within reasonable bounds)
            self.positions = np.clip(self.positions, self.min_bound, self.max_bound)
            
        return self.g_best_position, history
