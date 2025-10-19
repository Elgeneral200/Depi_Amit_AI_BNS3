"""
Gradient Descent Optimization for Profit Maximization with Linear Constraint.

This module implements a gradient descent algorithm to solve a linear programming
problem where we maximize profit subject to a resource constraint.

Problem Statement:
Maximize P = 7x1 + 4x2
subject to: 2.5x1 + 1.5x2 ≤ 80

Where:
- x1: quantity of smoothies
- x2: quantity of fresh juice
- 7: profit per smoothie ($)
- 4: profit per fresh juice ($)
- 2.5: fruit required per smoothie (kg)
- 1.5: fruit required per fresh juice (kg)
- 80: total fruit available (kg)

The algorithm uses gradient descent on the negative profit function with
projection to handle the linear constraint.

Author: Muhammad Fathi Kamal Ahmed 
ID: 42310346
Section: AI4
"""

def gradient_descent(alpha=0.1, iterations=5):
    """
    Solve the profit maximization problem using gradient descent with constraint projection.
    
    The method works by:
    1. Starting from an initial point (0, 0)
    2. Taking gradient steps toward higher profit (negative gradient of -P)
    3. Projecting infeasible points back onto the constraint boundary
    4. Repeating for specified number of iterations
    
    Args:
        alpha (float): Learning rate for gradient descent. Default is 0.1.
        iterations (int): Number of iterations to run. Default is 5.
    
    Returns:
        None: Results are printed to console.
        
    """
    # Initial point (start with zero production)
    x1, x2 = 0.0, 0.0

    # Gradients of the negative profit function
    # Since we maximize P = 7x1 + 4x2, we use negative gradients for ascent
    grad_x1 = -7  # ∂(-P)/∂x1 = -7
    grad_x2 = -4  # ∂(-P)/∂x2 = -4

    print("Gradient Descent with Constraint (2.5x1 + 1.5x2 <= 80)")
    print("--------------------------------------------------------")
    
    for i in range(1, iterations + 1):
        # Update step: move in direction of negative gradient (profit maximization)
        x1 = x1 - alpha * grad_x1
        x2 = x2 - alpha * grad_x2

        # Check constraint: total fruit consumption
        total_fruit = 2.5 * x1 + 1.5 * x2
        
        # Project back onto feasible region if constraint is violated
        if total_fruit > 80:
            # Scale down production to exactly meet the constraint
            scale = 80 / total_fruit
            x1 *= scale
            x2 *= scale

        # Calculate current profit
        profit = 7 * x1 + 4 * x2

        print(f"Iteration {i}: x1 = {x1:.2f}, x2 = {x2:.2f}, "
              f"Profit = ${profit:.2f}, Fruit used = {2.5*x1 + 1.5*x2:.2f} kg")

    print("--------------------------------------------------------")
    print(f"Final optimal solution after {iterations} iterations:")
    print(f"Smoothies = {x1:.2f}, Fresh Juice = {x2:.2f}, "
          f"Max Profit ≈ ${profit:.2f}")

# Run the function
if __name__ == "__main__":
    gradient_descent()