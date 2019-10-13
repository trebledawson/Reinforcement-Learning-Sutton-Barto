# ################################################################### #
# jacks-car-rental.py                                                 #
# Author: Glenn Dawson (2019)                                         #
# ---------------------------                                         #
# Implementation of Jack's Car Rental from Sutton and Barto (2018).   #
# ################################################################### #

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

MAX = 20  # Maximum number of cars at each location
EPSILON = 1e-8  # Policy evaluation convergence limit
GAMMA = 0.9  # Discount factor
POISSON_UB = 12  # Poisson pmf is negligible above this upper bound
POISSON2 = poisson.pmf(range(POISSON_UB), 2)  # lambda = 2
POISSON3 = poisson.pmf(range(POISSON_UB), 3)  # lambda = 3
POISSON4 = poisson.pmf(range(POISSON_UB), 4)  # lambda = 4
STATES = (np.array([[x, y] for x in range(MAX + 1) 
                           for y in range(MAX + 1)]))  # State space

def main():
    values = np.zeros((MAX + 1, MAX + 1))  # Initialize values arbitrarily in grid shape (21, 21)
    policy = np.zeros(STATES.shape[0])  # Initialize policy to move zero cars
    
    # Plot initial policy
    plt.figure()
    plt.title(f'Policy at step {0}')
    plt.xlabel('Cars at first location')
    plt.ylabel('Cars at second location')
    plt.contourf(policy.reshape(21, 21))
    plt.colorbar()
    
    # Policy Iteration
    stable = False
    steps = 0
    while not stable:
        # Policy Evaluation
        while True:
            values, delta = update_values(STATES, policy, values)
            print(delta)
            if delta < EPSILON:
                break

        # Policy Improvement
        policy, stable = update_policy(STATES, policy, values)
        
        steps += 1
        
        # Generate plots
        plt.figure()
        plt.title(f'Policy at step {steps}')
        plt.xlabel('Cars at second location')
        plt.ylabel('Cars at first location')
        plt.contourf(policy.reshape(21, 21))
        plt.colorbar()
    
    # Plot final value function
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title('Final value function')
    ax.set_xlabel('Cars at second location')
    ax.set_ylabel('Cars at first location')
    ax.set_zlabel('Expected profit')
    X, Y = np.meshgrid(range(21), range(21))
    ax.plot_surface(X, Y, values, rstride=1, cstride=1)
    
    plt.show()
    
def update_values(states, policy, values):
    """
    states : shape = (441, 2)
        Static array of states. Values between [0, 20].
        
    policy : shape = (441,)
        Static array of actions between [-5, 5].
        
    values : shape = (21, 21)
        Mutable array of state values.
    """
    delta = 0
    for state, action in zip(states, policy):
        # Get s_prime according to action
        # Note: p(s_prime | s, pi(s)) = 1
        s_prime = state.copy()
        s_prime[0] -= action
        s_prime[1] += action
        s_prime = np.clip(s_prime, 0, 20)
        
        # Store old value to check delta
        v = values[state[0], state[1]]
        
        # Get joint probability of r over range of rental/return rates
        # and calculate the updated value
        n_rentals_2, n_rentals_1, n_returns_2, n_returns_1 = (
            np.meshgrid(range(POISSON_UB), range(POISSON_UB), 
                        range(POISSON_UB), range(POISSON_UB)))
        rentals_joint_probs = np.outer(POISSON3, POISSON4)  # shape = (12, 12)
        n_rentals_1 = np.minimum(n_rentals_1, s_prime[0])  # shape = (12, 12, 12, 12)
        n_rentals_2 = np.minimum(n_rentals_2, s_prime[1])  # shape = (12, 12, 12, 12)
        
        rewards = (10 * np.add(n_rentals_1, n_rentals_2).flatten() 
                   - 2 * abs(action))  # shape = 12 ** 4 = 20736
        
        returns_joint_probs = np.outer(POISSON3, POISSON2)  # shape = (12, 12)
        n_final_1 = np.minimum((
            n_returns_1 - n_rentals_1 + s_prime[0]), 20).flatten()  # shape = 12 ** 4 = 20736
        n_final_2 = np.minimum((
            n_returns_2 - n_rentals_2 + s_prime[1]), 20).flatten()  # shape = 12 ** 4 = 20736
            
        v_prime = values[n_final_1, n_final_2]  # shape = 20736
        joint_probs = np.outer(rentals_joint_probs, 
                               returns_joint_probs).flatten()  # shape = 12 ** 4 = 20736
        values[state[0], state[1]] = (joint_probs 
                                      @ (rewards + GAMMA * v_prime))
        
        # Compare delta
        delta = max(delta, abs(v - values[state[0], state[1]]))
    return values, delta


def update_policy(states, policy, values):
    """
    states : shape = (441, 2)
        Static array of states. Values between [0, 20].
        
    policy : shape = (441,)
        Mutable array of actions between [-5, 5].
        
    values : shape = (21, 21)
        Static array of values.
    """
    stable = True
    for i in range(states.shape[0]):
        state = states[i]
        old_action = policy[i]
        
        # Build action space
        actions_lb = np.amax([state[0] - 20, -state[1], -5])
        actions_ub = np.amin([state[0], 20 - state[1], 5])
        actions = np.arange(actions_lb, actions_ub + 1)
        action_values = []
        
        for action in actions:
            # Get s_prime according to action
            # Note: p(s_prime | s, a) = 1
            s_prime = state.copy()
            s_prime[0] -= action
            s_prime[1] += action
            s_prime = np.clip(s_prime, 0, 20)
            
            n_rentals_2, n_rentals_1, n_returns_2, n_returns_1 = (
                np.meshgrid(range(POISSON_UB), range(POISSON_UB), 
                            range(POISSON_UB), range(POISSON_UB)))
                            
            rentals_joint_probs = np.outer(POISSON3, POISSON4)  # shape = (12, 12)
            n_rentals_1 = np.minimum(n_rentals_1, s_prime[0])  # shape = (12, 12, 12, 12)
            n_rentals_2 = np.minimum(n_rentals_2, s_prime[1])  # shape = (12, 12, 12, 12)
            
            rewards = (10 * np.add(n_rentals_1, n_rentals_2).flatten() 
                       - 2 * abs(action))  # shape = 12 ** 4 = 20736
            
            returns_joint_probs = np.outer(POISSON3, POISSON2)  # shape = (12, 12)
            n_final_1 = np.minimum((
                n_returns_1 - n_rentals_1 + s_prime[0]), 20).flatten()  # shape = 12 ** 4 = 20736
            n_final_2 = np.minimum((
                n_returns_2 - n_rentals_2 + s_prime[1]), 20).flatten()  # shape = 12 ** 4 = 20736
            v_prime = values[n_final_1, n_final_2]  # shape = 20736
            
            joint_probs = np.outer(rentals_joint_probs, 
                                   returns_joint_probs).flatten()  # shape = 12 ** 4 = 20736
            action_values.append((joint_probs  
                                  @ (rewards + GAMMA * v_prime)))
            
        policy[i] = actions[np.argmax(action_values)]
        
        if stable and policy[i] != old_action:
            stable = False
            
    return policy, stable
    
    
if __name__ == '__main__':
    main()
