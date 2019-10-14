# ################################################################### #
# gamblers-problem.py                                                 #
# Author: Glenn Dawson (2019)                                         #
# ---------------------------                                         #
# Implementation of the Gambler's Problem from Sutton and Barto       #
# (2018).                                                             #
# ################################################################### #

import numpy as np
import matplotlib.pyplot as plt
import time

PROBS = np.array([0.6, 0.4])  # p(loss), p(win)
WIN_CONDITION = 100
EPSILON = 0
GAMMA = 1
STATES = np.arange(1, 100)
POLICY = np.zeros(STATES.shape, dtype=np.int)

def main():
    values = np.zeros(STATES.shape[0] + 2)
    values[-1] = 1
    sweep = 1
    while True:
        delta = 0
        for state in STATES:
            # Store initial value
            v = values[state]
            
            # Generate action space
            actions = np.arange(0, min(state, 100-state) + 1)
            action_values = []
            
            for action in actions:
                # s_prime is state after either win or loss
                s_prime = np.array([state - action, state + action])
                # Get values at s_prime
                v_prime = values[s_prime]
                
                # Calculate action-value
                action_values.append(PROBS @ (GAMMA * v_prime))
            
            # Choose state value
            values[state] = np.amax(action_values)
            delta = max(delta, abs(v - values[state]))
        
        # Plot values over given sweeps
        if sweep in [1, 2, 3, 32]:
            plt.plot(STATES, values[1:-1], label=f'Sweep {sweep}')
            
        sweep += 1
        
        if delta <= EPSILON:
            break
        
    plt.plot(STATES, values[1:-1], label='Final value function')
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend()
    
    # Generate policy
    for i in range(STATES.shape[0]):
        # Get state
        state = STATES[i]
        
        # Generate action space
        actions = np.arange(0, min(state, 100-state) + 1)
        action_values = []
        
        for action in actions:
            # s_prime is state after either win or loss
            s_prime = np.array([state - action, state + action])
            
            # Get values at s_prime
            v_prime = values[s_prime]
            
            # Calculate action-value
            action_values.append(PROBS @ (GAMMA * v_prime))
        
        candidates = np.flatnonzero(action_values 
                                    == max(action_values))
        
        if len(candidates) > 1:
            idx = candidates[1]
        else:
            idx = 0
        POLICY[i] = (actions[idx])
        print(state, POLICY[i], actions, action_values)
        print('---')
    print(values)
    
    # Plot final policy
    plt.figure()
    plt.stem(STATES, POLICY, use_line_collection=True)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    
if __name__ == '__main__':
    main()
    plt.show()
