import sys
import gc
import numpy as np
import time
import copy
import matplotlib.pyplot as plt


def main(racetrack='A'):
    start = time.time()
    racetrack = str(racetrack).upper()
    
    Car = Agent(Racetrack(racetrack))
    rewards = []
    eval_returns = [[], [], []]
    eval_episodes = [[], [], []]
    for episode in range(10_000):
        # Evaluation logging
        if (episode + 1) % 1000 == 0:
            Car.noise = False
            Car.evaluate = True
            i = 0
            for start_position in Car.track.start_positions:
                Car.generate_episode(np.array([0, start_position]))
                reward = np.sum(Car.rewards_e)
                success = 'FAIL'
                if Car.success_e:
                    success = 'SUCCESS'
                    eval_episodes[i].append(episode + 1)
                    eval_returns[i].append(reward)
                track = Racetrack(racetrack).map
                for loc in Car.history_p:
                    track[loc[0], loc[1]] = 9                
                track = Racetrack(racetrack).map
                print('Episode:', episode + 1, '| Return:', reward, '| Steps:', len(Car.rewards_e), '| Restarts:', Car.n_restarts, '|', success)
                for loc in Car.history_p:
                    track[loc[0], loc[1]] = 9
                print(track)
                i += 1
            
            Car.noise = True
            Car.evaluate = False
        
        # Scheduling the epsilon-greedy exploration
        if (episode + 1) % 500 == 0 and Car.epsilon >= 0.01:
            Car.epsilon *= 0.9
            print(Car.epsilon)
        
        # Episode generation
        Car.generate_episode()
        reward = np.sum(Car.rewards_e)
        rewards.append(reward)
        if episode < 100:
            print(f'Episode {episode + 1} | Return: {reward}')
        
        # Sanity check
        if (episode + 1) % 200 == 0:
            print('.')
            
        gc.collect()
    
    # Plotting figures
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.grid()
    plt.xlabel('Number of episodes')
    plt.ylabel('Return')
    plt.title('Returns obtained over Expected Sarsa episodes | Track ' + racetrack)
    
    plt.figure()
    for i in range(3):
        plt.plot(eval_episodes[i], eval_returns[i], label=('Start at 0, ' + str(i+3)))
    plt.grid()
    plt.legend()
    plt.xlabel('Evaluation episode')
    plt.ylabel('Return')
    plt.title('Returns obtained over Expected Sarsa evaluations | Track ' + racetrack)
    
    # Final evaluations
    Car.noise = False  # Random acceleration stall disabled
    Car.evaluate = True  # Deterministic evaluation
    run = 1
    for start_position in Car.track.start_positions:
        Car.generate_episode(np.array([0, start_position]))
        track = Racetrack(racetrack).map
        for loc in Car.history_p:
            track[loc[0], loc[1]] = 9
        if Car.success_e:
            success = 'Success'
        else:
            success = 'Failure'
        plt.figure()
        plt.imshow(track, interpolation='none', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Evaluation run {run} | Steps: {len(Car.history_p) - 1} | Track {racetrack} | {success}')
        run += 1
    
    # Logging
    print('---')
    print(f'Runtime: {(time.time() - start):.6f} seconds')
    plt.show()
    
    
class Racetrack(object):
    def __init__(self, track='A'):
        if track == 'A':  # Simple
            self.height = 19
            self.width = 9
            self.map = np.ones((self.height + 1, self.width + 1), dtype=np.int8)
            
            # Set illegal squares
            self.map[15:, 0] = 0
            self.map[16:, 1] = 0
            self.map[19, 2] = 0
            self.map[:11, 0] = 0
            self.map[:7, 1] = 0
            self.map[:4, 2] = 0
            self.map[:14, 6] = 0
            self.map[:15, 7:] = 0
            
            # Set starting positions
            self.start_positions = [3, 4, 5]
            
            # Set finish positions
            self.finish_positions = [15, 16, 17, 18, 19]
        
        elif track == 'B':  # Larger, more precise
            self.height = 29
            self.width = 19
            self.map = np.ones((self.height + 1, self.width + 1), dtype=np.int8)
            
            # Set illegal squares
            self.map[15:, 0] = 0
            self.map[16:, 1] = 0
            self.map[19:, 2] = 0
            self.map[21:, 3] = 0
            self.map[22:, 4] = 0
            self.map[25:, 5] = 0
            self.map[27:, 6] = 0
            self.map[27:, 7] = 0
            self.map[28:, 8] = 0
            self.map[28:, 9] = 0
            self.map[28:, 10] = 0
            self.map[28:, 11] = 0
            self.map[28:, 12] = 0
            self.map[29:, 13] = 0
            self.map[29:, 14] = 0
            self.map[29:, 15] = 0
            self.map[29:, 16] = 0
            self.map[29:, 17] = 0
            self.map[29:, 18] = 0
            self.map[29:, 19] = 0
            
            self.map[:11, 0] = 0
            self.map[:7, 1] = 0
            self.map[:4, 2] = 0
            
            self.map[:14, 6] = 0
            self.map[:17, 7] = 0
            self.map[:19, 8] = 0
            self.map[:21, 9] = 0
            self.map[:22, 10] = 0
            self.map[:22, 11] = 0
            self.map[:23, 12] = 0
            self.map[:23, 13] = 0
            self.map[:24, 14] = 0
            self.map[:25, 15] = 0
            self.map[:26, 16] = 0
            self.map[:26, 17] = 0
            self.map[:27, 18] = 0
            self.map[:27, 19] = 0
            
            # Set starting positions
            self.start_positions = [3, 4, 5]
            
            # Set finish positions
            self.finish_positions = [28, 29]
        
        elif track == 'C':  # Hook, negative velocity required
            self.height = 29
            self.width = 19
            self.map = np.ones((self.height + 1, self.width + 1), dtype=np.int8)
            
            # Set illegal squares
            self.map[15:, 0] = 0
            self.map[16:, 1] = 0
            self.map[19:, 2] = 0
            self.map[21:, 3] = 0
            self.map[22:, 4] = 0
            self.map[25:, 5] = 0
            self.map[27:, 6] = 0
            self.map[27:, 7] = 0
            self.map[28:, 8] = 0
            self.map[29:, 9] = 0
            self.map[29:, 10] = 0
            self.map[29:, 11] = 0
            self.map[29:, 12] = 0
            self.map[29:, 13] = 0
            self.map[29:, 14] = 0
            self.map[28:, 15] = 0
            self.map[26:, 16] = 0
            self.map[24:, 17] = 0
            self.map[23:, 18] = 0
            self.map[22:, 19] = 0
            
            self.map[:11, 0] = 0
            self.map[:7, 1] = 0
            self.map[:4, 2] = 0
            
            self.map[:14, 6] = 0
            self.map[:17, 7] = 0
            self.map[:19, 8] = 0
            self.map[:21, 9] = 0
            self.map[:22, 10] = 0
            self.map[:23, 11] = 0
            self.map[:23, 12] = 0
            self.map[:23, 13] = 0
            self.map[:23, 14] = 0
            self.map[:21, 15] = 0
            self.map[:19, 16] = 0
            self.map[:17, 17] = 0
            self.map[:16, 18] = 0
            self.map[:15, 19] = 0
            
            # Set starting positions
            self.start_positions = [3, 4, 5]
            
            # Set finish positions
            self.finish_positions = [15, 22]
        
        elif track == 'D':  # Here be dragons
            self.height = 29
            self.width = 19
            self.map = np.ones((self.height + 1, self.width + 1), dtype=np.int8)
            
            # Set illegal squares
            self.map[15:, 0] = 0
            self.map[16:, 1] = 0
            self.map[19:, 2] = 0
            self.map[21:, 3] = 0
            self.map[22:, 4] = 0
            self.map[25:, 5] = 0
            self.map[27:, 6] = 0
            self.map[27:, 7] = 0
            self.map[28:, 8] = 0
            self.map[29:, 9] = 0
            self.map[29:, 10] = 0
            self.map[29:, 11] = 0
            self.map[29:, 12] = 0
            self.map[29:, 13] = 0
            self.map[29:, 14] = 0
            self.map[28:, 15] = 0
            self.map[26:, 16] = 0
            self.map[24:, 17] = 0
            self.map[23:, 18] = 0
            self.map[21:, 19] = 0
            
            self.map[:11, 0] = 0
            self.map[:7, 1] = 0
            self.map[:4, 2] = 0
            
            self.map[:14, 6] = 0
            self.map[:17, 7] = 0
            self.map[:5, 8] = 0
            self.map[:4, 9] = 0
            self.map[:3, 10] = 0
            self.map[:3, 11] = 0
            self.map[:3, 12] = 0
            self.map[:2, 13] = 0
            self.map[:2, 14] = 0
            self.map[:1, 15] = 0
            self.map[:1, 16] = 0
            self.map[:1, 17] = 0
            self.map[:1, 18] = 0
            self.map[:1, 19] = 0
            
            self.map[12:18, 8] = 0
            self.map[13:20, 9] = 0
            self.map[15:21, 10] = 0
            self.map[16:22, 11] = 0
            self.map[16:22, 12] = 0
            self.map[16:21, 13] = 0
            self.map[17:20, 14] = 0
            
            self.map[7:10, 13] = 0
            self.map[7:11, 14] = 0
            self.map[6:12, 15] = 0
            self.map[6:13, 16] = 0
            self.map[5:13, 17] = 0
            self.map[5:14, 18] = 0
            self.map[4:16, 19] = 0
            
            # Set starting positions
            self.start_positions = [3, 4, 5]
            
            # Set finish positions
            self.finish_positions = [1, 2, 3]
            
            
class Agent(object):
    def __init__(self, Track):
        self.track = Track
        
        # Hyperparameters
        self.alpha = 0.9
        self.epsilon = 0.9
        self.gamma = 0.9
        self.max_speed = 5
        
        # Action space restricted to (-1, 0, 1) in horizontal and vertical directions
        self.actions = [np.array([x, y]) for x in range(-1, 2) for y in range(-1, 2)]
        
        # State space is (pos[0], pos[1], vel[0], vel[1])
        self.states = np.empty((Track.map.shape[0], Track.map.shape[1], (self.max_speed * 2) + 1, (self.max_speed * 2) + 1), dtype=object)
        for i in range(self.states.shape[0]):
            for j in range(self.states.shape[1]):
                for k in range(self.states.shape[2]):
                    for l in range(self.states.shape[3]):
                        if j == self.track.width and i in self.track.finish_positions:
                            self.states[i, j, k, l] = State(self.epsilon, terminal=True)
                        else:
                            self.states[i, j, k, l] = State(self.epsilon, terminal=False)

        self.position = np.array([0, np.random.choice(self.track.start_positions)])
        self.velocity = np.zeros(2, dtype=np.int8)
        
        # Flags
        self.evaluate = False
        self.noise = True
        self.finished = False
        self.success_e = False
        self.restart = True
        
        # Noise
        self.noise_prob = 0.1
        
        # Episode logging
        self.rewards_e = []
        self.history_p = []
        self.n_restarts = 0
        
    def take_step(self, state):
        action = state.get_action(self.evaluate)
        if self.evaluate and not np.any(self.velocity) and action == 4:
            action = state.get_action(self.evaluate, stall=True)
            
        self.update_velocity(self.actions[action])
        velocity = copy.copy(self.velocity)
        
        # Update position
        while np.any(velocity):
            if velocity[0] > 0:
                self.position[0] += 1
                velocity[0] -= 1
            elif velocity[0] < 0:
                self.position [0] -= 1
                velocity[0] += 1
                
            if velocity[1] > 0:
                self.position[1] += 1
                velocity[1] -= 1
            elif velocity[1] < 0:
                self.position[1] -= 1
                velocity[1] += 1
            
            if not self.check():
                break
        
        return action
        
    def update_velocity(self, d_vel):
        """
        d_vel : [Horizontal, Vertical] change in velocity
        """
        if self.noise:
            if not self.restart:
                if np.random.uniform() > self.noise_prob:
                    self.velocity = np.clip(self.velocity + d_vel, -self.max_speed, self.max_speed)
            else:
                self.velocity = np.clip(self.velocity + d_vel, -self.max_speed, self.max_speed)
                self.restart = False
        else:
            self.velocity = np.clip(self.velocity + d_vel, -self.max_speed, self.max_speed)
    
    def check_velocity_update(self, d_vel):
        velocity_check = np.clip(self.velocity + np.array(d_vel), -self.max_speed, self.max_speed)
        if np.all(velocity_check == 0):
            return True
        else:
            return False
        
    def check(self):            
        # Check if position is illegal
        if np.any(self.position < 0):
            self.fail()
            return False
            
        elif self.position[0] > self.track.height:
            self.fail()
            return False
            
        elif self.position[1] > self.track.width:
            self.fail()
            return False
            
        elif self.track.map[self.position[0], self.position[1]] == 0:
            self.fail()
            return False
        
        # Else, check if success
        elif self.position[1] == self.track.width and self.position[0] in self.track.finish_positions:
            self.success()
            return False
        
        # Else, continue episode
        else: 
            self.rewards_e.append(-1)
            return True
        
    def fail(self):
        self.rewards_e.append(-1)
        self.reset()
        
        if self.evaluate:
            self.finished = True
        else:
            self.n_restarts += 1
    
    def success(self):
        self.success_e = True
        self.rewards_e.append(0)
        self.finished = True
        
    def reset(self):
        self.position = np.array([0, np.random.choice(self.track.start_positions)])
        self.velocity = np.zeros(2, dtype=np.int8)
        self.restart = True
    
    def generate_episode(self, evaluate=None):
        self.n_restarts = 0
        self.reset()
        if evaluate is not None:
            self.position = copy.copy(evaluate)
        self.success_e = False
        self.finished = False
        self.rewards_e = [-1]
        self.history_p = [copy.copy(self.position)]  # Copy is necessary
        
        state = self.states[self.position[0], self.position[1], self.velocity[0] + self.max_speed, self.velocity[1] + self.max_speed]
        
        # Logging
        if self.evaluate: 
            print('---')
            print(evaluate, state.get_action(self.evaluate))
            print(state.Q)
        
        # Episode generation
        while not self.finished: 
            action = self.take_step(state)
            reward = self.rewards_e[-1]
            state_prime = self.states[self.position[0], self.position[1], self.velocity[0] + self.max_speed, self.velocity[1] + self.max_speed]
            
            if not self.evaluate:
                # Update Q(S, A) with expectation
                expectation = state_prime.pi @ state_prime.Q
                update = reward + self.gamma * expectation - state.Q[action]
                state.Q[action] += self.alpha * update
                
                # Update behavioral policy with random tiebreaks
                max_Q = np.random.choice(np.flatnonzero(state.Q == state.Q.max()))
                for i in range(9):
                    if i == max_Q:
                        state.pi[i] = 1 - self.epsilon + (self.epsilon / 9)
                    else:
                        state.pi[i] = self.epsilon / 9
            
            self.history_p.append(copy.copy(self.position))
            state = self.states[self.position[0], self.position[1], self.velocity[0] + self.max_speed, self.velocity[1] + self.max_speed]
            if state.terminal:
                self.rewards_e = self.rewards_e[:-1]
                self.finished = True         
        
        
class State(object):
    def __init__(self, epsilon, terminal=False):
        self.terminal = terminal
        
        # Initialize Q(s, a) as zero for all actions in all states
        self.Q = np.array([0. for a in range(9)])
        
        # Epsilon-greedy behavioral policy with random initialization
        self.pi = np.empty(9) 
        max_Q = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
        for i in range(9):
            if i == max_Q:
                self.pi[i] = 1 - epsilon + (epsilon / 9)
            else:
                self.pi[i] = epsilon / 9
        
    def get_action(self, evaluate=False, stall=False):
        if not evaluate:
            return np.random.choice(range(9), p=self.pi)
        else:
            if not stall:
                return np.argmax(self.pi)  # Evaluate with greedy policy
            else:
                for a in np.random.choice(range(9), p=self.pi, replace=False, size=9):
                    if a != 4:
                        return a
        
        
if __name__ == '__main__':
    # Command: python racetrack_expected_sarsa.py [track]
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
