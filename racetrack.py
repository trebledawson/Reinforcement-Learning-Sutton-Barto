import numpy as np
import time
import copy
import matplotlib.pyplot as plt

def main():
    Car = Agent(Racetrack())
    rewards = []
    for episode in range(50000):
        Car.generate_episode()
        reward = np.sum(Car.rewards_e)
        rewards.append(reward)
        if (episode + 1) % 200 == 0:
            print('.')
        if (episode + 1) % 1000 == 0:
            track = Racetrack().map
            path = Car.history_p
            print('---')
            print('Episode:', episode + 1, '| Reward:', reward, '| Steps:', len(Car.rewards_e), 'Restarts:', Car.n_restarts)
            for loc in path:
                track[loc[0], loc[1]] = 9
            print(track)
            
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.grid()
    plt.xlabel('Number of episodes')
    plt.ylabel('Reward')
    plt.title('Reward obtained over Monte Carlo control episodes')
    
    Car.noise = False
    for _ in range(5):
        Car.generate_episode()
        track = Racetrack().map
        path = Car.history_p
        for loc in path:
            track[loc[0], loc[1]] = 9
        plt.figure()
        plt.imshow(track, interpolation='none', origin='lower')
    plt.show()
    
    
class Racetrack(object):
    def __init__(self):
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
        
class Agent(object):
    def __init__(self, Track):
        self.track = Track
        self.gamma = 0.9
        self.actions = [np.array([x, y]) for x in range(-1, 2) for y in range(-1, 2)]
        self.states = np.empty((Track.map.shape[0], Track.map.shape[1]), dtype=object)
        for i in range(self.states.shape[0]):
            for j in range(self.states.shape[1]):
                self.states[i, j] = State()

        self.position = np.array([0, np.random.choice(self.track.start_positions)])
        self.velocity = np.zeros(2, dtype=np.int8)
        self.noise = True
        self.noise_prob = 0.1
        self.states_e = []
        self.actions_e = []
        self.rewards_e = []
        self.history_p = []
        self.finished = False
        self.restart = True
        self.n_restarts = 0
        
    def take_step(self, state):
        action = state.get_action()
        illegal = self.check_velocity_update(self.actions[action])
        illegal_actions = []
        while illegal:
            illegal_actions.append(action)
            action = state.get_new_action(illegal_actions)
            illegal = self.check_velocity_update(self.actions[action])
        
        self.update_velocity(self.actions[action])
        velocity = copy.copy(self.velocity)
        
        # Check if velocity is legal:
        if np.all(self.velocity == 0):
            print('zero-v')
            print(self.velocity)
            print(action)
            print(self.actions[action])
            time.sleep(10)
            self.fail()
            
        while np.any(velocity):
            if velocity[0] > 0:
                self.position[0] += 1
                velocity[0] -= 1
            if velocity[1] > 0:
                self.position[1] += 1
                velocity[1] -= 1
            
            if not self.check():
                break
        
    def update_velocity(self, d_vel):
        """
        d_vel : [Horizontal, Vertical] change in velocity
        """
        if self.noise:
            if not self.restart:
                if np.random.uniform() > self.noise_prob:
                    self.velocity = np.clip(self.velocity + d_vel, 0, 5)
            else:
                self.velocity = np.clip(self.velocity + d_vel, 0, 5)
                self.restart = False
        else:
            self.velocity = np.clip(self.velocity + d_vel, 0, 5)
        
    def check_velocity_update(self, d_vel):
        velocity_check = np.clip(self.velocity + np.array(d_vel), 0, 5)
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
        self.n_restarts += 1
    
    def success(self):
        self.rewards_e.append(0)
        self.finished = True
        
    def reset(self):
        self.position = np.array([0, np.random.choice(self.track.start_positions)])
        self.velocity = np.zeros(2, dtype=np.int8)
        self.restart = True
    
    def generate_episode(self):
        self.n_restarts = 0
        self.reset()
        self.finished = False
        self.states_e = []
        self.actions_e = []
        self.rewards_e = []
        self.history_p = [copy.copy(self.position)]
        
        while not self.finished:
            state = self.states[self.position[0], self.position[1]]
            self.take_step(state)
            self.states_e.append(state)
            self.actions_e.append(state.get_action())
            self.history_p.append(copy.copy(self.position))
            
        G = 0
        for t in range(1, len(self.states_e) + 1):
            state = self.states_e[-1]
            action = self.actions_e[-1]
            
            self.states_e = self.states_e[:-1]
            self.actions_e = self.actions_e[:-1]
            
            G = self.gamma * G + self.rewards_e[-t]
            
            if state in self.states_e and action in self.actions_e:
                continue
            else:
                state.update(G, action)
    
class State(object):
    def __init__(self):
        self.epsilon = 0.1
        self.Q = np.random.uniform(size=9)
        self.returns = [[0] for _ in range(9)]
        self.A_star = np.random.choice(range(9))
        self.pi = np.empty(9)
        for i in range(9):
            if i == self.A_star:
                self.pi[i] = 1 - self.epsilon + (self.epsilon / 9)
            else:
                self.pi[i] = self.epsilon / 9
        
    def get_action(self):
        return np.random.choice(range(9), p=self.pi)
        
    def get_new_action(self, illegal):
        actions = np.random.choice(range(9), p=self.pi, replace=False, size=9)
        for a in actions:
            if a in illegal:
                continue
            else:
                return a
    
    def update(self, G, action):
        self.returns[action].append(G)
        self.Q = [np.mean(x) for x in self.returns]
        self.A_star = np.argmax(self.Q)
        for i in range(9):
            if i == self.A_star:
                self.pi[i] = 1 - self.epsilon + (self.epsilon / 9)
            else:
                self.pi[i] = self.epsilon / 9
        
        
if __name__ == '__main__':
    main()
