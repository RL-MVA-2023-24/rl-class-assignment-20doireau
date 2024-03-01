from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from joblib import dump, load
#from tqdm import tqdm

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def __init__(self, env):
        self.Q = None
        self.env = env
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        self.Q_list = []

    def greedy_action(self,s):
        Qsa = []
        nb_actions = self.env.action_space.n
        for a in range(nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)
    
    def greedy_action_ensemble(self,s):
        Qsa_mean = []
        nb_actions = self.env.action_space.n
        for Q in self.Q_list:
          Qsa = []
          for a in range(nb_actions):
              sa = np.append(s,a).reshape(1, -1)
              Qsa.append(Q.predict(sa))
          Qsa_mean.append(Qsa)
        Qsa_mean = np.mean(Qsa_mean, axis=0)
        return np.argmax(Qsa_mean)

    def collect_samples(self, horizon, epsilon=0.0, print_done_states=False):
        s = self.env.reset()[0]
        for _ in range(horizon):
            if np.random.rand() < epsilon:
                a = self.greedy_action(s)
            else:
                a = self.env.action_space.sample()

            s2, r, done, trunc, _ = self.env.step(a)
            self.S.append(s)
            self.A.append(a)
            self.R.append(r)
            self.S2.append(s2)
            self.D.append(done)

            if done or trunc:
                s = self.env.reset()[0]
                if done and print_done_states:
                    print("done!")
            else:
                s = s2

    def FQI(self, iterations, gamma=0.9):
        nb_actions = self.env.action_space.n
        nb_samples = len(self.S)

        S = np.array(self.S)
        A = np.array(self.A).reshape((-1, 1))
        SA = np.append(S, A, axis=1)
        Q = self.Q

        for iter in range(iterations):
            if iter == 0 and Q is None:
                value = self.R.copy()
            else:
                Q2 = np.zeros((nb_samples, nb_actions))
                for a2 in range(nb_actions):
                    A2 = np.full((len(self.S), 1), a2)
                    S2A2 = np.append(self.S2, A2, axis=1)
                    Q2[:, a2] = Q.predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = np.array(self.R) + gamma * (1 - np.array(self.D)) * max_Q2

            Q = ExtraTreesRegressor(n_estimators=50, n_jobs=-1)
            #Q = GradientBoostingRegressor(n_estimators=50)
            Q.fit(SA, value)
        self.Q = Q

    def train(self, horizon, n_update=500, nb_iterations=30, nb_steps=17):
        self.collect_samples(horizon)
        self.FQI(nb_iterations)

        for step in range(nb_steps):
            self.collect_samples(n_update, epsilon=0.85)
            self.FQI(nb_iterations)

        return self.Q
    
    def train_ensemble(self, horizon, n_ensemble = 5, n_update=500, nb_iterations=30, nb_steps=17):
        for _ in range(n_ensemble):
          self.S = []
          self.A = []
          self.R = []
          self.S2 = []
          self.D = []
          self.collect_samples(horizon)
          self.Q = self.FQI(nb_iterations)

          for step in range(nb_steps):
              self.collect_samples(n_update, epsilon=0.85)
              self.Q = self.FQI(nb_iterations)

          self.Q_list.append(self.Q)

        #self.save('listQmodels.joblib')

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.env.action_space.n)
        else:
            return self.greedy_action(observation)


    def save(self, path):
        self.path=path
        dump(self.Q, path)

    def load(self):
        self.Q = load('bestmodel.joblib')

