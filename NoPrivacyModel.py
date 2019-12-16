from mesa import Agent, Model
from mesa.time import RandomActivation
from typing import Tuple

import numpy as np
from scipy.stats import norm

class Transaction():
    """A transaction with uncertainty and payoffs"""
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_payout(self):
        """Generates payout from the transaction based on transaction's parameters"""
        return self.sigma * np.random.randn() + self.mu

class Patrician(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'patrician'
        self.wealth = 0
        self.num_transactions = 0

    def accepts(self, t: Transaction) -> bool:
        """Decides whether transaction is promising enough to participate in it

        Params:
            t: proposed transaction
        Returns:
            True if the probability of positive the payout is greater than acceptance threshold, False otherwise
        """
        expectation = norm(t.mu, t.sigma)
        if expectation.ppf(1-self.model.alpha) > 0:
            return True
        else:
            return False

    def step(self):
        """Performs a single step of the simulation"""

        # find an agent with whom to do transaction
        other_agent = self.random.choice(self.model.schedule.agents)

        # if transactions are allowed only between agents of two different types
        if self.model.symmetric:
            while other_agent.type != 'plebeian':
                other_agent = self.random.choice(self.model.schedule.agents)

        # create a transaction
        mu = np.random.uniform(self.model.mu_min, self.model.mu_max)
        sigma = self.model.sigma
        t = Transaction(mu, sigma)

        if self.accepts(t):
            if other_agent.accepts(t):
                self.num_transactions += 1
                self.wealth += t.sigma * np.random.randn() + t.mu
                other_agent.wealth += t.sigma * np.random.randn() + t.mu


class Plebeian(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'plebeian'
        self.wealth = 0
        self.num_transactions = 0

    def accepts(self, t: Transaction) -> bool:
        """Decides whether transaction is promising enough to participate in it. In contrast to Patricians,
        who have full knowledge of transaction parameters, Plebeians have skewed and biased knowledge
        about expected payouts

                Params:
                    t: proposed transaction
                Returns:
                    True if the probability of positive the payout is greater than acceptance threshold, False otherwise
        """

        # true percentile of distribution
        true_threshold = norm(t.mu, t.sigma).ppf(1-self.model.alpha)
        # what a plebeian sees
        visible_threshold = np.random.uniform(true_threshold-self.model.beta, true_threshold+self.model.gamma)

        if visible_threshold > 0:
            self.num_transactions += 1
            return True
        else:
            return False


class TransactionModel(Model):
    """A model with some number of agents."""
    def __init__(self,
                 n_plebeians: int,
                 n_patricians: int,
                 mu_range: Tuple,
                 sigma: float,
                 alpha: float = 0.5,
                 beta: float = 0,
                 gamma: float = 1,
                 symmetric: bool = False):

        self.n_plebeians = n_plebeians
        self.n_patricians = n_patricians
        self.mu_min, self.mu_max = mu_range
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.symmetric = symmetric

        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.n_plebeians):
            a = Plebeian(i, self)
            self.schedule.add(a)

        for i in range(self.n_patricians):
            a = Patrician(i+n_plebeians, self)
            self.schedule.add(a)

    def step(self):
        """Advance model by one step"""
        self.schedule.step()