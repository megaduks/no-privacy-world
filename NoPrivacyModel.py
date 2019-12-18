from mesa import Agent, Model
from mesa.time import RandomActivation
from typing import Tuple

import numpy as np
from scipy.stats import norm

class Transaction():
    """A transaction with uncertainty and payoffs

    Args:
        mu: expected value of the transaction
        sigma: standard deviation of the transaction
    """
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def get_offer(self) -> norm:
        """Creates an offer"""
        return norm(self.mu, self.sigma)

    def get_payoff(self, offer: norm) -> float:
        """Generates payout from the offer

        Params:
            offer: a distribution from which the payoff will be drawn
        Returns:
            random value drawn from the normal distribution described by the offer
        """
        return offer.rvs()


class Citizen(Agent):
    """A generic agent with fixed initial wealth and number of transactions

        Args:
            unique_id: unique identifier assigned to each agent
            model: reference to the model holding the state of the simulation

        Attributes:
            wealth (float): cummulative wealth obtained by the agent
            num_transactions (int): number of transactions conducted by the agent
            num_rejections (int): number of transactions rejected by the agent
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 0
        self.num_transactions = 0
        self.num_rejections = 0

    def accepts(self, offer: norm) -> bool:
        """Decides whether transaction is promising enough to participate in it

        Params:
            offer: distribution, from which payoff will be drawn
        Returns:
            True if the probability of positive the payout is greater than acceptance threshold, False otherwise
        """
        if offer.ppf(1-self.model.alpha) > 0:
            return True
        else:
            return False


class Patrician(Citizen):
    """An agent representing the Patrician group

        Args:
            unique_id: unique identifier assigned to each agent
            model: reference to the model holding the state of the simulation
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'patrician'

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

        pat_true = t.get_offer()
        pleb_true = t.get_offer()

        pleb_mu = np.random.uniform(pleb_true.mean(), pleb_true.mean() * self.model.beta)
        pleb_offer = norm(pleb_mu, sigma)

        if self.accepts(pat_true):
            if other_agent.accepts(pleb_offer):
                self.num_transactions += 1
                other_agent.num_transactions += 1

                self.wealth += t.get_payoff(pat_true)
                other_agent.wealth += t.get_payoff(pleb_true)
            else:
                self.num_rejections += 1
                other_agent.num_rejections += 1


class Plebeian(Citizen):
    """An agent which represents the Plebeian group

        Args:
            unique_id: unique identifier assigned to each agent
            model: reference to the model holding the state of the simulation
    """
    def __init__(self, unique_id: int, model: Model):
        super().__init__(unique_id, model)
        self.type = 'plebeian'


class TransactionModel(Model):
    """A model with some number of agents.

    Args:
        n_plebeians: number of agents representing the Plebeian class
        n_patricians: number of agents representing the Patrician class
        mu_range: range of expected value of the transaction
        sigma: standard deviation of the transaction
        alpha: threshold for the acceptance of a transaction
        beta: distortion of the expected value of transaction observed by the Plebeians
        symmetric: if True, transactions are allowed only between Patricians and Plebeians
    """
    def __init__(self,
                 n_plebeians: int,
                 n_patricians: int,
                 mu_range: Tuple,
                 sigma: float,
                 alpha: float = 0.5,
                 beta: float = 0,
                 symmetric: bool = False):

        self.n_plebeians = n_plebeians
        self.n_patricians = n_patricians
        self.mu_min, self.mu_max = mu_range
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
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