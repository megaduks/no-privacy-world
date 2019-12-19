from mesa import Agent, Model
from mesa.time import RandomActivation
from typing import Tuple

import numpy as np
from scipy.stats import norm


def create_offer(model: Model) -> norm:
    """Creates a normal distribution based on model's parameters

    Args:
        model: model holding the state of the simulation
    """
    mu = np.random.uniform(model.mu_min, model.mu_max)
    return norm(mu, model.sigma)


class Citizen(Agent):
    """A generic agent with fixed initial wealth and number of transactions

        Args:
            unique_id: unique identifier assigned to each agent
            model: model holding the state of the simulation

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
            offer: distribution, from which payout will be drawn
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

        # draw offers for the patrician and the plebeian
        offer_for_patrician = create_offer(self.model)
        offer_for_plebeian = create_offer(self.model)

        # create distorted offer for the plebeian
        mu = np.random.uniform(offer_for_plebeian.mean(), offer_for_plebeian.mean() * self.model.beta)
        distorted_offer_for_plebeian = norm(mu, self.model.sigma)

        if self.accepts(offer_for_patrician):
            if other_agent.accepts(distorted_offer_for_plebeian):
                self.num_transactions += 1
                other_agent.num_transactions += 1

                self.wealth += offer_for_patrician.rvs()
                other_agent.wealth += offer_for_plebeian.rvs()
            else:
                other_agent.num_rejections += 1
        else:
            self.num_rejections += 1


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