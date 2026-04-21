#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com
'''
from typing import Dict, List
import numpy as np

from .agent import Agent
from .constraints import Constraints

class CTNode:

    def __init__(self, constraints: Constraints,
                       solution: Dict[Agent, np.ndarray],
                       agent_start_times: Dict[int, int] = None):

        self.constraints = constraints
        self.solution = solution
        self.cost = self.sic(solution)
        # Store agent ID -> start_time mapping for pickling safety
        if agent_start_times is None:
            self.agent_start_times = {agent.agent_id: agent.start_time for agent in solution.keys()}
        else:
            self.agent_start_times = agent_start_times

    # Sum-of-Individual-Costs heuristics
    @staticmethod
    def sic(solution):
        return sum(len(sol) for sol in solution.items())

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return str(self.constraints.agent_constraints)

