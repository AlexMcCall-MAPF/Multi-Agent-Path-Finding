#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com
'''
from typing import Tuple
import numpy as np


class Agent:

    _agent_counter = 0

    def __init__(self, start: Tuple[int, int], goal: Tuple[int, int], start_time: int = 0):
        self.agent_id = Agent._agent_counter
        Agent._agent_counter += 1
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.start_time = start_time

    # Uniquely identify an agent by its immutable ID
    def __hash__(self):
        return hash(self.agent_id)

    def __eq__(self, other: 'Agent'):
        if not isinstance(other, Agent):
            return False
        return self.agent_id == other.agent_id

    def __str__(self):
        return str(self.start.tolist())

    def __repr__(self):
        return self.__str__()
