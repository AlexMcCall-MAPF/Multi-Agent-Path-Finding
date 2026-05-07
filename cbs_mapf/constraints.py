#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com
'''
from typing import Dict, Tuple, Set
from copy import deepcopy

from .agent import Agent

'''
Emulated dictionary of dictionaries for vertex and edge constraints
'''
class Constraints:

    def __init__(self):
        # Vertex constraints: Agent -> {time -> {forbidden_positions}}
        self.agent_constraints: Dict[Agent: Dict[int, Set[Tuple[int, int]]]] = dict()
        # Edge constraints: Agent -> {time -> {forbidden_edges}} where edge = ((x1,y1), (x2,y2))
        self.agent_edge_constraints: Dict[Agent: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]]] = dict()

    '''
    Deepcopy self with additional vertex constraints
    '''
    def fork(self, agent: Agent, obstacle: Tuple[int, int], start: int, end: int) -> 'Constraints':
        agent_constraints_copy = deepcopy(self.agent_constraints)
        agent_edge_constraints_copy = deepcopy(self.agent_edge_constraints)
        for time in range(start, end):
            agent_constraints_copy.setdefault(agent, dict()).setdefault(time, set()).add(obstacle)
        new_constraints = Constraints()
        new_constraints.agent_constraints = agent_constraints_copy
        new_constraints.agent_edge_constraints = agent_edge_constraints_copy
        return new_constraints

    '''
    Deepcopy self with additional edge constraints
    '''
    def fork_edge(self, agent: Agent, from_pos: Tuple[int, int], to_pos: Tuple[int, int], start: int, end: int) -> 'Constraints':
        agent_constraints_copy = deepcopy(self.agent_constraints)
        agent_edge_constraints_copy = deepcopy(self.agent_edge_constraints)
        edge = (from_pos, to_pos)
        for time in range(start, end):
            agent_edge_constraints_copy.setdefault(agent, dict()).setdefault(time, set()).add(edge)
        new_constraints = Constraints()
        new_constraints.agent_constraints = agent_constraints_copy
        new_constraints.agent_edge_constraints = agent_edge_constraints_copy
        return new_constraints

    def setdefault(self, key, default):
        return self.agent_constraints.setdefault(key, default)

    def __getitem__(self, agent):
        return self.agent_constraints[agent]

    def __iter__(self):
        for key in self.agent_constraints:
            yield key

    def __str__(self):
        return str(self.agent_constraints)


