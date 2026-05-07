#!/usr/bin/env python3
'''
Author: Haoran Peng
Email: gavinsweden@gmail.com

An implementation of multi-agent path finding using conflict-based search
[Sharon et al., 2015]
'''
from typing import List, Tuple, Dict, Callable, Set
import multiprocessing as mp
from heapq import heappush, heappop
from itertools import combinations
from copy import deepcopy
import numpy as np

# The low level planner for CBS is the Space-Time A* planner
# https://github.com/GavinPHR/Space-Time-AStar
from stastar.planner import Planner as STPlanner

from .constraint_tree import CTNode
from .constraints import Constraints
from .agent import Agent
from .assigner import *
class Planner:

    def __init__(self, grid_size: int,
                       robot_radius: int,
                       static_obstacles: List[Tuple[int, int]]):

        self.robot_radius = robot_radius
        self.st_planner = STPlanner(grid_size, robot_radius, static_obstacles)

    '''
    You can use your own assignment function, the default algorithm greedily assigns
    the closest goal to each start.
    '''
    def plan(self, starts: List[Tuple[int, int]],
                   goals: List[Tuple[int, int]],
                   start_times: List[int] = None,
                   assign:Callable = min_cost,
                   max_iter:int = 200,
                   low_level_max_iter:int = 100,
                   max_process:int = 10,
                   debug:bool = False) -> np.ndarray:

        self.low_level_max_iter = low_level_max_iter
        self.debug = debug

        # Handle start_times parameter
        if start_times is None:
            start_times = [0] * len(starts)
        
        if len(start_times) != len(starts):
            raise ValueError(f"start_times length ({len(start_times)}) must match starts length ({len(starts)})")

        # Do goal assignment
        self.agents = assign(starts, goals)
        
        # Apply start times to agents
        for agent, start_time in zip(self.agents, start_times):
            agent.start_time = start_time


        constraints = Constraints()

        # Compute path for each agent using low level planner
        solution = dict((agent, self.calculate_path(agent, constraints, None)) for agent in self.agents)

        open = []
        if all(len(path) != 0 for path in solution.values()):
            # Make root node with agent start_times mapping
            agent_start_times = {agent.agent_id: agent.start_time for agent in self.agents}
            node = CTNode(constraints, solution, agent_start_times)
            # Min heap for quick extraction
            open.append(node)

        manager = mp.Manager()
        iter_ = 0
        while open and iter_ < max_iter:
            iter_ += 1

            results = manager.list([])

            processes = []

            # Default to 10 processes maximum
            for _ in range(max_process if len(open) > max_process else len(open)):
                p = mp.Process(target=self.search_node, args=[heappop(open), results])
                processes.append(p)
                p.start()

            for p in processes:
                p.join()

            for result in results:
                if len(result) == 1:
                    if debug:
                        print('CBS_MAPF: Paths found after about {0} iterations'.format(4 * iter_))
                    return result[0]
                if result[0]:
                    heappush(open, result[0])
                if result[1]:
                    heappush(open, result[1])

        if debug:
            print('CBS-MAPF: Open set is empty, no paths found.')
        return np.array([])

    '''
    Abstracted away the cbs search for multiprocessing.
    The parameters open and results MUST BE of type ListProxy to ensure synchronization.
    '''
    def search_node(self, best: CTNode, results):
        # Restore agent start_times from CTNode mapping (they may be lost during pickling)
        for agent in best.solution.keys():
            if agent.agent_id in best.agent_start_times:
                agent.start_time = best.agent_start_times[agent.agent_id]
        
        # Also restore self.agents (they're also pickled)
        for agent in self.agents:
            if agent.agent_id in best.agent_start_times:
                agent.start_time = best.agent_start_times[agent.agent_id]
        
        agent_i, agent_j, conflict_info = self.validate_paths(self.agents, best)

        # If there is not conflict, validate_paths returns (None, None, (-1, ''))
        if agent_i is None:
            results.append((self.reformat(self.agents, best.solution),))
            return
        
        time_of_conflict, conflict_type = conflict_info
        
        # Debug: Print conflict information
        if self.debug:
            print(f'CBS: {conflict_type.upper()} CONFLICT detected between Agent {agent_i.agent_id} and Agent {agent_j.agent_id} at time {time_of_conflict}')
        
        # Calculate new constraints
        agent_i_constraint = self.calculate_constraints(best, agent_i, agent_j, time_of_conflict, conflict_type)
        agent_j_constraint = self.calculate_constraints(best, agent_j, agent_i, time_of_conflict, conflict_type)

        # Calculate new paths
        agent_i_path = self.calculate_path(agent_i,
                                           agent_i_constraint,
                                           self.calculate_goal_times(best, agent_i, self.agents))
        agent_j_path = self.calculate_path(agent_j,
                                           agent_j_constraint,
                                           self.calculate_goal_times(best, agent_j, self.agents))

        # Debug: Print which agents failed to find paths
        if self.debug:
            if len(agent_i_path) == 0:
                print(f'CBS: Agent {agent_i.agent_id} failed to find constrained path')
            if len(agent_j_path) == 0:
                print(f'CBS: Agent {agent_j.agent_id} failed to find constrained path')

        # Replace old paths with new ones in solution
        solution_i = best.solution
        solution_j = deepcopy(best.solution)
        solution_i[agent_i] = agent_i_path
        solution_j[agent_j] = agent_j_path

        node_i = None
        if all(len(path) != 0 for path in solution_i.values()):
            # Preserve agent_start_times in new node
            node_i = CTNode(agent_i_constraint, solution_i, best.agent_start_times)

        node_j = None
        if all(len(path) != 0 for path in solution_j.values()):
            # Preserve agent_start_times in new node
            node_j = CTNode(agent_j_constraint, solution_j, best.agent_start_times)

        results.append((node_i, node_j))


    '''
    Pair of agent, point of conflict, conflict type
    '''
    def validate_paths(self, agents, node: CTNode):
        # Check collision pair-wise
        for agent_i, agent_j in combinations(agents, 2):
            time_of_conflict, conflict_type = self.safe_distance(node.solution, agent_i, agent_j)
            # time_of_conflict=-1 if there is not conflict
            if time_of_conflict == -1:
                continue
            return agent_i, agent_j, (time_of_conflict, conflict_type)
        return None, None, (-1, '')


    def safe_distance(self, solution: Dict[Agent, np.ndarray], agent_i: Agent, agent_j: Agent) -> Tuple[int, str]:
        # Returns (time_of_conflict, conflict_type) where conflict_type is 'vertex' or 'edge'
        # Returns (-1, '') if no conflict
        start_i = agent_i.start_time
        start_j = agent_j.start_time
        end_i = start_i + len(solution[agent_i])
        end_j = start_j + len(solution[agent_j])
        
        # Check all absolute times where both agents exist
        for abs_time in range(max(start_i, start_j), min(end_i, end_j)):
            idx_i = abs_time - start_i
            idx_j = abs_time - start_j
            point_i = solution[agent_i][idx_i]
            point_j = solution[agent_j][idx_j]
            
            # Vertex conflict: same position at same time
            if self.dist(point_i, point_j) <= 2*self.robot_radius:
                if self.debug:
                    print(f'    COLLISION at time {abs_time}: Agent {agent_i.agent_id} at {point_i} vs Agent {agent_j.agent_id} at {point_j}')
                return abs_time, 'vertex'
        
        # Check for edge conflicts (head-on collisions)
        # Edge conflict: Agent i goes A->B, Agent j goes B->A at same time
        for abs_time in range(max(start_i, start_j) + 1, min(end_i, end_j)):
            idx_i = abs_time - start_i
            idx_j = abs_time - start_j
            
            # Get positions at current and previous times
            pos_i_prev = solution[agent_i][idx_i - 1]
            pos_i_curr = solution[agent_i][idx_i]
            pos_j_prev = solution[agent_j][idx_j - 1]
            pos_j_curr = solution[agent_j][idx_j]
            
            # Check if agents swapped positions (A->B and B->A)
            if (np.array_equal(pos_i_prev, pos_j_curr) and np.array_equal(pos_i_curr, pos_j_prev)):
                if self.debug:
                    print(f'    EDGE CONFLICT at time {abs_time}: Agent {agent_i.agent_id} {pos_i_prev}->{pos_i_curr} vs Agent {agent_j.agent_id} {pos_j_prev}->{pos_j_curr}')
                return abs_time, 'edge'
        
        return -1, ''

    @staticmethod
    def dist(point1: np.ndarray, point2: np.ndarray) -> int:
        return int(np.linalg.norm(point1-point2, 2))  # L2 norm

    def calculate_constraints(self, node: CTNode,
                                    constrained_agent: Agent,
                                    unchanged_agent: Agent,
                                    time_of_conflict: int,
                                    conflict_type: str = 'vertex') -> Constraints:
        constrained_path = node.solution[constrained_agent]
        unchanged_path = node.solution[unchanged_agent]

        # Convert absolute time to path indices
        idx_conflict_constrained = time_of_conflict - constrained_agent.start_time
        idx_conflict_unchanged = time_of_conflict - unchanged_agent.start_time
        
        if conflict_type == 'vertex':
            # Vertex conflict: agent cannot be at this position during this time range
            pivot = unchanged_path[idx_conflict_unchanged]
            conflict_end_time = time_of_conflict
            try:
                while idx_conflict_constrained < len(constrained_path) and \
                      self.dist(constrained_path[idx_conflict_constrained], pivot) < 2*self.robot_radius:
                    conflict_end_time += 1
                    idx_conflict_constrained += 1
            except IndexError:
                pass
            return node.constraints.fork(constrained_agent, tuple(pivot.tolist()), time_of_conflict, conflict_end_time)
        
        elif conflict_type == 'edge':
            # Edge conflict: agent cannot traverse this edge during this time range
            # At time_of_conflict, unchanged_agent is at position B
            # At time_of_conflict-1, unchanged_agent was at position A
            idx_prev = idx_conflict_unchanged - 1
            if idx_prev >= 0:
                from_pos = tuple(unchanged_path[idx_prev].tolist())
                to_pos = tuple(unchanged_path[idx_conflict_unchanged].tolist())
                # Constrain the reverse edge (to_pos -> from_pos)
                conflict_end_time = time_of_conflict + 1
                return node.constraints.fork_edge(constrained_agent, to_pos, from_pos, time_of_conflict, conflict_end_time)
            else:
                # Fallback to vertex constraint if can't extract edge
                pivot = unchanged_path[idx_conflict_unchanged]
                return node.constraints.fork(constrained_agent, tuple(pivot.tolist()), time_of_conflict, time_of_conflict + 1)
        
        # Default fallback
        pivot = unchanged_path[idx_conflict_unchanged]
        return node.constraints.fork(constrained_agent, tuple(pivot.tolist()), time_of_conflict, time_of_conflict + 1)

    def calculate_goal_times(self, node: CTNode, agent: Agent, agents: List[Agent]):
        solution = node.solution
        goal_times = dict()
        for other_agent in agents:
            if other_agent == agent:
                continue
            if len(solution[other_agent]) == 0:
                continue
            relative_time = len(solution[other_agent]) - 1
            absolute_time = other_agent.start_time + relative_time
            goal_times.setdefault(absolute_time, set()).add(tuple(solution[other_agent][relative_time]))
        return goal_times

    '''
    Calculate the paths for all agents with space-time constraints
    '''
    def calculate_path(self, agent: Agent, 
                       constraints: Constraints, 
                       goal_times: Dict[int, Set[Tuple[int, int]]]) -> np.ndarray:
        if self.debug:
            agent_constraints = constraints.agent_constraints.get(agent, {})
            agent_edge_constraints = constraints.agent_edge_constraints.get(agent, {})
            total_constraints = sum(len(obstacles) for obstacles in agent_constraints.values())
            total_edge_constraints = sum(len(edges) for edges in agent_edge_constraints.values())
            total_goal_obstacles = sum(len(obstacles) for obstacles in goal_times.values()) if goal_times else 0
            print(f'[calculate_path] Agent {agent.agent_id}: start={agent.start}, goal={agent.goal}, start_time={agent.start_time}, explicit_constraints={total_constraints}, edge_constraints={total_edge_constraints}, goal_time_obstacles={total_goal_obstacles}')
        
        # Prepare edge constraints for ST-A*
        edge_constraints = constraints.agent_edge_constraints.get(agent, {})
        
        path = self.st_planner.plan(agent.start, 
                                    agent.goal, 
                                    constraints.setdefault(agent, dict()), 
                                    semi_dynamic_obstacles=goal_times,
                                    edge_constraints=edge_constraints,
                                    start_time=agent.start_time,
                                    max_iter=self.low_level_max_iter, 
                                    debug=self.debug)
        
        if len(path) == 0 and self.debug:
            print(f'[calculate_path] Agent {agent.agent_id} FAILED - no path found!')
        
        return path

    '''
    Reformat the solution to a numpy array
    '''
    @staticmethod
    def reformat(agents: List[Agent], solution: Dict[Agent, np.ndarray]):
        solution = Planner.pad(solution, agents)
        reformatted_solution = []
        for agent in agents:
            reformatted_solution.append(solution[agent])
        return np.array(reformatted_solution)

    '''
    Pad paths to equal length with absolute time indexing, accounting for start times
    '''
    @staticmethod
    def pad(solution: Dict[Agent, np.ndarray], agents: List[Agent]):
        # Calculate maximum end time across all agents
        max_end_time = 0
        for agent, path in solution.items():
            agent_end_time = agent.start_time + len(path)
            max_end_time = max(max_end_time, agent_end_time)
        
        padded_solution = {}
        for agent, path in solution.items():
            if len(path) == 0:
                # If no path found, use start position throughout
                padded = np.array([agent.start] * max_end_time)
            else:
                # Prepend: start position for times before agent.start_time
                if agent.start_time > 0:
                    prepend = np.array([agent.start] * agent.start_time)
                else:
                    prepend = np.empty((0, 2))
                
                # Append: goal position for times after path ends
                append_count = max_end_time - agent.start_time - len(path)
                if append_count > 0:
                    append = np.array([path[-1]] * append_count)
                else:
                    append = np.empty((0, 2))
                
                # Concatenate all pieces
                padded = np.concatenate([prepend, path, append])
            padded_solution[agent] = padded
        
        return padded_solution

