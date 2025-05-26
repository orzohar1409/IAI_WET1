import numpy as np

from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict


class Action:
    DOWN = 0
    RIGHT = 1
    UP = 2
    LEFT = 3


class StateType:
    S = "S"
    G = "G"
    H = "H"
    P = "P"
    F = "F"
    T = "T"
    A = "A"
    L = "L"


StateCost = {StateType.S: 1,
             StateType.G: 1,
             StateType.H: np.inf,
             StateType.P: 100,
             StateType.F: 10,
             StateType.T: 3,
             StateType.A: 2,
             StateType.L: 1
             }


class StateGraphNode:
    def __init__(self, state_id: int, father: 'StateGraphNode', preceded_action: Action, cost: float,
                 ) -> None:
        self.actions = None
        self.preceded_action = preceded_action
        self.state_id = state_id
        self.father = father
        self.actions = father.actions + [preceded_action] if father else []
        self.cost = cost
        #check if cost is iinf
        try:
            self.total_path_cost = father.total_path_cost + cost if father else 0
        except TypeError:
            pass


class BFSGAgent():
    def __init__(self) -> None:
        self.CLOSE = {}
        self.OPEN = []

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        init_state = env.reset()
        init_node = StateGraphNode(init_state, father=None, preceded_action=None,
                                   cost=0)  # Initialize the first node
        self.OPEN.append(init_node)

        # Run on OPEN
        while self.OPEN:
            curr = self.OPEN.pop(0)  # Get first element
            self.CLOSE[curr.state_id] = curr  # Add to close

            # Search among all successors
            for action, successor in env.succ(curr.state_id).items():
                if successor == (None, None, None):
                    continue

                successor_node = StateGraphNode(successor[0], father=curr, preceded_action=action,
                                                cost=successor[1])

                # Check if the successor is already in close
                if successor_node.state_id in self.CLOSE:
                    continue
                # Check if the successor is already in open
                if successor_node.state_id in [node.state_id for node in self.OPEN]:
                    continue

                # Check if the successor is goal
                if env.is_final_state(successor_node.state_id):
                    # Return the path
                    return successor_node.actions, successor_node.total_path_cost, len(self.CLOSE)

                # Add to open
                self.OPEN.append(successor_node)

        # If we reach here, it means we didn't find a solution
        return [], 0, len(self.CLOSE)  # Return empty path and inf cost


class UCSAgent():

    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class WeightedAStarAgent():

    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class AStarAgent():

    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError
