import numpy as np

from HaifaEnv import HaifaEnv
from typing import List, Tuple, Optional
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
        # check if cost is iinf
        try:
            self.total_path_cost = father.total_path_cost + cost if father else 0
        except TypeError:
            pass


"""    def __hash__(self):
        return hash(self.state_id)

    def __eq__(self, other):
        return isinstance(other, StateGraphNode) and self.state_id == other.state_id
"""


class HaifaHeuristic:
    def __init__(self, env: HaifaEnv):
        self.env = env

    def manhattan_distance(self, state_id1: int, state_id2: int) -> float:
        state1_col, state1_row = self.env.to_row_col(state_id1)
        state2_col, state2_row = self.env.to_row_col(state_id2)
        return abs(state1_col - state2_col) + abs(state1_row - state2_row)

    def haifa_heuristic(self, state_id: int) -> float:
        min_manhattan_distance = np.inf
        for goal_state in self.env.get_goal_states():
            distance = self.manhattan_distance(state_id, goal_state)
            if distance < min_manhattan_distance:
                min_manhattan_distance = distance
        return min(min_manhattan_distance, StateCost[StateType.P])

    def __call__(self, state_id: int) -> float:
        return self.haifa_heuristic(state_id)


class priorityQueue:
    def __init__(self, ):
        self.elements = heapdict.heapdict()

    def insert(self, item, priority):
        """
        Insert an item into the priority queue with the given priority.
        """
        if item in self.elements:
            del self.elements[item]  # Remove existing item to update priority
        self.elements[item] = priority

    def pop(self):
        """
        Remove and return key-value pair with the lowest priority from the priority queue.
        """
        return self.elements.popitem()

    def is_empty(self) -> bool:
        """
        Check if the priority queue is empty.
        """
        return len(self.elements) == 0

    def contains(self, item) -> bool:
        """
        Check if the priority queue contains the specified item.
        """
        return item in self.elements

    def remove(self, item):
        """
        Remove the specified item from the priority queue.
        """
        if item in self.elements:
            del self.elements[item]
        else:
            raise KeyError(f"Item {item} not found in the priority queue.")


class BFSGAgent():
    def __init__(self) -> None:
        self.CLOSE = {}
        self.OPEN = []

    def clean(self):
        self.CLOSE = {}
        self.OPEN = []

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        self.clean()
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
        # priority queue for open nodes with heapdict
        self.OPEN = priorityQueue()
        self.CLOSE = {}
        self.nodes = {}
    def clean(self):
        self.OPEN = priorityQueue()
        self.CLOSE = {}
        self.nodes = {}
    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        self.clean()
        init_state = env.reset()
        init_node = StateGraphNode(init_state, father=None, preceded_action=None,
                                   cost=0)
        self.nodes[init_node.state_id] = init_node
        self.OPEN.insert(init_node.state_id, (init_node.cost, init_node.state_id))
        self.expanded_nodes = 0
        while not self.OPEN.is_empty():
            curr_node_id, curr_cost = self.OPEN.pop()
            curr_node = self.nodes[curr_node_id]
            self.CLOSE[curr_node.state_id] = curr_node_id  # TODO: is it search over the state graph or tree?

            if env.is_final_state(curr_node.state_id):
                return curr_node.actions, curr_node.total_path_cost, self.expanded_nodes

            self.expanded_nodes += 1
            for action, successor in env.succ(curr_node.state_id).items():
                if successor == (None, None, None):
                    continue
                successor_node = StateGraphNode(successor[0], father=curr_node, preceded_action=action,
                                                cost=successor[1])
                in_open = successor_node.state_id in self.OPEN.elements.keys()
                in_close = successor_node.state_id in self.CLOSE.keys()

                if not in_close and not in_open:
                    self.nodes[successor_node.state_id] = successor_node
                    self.OPEN.insert(successor_node.state_id, (successor_node.total_path_cost, successor_node.state_id))
                elif in_open and successor_node.total_path_cost < self.OPEN.elements[successor_node.state_id][0]:
                    self.nodes[successor_node.state_id] = successor_node
                    self.OPEN.insert(successor_node.state_id, (successor_node.total_path_cost, successor_node.state_id))

        return [], np.inf, len(self.CLOSE)  # Return empty path and inf cost


class WeightedAStarAgent():

    def __init__(self):
        self.OPEN = priorityQueue()
        self.CLOSE = {}
        self.nodes = {}
        self.expanded_nodes = set()
        self.heuristic = None  # to be initialized in search()

    def clean(self):
        """
        Clean the OPEN and CLOSE lists for a new search.
        """
        self.OPEN = priorityQueue()
        self.CLOSE = {}
        self.nodes = {}
        self.expanded_nodes = set()
    def f(self, node: StateGraphNode, h_weight: float) -> float:
        """
        f(n) = g(n) + w * h(n)
        """
        return node.total_path_cost + (h_weight * self.heuristic(node.state_id))

    def search(self, env: HaifaEnv, h_weight) -> Tuple[List[int], float, int]:
        self.clean()
        self.heuristic = HaifaHeuristic(env)
        init_state = env.reset()
        init_node = StateGraphNode(init_state, father=None, preceded_action=None,
                                   cost=0)
        self.nodes[init_node.state_id] = init_node
        self.OPEN.insert(init_node.state_id, (self.f(init_node, h_weight), init_node.state_id))

        while not self.OPEN.is_empty():
            curr_node_id, curr_cost = self.OPEN.pop()
            curr_node = self.nodes[curr_node_id]
            self.CLOSE[curr_node.state_id] = self.f(curr_node,
                                                    h_weight)  # TODO: is it search over the state graph or tree?

            if env.is_final_state(curr_node.state_id):
                return curr_node.actions, curr_node.total_path_cost, len(self.expanded_nodes)

            self.expanded_nodes.add(curr_node.state_id)
            for action, successor in env.succ(curr_node.state_id).items():
                if successor == (None, None, None):
                    continue

                successor_node = StateGraphNode(successor[0], father=curr_node, preceded_action=action,
                                                cost=successor[1])

                in_open = successor_node.state_id in self.OPEN.elements.keys()
                in_close = successor_node.state_id in self.CLOSE

                if not in_close and not in_open:
                    self.nodes[successor_node.state_id] = successor_node
                    self.OPEN.insert(successor_node.state_id, (self.f(successor_node, h_weight), successor_node.state_id))

                elif in_open and self.f(successor_node, h_weight) < self.OPEN.elements[successor_node.state_id][0]:
                    self.OPEN.remove(successor_node.state_id)
                    self.nodes[successor_node.state_id] = successor_node
                    self.OPEN.insert(successor_node.state_id, (self.f(successor_node, h_weight), successor_node.state_id))
                elif in_close and self.f(successor_node, h_weight) < self.CLOSE[successor_node.state_id]:
                    self.OPEN.remove(successor_node.state_id)
                    self.nodes[successor_node.state_id] = successor_node
                    self.OPEN.insert(successor_node.state_id, (self.f(successor_node, h_weight), successor_node.state_id))
                    del self.CLOSE[successor_node.state_id]


class AStarAgent(WeightedAStarAgent):
    def __init__(self):
        super().__init__()

    def search(self, env: HaifaEnv, h_weight: Optional[float] = 1.0) -> Tuple[List[int], float, int]:
        return super().search(env, h_weight=1.0)
