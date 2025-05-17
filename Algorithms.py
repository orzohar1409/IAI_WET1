import numpy as np

from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict


class GraphNode:
    def __init__(self, state_num, successors):
        self.state_num = state_num
        self.successors = successors


class BFSGAgent():
    def __init__(self) -> None:
        self.CLOSE = {}
        self.OPEN = []

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        init_state = env.reset()
        self.OPEN.append({"id": init_state,"actions": []})

        # Run on OPEN
        while self.OPEN:
            curr = self.OPEN.pop()
            self.CLOSE[curr] = []


            # Search among all successors
            for action, successor in env.succ(curr).items():
                pass
                # Check if child is goal


                # Add child
                #if succ in close
                #self.OPEN.append((successor, curr['actions']+action))





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

