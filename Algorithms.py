import numpy as np

from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict



class BFSGAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError
        


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

