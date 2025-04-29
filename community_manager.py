import numpy as np
from utils import community_configs
from prosumer import Prosumer

class CommunityManager:
    def __init__(self, community: str):
        self.community = community
        self.community_config = community_configs[community]
        self.prosumers = self.__make_prosumers()
        
    def __make_prosumers(self, ) -> list:
        return [Prosumer(prosumer) for prosumer in self.community_config["prosumers"]]
    
    def query_bundle(self, price: np.ndarray) -> list[np.ndarray]:
        """
        Query the utility maximizing bundle from prosumers at the given price
        """
        return [prosumer.bundle_query(price) for prosumer in self.prosumers]
    
    

