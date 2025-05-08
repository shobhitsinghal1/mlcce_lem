import numpy as np
from utils import community_configs
from prosumer import Prosumer
from mlcce import MLCCE

class CommunityManager:
    def __init__(self, community: str):
        self.community = community
        self.community_config = community_configs[community]
        self.prosumers = [Prosumer(prosumer) for prosumer in self.community_config["prosumers"]]
        self.mlcce = MLCCE(np.array(self.community_config["price_init"]), self.query_bundle, len(self.prosumers))

    def query_bundle(self, price: np.ndarray) -> tuple:
        """
        Query the utility maximizing bundle from prosumers at the given price
        """
        bundles = []
        values = []
        for prosumer in self.prosumers:
            bundle, value = prosumer.bundle_query(price)
            bundles.append(bundle)
            values.append(value)
        return bundles, values
    
    def clear_market(self, ):
        clearing_price, dispatch_bundles = self.mlcce.run_mlcce()

        print(clearing_price)
