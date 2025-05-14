import numpy as np
from utils import *
from prosumer import Prosumer, LogarithmicBidder
from mlcce import MLCCE
import gurobipy as gp
from gurobipy import GRB


class CommunityManager:
    
    def __init__(self, community: str):
        self.community = community
        self.community_config = community_configs[community]
        self.bidders = [eval(bidder_configs[b]['type'])(b) for b in self.community_config["bidders"]]
        assert len(set([b.intervals for b in self.bidders]))==1, "All bidders must have the same number of intervals"

        capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        self.mlcce = MLCCE([np.array(self.community_config["price_init"][i]) for i in range(len(self.community_config['price_init']))], self.query_bundle, len(self.bidders), price_max=self.community_config["price_max"], capacities=capacities, imbalance_tol_coef=self.community_config["imbalance_tol_coef"])

        self.full_info_mip = None

    def query_bundle(self, price: np.ndarray) -> tuple:
        """
        Query the utility maximizing bundle from prosumers at the given price
        """
        bundles = []
        values = []
        for prosumer in self.bidders:
            bundle, value = prosumer.bundle_query(price)
            bundles.append(bundle)
            values.append(value)
        return bundles, values
    
    def get_full_info_clearing(self, ) -> tuple:
        self.full_info_mip = gp.Model("Full information clearing MIP")
        objexpr = 0
        dispatch_variables = []
        for b in self.bidders:
            x, obj = b.add_model(self.full_info_mip)
            dispatch_variables.append(x)
            objexpr += obj
        self.full_info_mip.setObjective(objexpr, GRB.MAXIMIZE)
        [self.full_info_mip.addConstr(gp.quicksum(dispatch_variables[j][i] for j in range(len(self.bidders)))==0, name=f"balance_{i}") for i in range(self.bidders[0].intervals)]
        self.full_info_mip.optimize()

        dispatch_bundles = [np.array([dispatch_variables[j][i].x for i in range(b.intervals)]) for j in range(len(self.bidders))]
        clearing_price = np.array([self.full_info_mip.getConstrByName(f"balance_{i}").Pi for i in range(self.bidders[0].intervals)])

        return clearing_price, dispatch_bundles

    
    def clear_market(self, ):
        clearing_price, dispatch_bundles = self.mlcce.run_mlcce()

        print(clearing_price)

if __name__ == "__main__":
    community_manager = CommunityManager("dummy_logarithmic_community")
    price, bundles = community_manager.get_full_info_clearing()
    print('Full info clearing price:', price)
    print('Dispatch:')
    [print(bundles[i]) for i in range(len(bundles))]
    
    community_manager.clear_market()