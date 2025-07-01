import numpy as np
from utils import *
from Bidder import *
from Mechanisms import *
import gurobipy as gp
from gurobipy import GRB


class CommunityManager:
    
    def __init__(self, community: str):
        self.community = community
        self.community_config = community_configs[community]
        self.horizon = self.community_config["horizon"]
        self.bidders = [eval(bidder_configs[b]['type'])(b, self.horizon) for b in self.community_config["bidders"]]

        capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        self.mlcce = MLCCE(n_init_prices=10,
                           n_products=self.horizon,
                           query_func=self.query_bundle, 
                           num_participants=len(self.bidders), 
                           price_max=self.community_config["price_max"], 
                           capacities=capacities, 
                           imbalance_tol_coef=self.community_config["imbalance_tol_coef"],
                           logname=community+'_n'+str(self.horizon),
                           bidders=self.bidders)
        self.chp_fullinfo = CHP_fullinfo(self.bidders,
                                         capacities=capacities)

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
        balance_constrs = [self.full_info_mip.addConstr(gp.quicksum(dispatch_variables[j][i] for j in range(len(self.bidders)))==0, name=f"balance_{i}") for i in range(self.horizon)] # couple all bidders by balance constraint
        self.full_info_mip.update()
        self.full_info_mip.optimize()

        if self.full_info_mip.status != 2:
            self.full_info_mip.computeIIS()
            self.full_info_mip.write("full_info_clearing.ilp")
            raise ValueError(f"The problem is not feasible, status: {gurobi_status_converter(self.full_info_mip.status)}")

        dispatch_bundles = [np.array([dispatch_variables[j][i].getAttr('x') for i in range(self.horizon)]) for j in range(len(self.bidders))]
        
        if self.full_info_mip.IsMIP:
            fixedmodel = self.full_info_mip.fixed()        
            fixedmodel.optimize()
            clearing_price = np.array([fixedmodel.getConstrByName(f'balance_{i}').Pi for i in range(self.horizon)])
        else:
            clearing_price = np.array([self.full_info_mip.getConstrByName(f'balance_{i}').Pi for i in range(self.horizon)])


        return clearing_price, dispatch_bundles

    
    def clear_market(self, mechanism: str = 'MLCCE'):
        if mechanism == 'CHP':
            clearing_price = self.chp_fullinfo.run()
        elif mechanism == 'MLCCE':
            clearing_price, _ = self.mlcce.run()

        return clearing_price

if __name__ == "__main__":
    community_manager = CommunityManager("community2")
    chp_clearing_price = community_manager.clear_market('CHP')
    print(f'Convex hull clearing price: {np.round(chp_clearing_price, 2)}')
    
    mlcce_clearing_price = community_manager.clear_market('MLCCE')
    print('MLCCE clearing price:', mlcce_clearing_price)