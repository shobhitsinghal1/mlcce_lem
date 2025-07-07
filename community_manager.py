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

        wandb_run = wandb.init(project='mlcce', entity='shosi-danmarks-tekniske-universitet-dtu', name=community+'_n'+str(self.horizon)) 

        if type(self.community_config["price_max"]) is not list:
            price_max = np.array([self.community_config["price_max"]]*self.horizon)
        else:
            price_max = self.community_config["price_max"]
        if type(self.community_config["price_min"]) is not list:
            price_min = np.array([self.community_config["price_min"]]*self.horizon)
        else:
            price_min = self.community_config["price_min"]
        
        
        self.mlcce = MLCCE(n_init_prices=10,
                           n_products=self.horizon,
                           price_bound=(price_min, price_max),
                           imbalance_tol_coef=self.community_config["imbalance_tol_coef"],
                           wandb_run=wandb_run,
                           bidders=self.bidders)
        self.chp_fullinfo = CHP_fullinfo(self.bidders,
                                         price_bound=(price_min, price_max),
                                         wandb_run=wandb_run)

        self.full_info_mip = None


    def clear_market(self, mechanism: str = 'MLCCE'):
        if mechanism == 'CHP':
            clearing_price = self.chp_fullinfo.run()
        elif mechanism == 'MLCCE':
            clearing_price, _ = self.mlcce.run()

        return clearing_price

if __name__ == "__main__":
    community_manager = CommunityManager("community2")
    chp_clearing_price, dispatch = community_manager.clear_market('CHP')

    mlcce_clearing_price = community_manager.clear_market('MLCCE')
    print('MLCCE clearing price:', mlcce_clearing_price)