import numpy as np
from utils import *
from Bidder import *
from Mechanisms import *
import gurobipy as gp
from gurobipy import GRB
import datetime
import time
import wandb


class MarketOperator:
    """ Class to manage the market clearing process for a community of bidders. """
    def __init__(self, community: str, mechanism: str, seed: int):
        self.community = community

        # Params
        self.community_config = community_configs[community]
        self.n_products = self.community_config["n_products"]
        if type(self.community_config["price_max"]) is not list:
            price_max = np.array([self.community_config["price_max"]]*self.n_products)
        else:
            price_max = self.community_config["price_max"]
        if type(self.community_config["price_min"]) is not list:
            price_min = np.array([self.community_config["price_min"]]*self.n_products)
        else:
            price_min = self.community_config["price_min"]

        # WandB init
        self.wandb_run = wandb.init(project='mlcce',
                                    entity='shosi-danmarks-tekniske-universitet-dtu', 
                                    name=community + f"_{seed}_" + mechanism + datetime.datetime.fromtimestamp(time.time()).strftime("_%d_%H:%M"),
                                    config={"cce_params": cce_params,
                                            "mlcce_params": mlcce_params,
                                            "mvnn_params": mvnn_params,
                                            "community_config": self.community_config,
                                            "community_name": community,
                                            "mechanism": mechanism,
                                            "seed": seed},
                                    mode='online')

        # Spawn bidders
        with open(f'configs/{community}/{seed}.json', 'r') as f:
            bidder_configs = json.load(f)
        self.bidders = [eval(bidder_configs[b]['type'])(b, self.n_products, bidder_configs[b]) for b in bidder_configs.keys()]

        # Initialize mechanisms
        if mechanism == 'MLCCE':
            self.mechanism = MLCCE(n_products=self.n_products,
                                   price_bound=(price_min, price_max),
                                   wandb_run=self.wandb_run,
                                   bidders=self.bidders)
        elif mechanism == 'CCE':
            self.mechanism = CCE(self.bidders,
                                 price_bound=(price_min, price_max),
                                 wandb_run=self.wandb_run)
        elif mechanism == 'CHP':
            self.mechanism = CHP(self.bidders,
                                          price_bound=(price_min, price_max),
                                          wandb_run=self.wandb_run)
        elif mechanism == 'FullInfo':
            self.mechanism = FullInfo(self.bidders,
                                      n_products=self.n_products,
                                      wandb_run=self.wandb_run)
        elif mechanism == 'FullInfoSequential':
            self.mechanism = FullInfoSequential(self.bidders,
                                                n_products=self.n_products,
                                                wandb_run=self.wandb_run)


    def clear_market(self, ):
        clearing_price, dispatch = self.mechanism.run()
        if clearing_price is not None:
            self.wandb_run.log({f"clearing_price/Product {i}": clearing_price[i] for i in range(self.n_products)}, commit=False)
        self.wandb_run.log({f"dispatch/Participant {j}/Product {i}": dispatch[j][i] for j in range(len(self.bidders)) for i in range(self.n_products)}, commit=False)

        self.wandb_run.finish()

if __name__ == "__main__":
    for n in [40]:
        community = f'random6_{n}'
        print("\nN_products ", n)
        for i in range(0,1):
            print(f"\nSeed: {i}")
            # community_manager = MarketOperator(community, mechanism='CCE', seed=i)
            # community_manager.clear_market()
            community_manager = MarketOperator(community, mechanism='MLCCE', seed=i)
            community_manager.clear_market()
            print('\n')