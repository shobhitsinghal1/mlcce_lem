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
    
    def __init__(self, community: str, mechanism: str, seed: int = 0):
        self.community = community
        np.random.seed(seed)

        # Params
        self.community_config = community_configs[community]
        self.horizon = self.community_config["horizon"]
        if type(self.community_config["price_max"]) is not list:
            price_max = np.array([self.community_config["price_max"]]*self.horizon)
        else:
            price_max = self.community_config["price_max"]
        if type(self.community_config["price_min"]) is not list:
            price_min = np.array([self.community_config["price_min"]]*self.horizon)
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
                                            "mechanism": mechanism})

        # Spawn bidders
        self.make_bidders()

        # Initialize mechanisms
        if mechanism == 'MLCCE':
            self.mechanism = MLCCE(n_products=self.horizon,
                                   price_bound=(price_min, price_max),
                                   wandb_run=self.wandb_run,
                                   bidders=self.bidders)
        elif mechanism == 'CCE':
            self.mechanism = CCE(self.bidders,
                                 price_bound=(price_min, price_max),
                                 wandb_run=self.wandb_run)
        elif mechanism == 'CHP':
            self.mechanism = CHP_fullinfo(self.bidders,
                                          price_bound=(price_min, price_max),
                                          wandb_run=self.wandb_run)


    def make_bidders(self, ):
        if "bidders" in self.community_config.keys():
            self.bidders = [eval(static_bidder_configs[b]['type'])(b, self.horizon, static_bidder_configs[b]) for b in self.community_config["bidders"]]
        else:
            self.bidders = []
            avg_price = (self.community_config["price_max"] + self.community_config["price_min"]) / 2
            spread_limit = self.community_config["price_max"] - self.community_config["price_min"]
            configs = {}
            for i, n in enumerate(self.community_config["N"]):
                bidder_type = self.community_config["bidder_types"][i]
                for j in range(n):
                    config = {}
                    if bidder_type == 'HeatPump':
                        config['type'] = 'ProsumerStorage'
                        name = f'HeatPump{j}'
                        capacity = np.random.uniform(5, 50)
                        config['s0'] = np.random.uniform(0, capacity)
                        config['sn'] = capacity/2
                        config['alpha'] = np.random.uniform(1, 2)
                        config['eta'] = np.random.uniform(0.5, 0.7)
                        config['power_limit_up'] = np.random.uniform(capacity/5, capacity)
                        config['power_limit_down'] = 0
                        config['su'] = [capacity]*self.horizon
                        config['sl'] = [0]*self.horizon
                        config['gamma'] = [np.random.uniform(capacity/20, capacity/10)]*self.horizon
                        config['available'] = [1]*self.horizon
                        config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, self.horizon)
                        config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
                        config['oppcprice'] = list(config['oppcprice'])
                        config['beta'] = [0]*self.horizon
                    elif bidder_type == 'HomeStorage':
                        config['type'] = 'ProsumerStorage'
                        name = f'HomeStorage{j}'
                        capacity = np.random.uniform(5, 50)
                        config['s0'] = np.random.uniform(0, capacity)
                        config['sn'] = capacity/2
                        config['alpha'] = np.random.uniform(1, 2)
                        config['eta'] = np.random.uniform(0.85, 0.95)
                        config['power_limit_up'] = np.random.uniform(capacity/5, capacity)
                        config['power_limit_down'] = -config['power_limit_up']
                        config['su'] = [capacity]*self.horizon
                        config['sl'] = [0]*self.horizon
                        config['gamma'] = [0]*self.horizon
                        config['available'] = [1]*self.horizon
                        config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, self.horizon)
                        config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
                        config['oppcprice'] = list(config['oppcprice'])
                        config['beta'] = [np.random.uniform(0,0.4)]*self.horizon
                    # elif bidder_type == 'EV': # need to make sure the feasibility of the EV availability and gamma profile
                    #     config['type'] = 'ProsumerStorage'
                    #     name = f'EV{j}'
                    #     capacity = np.random.uniform(5, 50)
                    #     config['s0'] = np.random.uniform(0, capacity)
                    #     config['sn'] = capacity*0.7
                    #     config['alpha'] = np.random.uniform(2,8)
                    #     config['eta'] = np.random.uniform(0.85, 0.95)
                    #     config['power_limit_up'] = np.random.uniform(capacity/5, capacity)
                    #     config['power_limit_down'] = -config['power_limit_up']
                    #     config['su'] = [capacity]*self.horizon
                    #     config['sl'] = [0]*self.horizon
                    #     config['available'] = np.random.choice([0,1], self.horizon)
                    #     config['gamma'] = np.where(config['available'], 0, np.random.uniform(capacity/6, capacity/4, self.horizon))
                    #     config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, self.horizon)
                    #     config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
                    #     config['oppcprice'] = list(config['oppcprice'])
                    #     config['beta'] = [np.random.uniform(0.2,0.5)]*self.horizon
                    elif bidder_type == 'Wind':
                        config['type'] = 'ProsumerRenewable'
                        name = f'Wind{j}'
                        capacity = np.random.uniform(5, 50)
                        config['generation'] = list(np.random.uniform(0, capacity, self.horizon))
                        config['oppdprice'] = list(np.random.normal(avg_price*0.4, spread_limit/3, self.horizon))
                    elif bidder_type == 'Consumer':
                        config['type'] = 'ProsumerConsumer'
                        name = f'Consumer{j}'
                        capacity = np.random.uniform(5, 50)
                        config['consumption'] = list(np.random.uniform(0, capacity, self.horizon))
                        config['oppcprice'] = list(np.random.normal(avg_price*0.6, spread_limit/3, self.horizon))
                    elif bidder_type == 'Switch':
                        config['type'] = 'ProsumerSwitch'
                        name = f'Switch{j}'
                        config['capacity'] = np.random.uniform(5, 15)
                        config['oppcprice'] = list(np.random.normal(avg_price*0.6, spread_limit/3, self.horizon))
                        config['tr'] = int(np.random.randint(1, int(self.horizon/2) + 1))
                        config['tl'] = int(np.random.randint(1, self.horizon - config['tr'] + 2))
                        config['tu'] = int(np.random.randint(config['tl'] + config['tr'] - 1, self.horizon + 1))
                    configs[name] = config
            with open('configs/bidder_configs.json', 'w') as f:
                json.dump(configs, f, indent=4)
            self.bidders = [eval(configs[b]['type'])(b, self.horizon, configs[b]) for b in configs.keys()]
    
    
    # def plot_metrics(self, metrics: dict, metric_names):
    #     if self.wandb_run is not None:
    #         for metric in metric_names:
    #             xs = []
    #             ys = []
    #             keys = []
    #             for mechanism in metrics.keys():
    #                 if metric in metrics[mechanism].keys():
    #                     xs.append(metrics[mechanism][metric]['iter'])
    #                     ys.append(metrics[mechanism][metric][metric])
    #                     keys.append(mechanism)
    #             self.wandb_run.log({f"{metric}": wandb.plot.line_series(xs=xs,
    #                                                                     ys=ys,
    #                                                                     keys=keys,
    #                                                                     title=f"{metric}",
    #                                                                     xname="Iterations")}, commit=False)


    def clear_market(self, ):
        clearing_price, dispatch = self.mechanism.run()
        self.wandb_run.finish()

        return None

if __name__ == "__main__":
    community = 'random2'
    community_manager = MarketOperator(community, mechanism='CHP')
    community_manager.clear_market()
    community_manager = MarketOperator(community, mechanism='CCE')
    community_manager.clear_market()
    # community_manager = MarketOperator(community, mechanism='MLCCE')
    # community_manager.clear_market()