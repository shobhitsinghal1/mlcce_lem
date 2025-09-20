import numpy as np
import json
import os

mip_params = {
    'timeLimit': np.inf,
    'FeasibilityTol': 1e-6,
    'IntFeasTol': 1e-6,
    'MIPGap': 1e-6,
    'outputFlag': 1,
    'DualReductions': 1,
}

next_price_params = {
    'method': 'SLSQP',
    'prox_coef': 0,
}

cce_params = {
    'max_iter': 40,
    'base_step': 0.5,
    'decay': 0.6,
}

mlcce_params = {
    'cce_rounds': 5,
    'max_iter': 40
}


mvnn_params = {
    'ProsumerStorage': {
        'clip_grad_norm': 1,
        'use_gradient_clipping': False,
        'train_split': 0.7,
        'batch_size': 1000,
        'epochs': 238,
        'l2_reg': .026,
        'learning_rate': 0.0075,
        'stopping_condition': 'train_loss',
        'num_hidden_layers': 1,
        'num_hidden_units': 9,
        'layer_type': 'MVNNLayerReLUProjected',
        'lin_skip_connection': False,
        'dropout_prob': 0,
        'init_method': 'custom',
        'random_ts': [0, 1],
        'trainable_ts': True,
        'init_E': 1,
        'init_Var': 0.09,
        'init_b': 0.05,
        'init_bias': 0.05,
        'init_little_const': 0.1,
        'device': 'cpu'},

    'ProsumerRenewable': {
        'clip_grad_norm': 1,
        'use_gradient_clipping': False,
        'train_split': 0.7,
        'batch_size': 1000,
        'epochs': 288,
        'l2_reg': .3,
        'learning_rate': 0.003,
        'stopping_condition': 'early_stop',
        'num_hidden_layers': 1,
        'num_hidden_units': 9,
        'layer_type': 'MVNNLayerReLUProjected',
        'lin_skip_connection': True,
        'dropout_prob': 0,
        'init_method': 'custom',
        'random_ts': [0, 1],
        'trainable_ts': True,
        'init_E': 1,
        'init_Var': 0.09,
        'init_b': 0.05,
        'init_bias': 0.05,
        'init_little_const': 0.1,
        'device': 'cpu'},
    
    'ProsumerConsumer': {
        'clip_grad_norm': 1,
        'use_gradient_clipping': False,
        'train_split': 0.7,
        'batch_size': 1000,
        'epochs': 288,
        'l2_reg': .3,
        'learning_rate': 0.003,
        'stopping_condition': 'early_stop',
        'num_hidden_layers': 1,
        'num_hidden_units': 9,
        'layer_type': 'MVNNLayerReLUProjected',
        'lin_skip_connection': True,
        'dropout_prob': 0,
        'init_method': 'custom',
        'random_ts': [0, 1],
        'trainable_ts': True,
        'init_E': 1,
        'init_Var': 0.09,
        'init_b': 0.05,
        'init_bias': 0.05,
        'init_little_const': 0.1,
        'device': 'cpu'},
    
    'ProsumerSwitch': {
        'clip_grad_norm': 1,
        'use_gradient_clipping': False,
        'train_split': 0.7,
        'batch_size': 1000,
        'epochs': 135,
        'l2_reg': .06,
        'learning_rate': 0.12,
        'stopping_condition': 'train_loss',
        'num_hidden_layers': 2,
        'num_hidden_units': 8,
        'layer_type': 'MVNNLayerReLUProjected',
        'lin_skip_connection': True,
        'dropout_prob': 0,
        'init_method': 'custom',
        'random_ts': [0, 1],
        'trainable_ts': True,
        'init_E': 1,
        'init_Var': 0.09,
        'init_b': 0.05,
        'init_bias': 0.05,
        'init_little_const': 0.1,
        'device': 'cpu'}
    
}

mvnn_params_hpopt = {
    # 'use_gradient_clipping': ['categorical', {'choices': [True, False]}],
    # 'batch_size': ['int', {'low': 1, 'high': 10}],
    'epochs': ['int', {'low': 1, 'high': 300}],
    'l2_reg': ['float', {'low': 0.001, 'high': 0.5, 'log': True}],
    'learning_rate': ['float', {'low': 0.0001, 'high': 0.2, 'log': True}],

    'num_hidden_layers': ['int', {'low': 1, 'high': 3}],
    'num_hidden_units': ['int', {'low': 2, 'high': 10}],
    'lin_skip_connection': ['categorical', {'choices': [True, False]}],
    'stopping_condition': ['categorical', {'choices': ['train_loss', 'val_loss', 'early_stop']}],
}

community_configs = json.load(open('configs/community_configs.json'))

def gurobi_status_converter(int_status):
        status_table = ['woopsies!', 'LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL', 'INPROGRESS', 'USER_OBJ_LIMIT']
        return status_table[int_status]


def log_mech_metrics(self, price, bundle, value, step):
    [self.wandb_run.log({f"Price/Product {i}": price[i]}, step=step, commit=False) for i in range(self.n_products)] # log prices

    imb = np.sum(bundle, 0)
    total_capacity = np.sum(self.capacities, 0)
    rel_imb = np.where(imb >= 0, imb * 100 / total_capacity[1], imb * 100 / total_capacity[0])
    imbalance_norm = np.linalg.norm(rel_imb, 1)
    self.imbalance_norm.append(imbalance_norm)
    [self.wandb_run.log({f"Imbalance/Product {i}": rel_imb[i]}, step=step, commit=False) for i in range(self.n_products)] # log imbalance
    self.wandb_run.log({f"Imbalance_norm": imbalance_norm}, step=step, commit=False)

    self.wandb_run.log({"Lagrange_dual": value}, step=step, commit=False)


# --------------------- Generate bidder configurations ---------------------
def make_HeatPump(j, n_products, avg_price, spread_limit):
    config = {}
    config['type'] = 'ProsumerStorage'
    name = f'HeatPump{j}'
    capacity = np.random.uniform(5, 50)
    config['s0'] = np.random.uniform(0, capacity)
    config['sn'] = capacity/2
    config['alpha'] = np.random.uniform(0.1, 0.2)
    config['eta'] = np.random.uniform(0.5, 0.7)
    config['power_limit_up'] = np.random.uniform(capacity/5, capacity)
    config['power_limit_down'] = 0
    config['su'] = [capacity]*n_products
    config['sl'] = [0]*n_products
    config['gamma'] = [np.random.uniform(capacity/20, capacity/10)]*n_products
    config['available'] = [1]*n_products
    config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, n_products)
    config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
    config['oppcprice'] = list(config['oppcprice'])
    config['beta'] = [0]*n_products
    return config, name

def make_HomeStorage(j, n_products, avg_price, spread_limit):
    config = {}
    config['type'] = 'ProsumerStorage'
    name = f'HomeStorage{j}'
    capacity = np.random.uniform(5, 50)
    config['s0'] = np.random.uniform(0, capacity)
    config['sn'] = capacity/2
    config['alpha'] = np.random.uniform(0.1, 0.2)
    config['eta'] = np.random.uniform(0.85, 0.95)
    config['power_limit_up'] = np.random.uniform(capacity/5, capacity)
    config['power_limit_down'] = -config['power_limit_up']
    config['su'] = [capacity]*n_products
    config['sl'] = [0]*n_products
    config['gamma'] = [0]*n_products
    config['available'] = [1]*n_products
    config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, n_products)
    config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
    config['oppcprice'] = list(config['oppcprice'])
    config['beta'] = [np.random.uniform(0,0.4)]*n_products
    return config, name

def make_Wind(j, n_products, avg_price, spread_limit):
    config = {}
    config['type'] = 'ProsumerRenewable'
    name = f'Wind{j}'
    capacity = np.random.uniform(5, 50)
    config['generation'] = list(np.random.uniform(0, capacity, n_products))
    config['oppdprice'] = list(np.random.normal(avg_price*0.4, spread_limit/3, n_products))
    return config, name

def make_Consumer(j, n_products, avg_price, spread_limit):
    config = {}
    config['type'] = 'ProsumerConsumer'
    name = f'Consumer{j}'
    capacity = np.random.uniform(5, 50)
    config['consumption'] = list(np.random.uniform(0, capacity, n_products))
    config['oppcprice'] = list(np.random.normal(avg_price*0.6, spread_limit/3, n_products))
    return config, name

def make_Switch(j, n_products, avg_price, spread_limit):
    config = {}
    config['type'] = 'ProsumerSwitch'
    name = f'Switch{j}'
    config['capacity'] = np.random.uniform(5, 15)
    config['oppcprice'] = list(np.random.normal(avg_price*0.6, spread_limit/3, n_products))
    config['tr'] = int(np.random.randint(1, int(n_products/2) + 1))
    config['tl'] = int(np.random.randint(1, n_products - config['tr'] + 2))
    config['tu'] = int(np.random.randint(config['tl'] + config['tr'] - 1, n_products + 1))
    return config, name

def make_StorageFlex(j, n_products, avg_price, spread_limit):
    config = {}
    assert n_products % 2 == 0
    nH = int(n_products / 2)
    config['type'] = 'ProsumerStorageFlex'
    name = f'StorageFlex{j}'
    config['oppeprice'] = list(np.random.normal(avg_price*1.25, spread_limit/20, nH))
    config['oppfprice'] = list(np.random.normal(avg_price*0.625, spread_limit/20, nH))
    config['alphae'] = list(np.random.uniform(-0.3, -0.1, nH))
    config['alphaf'] = list(np.random.uniform(-0.3, -0.1, nH))
    config['su'] = np.random.uniform(5, 50)
    config['s0'] = np.random.uniform(0, config['su'])
    config['fixflex'] = False
    config['fixener'] = False
    return config, name

def make_DSO(j, n_products, avg_price, spread_limit):
    config = {}
    nH = int(n_products / 2)
    config['type'] = 'ProsumerDSO'
    name = f'DSO{j}'
    config['oppfprice'] = list(np.random.normal(avg_price, spread_limit/20, nH))
    config['alphaf'] = list(np.random.uniform(-0.5, -1, nH))
    config['flexlimit'] = list(np.random.uniform(5, 20, nH))
    config['fixflex'] = False
    config['fixener'] = False
    return config, name

def make_ConsumerFlex(j, n_products, avg_price, spread_limit):
    config = {}
    nH = int(n_products / 2)
    config['type'] = 'ProsumerConsumerFlex'
    name = f'ConsumerFlex{j}'
    config['oppcprice'] = list(np.random.normal(avg_price*1.25, spread_limit/20, nH))
    config['alphae'] = list(np.random.uniform(-0.3, -0.1, nH))
    config['consumption'] = list(np.random.uniform(5, 50, nH))
    config['fixener'] = False
    return config, name

def make_RenewableFlex(j, n_products, avg_price, spread_limit):
    config = {}
    nH = int(n_products / 2)
    config['type'] = 'ProsumerRenewableFlex'
    name = f'RenewableFlex{j}'
    config['oppdprice'] = list(np.random.normal(avg_price, spread_limit/20, nH))
    config['alphae'] = list(np.random.uniform(-0.3, -0.1, nH))
    config['generation'] = list(np.random.uniform(5, 50, nH))
    config['fixener'] = False
    return config, name


def make_bidder_configs(community, seed):
    """
    Generate bidder configurations for a given community and random number generator seed.
    """
    np.random.seed(seed)
    community_config = community_configs[community]
    n_products = community_config["n_products"]

    avg_price = (community_config["price_max"] + community_config["price_min"]) / 2
    spread_limit = community_config["price_max"] - community_config["price_min"]
    configs = {}
    for i, n in enumerate(community_config["N"]):
        bidder_type = community_config["bidder_types"][i]
        for j in range(n):
            config, name = eval(f'make_{bidder_type}({j}, {n_products}, {avg_price}, {spread_limit})')
            configs[name] = config
    
    if not os.path.exists(f'configs/{community}'):
        os.mkdir(f'configs/{community}')
    with open(f'configs/{community}/{seed}.json', 'w') as f:
        json.dump(configs, f, indent=4)

if __name__ == "__main__":
    for n in [10, 20, 40, 80, 120]:
        community = f'random6_{n}'
        for i in range(4):
            make_bidder_configs(community, i)
