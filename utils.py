import numpy as np
import json

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
    'base_step': 0.1,
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
        'init_little_const': 0.1,},

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
        'init_little_const': 0.1,},
    
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
        'init_little_const': 0.1,},
    
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
        'init_little_const': 0.1,}
    
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
    'stopping_condition': ['categorical', {'choices': ['train_loss', 'val_loss', 'early_stop']}]
    # 'dropout_prob': ['float', {'low': 0, 'high': 0.2}],
    # 'trainable_ts': ['categorical', {'choices': [True, False]}],
}

# asset_configs = json.load(open('configs/asset_configs.json'))
community_configs = json.load(open('configs/community_configs.json'))

def gurobi_status_converter(int_status):
        status_table = ['woopsies!', 'LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL', 'INPROGRESS', 'USER_OBJ_LIMIT']
        return status_table[int_status]


def make_bidder_configs(community):
    community_config = community_configs[community]
    horizon = community_config["horizon"]

    avg_price = (community_config["price_max"] + community_config["price_min"]) / 2
    spread_limit = community_config["price_max"] - community_config["price_min"]
    configs = {}
    for i, n in enumerate(community_config["N"]):
        bidder_type = community_config["bidder_types"][i]
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
                config['su'] = [capacity]*horizon
                config['sl'] = [0]*horizon
                config['gamma'] = [np.random.uniform(capacity/20, capacity/10)]*horizon
                config['available'] = [1]*horizon
                config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, horizon)
                config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
                config['oppcprice'] = list(config['oppcprice'])
                config['beta'] = [0]*horizon
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
                config['su'] = [capacity]*horizon
                config['sl'] = [0]*horizon
                config['gamma'] = [0]*horizon
                config['available'] = [1]*horizon
                config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, horizon)
                config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
                config['oppcprice'] = list(config['oppcprice'])
                config['beta'] = [np.random.uniform(0,0.4)]*horizon
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
            #     config['su'] = [capacity]*horizon
            #     config['sl'] = [0]*horizon
            #     config['available'] = np.random.choice([0,1], horizon)
            #     config['gamma'] = np.where(config['available'], 0, np.random.uniform(capacity/6, capacity/4, horizon))
            #     config['oppcprice'] = np.random.normal(avg_price, spread_limit/3, horizon)
            #     config['oppdprice'] = list(np.random.uniform(config['oppcprice'] - spread_limit/10, config['oppcprice']))
            #     config['oppcprice'] = list(config['oppcprice'])
            #     config['beta'] = [np.random.uniform(0.2,0.5)]*horizon
            elif bidder_type == 'Wind':
                config['type'] = 'ProsumerRenewable'
                name = f'Wind{j}'
                capacity = np.random.uniform(5, 50)
                config['generation'] = list(np.random.uniform(0, capacity, horizon))
                config['oppdprice'] = list(np.random.normal(avg_price*0.4, spread_limit/3, horizon))
            elif bidder_type == 'Consumer':
                config['type'] = 'ProsumerConsumer'
                name = f'Consumer{j}'
                capacity = np.random.uniform(5, 50)
                config['consumption'] = list(np.random.uniform(0, capacity, horizon))
                config['oppcprice'] = list(np.random.normal(avg_price*0.6, spread_limit/3, horizon))
            elif bidder_type == 'Switch':
                config['type'] = 'ProsumerSwitch'
                name = f'Switch{j}'
                config['capacity'] = np.random.uniform(5, 15)
                config['oppcprice'] = list(np.random.normal(avg_price*0.6, spread_limit/3, horizon))
                config['tr'] = int(np.random.randint(1, int(horizon/2) + 1))
                config['tl'] = int(np.random.randint(1, horizon - config['tr'] + 2))
                config['tu'] = int(np.random.randint(config['tl'] + config['tr'] - 1, horizon + 1))
            configs[name] = config
    with open(f'configs/bidder_configs_{community}.json', 'w') as f:
        json.dump(configs, f, indent=4)

if __name__ == "__main__":
    make_bidder_configs('random6')