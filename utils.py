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

#   prosumer
mvnn_params_n2 = {
    'clip_grad_norm': 1,
    'use_gradient_clipping': True,
    'train_split': 0.8,
    'batch_size': 3,
    'epochs': 15,
    'l2_reg': .01,
    'learning_rate': 0.05,
    'print_frequency': 101,

    'num_hidden_layers': 2,
    'num_hidden_units': 16,
    'layer_type': 'MVNNLayerReLUProjected',
    'target_max': 1,
    'lin_skip_connection': False,
    'dropout_prob': 0,
    'init_method': 'custom',
    'random_ts': [0, 1],
    'trainable_ts': False,
    'init_E': 1,
    'init_Var': 0.09,
    'init_b': 0.05,
    'init_bias': 0.05,
    'init_little_const': 0.1,
}

mvnn_params = {
    'clip_grad_norm': 1,
    'use_gradient_clipping': False,
    'train_split': 0.7,
    'batch_size': 1000,
    'epochs': 200,
    'l2_reg': .1,
    'learning_rate': 0.06,
    'print_frequency': 101,

    'num_hidden_layers': 2,
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
}

#   logarithmic bidder
# mvnn_params = {
#     'clip_grad_norm': 1,
#     'use_gradient_clipping': False,
#     'train_split': 0.8,
#     'batch_size': 4,
#     'epochs': 40,
#     'l2_reg': .002,
#     'learning_rate': 0.06,
#     'print_frequency': 101,

#     'num_hidden_layers': 2,
#     'num_hidden_units': 8,
#     'layer_type': 'MVNNLayerReLUProjected',
#     'target_max': 1,
#     'lin_skip_connection': True,
#     'dropout_prob': 0,
#     'init_method': 'custom',
#     'random_ts': [0, 1],
#     'trainable_ts': True,
#     'init_E': 1,
#     'init_Var': 0.09,
#     'init_b': 0.05,
#     'init_bias': 0.05,
#     'init_little_const': 0.1,
# }

mvnn_params_hpopt = {
    # 'use_gradient_clipping': ['categorical', {'choices': [True, False]}],
    # 'batch_size': ['int', {'low': 1, 'high': 10}],
    'epochs': ['int', {'low': 1, 'high': 300}],
    'l2_reg': ['float', {'low': 0.0001, 'high': 0.1, 'log': True}],
    'learning_rate': ['float', {'low': 0.0001, 'high': 0.1, 'log': True}],

    'num_hidden_layers': ['int', {'low': 2, 'high': 4}],
    'num_hidden_units': ['int', {'low': 5, 'high': 20}],
    'lin_skip_connection': ['categorical', {'choices': [True, False]}],
    # 'dropout_prob': ['float', {'low': 0, 'high': 0.2}],
    # 'trainable_ts': ['categorical', {'choices': [True, False]}],
}

static_bidder_configs = json.load(open('configs/bidder_configs_static.json'))
# asset_configs = json.load(open('configs/asset_configs.json'))
community_configs = json.load(open('configs/community_configs.json'))

def gurobi_status_converter(int_status):
        status_table = ['woopsies!', 'LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL', 'INPROGRESS', 'USER_OBJ_LIMIT']
        return status_table[int_status]