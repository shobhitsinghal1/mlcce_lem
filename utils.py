import numpy as np
import json

mip_params = {
    'timeLimit': np.inf,
    'FeasibilityTol': 1e-6,
    'IntFeasTol': 1e-6,
    'MIPGap': 1e-6,
    'outputFlag': 1,
    'DualReductions': 0, #1--------------
}

mvnn_params = {
    'clip_grad_norm': 1,
    'use_gradient_clipping': True,
    'train_split': 0.8,
    'batch_size': 1,
    'epochs': 100,
    'l2_reg': .0001,
    'learning_rate': 0.005,
    'print_frequency': 1,

    'num_hidden_layers': 2,
    'num_hidden_units': 20,
    'layer_type': 'MVNNLayerReLUProjected',
    'target_max': 1,
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

mvnn_params_big = {
    'clip_grad_norm': 1,
    'use_gradient_clipping': False,
    'train_split': 0.8,
    'batch_size': 1,
    'epochs': 10,
    'l2_reg': .0001,
    'learning_rate': 0.005,
    'print_frequency': 1,

    'num_hidden_layers': 20,
    'num_hidden_units': 20,
    'layer_type': 'MVNNLayerReLUProjected',
    'target_max': 1,
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
}

prosumer_configs = json.load(open('configs/prosumer_configs.json'))
asset_configs = json.load(open('configs/asset_configs.json'))
community_configs = json.load(open('configs/community_configs.json'))