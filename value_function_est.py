# Libs
import numpy as np
import torch
import sklearn.metrics
from scipy import stats as scipy_stats
import wandb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from line_profiler import profile
import optuna
import time
np.random.seed(0)
torch.manual_seed(0)

# Own Libs
from Bidder import *
from utils import *
from mvnns.mvnn_generic import MVNN_GENERIC
from gurobi_mip_mvnn_generic_single_bidder_util_max import GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX

PYTHONWARNINGS = 'error'

class ValueFunctionEstimateDQ():
    
    def __init__(self, capacity_generic_goods: list[np.ndarray], mvnn_params: dict, mip_params: dict, price_scale: float = 200):
        self.trained_model = None # supposed to point to the last trained model
        self.max_util_mvnn_model = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(mip_params=mip_params)
        self.price_scale = price_scale
        
        # Params
        self.mvnn_params = mvnn_params
        self.mip_params = mip_params
        self.capacity_generic_goods = capacity_generic_goods
        
        # Maintains a dataset
        self.dataset_price = []
        self.dataset_true_value = []
        self.dataset_bundle = []


    @profile
    def __dq_train_mvnn_helper(self, model, optimizer, train_loader_demand_queries, device):

        clip_grad_norm = self.mvnn_params['clip_grad_norm']
        use_gradient_clipping = self.mvnn_params['use_gradient_clipping']

        model.train()
        loss_dq_list = []
        dq_pred_error_list = []

        for batch_idx, (demand_vectors, price_vectors, _) in enumerate(train_loader_demand_queries):
            price_vectors, demand_vectors = price_vectors.to(device), demand_vectors.to(device)
            optimizer.zero_grad()

            #--------------------------------
            # IMPORTANT: we need to transform the weights of the MVNN to be non-negative.
            model.transform_weights()
            #--------------------------------

            # compute the network's predicted answer to the demand query
            self.max_util_mvnn_model.update_model(model)

            # computing loss
            loss = 0
            for price_vector, demand_vector in zip(price_vectors, demand_vectors):
                self.max_util_mvnn_model.update_prices_in_objective(price_vector.cpu().numpy())
                try:
                    predicted_demand = self.max_util_mvnn_model.get_max_util_bundle()
                    if np.any(self.capacity_generic_goods[0] - predicted_demand >= self.mip_params['FeasibilityTol']*10) or np.any(predicted_demand - self.capacity_generic_goods[1] >= self.mip_params['FeasibilityTol']*10):
                        print(f'domain violated: {predicted_demand}')
                except:
                    print('--- MIP is unbounded, skipping this sample! ---')
                    continue

                # get the predicted value for that answer
                predicted_value = model(torch.from_numpy(predicted_demand).float())

                predicted_utility = predicted_value - torch.dot(price_vector, torch.from_numpy(predicted_demand).to(device).float())

                # get the predicted utility for the actual demand vector
                predicted_value_at_true_demand = model(demand_vector)

                predicted_utility_at_true_demand = predicted_value_at_true_demand - torch.dot(price_vector, demand_vector)


                # compute the loss
                predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
                if predicted_utility_difference < -self.mip_params['MIPGap']*10: #if the difference is significant compared to the MIP solution tolerance
                    print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')

                loss += torch.relu(predicted_utility_difference)   # for numerical stability

                loss_dq_list.append(predicted_utility_difference.detach().numpy()[0])
                dq_pred_error_list.append(np.linalg.norm(predicted_demand - demand_vector.numpy(), ord = 2))
            
            
            loss = loss / len(price_vectors)
            loss.backward()

            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            model.transform_weights()

        return np.mean(loss_dq_list), np.mean(dq_pred_error_list)


    def __dq_val_mvnn(self, trained_model, val_loader, train_loader, device):
        trained_model.eval()
        metrics = {}

        # Validation data
        scaled_value_preds_val = []
        demand_vectors_val = []
        price_vectors_val = []
        true_values_val = []
        with torch.no_grad():
            if val_loader is not None:
                for demand_vector, price_vector, true_value in val_loader:
                    demand_vector = demand_vector.to(device)
                    scaled_value_prediction = trained_model(demand_vector)
                    scaled_value_preds_val.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
                    true_values_val.extend(list(true_value.numpy()))
                    demand_vectors_val.extend(list(demand_vector.cpu().numpy()))
                    price_vectors_val.extend(list(price_vector.numpy()))
        
                scaled_value_preds_val = np.array(scaled_value_preds_val)
                value_preds_val = scaled_value_preds_val * self.price_scale
                true_values_val = np.array(true_values_val)
                scaled_true_values_val = true_values_val/self.price_scale


        # Train data
        scaled_value_preds_train = []
        demand_vectors_train = []
        price_vectors_train = []
        true_values_train = []
        with torch.no_grad():
            for demand_vector, price_vector, true_value in train_loader:
                demand_vector = demand_vector.to(device)
                scaled_value_prediction = trained_model(demand_vector)
                scaled_value_preds_train.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
                true_values_train.extend(list(true_value.numpy()))
                demand_vectors_train.extend(demand_vector.cpu().numpy())
                price_vectors_train.extend(price_vector.numpy())
            
            scaled_value_preds_train = np.array(scaled_value_preds_train)
            value_preds_train = scaled_value_preds_train * self.price_scale
            true_values_train = np.array(true_values_train)
            scaled_true_values_train = true_values_train/self.price_scale

        # --------------------------------------

        # 1. generalization performance measures (on the validation set, that is drawn using price vectors)
        if val_loader is not None:
            metrics['kendall_tau'] = scipy_stats.kendalltau(value_preds_val, true_values_val).correlation
            if np.isnan(metrics['kendall_tau']):
                metrics['kendall_tau'] = 0.0
            metrics['r2_centered'] = sklearn.metrics.r2_score(y_true=true_values_val - np.mean(true_values_val), y_pred= value_preds_val - np.mean(value_preds_val)) # a centered R2, because constant shifts in model predictions should not really affect us  
            metrics['true_values'] = true_values_val  # also store all true /predicted values so that we can make true vs predicted plots
            metrics['predicted_values'] = value_preds_val

        # --------------------------------------
        
        # 2. generalization performance measures (on the training set, that is drawn using price vectors)            
        metrics['kendall_tau_train'] = scipy_stats.kendalltau(true_values_train, value_preds_train).correlation
        if np.isnan(metrics['kendall_tau_train']):
                metrics['kendall_tau_train'] = 0.0
        metrics['r2_centered_train'] = sklearn.metrics.r2_score(y_true=true_values_train - np.mean(true_values_train), y_pred= value_preds_train - np.mean(value_preds_train))
        metrics['true_values_train'] = true_values_train # also store all true /predicted values so that we can make true vs predicted plots
        metrics['predicted_values_train'] = value_preds_train

        # --------------------------------------
        
        # 3. DQ loss performance measure (same as training loss)
        if val_loader is not None:
            self.max_util_mvnn_model.update_model(trained_model)
            val_dq_loss = 0
            predicted_demands = []
            for (j, price_vector) in enumerate(price_vectors_val):
                # update the prices in the MIP objective to the price vector of the current datapoint
                self.max_util_mvnn_model.update_prices_in_objective(price_vector)
            
                try: 
                    predicted_demand = self.max_util_mvnn_model.get_max_util_bundle()
                except:
                    print('MIP is unbounded, something is wrong!')
                    predicted_demand = np.ones(demand_vector.shape[0])
                predicted_demands.append(predicted_demand)

                with torch.no_grad():
                    predicted_value = trained_model(torch.from_numpy(predicted_demand).float()).item()
                predicted_utility = predicted_value - np.dot(price_vector, predicted_demand)

                predicted_value_at_true_demand = scaled_value_preds_val[j]
                predicted_utility_at_true_demand = predicted_value_at_true_demand - np.dot(price_vector, demand_vectors_val[j])

                # compute the loss
                predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
                val_dq_loss += predicted_utility_difference
                if predicted_utility_difference < - self.mip_params['MIPGap']*10:
                    print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')

            metrics['val_dq_loss_scaled'] = val_dq_loss / len(price_vectors_val)
        # --------------------------------------


        # 3. Regret performance measure 
        # --------------------------------------
        # regret = 0
        # for (j, price_vector) in enumerate(price_vectors):
        #     # calculate the optimal true utility for the true demand vector
        #     scaled_true_value =  scaled_true_values[j]
        #     scaled_true_opt_utility = scaled_true_value - np.dot(price_vector, demand_vectors[j])

        #     # calculate the true utility for the predicted demand vector
        #     predicted_demand = predicted_demands[j]
        #     scaled_value_at_predicted_demand = bidder.calculate_value(predicted_demand) / scale
        #     scaled_utility_at_predicted_demand = scaled_value_at_predicted_demand - np.dot(price_vector, predicted_demand)

        #     regret = regret + (scaled_true_opt_utility - scaled_utility_at_predicted_demand)


        # val_metrics['mean_regret'] = (regret * scale) / len(price_vectors)
        # val_metrics['mean_regret_scaled'] = val_metrics['mean_regret'] / common_scale_generalazation # regret scaled by the common scale of the generalization set to make numbers interpretable
        # --------------------------------------

        return metrics


    def __get_scaled_data_split(self, ):
        """
        Splits the dataset into training and validation set
        """
        random_state = np.random.randint(0, 100)
        train_split_proportion = self.mvnn_params['train_split']
        
        train_prices, val_prices = train_test_split(self.dataset_price, train_size=train_split_proportion, random_state=random_state)
        train_true_values, val_true_values = train_test_split(self.dataset_true_value, train_size=train_split_proportion, random_state=random_state)
        train_bundles, val_bundles = train_test_split(self.dataset_bundle, train_size=train_split_proportion, random_state=random_state)

        if train_split_proportion == 1:
            return np.asarray(train_prices)/self.price_scale, None, np.asarray(train_bundles), None, np.asarray(train_true_values), None
        else:
            return np.asarray(train_prices)/self.price_scale, np.asarray(val_prices)/self.price_scale, np.asarray(train_bundles), np.asarray(val_bundles), np.asarray(train_true_values), np.asarray(val_true_values)


    def add_data_point(self, price, bundle, true_value):
        self.dataset_price.append(price)
        self.dataset_true_value.append(true_value)
        self.dataset_bundle.append(bundle)

        values_lower_bound = [np.dot(self.dataset_bundle[i], self.dataset_price[i]) for i in range(len(self.dataset_bundle))]
        # print(f'Max: {np.max(values_lower_bound)}')
        # print(f'Mean: {np.mean(values_lower_bound)}')
        self.price_scale = np.mean(np.abs(values_lower_bound)) + 1e-5
        return


    def dq_train_mvnn(self, ) -> tuple:
        """
        Train a new MVNN model using the bundle query data and return the trained model and validation metrics.
        """

        # get the data
        P_train, P_val, X_train, X_val, V_train, V_val = self.__get_scaled_data_split()


        if type(self.mvnn_params['batch_size']) == str:
            batch_size = len(P_train)
        else:
            batch_size = self.mvnn_params['batch_size']

        epochs = self.mvnn_params['epochs'] 
        l2_reg = self.mvnn_params['l2_reg']
        learning_rate = self.mvnn_params['learning_rate']


        train_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                      torch.from_numpy(P_train).float(),
                                                                      torch.from_numpy(V_train).float())
        train_loader_demand_queries = torch.utils.data.DataLoader(train_dataset_demand_queries,
                                                                batch_size=batch_size,
                                                                shuffle=True)
        if P_val is not None and X_val is not None:
            val_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                        torch.from_numpy(P_val).float(),
                                                                        torch.from_numpy(V_val).float())
            val_loader_demand_queries = torch.utils.data.DataLoader(val_dataset_demand_queries,
                                                                    batch_size=1,
                                                                    shuffle=True)
        else:
            val_loader_demand_queries = None


        model = MVNN_GENERIC(input_dim=len(self.capacity_generic_goods[0]),
                            num_hidden_layers=self.mvnn_params['num_hidden_layers'],
                            num_hidden_units=self.mvnn_params['num_hidden_units'],
                            layer_type=self.mvnn_params['layer_type'],
                            lin_skip_connection = self.mvnn_params['lin_skip_connection'],
                            dropout_prob = self.mvnn_params['dropout_prob'],
                            init_method = self.mvnn_params['init_method'],
                            random_ts = self.mvnn_params['random_ts'],
                            trainable_ts = self.mvnn_params['trainable_ts'],
                            init_E = self.mvnn_params['init_E'],
                            init_Var = self.mvnn_params['init_Var'],
                            init_b = self.mvnn_params['init_b'],
                            init_bias = self.mvnn_params['init_bias'],
                            init_little_const = self.mvnn_params['init_little_const'],
                            capacity_generic_goods=self.capacity_generic_goods
                            )
        self.trained_model = model #update the trained model


        # make sure ts have no regularisation (the bigger t the more regular)
        l2_reg_parameters = {'params': [], 'weight_decay': l2_reg}
        no_l2_reg_parameters = {'params': [], 'weight_decay': 0.0}
        for p in [*model.named_parameters()]:
            if 'ts' in p[0]:
                no_l2_reg_parameters['params'].append(p[1])
            else:
                l2_reg_parameters['params'].append(p[1])


        optimizer = torch.optim.Adam([l2_reg_parameters,no_l2_reg_parameters],
                                    lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            float(epochs))

        metrics = []
        best_val_loss = np.inf
        patience = 0
        for epoch in range(epochs):
            train_loss_dq, dq_pred_error = self.__dq_train_mvnn_helper(model,
                                                        optimizer,
                                                        train_loader_demand_queries,
                                                        device=torch.device('cpu')
                                                        )
            
            # if val_loader_demand_queries is not None:
            val_metrics = self.__dq_val_mvnn(trained_model = model,
                                             val_loader = val_loader_demand_queries,
                                             train_loader = train_loader_demand_queries,
                                             device=torch.device('cpu'))
            metrics.append(val_metrics)
            metrics[-1]["train_dq_loss_scaled"] = train_loss_dq
            metrics[-1]["dq_pred_error"] = dq_pred_error

            scheduler.step()

            
            if self.mvnn_params['stopping_condition'] == 'early_stop':
                if metrics[-1]['val_dq_loss_scaled'] < best_val_loss:
                    best_val_loss = metrics[-1]['val_dq_loss_scaled']
                    best_checkpoint = model.state_dict()
                else:
                    patience += 1
                    if patience >= 5:  # early stopping patience
                        self.trained_model.load_state_dict(best_checkpoint)
                        break
            elif self.mvnn_params['stopping_condition'] == 'train_loss':
                if metrics[-1]["train_dq_loss_scaled"] <= 1e-2:
                    break
            elif self.mvnn_params['stopping_condition'] == 'val_loss':
                if metrics[-1]['val_dq_loss_scaled'] <= 1e-2:
                    break

        
        return model, metrics[-1]  # return the last epoch metrics


    def get_bundle_value(self, bundle: np.ndarray) -> float:
        """
        Get the value of a bundle using the learned value function.
        """
        self.trained_model.eval()
        return self.trained_model(torch.from_numpy(bundle).float()).detach().numpy() * self.price_scale


    def get_max_util_bundle(self, price: np.ndarray) -> np.ndarray:
        """
        Query the utility maximizing bundle using the learned value function and for given price.
        """
        scaled_price = price / self.price_scale
        if self.trained_model is None:
            raise ValueError("Model not trained yet.")
        self.max_util_mvnn_model.update_model(self.trained_model)
        self.max_util_mvnn_model.update_prices_in_objective(scaled_price)
        return self.max_util_mvnn_model.get_max_util_bundle()


    def plot_nn(self, bidder: Bidder, step, id, wandb_run = None):
        assert len(self.capacity_generic_goods[0]) == 2, "Plotting is only supported for 2D"

        bounds = self.capacity_generic_goods
        x1 = np.linspace(bounds[0][0], bounds[1][0], 50)
        x2 = np.linspace(bounds[0][1], bounds[1][1], 50)
        x1, x2 = np.meshgrid(x1, x2)
        X1, X2 = x1.flatten(), x2.flatten()
        X3_pred = []
        X3_true = []
        for x11, x22 in zip(X1, X2):
            bundle = np.array([x11, x22])
            X3_pred.append(self.trained_model.to(torch.float64)(torch.from_numpy(bundle)).detach().numpy()[0]*self.price_scale)
            X3_true.append(bidder.get_value(bundle))

        X3_pred = np.array(X3_pred)
        X3_true = np.array(X3_true)
        X1, X2, X3_true, X3_pred = X1.reshape(x1.shape), X2.reshape(x1.shape), X3_true.reshape(x1.shape), X3_pred.reshape(x1.shape)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(X1, X2, X3_true, label='True', alpha=0.5)
        ax.plot_surface(X1, X2, X3_pred, label='Predicted')
        ax.legend()
        if wandb_run is not None:
            wandb_run.log({f"Value_estimator_plot/Participant {id}": wandb.Image(fig)}, step=step, commit=False)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    # unit test for value function estimator - fully substitute value function - 2 intervals

    # run = wandb.init(project='mlcce', entity='shosi-danmarks-tekniske-universitet-dtu')

    nprices = 20
    price_scale = 100
    horizon = 6
    price_list = [np.random.rand(horizon) * price_scale for _ in range(nprices)]
    bidder_configs = 'configs/bidder_configs_random6.json'
    bidder_type = 'ProsumerSwitch'
    bidder_name = 'Switch0'

    with open(bidder_configs, 'r') as f:
            configs = json.load(f)
    bidders = [eval(bidder_type)(bidder_name, horizon=horizon, config=configs[bidder_name])]

    def objective(trial):
        for key in mvnn_params[bidder_type].keys():
            if key in mvnn_params_hpopt.keys():
                mvnn_params[bidder_type][key] = eval('trial.suggest_' + mvnn_params_hpopt[key][0])(key, **mvnn_params_hpopt[key][1])

        estimators = [ValueFunctionEstimateDQ(bidder.get_capacity_generic_goods(), mvnn_params[bidder_type], mip_params, price_scale=price_scale) for bidder in bidders]

        queried_bundles = []
        true_values = []
        train_loss = []
        val_loss = []
        kendall_tau = []
        dq_pred_error = []
        r2 = []
        times = []
        for bidder, estimator in zip(bidders, estimators):
            queried_bundles.append([])
            true_values.append([])
            for i in range(len(price_list)):
                bundle, value = bidder.bundle_query(price_list[i])
                queried_bundles.append(bundle)
                true_values.append(value)
                estimator.add_data_point(price_list[i], bundle, value)

                if i in list(np.arange(5, nprices, 2, dtype=int)):
                    start = time.time()
                    model, metrics = estimator.dq_train_mvnn()
                    times.append(time.time() - start)
                    train_loss.append(metrics['train_dq_loss_scaled'])
                    val_loss.append(metrics['val_dq_loss_scaled'])
                    kendall_tau.append(metrics['kendall_tau'])
                    r2.append(metrics['r2_centered'])
                    dq_pred_error.append(metrics['dq_pred_error'])

        return np.percentile(dq_pred_error, 75), np.percentile(kendall_tau, 25), np.percentile(r2, 25), np.mean(times)

    study = optuna.create_study(directions=["minimize", "maximize", "maximize", "minimize"], sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, timeout=1800)

    trials = sorted(study.best_trials, key=lambda t: t.values[0])
    [print(trial.params) for trial in trials]
    [print(trial.values) for trial in trials]
    optuna.visualization.plot_param_importances(study).show(redered='browser')
    # print("Accuracy: {}".format(trial.value))
    # print("Best hyperparameters: {}".format(trial.params))


    # data = [[v1, v2] for v1, v2 in zip(metrics[list(metrics.keys())[-1]]['predicted_values'], metrics[list(metrics.keys())[-1]]['true_values'])]
    # loss_data = [[i+1, metrics[i]['train_dq_loss_scaled'], 'train_loss'] for i in range(len(metrics.keys()))]
    # loss_data.extend([[i+1, metrics[i]['val_dq_loss_scaled'], 'val_loss'] for i in range(len(metrics.keys()))])
    # table = wandb.Table(data=data, columns=["predicted_values", "true_values"])
    # table_loss = wandb.Table(data=loss_data, columns=['step', 'loss_scaled', 'series'])
    # run.log({'Prediction_plot': wandb.plot.scatter(table, "predicted_values", "true_values")})
    # run.log({'Loss_plot': wandb.plot.line(table_loss, "step", "loss_scaled", "series")})
    # print(f'Model trained with metrics: {metrics[list(metrics.keys())[-1]]}')
