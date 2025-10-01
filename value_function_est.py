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
    
    def __init__(self, capacity_generic_goods: list[np.ndarray], mvnn_params: dict, mip_params: dict, price_scale: float):
        self.trained_model = None # points to the last trained model
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
                predicted_value = model(torch.from_numpy(predicted_demand).float().to(device))

                predicted_utility = predicted_value - torch.dot(price_vector, torch.from_numpy(predicted_demand).float().to(device))

                # get the predicted utility for the actual demand vector
                predicted_value_at_true_demand = model(demand_vector)

                predicted_utility_at_true_demand = predicted_value_at_true_demand - torch.dot(price_vector, demand_vector)


                # compute the loss
                predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
                if predicted_utility_difference < -self.mip_params['MIPGap']*10: #if the difference is significant compared to the MIP solution tolerance
                    print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')

                loss += torch.relu(predicted_utility_difference)   # for numerical stability

                loss_dq_list.append(predicted_utility_difference.detach().cpu().numpy()[0])
                dq_pred_error_list.append(np.linalg.norm(predicted_demand - demand_vector.cpu().numpy(), ord = 2))
            
            
            loss = loss / len(price_vectors)
            loss.backward()

            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            model.transform_weights()

        return np.mean(loss_dq_list), np.mean(dq_pred_error_list)

    @profile
    def __dq_val_mvnn(self, trained_model, val_loader, train_loader, device):
        """
        Validate the MVNN model using the demand query data and return validation metrics."""
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
                    
                    scaled_value_preds_val.extend(scaled_value_prediction.detach().cpu().numpy())
                    true_values_val.extend(true_value.numpy())
                    demand_vectors_val.extend(demand_vector.cpu().numpy())
                    price_vectors_val.extend(price_vector.numpy())
        
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

                scaled_value_preds_train.extend(scaled_value_prediction.detach().cpu().numpy())
                true_values_train.extend(true_value.numpy())
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
                    predicted_value = trained_model(torch.from_numpy(predicted_demand).float().to(device))
                predicted_utility = predicted_value.detach().cpu().numpy()[0] - np.dot(price_vector, predicted_demand)

                predicted_value_at_true_demand = scaled_value_preds_val[j]
                predicted_utility_at_true_demand = predicted_value_at_true_demand - np.dot(price_vector, demand_vectors_val[j])

                # compute the loss
                predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
                val_dq_loss += predicted_utility_difference
                if predicted_utility_difference < - self.mip_params['MIPGap']*10:
                    print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')

            metrics['val_dq_loss_scaled'] = val_dq_loss / len(price_vectors_val)
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
        self.price_scale = np.mean(np.abs(values_lower_bound)) + 1e-5
        return

    @profile
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


        if self.trained_model is None:
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
            self.trained_model = model # update the trained model


        # make sure ts have no regularisation (the bigger t the more regular)
        l2_reg_parameters = {'params': [], 'weight_decay': l2_reg}
        no_l2_reg_parameters = {'params': [], 'weight_decay': 0.0}
        for p in [*self.trained_model.named_parameters()]:
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
            train_loss_dq, dq_pred_error = self.__dq_train_mvnn_helper(self.trained_model.to(torch.device(self.mvnn_params['device'])),
                                                        optimizer,
                                                        train_loader_demand_queries,
                                                        device=torch.device(self.mvnn_params['device'])
                                                        )
            
            val_metrics = self.__dq_val_mvnn(trained_model = self.trained_model.to(torch.device(self.mvnn_params['device'])),
                                             val_loader = val_loader_demand_queries,
                                             train_loader = train_loader_demand_queries,
                                             device=torch.device(self.mvnn_params['device']))
            metrics.append(val_metrics)
            metrics[-1]["train_dq_loss_scaled"] = train_loss_dq
            metrics[-1]["dq_pred_error"] = dq_pred_error

            scheduler.step()

            
            if self.mvnn_params['stopping_condition'] == 'early_stop':
                if metrics[-1]['val_dq_loss_scaled'] < best_val_loss:
                    best_val_loss = metrics[-1]['val_dq_loss_scaled']
                    best_checkpoint = self.trained_model.state_dict()
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

        
        return self.trained_model, metrics[-1]  # return the last epoch metrics


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
        scaled_price = price / self.price_scale # value function is trained on scaled prices
        if self.trained_model is None:
            raise ValueError("Model not trained yet.")
        self.max_util_mvnn_model.update_model(self.trained_model)
        self.max_util_mvnn_model.update_prices_in_objective(scaled_price)
        return self.max_util_mvnn_model.get_max_util_bundle()


    def plot_nn(self, bidder: Bidder, step, id, wandb_run = None):
        """
        Plot the learned value function for 2D economy.
        """
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
    """
    Use optuna to optimize hyperparameters for value function estimation
    """

    price_dict = {24: [
            [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], 
            [154.6360430477, 133.148891428795, 129.87959703393, 132.706246549863, 90.8912415350016, 15.8878608160352, 165.340649251146, 54.8003081292853, 93.0105445644528, 131.308228553092, 169.329519856823, 115.220479765611, 182.997285237801, 92.4823830621568, 193.963002544446, 49.7278015074743, 130.965775415486, 186.829882340393, 123.355591406136, 144.567342707373, 125.208727438881, 187.786320218883, 165.372563760997, 153.529723517257],
            [57.7949069727683, 82.3439212767093, 42.1597531123134, 37.8973321880314, 61.3829019909237, 230.957180858805, 20.6413347865573, 136.271956191138, 101.223982238195, 73.0845167686794, 53.0030784052824, 151.573663680701, 60.6421335272442, 153.07107485084, 57.2234552628289, 201.426582430493, 95.1663511441493, 21.6155458746111, 141.724462317953, 99.5233901454338, 103.468342411484, 56.9915435265799, 90.8918319637835, 46.858509112845],
            [35.5011272346944, -3.43481735913563, 75.959830119835, 94.4586110373153, 33.6292548581346, 113.562140368418, 141.880420638768, 22.4759564396476, 65.3892415597495, 93.3159744359288, 107.915138214717, 40.497186348764, 128.73934594096, 48.7027046386253, 124.421838580542, 87.2447477694169, 58.8800588665567, 199.145106683356, 55.9513596496032, 65.0711326963946, 21.1304161175206, 90.5054766379371, 1.07719428842601, 46.3776791892007],
            [42.6349003246692, 153.242192611181, 2.29586091904157, 25.64469002617, 89.2367504500906, 37.1268877456547, 63.5123735073415, 134.677203868553, 66.0550330409871, 53.0860776787782, 47.6691283319506, 121.467826301194, 41.4410922685028, 94.7983733802078, 52.5157956215689, 55.6411951532324, 70.8502209196558, 84.8945574579867, 79.1249914242255, 37.3461607211447, 142.494441992156, -5.43929604511858, 164.120312687981, 17.4435799063999],
            [62.7752518846255, 67.279545260093, 80, 75.9099547355056, 16.9813365653466, 77.3203506728119, 62.4618064868638, 52.6752692114848, 57.4323009135778, 64.3616849647214, 70.3160533497435, 50.5382995509405, 80, 46.0546406639085, 80, 53.8256613509801, 59.7746509862577, 38.7980057046912, 58.6152653716398, 80, 67.4422060007224, 80, 80, 80],
            [45.5219132440708, 52.1380516264621, 33.2428120138332, 34.7061669358271, 80, 24.5116245892341, 62.124481483952, 42.5374028564095, 51.9233609011596, 54.2279849728316, 46.3072472427286, 67.893401018597, 49.5913711849902, 57.4425064625797, 59.2518957084302, 58.2623723456427, 45.1072962814319, 80, 44.7462757557791, 48.0041707652121, 40.007073242392, 62.3800181132585, 53.0551860269673, 27.596620185355],
            [37.1370358617502, 28.0184823184621, 59.2602294450149, 52.5111614742536, 18.5040958929991, 80, 43.0203726253678, 45.4086888387465, 38.440530870826, 49.8815340217242, 57.7226043522774, 34.3399711034597, 73.5669729909929, 46.568577214952, 66.3104985532418, 38.2276988274061, 57.946413575198, 26.9184418512033, 61.5989391211527, 42.2949812328477, 48.7112428484193, 47.6441435349167, 40.4106158472888, 62.0674116270829],
            [41.5649883202107, 50.166397320678, 25.2905784012959, 39.4899851259802, 66.6649530172071, 26.2569024001191, 63.1430612308985, 35.7139368680304, 52.2055222564137, 49.6218331025724, 37.9883733777813, 80, 45.8613330259758, 49.8366635731835, 54.8368813071998, 54.8266940016852, 30.4183486618484, 80, 36.7058840899151, 51.8420058206175, 29.557620336905, 53.5982525354469, 47.1066886813276, 26.4563797516999],
            [43.2811881137395, 25.6535970975207, 63.3117594661419, 45.759462288001, 20.2206855921821, 74.1741370278769, 43.8529292071586, 52.5364326952449, 35.0816257718305, 49.0172064509578, 58.1893091464034, 39.2906987584423, 65.0227646986805, 43.3917568787561, 62.671988799723, 39.1292901172295, 56.1015457736061, 29.2176666129635, 58.76263057136, 40.438158207736, 55.7161352427255, 52.5165812824861, 39.2640406196349, 53.5660692917595],
            [38.3983029360259, 54.2711278666666, 30.9082099063815, 43.4956226043882, 56.5591494060345, 29.9899882268825, 62.4836348198568, 29.5191782045832, 54.0957965400766, 48.893812432958, 43.9528167910191, 66.1042073687312, 49.3095997876764, 48.6623784690122, 56.0388728450232, 49.8581565029896, 35.9534855703478, 75.8550585584386, 38.1700173789584, 58.5538814858576, 23.8091069464689, 51.6238201318722, 54.691470698159, 26.6877416561588],
            [43.0563788004402, 27.808995496094, 52.2360666732401, 44.5357143543491, 25.9760527698453, 60.5062615867776, 46.5460800036583, 56.1231507272211, 37.1704015700693, 50.2336633056865, 50.3267453088711, 37.6851848347829, 62.3118564466777, 43.1959360588518, 60.4192879613955, 38.232844914814, 53.6065717405376, 34.2391184241493, 54.4440644098734, 43.3112161922955, 62.1511929247325, 53.4642327399062, 40.5472255309044, 52.2107428130048],
            [39.0788298702128, 47.0805723351223, 31.8893555170706, 46.3919927568743, 46.7838436824666, 27.3192979036703, 61.7289813250408, 29.4240535812257, 52.0029852558333, 49.2946640972631, 45.8879852492107, 61.9248070129145, 48.9370694615975, 47.5830151341376, 55.9131341833146, 47.9034826299354, 38.4738130965994, 68.9722263444317, 40.0328079201505, 54.6005117453391, 25.8435030674725, 53.8753370601442, 48.9678791157095, 28.7968509623991],
            [40.9473343862856, 30.7579695653698, 44.7309337799041, 43.6253906819425, 31.5169901229459, 53.1849844783593, 45.0034358686685, 52.7597309433059, 38.8022317232048, 50.4501870988437, 46.1022548518685, 42.6329509758415, 60.9535841618169, 43.2547463654488, 59.2702265546238, 40.5573875713804, 52.0640294599728, 36.7617738488282, 49.1358036451919, 42.9923197453806, 53.3715433641087, 53.4538922258745, 40.782511517327, 46.2783081684849],
            [40.984496077374, 40.9864478758059, 33.6322906406457, 41.6286775207932, 37.2631263013912, 30.5554288387177, 56.0211381847661, 33.3270899517908, 48.4102509408651, 49.8408234055759, 45.3263853408714, 57.1693110482195, 49.2870672297443, 47.0167019683401, 55.2242845674825, 46.2747437360405, 42.4339926713031, 58.4943934616461, 43.4163619466685, 50.8258814927215, 30.7581467722569, 52.2359674222325, 46.6960626380603, 33.9893015936291],
            [38.7664689244452, 37.7826521436475, 42.5796867596981, 41.6780034681718, 33.00886403457, 46.0489116206522, 45.9278530950889, 47.0180040582957, 38.3616790629597, 49.9564990682095, 47.9573678637535, 42.8882302393784, 58.6526791794553, 43.7184495805771, 58.5749647077328, 40.8989082153532, 44.078214757915, 39.1087933770074, 47.6490994268367, 42.5345014230326, 49.6900001805462, 51.8571627144562, 44.3977442674407, 41.2552562300314],
            [39.4602519806022, 39.9054182262142, 36.6500935193764, 40.6981399348763, 37.9630906262897, 27.8458401446566, 54.7967022429104, 37.0239858917335, 46.7696337171405, 49.0531476670528, 46.2087851359387, 55.5094366048001, 48.6497797964833, 46.618794392531, 54.5827151946364, 44.3196498699859, 45.0731933662286, 51.9511875612524, 44.2349257888545, 48.2869152940736, 33.0078516910599, 51.8892509642724, 45.9726014589999, 34.1554430461349],
            [39.0674930265746, 39.570505479914, 34.8430937143645, 42.6790644814767, 33.3643946091127, 46.0571257355865, 45.1780469040138, 44.4333616225058, 40.8306250243179, 49.5521386569678, 47.815039820505, 42.7051725538903, 57.1400965842746, 43.3996226840781, 57.6063525319685, 41.9567409252721, 44.6148584806904, 44.0376715385221, 47.1582839153708, 41.3899828058204, 48.0128899481432, 50.8027107736504, 44.5189213688982, 41.0893798876203],
            [39.0969773510141, 39.2842168805986, 36.3160098775797, 42.2399989919631, 35.6420561147517, 38.5391639366785, 48.9697471027061, 40.8703187918313, 43.4582015728227, 49.6944077685601, 46.8231637718894, 48.4422967642207, 53.4527836862006, 44.5469774373395, 57.0812370351089, 41.5020329605301, 44.4885391033299, 47.1405300927422, 46.4642365984703, 43.8429964374352, 41.4902246438765, 51.1575109522536, 44.9800716628056, 38.4843367480474],
            [39.1010190364411, 39.5004971614919, 36.4055825816127, 42.0263942358565, 36.1482998173783, 36.506294382499, 49.2589254222112, 39.9622578945056, 44.3752474299013, 49.7049889429896, 47.0202422432319, 47.2537883673311, 51.2950039546835, 45.3778655141274, 57.0173441674387, 41.3280585823298, 44.3251909400316, 47.9906606874453, 46.0757434654863, 43.6963810440744, 41.3991151513216, 51.2133187028132, 44.9606894624923, 38.0043221296874],
        ],
        6: [[16, 16, 16, 16, 16, 16],
            [126.418597378141, 66.6255694376814, 61.7330421813556, 121.173804965065, 116.758358583083, 147.44040114742],
            [58.427274153984, 99.1412109829831, 91.8681856031174, 71.1210546621027, 92.2267636833408, 67.1574740450958],
            [68.615030877751, 62.5405059840067, 56.3155006604463, 66.0023053017324, 61.0927024003128, 43.7285422507693],
            [60.0861214275798, 50.4276664972866, 41.9814425803331, 56.2585997129669, 45.5281968967933, 58.0928515306457],
            [57.4065865238508, 48.6101986683798, 44.4115464375701, 51.4176507114227, 57.9652986384421, 38.5553658190849],
            [56.8125806004815, 48.9448398440147, 44.0397558764182, 48.7802078069861, 52.0559945005535, 46.6130924545934],
            [56.3623003635399, 49.0162748457002, 44.6635834552191, 45.359879499315, 58.129820532754, 40.5933480063493],
            [56.430340448084, 49.2672556170924, 44.0584279346411, 44.6870820442844, 53.6708910295855, 45.7579696453162],
            [56.4984167727485, 49.2943148882421, 44.0815829148375, 44.0718574557305, 54.0771771529278, 44.9051968335999],
            [56.5659164150833, 49.4020244792002, 43.0856296276137, 43.6833942020116, 54.1283381572307, 45.9454721063676],
            [56.5837398254574, 48.7840113805754, 43.3206536055488, 43.14820537332, 53.8861676087923, 45.212508454912],
            [56.3795707297537, 49.010230641833, 42.64190183636, 43.1359821890962, 54.1592227553738, 45.6816364230264],
            [56.4443392651642, 48.8692694123997, 43.1513417735848, 42.4055920358612, 53.55634562721, 44.5418747224261],
            [56.4048705732803, 48.911469806347, 42.5524173429933, 43.3538550108938, 53.9436612867752, 45.5015191617855],
            [56.4124926801764, 48.8061621269726, 43.116272969091, 42.3196150938332, 54.2112975551619, 44.6723605104863],
            [56.3817906139284, 48.8986868638953, 42.5757754700384, 43.1945518886318, 53.400322430423, 45.5910209019727],
            [56.4025315989903, 48.7657580682643, 43.0407492232578, 42.5438935716248, 54.1049250686705, 44.5776691176196],
            [56.3697174899887, 48.8502808246332, 42.5466754958854, 43.3593565473716, 53.4040129341115, 45.4809755793335],
            [56.3843562180124, 48.7399775767, 43.0282000482966, 42.4591293045453, 54.0345464762894, 45.4680082221757],
            [56.3591152467675, 48.8222609859459, 42.5945385560787, 43.225172999577, 54.1786334533915, 45.5984247898558],
            [56.3727150625869, 48.7805189729456, 42.9868706039798, 42.4107641402577, 53.690617116215, 44.8474197016741],
            [56.3666145736394, 48.8112561408346, 42.5901237941086, 43.1349465585051, 53.9460373504495, 45.3721049826298],
            [56.3720697685854, 48.7699555756956, 42.9428594089322, 42.6037456198076, 54.1219079904288, 45.5868921376487],
            [56.3641400415961, 48.8150724984225, 42.5776752195017, 43.2919101801368, 53.70244779779, 44.8953052141362],
            [56.3767664260599, 48.7185083887089, 43.2024769115461, 42.0433576615927, 54.1022512806172, 45.6565866525785],
            [56.3546787522982, 48.8242138043753, 42.7899023177728, 42.7120225315069, 53.7251239198698, 44.9180382023747],
            [56.3694061191678, 48.7996848678313, 42.9378203214328, 42.9806530115957, 53.9449396880319, 45.3457442200455],
            [56.3701655273279, 48.8243797070295, 43.0628566473426, 42.5961893519126, 54.0974325025484, 45.563592177902],
            [56.3719269456536, 48.8374676628052, 42.9555924499955, 42.7892788514217, 53.9826542605554, 45.3756868471679],
            [56.3792421522816, 48.8526882757156, 42.6428649039345, 43.3937762363022, 54.1141713845218, 45.5619687595545],
            [56.3856800283707, 48.8216544634698, 42.9699768089491, 42.730139026525, 53.7527111382335, 44.9889048783593],
            [56.3823555539979, 48.8351336945907, 42.6635644698932, 43.3125716985164, 53.9473341083319, 45.3444423209217],
            [56.3833255347622, 48.8111595967259, 42.9147531149749, 42.7708730769178, 54.0676560953344, 45.516876715009],
            [56.3792452209673, 48.8341649465491, 42.6273729507145, 43.3333814209847, 54.1645257314083, 45.00820657603],
            [56.3816815830062, 48.8083047126917, 42.9257933200107, 42.7196171506956, 53.7228899011784, 45.3962736048255],
            [56.3779924531773, 48.8169006609452, 42.6453102580592, 43.2640222674058, 53.9778323601625, 45.5085500822084],
            [56.3777331146113, 48.7949589073235, 42.9137590857556, 42.6810129634637, 54.1146168331893, 45.0185899125136],
            [56.3735691399564, 48.8198076216512, 42.6420359149838, 43.2088790202013, 53.717515441335, 45.3771122982062],
            [56.3775911908434, 48.758975594864, 43.078863299572, 42.2441427465473, 54.1384077524413, 45.5889699617041]]
    }
    nprices = 20
    price_scale = 80
    bidders_info = {
        # 'configs/random24_80/0.json': {'names': ['HomeStorage0', 'HomeStorage1', 'HomeStorage2', 'HomeStorage3', 'HomeStorage4', 'HomeStorage5', 'HomeStorage6', 'HomeStorage7', 
        #                                          'HomeStorage8', 'HomeStorage9', 'HomeStorage10', 'HomeStorage11', 'HomeStorage12', 'HomeStorage13', 'HomeStorage14', 'HomeStorage15', 
        #                                          'HeatPump0', 'HeatPump1', 'HeatPump2', 'HeatPump3', 'HeatPump4', 'HeatPump5', 'HeatPump6', 'HeatPump7', 
        #                                          'HeatPump8', 'HeatPump9', 'HeatPump10', 'HeatPump11', 'HeatPump12', 'HeatPump13', 'HeatPump14', 'HeatPump15',
        #                                          'Consumer0', 'Consumer1', 'Consumer2', 'Consumer3', 'Consumer4', 'Consumer5', 'Consumer6', 'Consumer7', 
        #                                          'Consumer8', 'Consumer9', 'Consumer10', 'Consumer11', 'Consumer12', 'Consumer13', 'Consumer14', 'Consumer15', 
        #                                          'Wind0', 'Wind1', 'Wind2', 'Wind3', 'Wind4', 'Wind5', 'Wind6', 'Wind7', 
        #                                          'Wind8', 'Wind9', 'Wind10', 'Wind11', 'Wind12', 'Wind13', 'Wind14', 'Wind15',
        #                                          'Switch0', 'Switch1', 'Switch2', 'Switch3', 'Switch4', 'Switch5', 'Switch6', 'Switch7',
        #                                          'Switch8', 'Switch9', 'Switch10', 'Switch11', 'Switch12', 'Switch13', 'Switch14', 'Switch15'],
        #                                 'n_prod': 24},
        'configs/random24_120/0.json': {'names': ['Switch0'],
                                        'n_prod': 24},
        'configs/random6_120/0.json': {'names': ['Switch1'],
                                        'n_prod': 6},
    }
    bidder_type = 'ProsumerRenewable'


    bidders = []
    for bidder_info, config_file in zip(bidders_info.values(), bidders_info.keys()):
        with open(config_file, 'r') as f:
            config = json.load(f)
        for bidder_name in bidder_info['names']:
            bidders.append(eval(config[bidder_name]['type'])(bidder_name, bidder_info['n_prod'], config=config[bidder_name]))

    def objective(trial):
        for key in mvnn_params[bidder_type].keys():
            if key in mvnn_params_hpopt.keys():
                mvnn_params[bidder_type][key] = eval('trial.suggest_' + mvnn_params_hpopt[key][0])(key, **mvnn_params_hpopt[key][1])

        estimators = [ValueFunctionEstimateDQ(bidder.get_capacity_generic_goods(), mvnn_params[bidder.__class__.__name__], mip_params, price_scale=price_scale) for bidder in bidders]

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
            for i in range(nprices):
                prices = price_dict[bidder.n_product][i]
                bundle, value = bidder.bundle_query(prices)
                queried_bundles.append(bundle)
                true_values.append(value)
                estimator.add_data_point(prices, bundle, value)

                if i in list(np.arange(5, nprices, 2, dtype=int)):
                    start = time.time()
                    _, metrics = estimator.dq_train_mvnn()
                    times.append(time.time() - start)
                    train_loss.append(metrics['train_dq_loss_scaled'])
                    val_loss.append(metrics['val_dq_loss_scaled'])
                    kendall_tau.append(metrics['kendall_tau'])
                    r2.append(metrics['r2_centered'])
                    dq_pred_error.append(metrics['dq_pred_error'])

        return np.percentile(dq_pred_error, 75), np.percentile(kendall_tau, 25), np.mean(times)

    study = optuna.create_study(directions=["minimize", "maximize", "minimize"], sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, timeout=3600)

    trials = sorted(study.best_trials, key=lambda t: t.values[0])
    [print(trial.params) for trial in trials]
    [print(trial.values) for trial in trials]
    optuna.visualization.plot_param_importances(study).show(redered='browser')
