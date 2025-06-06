import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from line_profiler import profile
from codetiming import Timer

#own libs
from value_function_est import ValueFunctionEstimateDQ
from utils import mvnn_params, mip_params, next_price_params


class MLCCE:
    def __init__(self, price_init: list[np.ndarray], query_func: callable, num_participants: int, price_max: float, capacities: list, imbalance_tol_coef: float, bidders: list = None):
        """
        :param query_func: function to query the utility maximizing bundle from participants - returns a list of bundles
        """
        self.queried_bundles = [] # list (clock iterations) of list (prosumers) of queried bundles
        self.queried_values = [] # list (clock iterations) of list (prosumers) of true values
        self.price_iterates = price_init
        self.num_participants = num_participants
        self.imbalance_tol = np.sum(capacities, 0)*imbalance_tol_coef
        self.value_estimator_metrics = [{'train_dq_loss_scaled': [], 'val_dq_loss_scaled': [], 'r2_centered': []} for i in range(num_participants)]
        
        self.bundle_query = query_func # returns list of bundles and list of true values
        self.value_estimators = [ValueFunctionEstimateDQ(capacity_generic_goods=capacities[i],
                                                         mvnn_params=mvnn_params,
                                                         mip_params=mip_params,
                                                         price_scale=price_max) for i in range(num_participants)]
        self.capacities = capacities
        self.bidders = bidders # only for plotting the estimated and true value functions
        self.price_max = price_max
        self.eq_tol = 1e-3
        self.mlcce_iter = 0

        self.next_price_params = next_price_params

        self.train_timer = Timer('train_timer', logger=None)
        self.next_price_timer = Timer('next_price_timer', logger=None)


    def __get_pred_imb(self, price: np.ndarray) -> np.ndarray:
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        pred_imb = np.sum(predicted_bundles, 0)
        return pred_imb


    def __next_price(self, ):
        """
        Calculate the next price iterate based on the bundles predicted by the learned value functions
        """

        def gradient(price: np.ndarray) -> np.ndarray:
            predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
            return -np.sum(predicted_bundles, axis=0) # sum over all participants
        
        def objective(price: np.ndarray) -> float:
            predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
            predicted_values = np.array([value_estimator.get_bundle_value(bundle) for value_estimator, bundle in zip(self.value_estimators, predicted_bundles)])
            payments = np.array([np.dot(price, bundle) for bundle in predicted_bundles])
            return np.sum(predicted_values - payments) # sum over all participants
        
        initial_price = self.price_iterates[-1]
        trust_region_rad = self.next_price_params['trust_region_radius_coef']*self.price_max*np.pow(self.mlcce_iter+1, self.next_price_params['trust_region_decay_pow'])
        bounds=[(self.price_iterates[-1][i]-trust_region_rad, 
                 self.price_iterates[-1][i]+trust_region_rad) for i in range(len(self.price_iterates[-1]))]
        solution = opt.minimize(objective, initial_price, jac=gradient, bounds=bounds, method=self.next_price_params['method'])

        pred_imb = self.__get_pred_imb(solution.x)
        print(f'Predicted imbalance: {pred_imb}')
        
        return solution.x


        # def pred_imbalance(bundles: list) -> float:
        #     return np.abs(np.sum(bundles, 0))

        # price_iterates = [np.multiply(self.price_iterates[-1], np.random.uniform(1, 1, len(self.price_iterates[-1])))]
        # best_iter = {'W': np.inf, 'price': None}
        # bbest_iter = {'W': np.inf, 'price': None, 'imb': np.array([np.inf]*len(self.price_iterates[-1]))}

        # lr = self.next_price_params['base_learning_rate']
        # decay = self.next_price_params['lr_decay']
        
        # # for t in range(self.next_price_params['max_iter']):
        # while True:
        #     predicted_bundles = [value_estimator.get_max_util_bundle(price_iterates[-1]) for value_estimator in self.value_estimators]
        #     predicted_utilities = np.array([value_estimator.get_bundle_value(bundle) - np.dot(price_iterates[-1], bundle) for value_estimator, bundle in zip(self.value_estimators, predicted_bundles)])
            
        #     W = np.sum(predicted_utilities)
        #     if W < best_iter['W']:
        #         best_iter['W'] = W
        #         best_iter['price'] = price_iterates[-1]
        #     else:
        #         lr = lr * (1 - decay)
            
        #     if np.all(pred_imbalance(predicted_bundles) < bbest_iter['imb']):
        #         bbest_iter['W'] = W
        #         bbest_iter['price'] = price_iterates[-1]
        #         bbest_iter['imb'] = pred_imbalance(predicted_bundles)
        #     else:    
        #         lr = lr * (1 - decay)

        #     grad = -np.sum(predicted_bundles, axis=0)
        #     print(np.linalg.norm(grad))
        #     print(lr)
        #     if np.linalg.norm(grad) < 1e-3:
        #         break
        #     price_iterates.append(price_iterates[-1] - lr * grad)

        # return bbest_iter['price']


    def __is_equilibrium(self, ):
        if np.all(np.abs(np.sum(self.queried_bundles[-1], 0))<=self.imbalance_tol):
            return True
        else:
            return False

    
    def __plot_info(self, ):
        fig, ax = plt.subplots(1, 1)
        n = len(self.price_iterates)
        for i in range(len(self.price_iterates[0])):
            ax.plot(np.linspace(1, n, n), [self.price_iterates[j][i] for j in range(n)], label=f'Product {i}')
        ax.set_xlabel('Clock Iteration')
        ax.set_ylabel('Price')
        ax.legend()
        plt.show()


    def __log_info(self, run):
        run.log({"Train time": self.train_timer.timers.total('train_timer')})
        run.log({"Next price time": self.next_price_timer.timers.total('next_price_timer')})
        
        n_prod = len(self.price_iterates[0])
        data_price_iterates = [[j+1]+[self.price_iterates[j][i] for i in range(n_prod)] for j in range(len(self.price_iterates))]
        data_imbalance = [[j+1]+list(np.sum(self.queried_bundles[j], 0)) for j in range(len(self.queried_bundles))]
        table_price_iterates = pd.DataFrame(data_price_iterates, columns=['iter']+[f'Product {i}' for i in range(n_prod)])
        table_imbalance = pd.DataFrame(data_imbalance, columns=['iter']+[f'Product {i}' for i in range(n_prod)])

        run.log({"price_iterates_plot": wandb.plot.line_series(xs=table_price_iterates['iter'],
                                                                    ys=[table_price_iterates[f'Product {i}'] for i in range(n_prod)],
                                                                    keys=[f'Product {i}' for i in range(n_prod)],
                                                                    title="Price Iterates",
                                                                    xname="Iteration"
                                                                   )})
        run.log({"imbalance_plot": wandb.plot.line_series(xs=table_imbalance['iter'],
                                                               ys=[table_imbalance[f'Product {i}'] for i in range(n_prod)],
                                                               keys=[f'Product {i}' for i in range(n_prod)],
                                                               title="Imbalance",
                                                               xname="Iteration"
                                                              )})
        
        data_train_dq_loss_scaled = [[i+1]+[self.value_estimator_metrics[j]['train_dq_loss_scaled'][i] for j in range(self.num_participants)] for i in range(self.mlcce_iter)]
        data_val_dq_loss_scaled = [[i+1]+[self.value_estimator_metrics[j]['val_dq_loss_scaled'][i] for j in range(self.num_participants)] for i in range(self.mlcce_iter)]
        data_r2_centered = [[i+1]+[self.value_estimator_metrics[j]['r2_centered'][i] for j in range(self.num_participants)] for i in range(self.mlcce_iter)]
        
        table_train_dq_loss_scaled = pd.DataFrame(data=data_train_dq_loss_scaled, columns=['iter']+[f'Participant {i+1}' for i in range(self.num_participants)])
        table_val_dq_loss_scaled = pd.DataFrame(data=data_val_dq_loss_scaled, columns=['iter']+[f'Participant {i+1}' for i in range(self.num_participants)])
        table_r2_centered = pd.DataFrame(data=data_r2_centered, columns=['iter']+[f'Participant {i+1}' for i in range(self.num_participants)])

        run.log({"train_dq_loss_scaled_plot": wandb.plot.line_series(xs=table_train_dq_loss_scaled['iter'],
                                                                          ys=[table_train_dq_loss_scaled[f'Participant {i+1}'] for i in range(self.num_participants)],
                                                                          keys=[f'Participant {i+1}' for i in range(self.num_participants)],
                                                                          title="Train DQ Loss Scaled",
                                                                          xname="MLCCE iteration"
                                                                         )})
        run.log({"val_dq_loss_scaled_plot": wandb.plot.line_series(xs=table_val_dq_loss_scaled['iter'],
                                                                        ys=[table_val_dq_loss_scaled[f'Participant {i+1}'] for i in range(self.num_participants)],
                                                                        keys=[f'Participant {i+1}' for i in range(self.num_participants)],
                                                                        title="Validation DQ Loss Scaled",
                                                                        xname="MLCCE iteration"
                                                                       )})
        run.log({"r2_centered_plot": wandb.plot.line_series(xs=table_r2_centered['iter'],
                                                                 ys=[table_r2_centered[f'Participant {i+1}'] for i in range(self.num_participants)],
                                                                 keys=[f'Participant {i+1}' for i in range(self.num_participants)],
                                                                 title="R2 Centered",
                                                                 xname="MLCCE iteration"
                                                                )})
        
        [estimator.plot_nn(bidder, wandb_run=run) for bidder, estimator in zip(self.bidders, self.value_estimators)]


    @profile
    def run_mlcce(self, ):
        """
        Machine learning based combinatorial clock exchange. This function conducts the exchange auction and returns a clearing price and dispatch schedule.
        """
        
        run = wandb.init(entity='shosi-danmarks-tekniske-universitet-dtu', project='mlcce')

        # initial queries
        for i in range(len(self.price_iterates)-1):
            bundle_queries, true_values = self.bundle_query(self.price_iterates[i])
            self.queried_bundles.append(bundle_queries)
            self.queried_values.append(true_values)
            [self.value_estimators[j].add_data_point(self.price_iterates[i], bundle_queries[j], true_values[j]) for j in range(self.num_participants)]
        
        while True:
            # Query the utility maximizing bundle from prosumers at the current price
            bundle_queries, true_values = self.bundle_query(self.price_iterates[-1])
            self.queried_bundles.append(bundle_queries)
            self.queried_values.append(true_values)

            imb = np.sum(bundle_queries, 0)
            print(f'MLCCE iteration {self.mlcce_iter}: Price: {self.price_iterates[-1]}. Imbalance product-wise: {imb}')


            if self.__is_equilibrium():
                break

            # Else learn new information and update prices
            # Learn value function
            with self.train_timer:
                for i in range(self.num_participants):
                    self.value_estimators[i].add_data_point(self.price_iterates[-1], bundle_queries[i], true_values[i])
                    _, metrics = self.value_estimators[i].dq_train_mvnn() #true values only for validation purposes
                    self.value_estimator_metrics[i]['train_dq_loss_scaled'].append(metrics[list(metrics)[-1]]['train_dq_loss_scaled'])
                    self.value_estimator_metrics[i]['val_dq_loss_scaled'].append(metrics[list(metrics)[-1]]['val_dq_loss_scaled'])
                    self.value_estimator_metrics[i]['r2_centered'].append(metrics[list(metrics)[-1]]['r2_centered'])
            
            with self.next_price_timer:
                noise_scale = np.abs(self.__get_pred_imb(self.price_iterates[-1]) - np.abs(np.sum(bundle_queries, 0))) * self.price_max / np.sum(self.capacities, 0)
                noise = np.random.normal(0, noise_scale/(self.mlcce_iter+1), len(self.price_iterates[-1]))
                price = np.clip(self.__next_price() + noise, -self.price_max, self.price_max)
                self.price_iterates.append(price)
                self.mlcce_iter += 1
        
            if self.mlcce_iter%5==0:
                self.__log_info(run)

        return self.price_iterates[-1], self.queried_bundles[-1]
