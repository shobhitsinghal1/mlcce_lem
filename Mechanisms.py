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
    def __init__(self, n_init_prices: int, n_products: int, query_func: callable, num_participants: int, price_max: float, capacities: list, imbalance_tol_coef: float, logname: str, bidders: list = None):
        """
        :param query_func: function to query the utility maximizing bundle from participants - returns a list of bundles
        """
        self.queried_bundles = [] # list (clock iterations) of list (prosumers) of queried bundles
        self.queried_values = [] # list (clock iterations) of list (prosumers) of true values
        self.n_init_prices = n_init_prices
        self.n_products = n_products
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

        self.wandb_run = wandb.init(entity='shosi-danmarks-tekniske-universitet-dtu', project='mlcce', name=logname)


    def __next_price_grad(self, price: np.ndarray):
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        return -np.sum(predicted_bundles, axis=0) # sum over all participants
    

    def __next_price_objective(self, price: np.ndarray):
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        predicted_values = np.array([value_estimator.get_bundle_value(bundle) for value_estimator, bundle in zip(self.value_estimators, predicted_bundles)])
        payments = np.array([np.dot(price, bundle) for bundle in predicted_bundles])
        return np.sum(predicted_values - payments) # sum over all participants


    def __next_price(self, ):
        """
        Calculate the next price iterate based on the bundles predicted by the learned value functions
        """
        
        initial_price = self.price_iterates[-1]
        trust_region_rad = self.next_price_params['trust_region_radius_coef']*self.price_max*np.pow(self.mlcce_iter+1, self.next_price_params['trust_region_decay_pow'])
        bounds=[(self.price_iterates[-1][i]-trust_region_rad, 
                 self.price_iterates[-1][i]+trust_region_rad) for i in range(len(self.price_iterates[-1]))]
        solution = opt.minimize(self.__next_price_objective, initial_price, jac=self.__next_price_grad, bounds=bounds, method=self.next_price_params['method'])

        pred_imb = -self.__next_price_grad(solution.x)
        print(f'Predicted imbalance: {pred_imb}')
        
        return solution.x


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


    def __log_info(self, ):
        self.wandb_run.log({"Train time": self.train_timer.timers.total('train_timer')})
        self.wandb_run.log({"Next price time": self.next_price_timer.timers.total('next_price_timer')})
        
        n_prod = len(self.price_iterates[0])
        data_price_iterates = [[j+1]+[self.price_iterates[j][i] for i in range(n_prod)] for j in range(len(self.price_iterates))]
        data_imbalance = [[j+1]+list(np.sum(self.queried_bundles[j], 0)) for j in range(len(self.queried_bundles))]
        table_price_iterates = pd.DataFrame(data_price_iterates, columns=['iter']+[f'Product {i}' for i in range(n_prod)])
        table_imbalance = pd.DataFrame(data_imbalance, columns=['iter']+[f'Product {i}' for i in range(n_prod)])

        self.wandb_run.log({"price_iterates_plot": wandb.plot.line_series(xs=table_price_iterates['iter'],
                                                                    ys=[table_price_iterates[f'Product {i}'] for i in range(n_prod)],
                                                                    keys=[f'Product {i}' for i in range(n_prod)],
                                                                    title="Price Iterates",
                                                                    xname="Iteration"
                                                                   )})
        self.wandb_run.log({"imbalance_plot": wandb.plot.line_series(xs=table_imbalance['iter'],
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

        self.wandb_run.log({"train_dq_loss_scaled_plot": wandb.plot.line_series(xs=table_train_dq_loss_scaled['iter'],
                                                                          ys=[table_train_dq_loss_scaled[f'Participant {i+1}'] for i in range(self.num_participants)],
                                                                          keys=[f'Participant {i+1}' for i in range(self.num_participants)],
                                                                          title="Train DQ Loss Scaled",
                                                                          xname="MLCCE iteration"
                                                                         )})
        self.wandb_run.log({"val_dq_loss_scaled_plot": wandb.plot.line_series(xs=table_val_dq_loss_scaled['iter'],
                                                                        ys=[table_val_dq_loss_scaled[f'Participant {i+1}'] for i in range(self.num_participants)],
                                                                        keys=[f'Participant {i+1}' for i in range(self.num_participants)],
                                                                        title="Validation DQ Loss Scaled",
                                                                        xname="MLCCE iteration"
                                                                       )})
        self.wandb_run.log({"r2_centered_plot": wandb.plot.line_series(xs=table_r2_centered['iter'],
                                                                 ys=[table_r2_centered[f'Participant {i+1}'] for i in range(self.num_participants)],
                                                                 keys=[f'Participant {i+1}' for i in range(self.num_participants)],
                                                                 title="R2 Centered",
                                                                 xname="MLCCE iteration"
                                                                )})
        
        if len(self.price_iterates[-1]) <=2:
            [estimator.plot_nn(bidder, wandb_run=self.wandb_run) for bidder, estimator in zip(self.bidders, self.value_estimators)]


    @profile
    def run(self, ):
        """
        Machine learning based combinatorial clock exchange. This function conducts the exchange auction and returns a clearing price and dispatch schedule.
        """

        # initial queries
        self.price_iterates = [np.zeros(self.n_products)] # start with zero price
        for i in range(1, self.n_init_prices):
            bundle_queries, true_values = self.bundle_query(self.price_iterates[i-1])
            self.queried_bundles.append(bundle_queries)
            self.queried_values.append(true_values)
            [self.value_estimators[j].add_data_point(self.price_iterates[i-1], bundle_queries[j], true_values[j]) for j in range(self.num_participants)]
            self.price_iterates.append(self.price_iterates[i-1] + np.sum(bundle_queries, 0))
        
        while True:
            # Query the utility maximizing bundle from prosumers at the current price
            bundle_queries, true_values = self.bundle_query(self.price_iterates[-1])
            self.queried_bundles.append(bundle_queries)
            self.queried_values.append(true_values)

            imb = np.sum(bundle_queries, 0)
            print(f'MLCCE iteration {self.mlcce_iter}: Price: {np.round(self.price_iterates[-1], 2)}. Imbalance product-wise: {np.round(imb, 2)}')


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
                noise_scale = np.abs(-self.__next_price_grad(self.price_iterates[-1]) - np.abs(np.sum(bundle_queries, 0))) * self.price_max / np.sum(self.capacities, 0)
                noise = np.random.normal(0, noise_scale/(self.mlcce_iter+1), len(self.price_iterates[-1]))
                price = np.clip(self.__next_price() + noise, -self.price_max, self.price_max)
                self.price_iterates.append(price)
                self.mlcce_iter += 1
        
            if self.mlcce_iter%5==0:
                self.__log_info()

        return self.price_iterates[-1], self.queried_bundles[-1]


class CHP_fullinfo:
    def __init__(self, bidders: list, capacities: list):
        self.bidders = bidders
        self.capacities = capacities
        pass


    def run(self, ):
        """
        Calculate the convex hull price based on full access to the bidders' value functions.
        """

        def gradient(price: np.ndarray) -> np.ndarray:
            gradient = np.zeros_like(price)
            for i, bidder in enumerate(self.bidders):
                bundle, _ = bidder.bundle_query(price)
                gradient -= bundle

            return gradient
        
        def objective(price: np.ndarray) -> float:
            objective = 0
            for i, bidder in enumerate(self.bidders):
                bundle, value = bidder.bundle_query(price)
                objective += value - np.dot(price, bundle)

            return objective
        
        initial_price = np.zeros_like(self.capacities[0])
        solution = opt.minimize(objective, initial_price, jac=gradient)

        imb = -gradient(solution.x)
        print(f'CHP imbalance: {imb}')
        
        return solution.x
    