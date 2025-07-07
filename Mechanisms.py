import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from line_profiler import profile
from codetiming import Timer
import concurrent.futures

#own libs
from value_function_est import ValueFunctionEstimateDQ
from utils import mvnn_params, mip_params, next_price_params


class MLCCE:
    def __init__(self, n_init_prices: int, n_products: int, price_bound: tuple, imbalance_tol_coef: float, wandb_run, bidders: list = None):
        """
        :param query_func: function to query the utility maximizing bundle from participants - returns a list of bundles
        """
        self.bidders = bidders
        self.capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        self.num_participants = len(self.bidders)

        self.queried_bundles = [] # list (clock iterations) of list (prosumers) of queried bundles
        self.queried_values = [] # list (clock iterations) of list (prosumers) of true values
        self.true_lagrangian = []
        self.n_init_prices = n_init_prices
        self.n_products = n_products
        self.imbalance_tol = np.sum(self.capacities)*imbalance_tol_coef
        self.value_estimator_metrics = [{'train_dq_loss_scaled': [], 'val_dq_loss_scaled': [], 'r2_centered': []} for i in range(self.num_participants)]
        
        self.value_estimators = [ValueFunctionEstimateDQ(capacity_generic_goods=self.capacities[i],
                                                         mvnn_params=mvnn_params,
                                                         mip_params=mip_params,
                                                        ) for i in range(self.num_participants)]
        
        self.price_bound = price_bound
        self.eq_tol = 1e-3
        self.mlcce_iter = 0

        self.next_price_params = next_price_params

        self.train_timer = Timer('train_timer', logger=None)
        self.next_price_timer = Timer('next_price_timer', logger=None)

        self.wandb_run = wandb_run


    def __next_price_grad(self, price: np.ndarray):
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        return -np.sum(predicted_bundles, axis=0) + self.next_price_params['prox_coef']*(price - self.price_iterates[-1])
    

    def __next_price_objective(self, price: np.ndarray):
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        predicted_values = np.array([value_estimator.get_bundle_value(bundle) for value_estimator, bundle in zip(self.value_estimators, predicted_bundles)])
        payments = np.array([np.dot(price, bundle) for bundle in predicted_bundles])
        return np.sum(predicted_values - payments) + self.next_price_params['prox_coef'] * np.linalg.norm(price - self.price_iterates[-1], 2)**2


    def __next_price(self, ):
        """
        Calculate the next price iterate based on the bundles predicted by the learned value functions
        """
        

        initial_price = self.price_iterates[-1]
        # trust_region_rad = [self.next_price_params['trust_region_radius_coef'] * (self.price_bound[1][i]-self.price_bound[0][i]) * np.pow(self.mlcce_iter+1, self.next_price_params['trust_region_decay_pow']) for i in range(self.n_products)]
        # bounds=[(initial_price[i] - trust_region_rad[i], initial_price[i] + trust_region_rad[i]) for i in range(self.n_products)]
        solution = opt.minimize(self.__next_price_objective,
                                initial_price,
                                jac=self.__next_price_grad,
                                # bounds=bounds,
                                constraints=[opt.LinearConstraint(np.diag(self.current_imb), lb=self.current_imb*initial_price)],
                                method=self.next_price_params['method'])

        pred_imb = -self.__next_price_grad(solution.x)
        print(f'Predicted imbalance: {np.round(pred_imb, 2)}')
        
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
        if len(self.price_iterates[-1]) <=2:
            [estimator.plot_nn(bidder, self.mlcce_iter, wandb_run=self.wandb_run) for bidder, estimator in zip(self.bidders, self.value_estimators)]
        
        self.wandb_run.log({"MLCCE Train time": self.train_timer.timers.total('train_timer')}, step=self.mlcce_iter)
        self.wandb_run.log({"MLCCE Next price time": self.next_price_timer.timers.total('next_price_timer')}, step=self.mlcce_iter)
        
        data_price_iterates = [[j+1]+[self.price_iterates[j][i] for i in range(self.n_products)] for j in range(len(self.price_iterates))]
        data_imbalance = [[j+1]+list(np.sum(self.queried_bundles[j], 0)) for j in range(len(self.queried_bundles))]
        table_price_iterates = pd.DataFrame(data_price_iterates, columns=['iter']+[f'Product {i}' for i in range(self.n_products)])
        table_imbalance = pd.DataFrame(data_imbalance, columns=['iter']+[f'Product {i}' for i in range(self.n_products)])

        self.wandb_run.log({"MLCCE_price_iterates_plot": wandb.plot.line_series(xs=table_price_iterates['iter'],
                                                                    ys=[table_price_iterates[f'Product {i}'] for i in range(self.n_products)],
                                                                    keys=[f'Product {i}' for i in range(self.n_products)],
                                                                    title="Price Iterates",
                                                                    xname="Iteration"
                                                                   )}, commit=False)
        self.wandb_run.log({"MLCCE_imbalance_plot": wandb.plot.line_series(xs=table_imbalance['iter'],
                                                               ys=[table_imbalance[f'Product {i}'] for i in range(self.n_products)],
                                                               keys=[f'Product {i}' for i in range(self.n_products)],
                                                               title="Imbalance",
                                                               xname="Iteration"
                                                              )}, commit=False)
        
        data_train_dq_loss_scaled = [[i+1]+[self.value_estimator_metrics[j]['train_dq_loss_scaled'][i] for j in range(self.num_participants)] for i in range(self.mlcce_iter)]
        # data_val_dq_loss_scaled = [[i+1]+[self.value_estimator_metrics[j]['val_dq_loss_scaled'][i] for j in range(self.num_participants)] for i in range(self.mlcce_iter)]
        # data_r2_centered = [[i+1]+[self.value_estimator_metrics[j]['r2_centered'][i] for j in range(self.num_participants)] for i in range(self.mlcce_iter)]
        data_true_lagrangian = [[i+1, self.true_lagrangian[i]] for i in range(len(self.true_lagrangian))]
        
        table_train_dq_loss_scaled = pd.DataFrame(data=data_train_dq_loss_scaled, columns=['iter']+[f'Participant {i+1}' for i in range(self.num_participants)])
        # table_val_dq_loss_scaled = pd.DataFrame(data=data_val_dq_loss_scaled, columns=['iter']+[f'Participant {i+1}' for i in range(self.num_participants)])
        # table_r2_centered = pd.DataFrame(data=data_r2_centered, columns=['iter']+[f'Participant {i+1}' for i in range(self.num_participants)])
        table_true_lagrangian = pd.DataFrame(data=data_true_lagrangian, columns=['iter', 'true_lagrangian'])

        self.wandb_run.log({"MLCCE_train_dq_loss_scaled_plot": wandb.plot.line_series(xs=table_train_dq_loss_scaled['iter'],
                                                                                      ys=[table_train_dq_loss_scaled[f'Participant {i+1}'] for i in range(self.num_participants)],
                                                                                      keys=[f'Participant {i+1}' for i in range(self.num_participants)],
                                                                                      title="Train DQ Loss Scaled",
                                                                                      xname="MLCCE iteration"
                                                                                     )}, commit=False)
        # self.wandb_run.log({"MLCCE_val_dq_loss_scaled_plot": wandb.plot.line_series(xs=table_val_dq_loss_scaled['iter'],
        #                                                                             ys=[table_val_dq_loss_scaled[f'Participant {i+1}'] for i in range(self.num_participants)],
        #                                                                             keys=[f'Participant {i+1}' for i in range(self.num_participants)],
        #                                                                             title="Validation DQ Loss Scaled",
        #                                                                             xname="MLCCE iteration"
        #                                                                            )})
        # self.wandb_run.log({"MLCCE_r2_centered_plot": wandb.plot.line_series(xs=table_r2_centered['iter'],
        #                                                                      ys=[table_r2_centered[f'Participant {i+1}'] for i in range(self.num_participants)],
        #                                                                      keys=[f'Participant {i+1}' for i in range(self.num_participants)],
        #                                                                      title="R2 Centered",
        #                                                                      xname="MLCCE iteration"
        #                                                                     )})
        self.wandb_run.log({"MLCCE_true_lagrangian": wandb.plot.line_series(xs=table_true_lagrangian['iter'],
                                                                            ys=[table_true_lagrangian['true_lagrangian']],
                                                                            title="True lagrangian",
                                                                            xname="Iteration"
                                                                            )})
    
    @profile
    def run(self, ):
        """
        Machine learning based combinatorial clock exchange. This function conducts the exchange auction and returns a clearing price and dispatch schedule.
        """

        # initial queries
        self.price_iterates = [np.zeros(self.n_products)] # start with zero price

        for i in range(1, self.n_init_prices):
            self.queried_bundles.append([])
            self.queried_values.append([])
            self.true_lagrangian.append(0)
            for j in range(self.num_participants):
                bundle, true_value = self.bidders[j].bundle_query(self.price_iterates[i-1])
                self.queried_bundles[-1].append(bundle)
                self.queried_values[-1].append(true_value)
                self.true_lagrangian[-1] += true_value - np.dot(self.price_iterates[i-1], bundle)
                self.value_estimators[j].add_data_point(self.price_iterates[i-1], bundle, true_value)
            
            self.price_iterates.append(self.price_iterates[i-1] + np.sum(self.queried_bundles[-1], 0))
        
        while True:
            # Query the utility maximizing bundle from prosumers at the current price

            self.queried_bundles.append([])
            self.queried_values.append([])
            self.true_lagrangian.append(0)
            for j in range(self.num_participants):
                bundle, true_value = self.bidders[j].bundle_query(self.price_iterates[-1])
                self.queried_bundles[-1].append(bundle)
                self.queried_values[-1].append(true_value)
                self.true_lagrangian[-1] += true_value - np.dot(self.price_iterates[-1], bundle)

            self.current_imb = np.sum(self.queried_bundles[-1], 0)
            print(f'MLCCE iteration {self.mlcce_iter}: Price: {np.round(self.price_iterates[-1], 2)}. Imbalance product-wise: {np.round(self.current_imb, 2)} \n')

            if self.__is_equilibrium():
                break

            # Else learn new information and update prices
            # Learn value function - can parallelize
            def train_helper(i):
                _, metrics = self.value_estimators[i].dq_train_mvnn() #true values only for validation purposes
                return {'train_dq_loss_scaled': metrics[list(metrics)[-1]]['train_dq_loss_scaled'],
                        # 'val_dq_loss_scaled': metrics[list(metrics)[-1]]['val_dq_loss_scaled'],
                        # 'r2_centered': metrics[list(metrics)[-1]]['r2_centered']
                        }
    
            with self.train_timer:
                [self.value_estimators[i].add_data_point(self.price_iterates[-1], self.queried_bundles[-1][i], self.queried_values[-1][i]) for i in range(self.num_participants)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                        for i, metrics in enumerate(executor.map(train_helper, range(len(self.value_estimators)))):
                # for i in range(self.num_participants):
                #     metrics = train_helper(i)
                            self.value_estimator_metrics[i]['train_dq_loss_scaled'].append(metrics['train_dq_loss_scaled'])
                            # self.value_estimator_metrics[i]['val_dq_loss_scaled'].append(metrics['val_dq_loss_scaled'])
                            # self.value_estimator_metrics[i]['r2_centered'].append(metrics['r2_centered'])

            
            with self.next_price_timer:
                # noise_scale = np.abs(-self.__next_price_grad(self.price_iterates[-1]) - np.abs(np.sum(self.queried_bundles[-1], 0))) * self.price_max / np.sum(self.capacities, 0)
                # noise = np.random.normal(0, noise_scale/(self.mlcce_iter+1), len(self.price_iterates[-1]))
                noise = 0
                price = np.clip(self.__next_price() + noise, self.price_bound[0], self.price_bound[1])

                self.price_iterates.append(price)

                self.mlcce_iter += 1
        
            if self.mlcce_iter%5==0:
                self.__log_info()

        return self.price_iterates[-1], self.queried_bundles[-1]


class CHP_fullinfo:
    def __init__(self, bidders: list, price_bound: tuple, wandb_run):
        self.bidders = bidders
        self.capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        self.n_products = len(self.capacities[0][0])
        self.price_bound = price_bound
        self.wandb_run = wandb_run


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
        
        initial_price = np.zeros(self.n_products)
        bounds=[(self.price_bound[0][i], self.price_bound[1][i]) for i in range(self.n_products)]
        solution = opt.minimize(objective, initial_price, jac=gradient, bounds=bounds)

        imb = -gradient(solution.x)
        print(f'CHP imbalance: {imb}')
        print(f'CHP clearing price: {np.round(solution.x, 2)}')
        print(f'CHP Lagrangian: {objective(solution.x)}')

        dispatch = [bidder.bundle_query(solution.x)[0] for bidder in self.bidders]

        fig, ax = plt.subplots()
        [ax.plot(np.arange(1, self.n_products+1, 1), dispatch[i], label=f'{self.bidders[i].get_name()}') for i in range(len(dispatch))]
        ax.legend()
        
        if self.wandb_run is not None:
            self.wandb_run.log({f"CHP_dispatch": wandb.Image(fig)})
        
        return solution.x, dispatch
    