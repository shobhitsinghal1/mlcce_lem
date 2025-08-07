import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import wandb
from line_profiler import profile
from codetiming import Timer
import concurrent.futures
import logging

#own libs
from value_function_est import ValueFunctionEstimateDQ
from utils import mvnn_params, mip_params, next_price_params, cce_params, mlcce_params

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

class MLCCE:
    def __init__(self, n_products: int, price_bound: tuple, wandb_run, bidders: list = None):
        self.bidders = bidders
        self.wandb_run = wandb_run

        # params
        self.n_init_prices = mlcce_params['cce_rounds']
        self.n_products = n_products
        self.capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        self.num_participants = len(self.bidders)
        self.price_bound = price_bound
        self.eq_tol = 1e-3
        self.mlcce_iter = 0
        self.next_price_params = next_price_params
        self.cce_params = cce_params
        self.mlcce_params = mlcce_params

        # variables and metrics
        self.price_iterates = []
        self.queried_bundles = [] # list (clock iterations) of list (prosumers) of queried bundles
        self.queried_values = [] # list (clock iterations) of list (prosumers) of true values
        self.Lagrange_dual = []
        self.imbalance_norm = []
        self.value_estimator_metrics = [{'train_dq_loss_scaled': [], 'val_dq_loss_scaled': [],
                                         'r2_centered': [], 'r2_centered_train': [],
                                         'kendall_tau': [], 'kendall_tau_train': []} for _ in range(self.num_participants)] # mentioned metrics are being tracked
        self.train_timer = Timer('train_timer', logger=None)
        self.next_price_timer = Timer('next_price_timer', logger=None)
        
        # value estimators initialized
        self.value_estimators = [ValueFunctionEstimateDQ(capacity_generic_goods=self.capacities[i],
                                                         mvnn_params=mvnn_params,
                                                         mip_params=mip_params,
                                                        ) for i in range(self.num_participants)]


    def __next_price_grad(self, decvar: np.ndarray):
        price = decvar[:-1]
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        grad = np.append(-np.sum(predicted_bundles, axis=0) + 2*self.next_price_params['prox_coef']*(price - self.price_iterates[-1]), 0)
        return grad
    

    def __next_price_objective(self, decvar: np.ndarray):
        price = decvar[:-1]
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        predicted_values = np.array([value_estimator.get_bundle_value(bundle) for value_estimator, bundle in zip(self.value_estimators, predicted_bundles)])
        payments = np.array([np.dot(price, bundle) for bundle in predicted_bundles])
        return np.sum(predicted_values - payments) + self.next_price_params['prox_coef'] * np.linalg.norm(price - self.price_iterates[-1], 2)**2


    def __next_price(self, ):
        initial_price = self.price_iterates[-1]
        step_bound = [self.cce_params['base_step'] / (self.mlcce_iter+1), self.cce_params['base_step'] / np.pow(self.mlcce_iter+1, 0.5)]
        bounds=[(self.price_bound[0][i], self.price_bound[1][i]) for i in range(self.n_products)].append((step_bound[0], step_bound[1]))
        solution = opt.minimize(self.__next_price_objective,
                                np.append(initial_price, self.cce_params['base_step'] / (self.mlcce_iter+1)),
                                jac=self.__next_price_grad,
                                bounds=bounds,
                                constraints=[opt.LinearConstraint(np.append(np.eye(self.n_products), -self.current_imb.reshape(-1, 1), 1), 
                                                                  lb=initial_price,
                                                                  ub=initial_price)],
                                method=self.next_price_params['method'])

        pred_imb = -self.__next_price_grad(solution.x)[:-1]
        print(f'Predicted imbalance: {np.round(pred_imb, 2)}')
        
        return solution.x[:-1]


    def __log_price(self, price, step):
        [self.wandb_run.log({f"Price/Product {i}": price[i]}, step=step, commit=False) for i in range(self.n_products)] # log prices
    
    def __log_imbalance(self, bundle, step):
        imb = np.sum(bundle, 0)
        total_capacity = np.sum(self.capacities, 0)
        rel_imb = np.where(imb >= 0, imb * 100 / total_capacity[1], imb * 100 / total_capacity[0])

        imbalance_norm = np.linalg.norm(rel_imb, 1)
        self.imbalance_norm.append(imbalance_norm)
        [self.wandb_run.log({f"Imbalance/Product {i}": rel_imb[i]}, step=step, commit=False) for i in range(self.n_products)] # log imbalance
        self.wandb_run.log({f"Imbalance_norm": imbalance_norm}, step=step, commit=False)
    
    def __log_Lagrange_dual(self, value, step):
        self.wandb_run.log({"Lagrange_dual": value}, step=step, commit=False)


    def run(self, ):
        # CCE
        self.price_iterates.append(self.price_bound[0]*0.8 + self.price_bound[1]*0.2) # initial prices

        for i in range(1, self.n_init_prices):
            self.queried_bundles.append([])
            self.queried_values.append([])
            self.Lagrange_dual.append(0)
            for j in range(self.num_participants):
                bundle, true_value = self.bidders[j].bundle_query(self.price_iterates[i-1])
                self.queried_bundles[-1].append(bundle)
                self.queried_values[-1].append(true_value)
                self.Lagrange_dual[-1] += true_value - np.dot(self.price_iterates[i-1], bundle)
                self.value_estimators[j].add_data_point(self.price_iterates[i-1], bundle, true_value)
            self.__log_price(self.price_iterates[-1], step=i)
            self.__log_imbalance(self.queried_bundles[-1], step=i)
            self.__log_Lagrange_dual(self.Lagrange_dual[-1], step=i)
            # self.wandb_run.log({}) # Commit

            step = self.cce_params['base_step'] / (np.pow(i, self.cce_params['decay']))
            self.price_iterates.append(self.price_iterates[i-1] + np.sum(self.queried_bundles[-1], 0) * step)


        # MLCCE
        while True:
            # Query the utility maximizing bundle from prosumers at the current price
            self.queried_bundles.append([])
            self.queried_values.append([])
            self.Lagrange_dual.append(0)
            for j in range(self.num_participants):
                bundle, true_value = self.bidders[j].bundle_query(self.price_iterates[-1])
                self.queried_bundles[-1].append(bundle)
                self.queried_values[-1].append(true_value)
                self.Lagrange_dual[-1] += true_value - np.dot(self.price_iterates[-1], bundle)
            self.__log_price(self.price_iterates[-1], step=self.n_init_prices + self.mlcce_iter)
            self.__log_imbalance(self.queried_bundles[-1], step=self.n_init_prices + self.mlcce_iter)
            self.__log_Lagrange_dual(self.Lagrange_dual[-1], step=self.n_init_prices + self.mlcce_iter)


            # Print information
            self.current_imb = np.sum(self.queried_bundles[-1], 0)
            print(f'MLCCE iteration {self.mlcce_iter}: Price: {np.round(self.price_iterates[-1], 2)}. Imbalance product-wise: {np.round(self.current_imb, 2)} \n')

            # Stopping condition
            if self.mlcce_iter >= self.mlcce_params['max_iter'] - self.mlcce_params['cce_rounds']:
                break


            # Else learn new information
            def train_helper(i):
                _, metrics = self.value_estimators[i].dq_train_mvnn() #true values only for validation purposes
                return metrics
    
            with self.train_timer:
                [self.value_estimators[i].add_data_point(self.price_iterates[-1], self.queried_bundles[-1][i], self.queried_values[-1][i]) for i in range(self.num_participants)]
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    for i, metrics in enumerate(executor.map(train_helper, range(self.num_participants))):
                        for key in self.value_estimator_metrics[i].keys():
                            self.value_estimator_metrics[i][key].append(metrics[key])


            # Update prices
            with self.next_price_timer:
                price = np.clip(self.__next_price(), self.price_bound[0], self.price_bound[1])
                self.price_iterates.append(price)
                self.mlcce_iter += 1


            # Value estimation and next price metric logging
            [self.wandb_run.log({f"{key}/Participant {i}": self.value_estimator_metrics[i][key][-1]}, step=self.n_init_prices+self.mlcce_iter, commit=False) for key in self.value_estimator_metrics[i].keys() for i in range(self.num_participants)]
            self.wandb_run.log({"MLCCE Train time": self.train_timer.timers.total('train_timer')}, step=self.n_init_prices+self.mlcce_iter, commit=False)
            if self.n_products == 2:
                [estimator.plot_nn(bidder, self.n_init_prices+self.mlcce_iter, wandb_run=self.wandb_run) for bidder, estimator in zip(self.bidders, self.value_estimators)]
            self.wandb_run.log({"MLCCE Next price time": self.next_price_timer.timers.total('next_price_timer')}, step=self.n_init_prices+self.mlcce_iter, commit=False)
        
        # Post-processing
        best_iter = np.argmin(self.imbalance_norm)
        dispatch = self.queried_bundles[best_iter]
        clearing_price = self.price_iterates[best_iter]
        
        
        return clearing_price, dispatch


class CCE:
    def __init__(self, bidders: list, price_bound: tuple, wandb_run):
        self.bidders = bidders
        self.price_bound = price_bound
        self.wandb_run = wandb_run

        # Params
        self.capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        self.n_products = len(self.capacities[0][0])
        self.cce_params = cce_params

        # Variables and metrics
        self.price_iterates = []
        self.queried_bundles = []
        self.Lagrange_dual = []
        self.imbalance_norm = []
        self.cce_iter = 0


    def __log_price(self, price, step):
        [self.wandb_run.log({f"Price/Product {i}": price[i]}, step=step, commit=False) for i in range(self.n_products)] # log prices
    
    def __log_imbalance(self, bundle, step):
        imb = np.sum(bundle, 0)
        total_capacity = np.sum(self.capacities, 0)
        rel_imb = np.where(imb >= 0, imb * 100 / total_capacity[1], imb * 100 / total_capacity[0])

        imbalance_norm = np.linalg.norm(rel_imb, 1)
        self.imbalance_norm.append(imbalance_norm)
        [self.wandb_run.log({f"Imbalance/Product {i}": rel_imb[i]}, step=step, commit=False) for i in range(self.n_products)] # log imbalance
        self.wandb_run.log({f"Imbalance_norm": imbalance_norm}, step=step, commit=False)
    
    def __log_Lagrange_dual(self, value, step):
        self.wandb_run.log({"Lagrange_dual": value}, step=step, commit=False)


    def run(self, ):
        self.price_iterates.append(self.price_bound[0]*0.8 + self.price_bound[1]*0.2) # Initial prices
        
        while True:
            self.queried_bundles.append([])
            self.Lagrange_dual.append(0)
            gradient = np.zeros(self.n_products)
            for i, bidder in enumerate(self.bidders):
                bundle, true_value = bidder.bundle_query(self.price_iterates[-1])
                self.queried_bundles[-1].append(bundle)
                self.Lagrange_dual[-1] += true_value - np.dot(self.price_iterates[-1], bundle)
                gradient -= bundle
            
            self.cce_iter += 1 # increment iteration
            self.__log_price(self.price_iterates[-1], step=self.cce_iter)
            self.__log_imbalance(self.queried_bundles[-1], step=self.cce_iter)
            self.__log_Lagrange_dual(self.Lagrange_dual[-1], step=self.cce_iter)
            self.wandb_run.log({}) # Commit
            
            if self.cce_iter >= self.cce_params['max_iter']:
                break

            step = self.cce_params['base_step'] / np.pow(len(self.price_iterates), self.cce_params['decay'])
            self.price_iterates.append(self.price_iterates[-1] - step * gradient)
        
        # Post-processing
        best_iter = np.argmin(self.imbalance_norm)
        dispatch = self.queried_bundles[best_iter]
        clearing_price = self.price_iterates[best_iter]

        print(f'CCE imbalance: {np.round(np.sum(dispatch, 0), 2)}')
        print(f'CCE clearing price: {np.round(clearing_price, 2)}')
        print(f'CCE Lagrangian: {self.Lagrange_dual[best_iter]:.2f}')

        fig, ax = plt.subplots()
        [ax.plot(np.arange(1, self.n_products+1, 1), dispatch[i], label=f'{self.bidders[i].get_name()}') for i in range(len(dispatch))]
        ax.legend()

        if self.wandb_run is not None:
            self.wandb_run.log({f"CCE_dispatch": wandb.Image(fig)}, commit=False)
            plt.close(fig)
        
        return clearing_price, dispatch


class CHP_fullinfo:
    def __init__(self, bidders: list, price_bound: tuple, wandb_run):
        self.bidders = bidders
        self.capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        print(f'Capacities: {np.round(np.sum(self.capacities, 0), 0)}')
        self.n_products = len(self.capacities[0][0])
        self.price_bound = price_bound
        self.wandb_run = wandb_run


    def run(self, ):
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
        print(f'CHP imbalance: {np.round(imb, 2)}')
        print(f'CHP clearing price: {np.round(solution.x, 2)}')
        print(f'CHP Lagrangian: {objective(solution.x):.2f}')

        dispatch = [bidder.bundle_query(solution.x)[0] for bidder in self.bidders]

        fig, ax = plt.subplots()
        [ax.plot(np.arange(1, self.n_products+1, 1), dispatch[i], label=f'{self.bidders[i].get_name()}') for i in range(len(dispatch))]
        ax.legend()
        
        if self.wandb_run is not None:
            self.wandb_run.log({f"CHP_dispatch": wandb.Image(fig)}, commit=False)
        
        return solution.x, dispatch
    