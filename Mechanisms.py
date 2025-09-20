import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import wandb
from line_profiler import profile
from codetiming import Timer
import concurrent.futures
import logging
import gurobipy as gp

#own libs
from value_function_est import ValueFunctionEstimateDQ
from utils import mvnn_params, mip_params, next_price_params, cce_params, mlcce_params, log_mech_metrics

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
                                         'kendall_tau': [], 'kendall_tau_train': [],
                                         'dq_pred_error': []} for _ in range(self.num_participants)] # mentioned metrics are being tracked
        self.train_timer = Timer('train_timer', logger=None)
        self.next_price_timer = Timer('next_price_timer', logger=None)
        
        # value estimators initialized
        self.value_estimators = []
        for i in range(self.num_participants):
            mvnn_param = mvnn_params[bidders[i].__class__.__name__] if bidders[i].__class__.__name__ in mvnn_params.keys() else mvnn_params['ProsumerStorage']
            self.value_estimators.append(ValueFunctionEstimateDQ(capacity_generic_goods=self.capacities[i],
                                                                 mvnn_params=mvnn_param,
                                                                 mip_params=mip_params,
                                                                 price_scale=np.max(np.abs(price_bound)) # TODO: see if this works as expected
                                                                ))

    @profile
    def __next_price_grad(self, decvar: np.ndarray, prev_imb: np.ndarray, prev_price: np.ndarray):
        """
        Gradient of the next price objective with respect to the decision variables.
        """
        price = decvar[0] * prev_imb + prev_price
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        grad = -np.sum(predicted_bundles, axis=0)

        return np.dot(grad, prev_imb)
    
    @profile
    def __next_price_objective(self, decvar: np.ndarray, prev_imb: np.ndarray, prev_price: np.ndarray):
        """
        Objective for the next price optimization problem.
        """
        price = decvar[0] * prev_imb + prev_price
        predicted_bundles = [value_estimator.get_max_util_bundle(price) for value_estimator in self.value_estimators]
        predicted_values = np.array([value_estimator.get_bundle_value(bundle) for value_estimator, bundle in zip(self.value_estimators, predicted_bundles)])
        payments = np.array([np.dot(price, bundle) for bundle in predicted_bundles])
        return np.sum(predicted_values - payments)

    @profile
    def __next_price(self, ):
        """
        Solve the next price optimization problem.
        """
        initial_price = self.price_iterates[-1]
        step_bound = [self.cce_params['base_step'] / (self.mlcce_iter+self.n_init_prices), self.cce_params['base_step'] / np.pow(self.mlcce_iter+self.n_init_prices, 0.5)]
        bounds = []
        bounds.append((step_bound[0], step_bound[1]))
        solution = opt.minimize(self.__next_price_objective,
                                [step_bound[0]/2 + step_bound[1]/2],
                                jac=self.__next_price_grad,
                                args = (self.current_imb, initial_price),
                                bounds=bounds,
                                method=self.next_price_params['method'])

        print(f'Step: {solution.x[0]:.2f}, bounds: {step_bound[0]:.2f} - {step_bound[1]:.2f}')
        print(f'Solution status: {solution.success}: {solution.message}')
        
        return solution.x[0]*self.current_imb+initial_price, solution.x[0]


    def run(self, ):
        # First few rounds of CCE
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
            log_mech_metrics(self, self.price_iterates[-1], self.queried_bundles[-1], self.Lagrange_dual[-1], step=i)

            step = self.cce_params['base_step'] / (np.pow(i, self.cce_params['decay']))
            self.wandb_run.log({"Gradient step size": step}, step=i, commit=False)
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
            log_mech_metrics(self, self.price_iterates[-1], self.queried_bundles[-1], self.Lagrange_dual[-1], step=self.n_init_prices + self.mlcce_iter)


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
                price, step = self.__next_price()
                price = np.clip(price, self.price_bound[0], self.price_bound[1])
                self.price_iterates.append(price)
                self.mlcce_iter += 1


            # Value estimation and next price metric logging
            [self.wandb_run.log({f"{key}/Participant {i}": self.value_estimator_metrics[i][key][-1]}, step=self.n_init_prices+self.mlcce_iter, commit=False) for key in self.value_estimator_metrics[i].keys() for i in range(self.num_participants)]
            self.wandb_run.log({"MLCCE Train time": self.train_timer.timers.total('train_timer')}, step=self.n_init_prices+self.mlcce_iter, commit=False)
            if self.n_products == 2:
                [estimator.plot_nn(bidder, self.n_init_prices+self.mlcce_iter, id=i, wandb_run=self.wandb_run) for i, (bidder, estimator) in enumerate(zip(self.bidders, self.value_estimators))]
            self.wandb_run.log({"MLCCE Next price time": self.next_price_timer.timers.total('next_price_timer')}, step=self.n_init_prices+self.mlcce_iter, commit=False)
            self.wandb_run.log({"Gradient step size": step}, step=self.n_init_prices+self.mlcce_iter, commit=False)
        
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
            log_mech_metrics(self, self.price_iterates[-1], self.queried_bundles[-1], self.Lagrange_dual[-1], step=self.cce_iter)
            
            if self.cce_iter >= self.cce_params['max_iter']:
                break

            step = self.cce_params['base_step'] / np.pow(len(self.price_iterates), self.cce_params['decay'])
            self.price_iterates.append(self.price_iterates[-1] - step * gradient)
            self.wandb_run.log({"Gradient step size": step}, step=self.cce_iter, commit=False)
        
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


class CHP:
    """
    Computes convex hull prices"""
    def __init__(self, bidders: list, price_bound: tuple, wandb_run):
        self.bidders = bidders
        self.wandb_run = wandb_run

        # Params
        self.capacities = [bidder.get_capacity_generic_goods() for bidder in self.bidders]
        self.n_products = len(self.capacities[0][0])
        self.price_bound = price_bound

        self.imbalance_norm = []


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
        print(f'imbalance: {np.round(imb, 2)}')
        print(f'clearing price: {np.round(solution.x, 2)}')
        print(f'Lagrange dual: {objective(solution.x):.2f}')
        dispatch = [bidder.bundle_query(solution.x)[0] for bidder in self.bidders]
        
        log_mech_metrics(self, price=solution.x, bundle=dispatch, value=objective(solution.x), step=1)
        
        return solution.x, dispatch


class FullInfo:
    def __init__(self, bidders: list, n_products: int, wandb_run):
        """
        Full information benchmark - maximizes social welfare with complete information.
        """
        self.bidders = bidders
        self.wandb_run = wandb_run

        # Params
        self.n_products = n_products

        self.model = gp.Model("FullInfo", env=gp.Env(params={'LogToConsole': 0, 'OutputFlag': 0}))

    def run(self, ):
        dispatch_vars = []
        obj = 0
        for bidder in self.bidders:
            dispatch_var, obj_expr = bidder.add_model(self.model)
            dispatch_vars.append(dispatch_var)
            obj += obj_expr
        self.model.setObjective(obj, gp.GRB.MAXIMIZE)
        self.model.addConstrs((gp.quicksum(dispatch_vars[i][j] for i in range(len(self.bidders))) == 0 for j in range(self.n_products)), name="Trade balance")
        self.model.optimize()

        if self.model.status == gp.GRB.OPTIMAL:
            dispatch = [[dispatch_var[n].x for n in range(self.n_products)] for dispatch_var in dispatch_vars]
            primal_obj = self.model.ObjVal
        
        print(f'FullInfo objective: {primal_obj:.2f}')
        self.wandb_run.log({"Welfare": primal_obj}, commit=False)
        [self.wandb_run.log({f"Product_traded {j}": np.sum(np.abs(dispatch), 0)[j]}, commit=False) for j in range(self.n_products)]
        
        return None, dispatch


class FullInfoSequential:
    """ Sequential energy and flexibility market with full information - maximizes social welfare with complete information. """
    def __init__(self, bidders: list, n_products: int, wandb_run):
        self.bidders = bidders
        self.wandb_run = wandb_run

        # Params
        self.n_products = n_products
        assert n_products % 2 == 0, "Number of products must be even for FullInfoSequential mechanism"
        self.nH = int(n_products / 2)  # number of hours

        self.enermodel = gp.Model("FullInfo", env=gp.Env(params={'LogToConsole': 0, 'OutputFlag': 0}))
        self.flexmodel = gp.Model("FullInfo", env=gp.Env(params={'LogToConsole': 0, 'OutputFlag': 0}))

    def run(self, ):
        # Run energy only market
        dispatch_vars = []
        obj = 0
        for bidder in self.bidders:
            bidder.config['fixflex'] = [0] * self.nH  # disable flexibility
            bidder.config['fixener'] = False

            dispatch_var, obj_expr = bidder.add_model(self.enermodel)
            dispatch_vars.append(dispatch_var)
            obj += obj_expr
        self.enermodel.setObjective(obj, gp.GRB.MAXIMIZE)
        self.enermodel.addConstrs((gp.quicksum(dispatch_vars[i][j] for i in range(len(self.bidders))) == 0 for j in range(self.n_products)), name="Trade balance")
        self.enermodel.optimize()

        if self.enermodel.status == gp.GRB.OPTIMAL:
            enerdispatch = [[dispatch_var[n].x for n in range(self.n_products)] for dispatch_var in dispatch_vars]
            enerprimal_obj = self.enermodel.ObjVal


        # Run flexibility only market
        dispatch_vars = []
        obj = 0
        for i, bidder in enumerate(self.bidders):
            bidder.config['fixener'] = enerdispatch[i][:self.nH]  # fix energy dispatch
            bidder.config['fixflex'] = False

            dispatch_var, obj_expr = bidder.add_model(self.flexmodel)
            dispatch_vars.append(dispatch_var)
            obj += obj_expr
        self.flexmodel.setObjective(obj, gp.GRB.MAXIMIZE)
        self.flexmodel.addConstrs((gp.quicksum(dispatch_vars[i][j] for i in range(len(self.bidders))) == 0 for j in range(self.n_products)), name="Trade balance")
        self.flexmodel.optimize()

        if self.flexmodel.status == gp.GRB.OPTIMAL:
            flexdispatch = [[dispatch_var[n].x for n in range(self.n_products)] for dispatch_var in dispatch_vars]
            flexprimal_obj = self.flexmodel.ObjVal
        
        
        print(f'Sequential welfare: {flexprimal_obj:.2f}')
        self.wandb_run.log({"Welfare": flexprimal_obj}, commit=False)
        [self.wandb_run.log({f"Product_traded {j}": np.sum(np.abs(flexdispatch), 0)[j]}, commit=False) for j in range(self.n_products)]
        
        return None, flexdispatch