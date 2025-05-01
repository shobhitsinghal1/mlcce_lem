import numpy as np
from value_function_est import ValueFunctionEstimator
import scipy.optimize as opt

class MLCCE:
    def __init__(self, price_init: np.ndarry, query_func: callable, num_participants: int):
        """
        :param query_func: function to query the utility maximizing bundle from participants - returns a list of bundles
        """
        self.queried_bundles = [] # list (clock iterations) of list (prosumers) of queried bundles
        self.queried_values = [] # list (clock iterations) of list (prosumers) of true values
        self.price_iterates = [price_init]
        self.bundle_query = query_func # returns list of bundles and list of true values
        self.num_participants = num_participants
        self.value_estimators = [ValueFunctionEstimator() for i in range(num_participants)]

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
        
        solution = opt.minimize(objective, self.price_iterates[-1], method='BFGS', jac=gradient, bounds=[(-150, 150)]*len(self.price_iterates[-1]))
        return solution.x

    
    def run_mlcce(self, ):
        """
        Machine learning based combinatorial clock exchange. This function conducts the exchange auction and returns a clearing price and dispatch schedule.
        """
        while True:
            # Query the utility maximizing bundle from prosumers at the current price
            bundle_queries, true_values = self.bundle_query(self.price_iterates[-1])
            self.queried_bundles.append(bundle_queries)
            self.queried_values.append(true_values)

            if self.__is_equilibrium():
                break

            # Else learn new information and update prices
            # Learn value function
            for i in range(self.num_participants):
                self.value_estimators[i].add_data_point(bundle_queries[i], true_values[i])
                self.value_estimators[i].dq_train_mvnn(self.queried_bundles, self.queried_values) #true values only for validation purposes
            
            self.price_iterates.append(self.__next_price())
        
        return self.price_iterates[-1], self.queried_bundles[-1]
