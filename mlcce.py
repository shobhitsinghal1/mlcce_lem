import numpy as np
from value_function_est import ValueFunctionEstimator

class MLCCE:
    def __init__(self, price_init: np.ndarry, query_func: callable, num_participants: int):
        """
        :param query_func: function to query the utility maximizing bundle from participants - returns a list of bundles
        """
        self.queried_bundles = [] # list of list of queried bundles
        self.price_iterates = [price_init]
        self.bundle_query = query_func
        self.num_participants = num_participants
        self.value_estimators = [ValueFunctionEstimator() for i in range(num_participants)]

    def run_mlcce(self, ):
        """
        Machine learning based combinatorial clock exchange. This function conducts the exchange auction and returns a clearing price and dispatch schedule.
        """
        while True:
            # Query the utility maximizing bundle from prosumers at the current price
            bundle_queries = self.bundle_query(self.price_iterates[-1])
            self.queried_bundles.append(bundle_queries)

            if self.__is_equilibrium():
                break

            # Else learn new information and update prices
            # Learn value function
            for i in range(self.num_participants):
                self.value_estimators[i].train(self.queried_bundles)
            
            self.price_iterates.append(self.next_price())
        
        return self.price_iterates[-1], self.queried_bundles[-1]
