import scipy.optimize as opt
import numpy as np

class logarithmic_bidder:
    def __init__(self, intervals, flowlimit, scale=1):
        self.intervals = intervals
        self.flowlimit = flowlimit
        self.scale = scale
        

    def bundle_query(self, price):
        # c1 = opt.LinearConstraint(np.ones((1,self.intervals)), 0, 0)
        res = opt.minimize(self.neg_utility_func, x0=np.zeros(self.intervals), args=(price), constraints=[], bounds=[(-self.flowlimit, self.flowlimit)]*self.intervals)
        return res.x, -res.fun+np.dot(price, res.x)


    def get_value(self, x):
        return np.sum(np.log(x[i]+self.flowlimit*1.5) for i in range(self.intervals))*self.scale


    def neg_utility_func(self, x, price):
        return -self.get_value(x) + np.dot(price, x)


    def get_capacity_generic_goods(self, ):
        return np.array([self.flowlimit]*self.intervals)