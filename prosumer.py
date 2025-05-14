from utils import *
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.repn.plugins.lp_writer import LPWriter
import logging
from abc import ABC, abstractmethod
from scipy import optimize as opt
import gurobipy as gp
from gurobipy import GRB

logging.getLogger('pyomo.core').setLevel(logging.ERROR)

class Bidder(ABC):
    @abstractmethod
    def bundle_query(self, price: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def get_value(self, bundle: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_capacity_generic_goods(self, ) -> np.ndarray:
        pass



class Prosumer(Bidder):
    def __init__(self, prosumer: str):
        self.prosumer_config = bidder_configs[prosumer]
        self.asset_configs = asset_configs
        self.asset_configs_by_type = self.__group_configs_by_type()
        self.dq_model = self.__construct_demand_query_model()
    
    def __group_configs_by_type(self, ) -> dict:
        """
        Group the asset configurations by their asset type.
        :return: dictionary with asset types as keys and lists of asset configurations as values.
        """
        assets_by_type = {}
        for asset in self.prosumer_config["assets"]:
            asset_type = self.asset_configs[asset]["type"]
            if asset_type not in assets_by_type.keys():
                assets_by_type[asset_type] = []
            assets_by_type[asset_type].append(self.asset_configs[asset])
        return assets_by_type

    def __get_param_values(self, param_name: str, asset_type: str) -> dict:
        param_values = {}
        if asset_type in self.asset_configs_by_type.keys():
            asset_configs = self.asset_configs_by_type[asset_type]
            for i in range(len(asset_configs)):
                if isinstance(asset_configs[i][param_name], list):
                    for j in range(len(asset_configs[i][param_name])):
                        param_values[(i, j)] = asset_configs[i][param_name][j]
                else:
                    param_values[i] = asset_configs[i][param_name]
        return param_values
        
    def __add_asset_constraints(self, model: pyo.AbstractModel):
        def conp(m, h):
            return m.p[h] == sum(m.pf[i, h] for i in m.If) + sum(m.ps[i, h] for i in m.Is)
        
        def conf1(m, i, h):
            return (m.gl[i,h], m.pf[i,h], m.gu[i,h])

        def cons1(m, i, h):
            return m.ps[i,h] <= m.M * m.deltap[i,h]

        def cons2(m, i, h):
            return m.ps[i,h] >= -m.M*(1-m.deltap[i,h])

        def cons3l(m, i, h):
            return m.s[i,h-1] + m.eta[i]*m.ps[i,h] - m.gamma[i,h] - m.M*(1-m.deltap[i,h]) <= m.s[i,h]

        def cons3u(m, i, h):
            return m.s[i,h] <= m.s[i,h-1] + m.eta[i]*m.ps[i,h] - m.gamma[i,h] + m.M*(1-m.deltap[i,h])

        def cons4l(m, i, h):
            return m.s[i,h-1] + m.ps[i,h]/m.eta[i] - m.gamma[i,h] - m.M*m.deltap[i,h] <= m.s[i,h]

        def cons4u(m, i, h):
            return m.s[i,h] <= m.s[i,h-1] + m.ps[i,h]/m.eta[i] - m.gamma[i,h] + m.M*m.deltap[i,h]

        def cons5(m, i, h):
            return m.s[i,h] <= m.su[i,h] + m.tu[i,h]

        def cons6(m, i, h):
            return m.s[i,h] >= m.sl[i,h] - m.tl[i,h]

        def cons7(m, i, h): #ramp limit
            return m.P[i,h] <= m.ramp_limit[i]
        
        def cons8(m, i, h): #capacity limit
            return (0, m.s[i,h], m.capacity[i])

        def cons9(m, i, h):
            return m.P[i,h] >= m.ps[i,h]

        def cons10(m, i, h):
            return m.P[i,h] >= -m.ps[i,h]

        def cons11(m, i):
            return m.s[i,-1] == m.s0[i]

        model.conp = pyo.Constraint(model.H, rule=conp, name='conp')
        model.conf1 = pyo.Constraint(model.If, model.H, rule=conf1, name='conf1')
        model.cons1 = pyo.Constraint(model.Is, model.H, rule=cons1, name='cons1')
        model.cons2 = pyo.Constraint(model.Is, model.H, rule=cons2, name='cons2')
        model.cons3u = pyo.Constraint(model.Is, model.H, rule=cons3u, name='cons3u')
        model.cons3l = pyo.Constraint(model.Is, model.H, rule=cons3l, name='cons3l')
        model.cons4u = pyo.Constraint(model.Is, model.H, rule=cons4u, name='cons4u')
        model.cons4l = pyo.Constraint(model.Is, model.H, rule=cons4l, name='cons4l')
        model.cons5 = pyo.Constraint(model.Is, model.H, rule=cons5, name='cons5')
        model.cons6 = pyo.Constraint(model.Is, model.H, rule=cons6, name='cons6')
        model.cons7 = pyo.Constraint(model.Is, model.H, rule=cons7, name='cons7')
        model.cons8 = pyo.Constraint(model.Is, model.H, rule=cons8, name='cons8')
        model.cons9 = pyo.Constraint(model.Is, model.H, rule=cons9, name='cons9')
        model.cons10 = pyo.Constraint(model.Is, model.H, rule=cons10, name='cons10')
        model.cons11 = pyo.Constraint(model.Is, rule=cons11, name='cons11')

        return
    
    def __add_dq_objective(self, model: pyo.AbstractModel):
        #objective

        def objf(m, i):
            return sum(m.pf[i, h] * m.lamb[i, h] for h in m.H)

        def objs(m, i):
            return sum(-m.tu[i,h]*m.lambu[i,h] - m.tl[i,h]*m.lambl[i,h] - m.P[i,h]*m.beta[i,h] for h in m.H)

        def obj_expression(m):
            return sum(objf(m, i) for i in m.If) + sum(objs(m, i) for i in m.Is) - sum(m.p[h]*m.price[h] for h in m.H)

        model.obj = pyo.Objective(rule=obj_expression, sense=pyo.maximize)
    
    def __construct_demand_query_model(self, ) -> pyo.AbstractModel:
        """
        Constructs the demand query optimization model parameterized in price.
        :return: demand query model.
        """
        model = pyo.AbstractModel()

        nIf = len(self.asset_configs_by_type['fixedload'] if 'fixedload' in self.asset_configs_by_type.keys() else [])
        nIs = len(self.asset_configs_by_type['storage'] if 'storage' in self.asset_configs_by_type.keys() else [])
        nH = len(self.asset_configs_by_type['fixedload'][0]['lamb']) if nIf > 0 else len(self.asset_configs_by_type['storage'][0]['lambl'])

        #sets
        model.H = pyo.RangeSet(0, nH-1)
        model.Hs = pyo.RangeSet(-1, nH-1)
        model.If = pyo.RangeSet(0, nIf-1)
        model.Is = pyo.RangeSet(0, nIs-1)

        model.M = pyo.Param(default=1000) # Is there a better way to do this - like the one in mvnn mip?

        #price as parameter
        model.price = pyo.Param(model.H, name='price') # price vector

        #preference parameters
        #   fixed load
        model.lamb = pyo.Param(model.If, model.H, default=self.__get_param_values('lamb', 'fixedload'), name='lamb') # linear hourly coefficients
        model.gu = pyo.Param(model.If, model.H, default=self.__get_param_values('gu', 'fixedload'), name='gu') # upper limit on production
        model.gl = pyo.Param(model.If, model.H, default=self.__get_param_values('gl', 'fixedload'), name='gl') # lower limit on production
        #   storage
        model.lambl = pyo.Param(model.Is, model.H, default=self.__get_param_values('lambl', 'storage')) # lower limit violation penalty coeff
        model.lambu = pyo.Param(model.Is, model.H, default=self.__get_param_values('lambu', 'storage')) # upper limit violation penalty coeff
        model.su = pyo.Param(model.Is, model.H, default=self.__get_param_values('su', 'storage')) # SoC upper limit
        model.sl = pyo.Param(model.Is, model.H, default=self.__get_param_values('sl', 'storage')) # SoC lower limit
        model.s0 = pyo.Param(model.Is, default=self.__get_param_values('s0', 'storage')) # initial SoC
        model.eta = pyo.Param(model.Is, default=self.__get_param_values('eta', 'storage')) # charging efficiency
        model.gamma = pyo.Param(model.Is, model.H, default=self.__get_param_values('gamma', 'storage')) # dissipation rate
        model.beta = pyo.Param(model.Is, model.H, default=self.__get_param_values('beta', 'storage')) # storage degradation coeff
        model.capacity = pyo.Param(model.Is, default=self.__get_param_values('capacity', 'storage')) # storage capacity
        model.ramp_limit = pyo.Param(model.Is, default=self.__get_param_values('ramp_limit', 'storage')) # ramp limit

        #variables
        model.p = pyo.Var(model.H, name='p') # net power
        model.pf = pyo.Var(model.If, model.H, name='pf') # power for fixed load
        model.ps = pyo.Var(model.Is, model.H, name='ps') # power for storage
        #   auxiliary: storage
        model.deltap = pyo.Var(model.Is, model.H, domain=pyo.Binary, name='deltap')
        model.s = pyo.Var(model.Is, model.Hs, domain=pyo.NonNegativeReals, name='s')
        model.tu = pyo.Var(model.Is, model.H, domain=pyo.NonNegativeReals, name='tu')
        model.tl = pyo.Var(model.Is, model.H, domain=pyo.NonNegativeReals, name='tl')
        model.P = pyo.Var(model.Is, model.H, name='P')

        self.__add_dq_objective(model)
        self.__add_asset_constraints(model)

        return model
    
    def bundle_query(self, price: np.ndarray) -> tuple:
        """
        Query the utility maximizing demand of the prosumer for given prices.
        :param prices: price vector.
        :return: bundle, utility.
        """
        instance = self.dq_model.create_instance({None: {
                                                         'price': {h: price[h] for h in range(len(price))}
                                                        }})

        solver = pyo.SolverFactory('gurobi_direct')
        solution = solver.solve(instance, tee=False, options={"MIPGap": 0})

        if solution.Solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
            raise ValueError("The prosumer greedy bundle model is infeasible or unbounded.")
        
        bundle = np.array([instance.p[h].value for h in instance.H])
        return bundle, pyo.value(instance.obj)+np.dot(price, bundle)

    def get_value(self, bundle: np.ndarray) -> float:
        model = self.dq_model.clone()
        solver = pyo.SolverFactory('gurobi_direct')
        def con_bundle(m, h):
            return m.p[h] == bundle[h]
        model.cons_bundle = pyo.Constraint(model.H, rule=con_bundle, name='cons_bundle')
        instance = model.create_instance({None: {
                                                    'price': {h: 0 for h in model.H}
                                                }})

        solution = solver.solve(instance, tee=False)
        if solution.solver.status != SolverStatus.ok:
            # print(f"Termination condition for bundle {bundle}: {solution.solver.termination_condition}")
            return np.nan

        return pyo.value(instance.obj)

    def get_capacity_generic_goods(self, ) -> np.ndarray:
        #debugging
        # writer = LPWriter()
        model = self.dq_model.clone()
        solver = pyo.SolverFactory('gurobi_direct')
        bounds = []
        for i in range(len(model.H)):
            bounds.append([])
            for sense in [-1, 1]:
                model.obj = pyo.Objective(rule=lambda m: m.p[i]*sense, sense=pyo.maximize)
                instance = model.create_instance()
                solver.solve(instance, tee=False)
                bounds[i].append(np.abs(instance.p[i].value))
                # writer.write(instance, open('model1.lp', 'w'), symbolic_solver_labels=True)
        return np.max(bounds, axis=1)


class LogarithmicBidder(Bidder):
    def __init__(self, name: str):
        self.name = name
        self.intervals = bidder_configs[name]['intervals']
        self.flowlimit = bidder_configs[name]['flowlimit']
        self.scale = bidder_configs[name]['scale']
        self.shift = bidder_configs[name]['shift']
        self.full_info_imp = True

        # piecewise linear approximation of the logarithmic function
        self.x_pts = np.linspace(-self.flowlimit, self.flowlimit, 100)
        self.y_pts = np.log(self.x_pts - self.shift)
        self.xl = self.x_pts[:-1]
        self.xu = self.x_pts[1:]
        self.yl = np.log(self.xl - self.shift)
        self.yu = np.log(self.xu - self.shift)
        self.slope = (self.yu - self.yl) / (self.xu - self.xl)

        self.value_model = self.__get_value_model()


    def __neg_utility_func(self, x, price):
        return -self.get_value(x) + np.dot(price, x)        


    def add_model(self, model: gp.Model):
        x = model.addVars([i for i in range(self.intervals)], name=f'{self.name}_x', vtype=GRB.CONTINUOUS, lb=-self.flowlimit, ub=self.flowlimit)

        y = model.addVars([i for i in range(self.intervals)], name=f'{self.name}_y', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        [[model.addConstr(y[i] <= self.yl[j] + (x[i] - self.xl[j])*self.slope[j]) for j in range(len(self.xl))] for i in range(self.intervals)]

        return x, gp.quicksum(y[i]*self.scale for i in range(self.intervals))


    def bundle_query(self, price):
        res = opt.minimize(self.__neg_utility_func, x0=np.zeros(self.intervals), args=(price), constraints=[], bounds=[(-self.flowlimit, self.flowlimit)]*self.intervals)
        return res.x, -res.fun + np.dot(price, res.x)


    def __get_value_model(self, ):
        valuemodel = gp.Model('Value model')
        valuemodel.setParam('OutputFlag', 0)
        xvar, obj = self.add_model(valuemodel)
        [valuemodel.addConstr(xvar[i] == 0, name=f'valuecon_{i}') for i in range(self.intervals)]
        valuemodel.setObjective(obj, GRB.MAXIMIZE)
        valuemodel.update()
        return valuemodel
    

    def get_value(self, x):
        for i in range(self.intervals):
            self.value_model.getConstrByName(f'valuecon_{i}').RHS = x[i]
        self.value_model.update()
        self.value_model.optimize()

        return self.value_model.getObjective().getValue()
        # return np.sum(np.log(x[i] - self.shift) for i in range(self.intervals))*self.scale


    def get_capacity_generic_goods(self, ):
        return np.array([self.flowlimit]*self.intervals)
