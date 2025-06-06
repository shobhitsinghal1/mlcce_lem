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

    @abstractmethod
    def get_name(self, ) -> str:
        pass


class Prosumer(Bidder):
    def __init__(self, name: str, horizon: int):
        self.name = name
        self.horizon = horizon

        self.prosumer_config = bidder_configs[name]
        self.asset_configs_by_type = self.__group_configs_by_type()
        self.gpenv = gp.Env(params={'LogToConsole': 0, 'OutputFlag': 0})
        self.dq_model = self.__construct_dq_model()
        self.value_model = self.__construct_value_model()
        self.capacity_model = self.__construct_dq_model()
        self.full_info_imp = False


    def __group_configs_by_type(self, ) -> dict:
        """
        Group the asset configurations by their asset type.
        :return: dictionary with asset types as keys and lists of asset configurations as values.
        """
        assets_by_type = {}
        for asset in self.prosumer_config["assets"]:
            asset_type = asset_configs[asset]["type"]
            if asset_type not in assets_by_type.keys():
                assets_by_type[asset_type] = []
            assets_by_type[asset_type].append(asset_configs[asset])
        return assets_by_type


    def __get_param_values(self, param_name: str, asset_type: str) -> dict:
        param_values = {}
        if asset_type in self.asset_configs_by_type.keys():
            asset_configs = self.asset_configs_by_type[asset_type]
            for i in range(len(asset_configs)):
                if isinstance(asset_configs[i][param_name], list):
                    for j in range(self.horizon):
                        param_values[(i, j)] = asset_configs[i][param_name][j]
                else:
                    param_values[i] = asset_configs[i][param_name]
        return param_values


    def __add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['gl'][i, h] <= m['pf'][i, h] for i in m['If'] for h in m['H']), name=f'conf1l_{self.name}')
        m['model'].addConstrs((m['pf'][i, h] <= m['gu'][i, h] for i in m['If'] for h in m['H']), name=f'conf1u_{self.name}')

        m['model'].addConstrs((m['ps'][i, h] <= m['M'] * m['deltap'][i, h] for i in m['Is'] for h in m['H']), name=f'cons1_{self.name}')
        m['model'].addConstrs((m['ps'][i, h] >= -m['M'] * (1 - m['deltap'][i, h]) for i in m['Is'] for h in m['H']), name=f'cons2_{self.name}')
        m['model'].addConstrs((m['s'][i, h-1] + m['eta'][i] * m['ps'][i, h] - m['gamma'][i, h] - m['M'] * (1 - m['deltap'][i, h]) <= m['s'][i, h] for i in m['Is'] for h in m['H']), name=f'cons3l_{self.name}')
        m['model'].addConstrs((m['s'][i, h] <= m['s'][i, h-1] + m['eta'][i] * m['ps'][i, h] - m['gamma'][i, h] + m['M'] * (1 - m['deltap'][i, h]) for i in m['Is'] for h in m['H']), name=f'cons3u_{self.name}')
        m['model'].addConstrs((m['s'][i, h-1] + m['ps'][i, h] / m['eta'][i] - m['gamma'][i, h] - m['M'] * m['deltap'][i, h] <= m['s'][i, h] for i in m['Is'] for h in m['H']), name=f'cons4l_{self.name}')
        m['model'].addConstrs((m['s'][i, h] <= m['s'][i, h-1] + m['ps'][i, h] / m['eta'][i] - m['gamma'][i, h] + m['M'] * m['deltap'][i, h] for i in m['Is'] for h in m['H']), name=f'cons4u_{self.name}')
        m['model'].addConstrs((m['s'][i, h] <= m['su'][i, h] + m['tu'][i, h] for i in m['Is'] for h in m['H']), name=f'cons5_{self.name}')
        m['model'].addConstrs((m['s'][i, h] >= m['sl'][i, h] - m['tl'][i, h] for i in m['Is'] for h in m['H']), name=f'cons6_{self.name}')
        m['model'].addConstrs((m['P'][i, h] <= m['ramp_limit'][i] for i in m['Is'] for h in m['H']), name=f'cons7_{self.name}')
        m['model'].addConstrs((0 <= m['s'][i, h] for i in m['Is'] for h in m['H']), name=f'cons8l_{self.name}')
        m['model'].addConstrs((m['s'][i, h] <= m['capacity'][i] for i in m['Is'] for h in m['H']), name=f'cons8u_{self.name}')
        m['model'].addConstrs((m['P'][i, h] >= m['ps'][i, h] for i in m['Is'] for h in m['H']), name=f'cons9_{self.name}')
        m['model'].addConstrs((m['P'][i, h] >= -m['ps'][i, h] for i in m['Is'] for h in m['H']), name=f'cons10_{self.name}')
        m['model'].addConstrs((m['s'][i, -1] == m['s0'][i] for i in m['Is']), name=f'cons11_{self.name}')

        return


    def __get_value_obj_expr(self, m: dict):
        return gp.quicksum(m['pf'][i,h] for h in m['H'] for i in m['If']) + gp.quicksum(-m['tu'][i,h]*m['lambu'][i,h] - m['tl'][i,h]*m['lambl'][i,h] - m['P'][i,h]*m['beta'][i,h] for h in m['H'] for i in m['Is'])


    def __add_dq_objective(self, m: dict, price: np.ndarray):
        #objective

        m['model'].setObjective(
            self.__get_value_obj_expr(m)
            - gp.quicksum(m['p'][h]*price[h] for h in m['H']),
            GRB.MAXIMIZE
        )

        return


    def __add_asset_params_vars(self, m: dict):
        
        nIf = len(self.asset_configs_by_type['fixedload'] if 'fixedload' in self.asset_configs_by_type.keys() else [])
        nIs = len(self.asset_configs_by_type['storage'] if 'storage' in self.asset_configs_by_type.keys() else [])
        nH = self.horizon

        #sets
        m['If'] = list(range(nIf))
        m['Is'] = list(range(nIs))
        m['H'] = list(range(nH))
        m['Hs'] = list(range(-1, nH)) # for storage SoC

        m['M'] = 1000 # Is there a better way to do this - like the one in mvnn mip?

        # preference parameters
        #   fixed load
        m['lamb'] = self.__get_param_values('lamb', 'fixedload') # (if,h) linear hourly coefficients
        m['gu'] = self.__get_param_values('gu', 'fixedload') # (if,h) upper limit on production
        m['gl'] = self.__get_param_values('gl', 'fixedload') # (if,h) lower limit on production
        #   storage
        m['lambl'] = self.__get_param_values('lambl', 'storage') # (is,h) lower limit violation penalty coeff
        m['lambu'] = self.__get_param_values('lambu', 'storage') # (is,h) upper limit violation penalty coeff
        m['su'] = self.__get_param_values('su', 'storage') # (is,h) SoC upper limit
        m['sl'] = self.__get_param_values('sl', 'storage') # (is,h) SoC lower limit
        m['s0'] = self.__get_param_values('s0', 'storage') # (is) initial SoC
        m['eta'] = self.__get_param_values('eta', 'storage') # (is) charging efficiency
        m['gamma'] = self.__get_param_values('gamma', 'storage') # (is,h) dissipation rate
        m['beta'] = self.__get_param_values('beta', 'storage') # (is,h) storage degradation coeff
        m['capacity'] = self.__get_param_values('capacity', 'storage') # (is) storage capacity
        m['ramp_limit'] = self.__get_param_values('ramp_limit', 'storage') # (is) ramp limit

        # variables
        m['pf'] = m['model'].addVars(m['If'], m['H'], name=f'pf_{self.name}') # power for fixed load
        m['ps'] = m['model'].addVars(m['Is'], m['H'], name=f'ps_{self.name}') # power for storage
        #   auxiliary: storage
        m['deltap'] = m['model'].addVars(m['Is'], m['H'], vtype=GRB.BINARY, name=f'deltap_{self.name}')
        m['s'] = m['model'].addVars(m['Is'], m['Hs'], name=f's_{self.name}')
        m['tu'] = m['model'].addVars(m['Is'], m['H'], name=f'tu_{self.name}')
        m['tl'] = m['model'].addVars(m['Is'], m['H'], name=f'tl_{self.name}')
        m['P'] = m['model'].addVars(m['Is'], m['H'], name=f'P_{self.name}')

        return


    def __construct_dq_model(self, ) -> dict:
        """
        Constructs the demand query optimization model parameterized in price.
        :return: demand query model.
        """
        model = {'model': gp.Model('Prosumer demand query model', self.gpenv)}

        self.__add_asset_params_vars(model)
        self.__add_asset_constraints(model)
        model['p'] = model['model'].addVars(model['H'], name=f'p_{self.name}') # net power
        model['model'].addConstrs((model['p'][h] == gp.quicksum(model['pf'][i, h] for i in model['If']) + gp.quicksum(model['ps'][i, h] for i in model['Is']) for h in model['H']), name=f'conp_{self.name}') # net power flow

        return model


    def __construct_value_model(self, ) -> dict:
        model = {'model': gp.Model('Prosumer value query model', self.gpenv)}

        self.__add_asset_params_vars(model)
        self.__add_asset_constraints(model)
        model['p'] = model['model'].addVars(model['H'], name=f'p_{self.name}') # net power
        model['model'].addConstrs((model['p'][h] == gp.quicksum(model['pf'][i, h] for i in model['If']) + gp.quicksum(model['ps'][i, h] for i in model['Is']) for h in model['H']), name=f'conp_{self.name}') # net power flow
        model['con_bundle'] = model['model'].addConstrs((model['p'][h] == 0 for h in model['H']), name=f'con_bundle_{self.name}')
        self.__add_dq_objective(model, np.zeros(self.horizon))

        return model
    
    
    def add_model(self, model: gp.Model) -> tuple:
        """
        Adds the dispatch variables and constraints corresponding to the prosumer to the gurobi model.
        Returns dispatch variables and objective expression.
        """

        model = {'model': model}
        self.__add_asset_params_vars(model)
        self.__add_asset_constraints(model)
        model['p'] = model['model'].addVars(model['H'], name=f'p_{self.name}') # net power
        model['model'].addConstrs((model['p'][h] == gp.quicksum(model['pf'][i, h] for i in model['If']) + gp.quicksum(model['ps'][i, h] for i in model['Is']) for h in model['H']), name=f'conp_{self.name}') # net power flow
        
        return model['p'], self.__get_value_obj_expr(model)


    def bundle_query(self, price: np.ndarray) -> tuple:
        """
        Query the utility maximizing demand of the prosumer for given prices.
        :param prices: price vector.
        :return: bundle, utility.
        """
        assert len(price) == self.horizon, "Price vector length must match the horizon length."
        
        self.__add_dq_objective(self.dq_model, price)
        
        self.dq_model['model'].update()
        self.dq_model['model'].optimize()

        if self.dq_model['model'].status != 2:
            raise ValueError(f"The prosumer greedy bundle model is {gurobi_status_converter(self.dq_model['model'].status)}.")
        
        bundle = np.array([self.dq_model['p'][h].x for h in self.dq_model['H']])
        return bundle, self.dq_model['model'].ObjVal + np.dot(price, bundle)


    def get_value(self, bundle: np.ndarray) -> float:
        assert len(bundle) == self.horizon, "Bundle length must match the horizon length."
        
        [self.value_model['con_bundle'][i].setAttr(GRB.Attr.RHS, bundle[i]) for i in range(self.horizon)]

        self.value_model['model'].update()
        self.value_model['model'].optimize()
        
        if self.value_model['model'].status != SolverStatus.ok:
            return np.nan

        return self.value_model['model'].ObjVal


    def get_capacity_generic_goods(self, ) -> np.ndarray:

        bounds = []
        for i in range(self.horizon):
            bounds.append([])
            for sense in [-1, 1]:
                self.capacity_model['model'].setObjective(self.capacity_model['p'][i] * sense, GRB.MAXIMIZE)
                self.capacity_model['model'].update()
                self.capacity_model['model'].optimize()
                bounds[i].append(np.abs(self.capacity_model['p'][i].x))
        return np.max(bounds, axis=1)


    def get_name(self, ):
        return self.name


class LogarithmicBidder(Bidder):
    def __init__(self, name: str, horizon: int):
        self.name = name
        self.intervals = horizon
        
        # class specific parameters
        self.flowlimit = bidder_configs[name]['flowlimit']
        self.scale = bidder_configs[name]['scale']
        self.shift = bidder_configs[name]['shift']
        self.full_info_imp = True

        # piecewise linear approximation of the logarithmic function
        self.x_pts = np.linspace(-self.flowlimit, self.flowlimit, bidder_configs[name]['granularity'])
        self.y_pts = np.log(self.x_pts - self.shift)
        self.xl = self.x_pts[:-1]
        self.xu = self.x_pts[1:]
        self.yl = np.log(self.xl - self.shift)
        self.yu = np.log(self.xu - self.shift)
        self.slope = (self.yu - self.yl) / (self.xu - self.xl)

        self.value_model = self.__get_value_model()


    def __neg_utility_func(self, x, price):
        return -self.get_value(x) + np.dot(price, x)        


    def __get_value_model(self, ):
        valuemodel = gp.Model('Value model')
        valuemodel.setParam('OutputFlag', 0)
        xvar, obj = self.add_model(valuemodel)
        [valuemodel.addConstr(xvar[i] == 0, name=f'valuecon_{i}') for i in range(self.intervals)]
        valuemodel.setObjective(obj, GRB.MAXIMIZE)
        valuemodel.update()
        return valuemodel


    def add_model(self, model: gp.Model):
        """
        Adds the dispatch variables and constraints to the gurobi model
        Returns dispatch variables and objective expression
        """
        x = model.addVars([i for i in range(self.intervals)], name=f'{self.name}_x', vtype=GRB.CONTINUOUS, lb=-self.flowlimit, ub=self.flowlimit)

        y = model.addVars([i for i in range(self.intervals)], name=f'{self.name}_y', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        [[model.addConstr(y[i] <= self.yl[j] + (x[i] - self.xl[j])*self.slope[j]) for j in range(len(self.xl))] for i in range(self.intervals)]

        return x, gp.quicksum(y[i]*self.scale for i in range(self.intervals))


    def bundle_query(self, price):
        assert len(price) == self.intervals, "Price vector length must match the horizon."

        res = opt.minimize(self.__neg_utility_func, x0=np.zeros(self.intervals), args=(price), constraints=[], bounds=[(-self.flowlimit, self.flowlimit)]*self.intervals)
        return res.x, -res.fun + np.dot(price, res.x)


    def get_value(self, x):
        assert len(x) == self.intervals, "x must have the same length as horizon."
        
        for i in range(self.intervals):
            self.value_model.getConstrByName(f'valuecon_{i}').RHS = x[i]
        self.value_model.update()
        self.value_model.optimize()

        return self.value_model.getObjective().getValue()
        # return np.sum(np.log(x[i] - self.shift) for i in range(self.intervals))*self.scale


    def get_capacity_generic_goods(self, ):
        return np.array([self.flowlimit]*self.intervals)


    def get_name(self, ):
        return self.name
