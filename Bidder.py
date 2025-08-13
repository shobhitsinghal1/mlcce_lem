from utils import *
import numpy as np
import logging
from abc import ABC, abstractmethod
from scipy import optimize as opt
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


class Bidder(ABC):
    
    @abstractmethod
    def bundle_query(self, price: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def get_value(self, bundle: np.ndarray) -> float:
        pass

    @abstractmethod
    def get_capacity_generic_goods(self, ) -> list:
        pass

    @abstractmethod
    def get_name(self, ) -> str:
        pass


class Prosumer(Bidder):

    def __init__(self, name: str, n_product: int, config: dict):
        self.name = name
        self.n_product = n_product
        self.config = config

        self.gpenv = gp.Env(params={'LogToConsole': 0, 'OutputFlag': 0})
        self.dq_model = self.construct_dq_model()
        self.value_model = self.construct_value_model() # just an additional constraint for bundle, and objective with zero prices
        self.capacity_model = self.construct_dq_model()
        self.full_info_imp = False


    def get_param_values(self, param_name: str, length: int = 0) -> dict:
        param_values = {}
        if length>0:
            for j in range(length):
                param_values[j] = self.config[param_name][j]
        else:
            param_values = self.config[param_name]
        return param_values

    # @abstractmethod
    def add_asset_params_vars(self, m: dict):
        pass
    
    # @abstractmethod
    def add_asset_constraints(self, m: dict):
        pass

    # @abstractmethod
    def get_value_obj_expr(self, m: dict):
        pass

    
    def add_model(self, model: gp.Model):
        m = {'model': model}
        self.add_asset_params_vars(m)
        self.add_asset_constraints(m)
        expr = self.get_value_obj_expr(m)

        return m['p'], expr


    def add_dq_objective(self, m: dict, price: np.ndarray):
        #objective

        m['model'].setObjective(
            self.get_value_obj_expr(m)
            - gp.quicksum(m['p'][h]*price[h] for h in range(self.n_product)),
            GRB.MAXIMIZE
        )

        return


    def construct_dq_model(self, ) -> dict:
        """
        Constructs the demand query optimization model parameterized in price.
        :return: demand query model.
        """
        model = {'model': gp.Model('Prosumer demand query model', self.gpenv)}

        self.add_asset_params_vars(model)
        self.add_asset_constraints(model)
        return model


    def construct_value_model(self, ) -> dict:
        model = {'model': gp.Model('Prosumer value query model', self.gpenv)}

        self.add_asset_params_vars(model)
        self.add_asset_constraints(model)
        model['con_bundle'] = model['model'].addConstrs((model['p'][h] == 0 for h in model['H']), name=f'con_bundle_{self.name}')
        self.add_dq_objective(model, np.zeros(self.n_product))

        return model


    def bundle_query(self, price: np.ndarray) -> tuple:
        """
        Query the utility maximizing demand of the prosumer for given prices.
        :param prices: price vector.
        :return: bundle, utility.
        """
        assert len(price) == self.n_product, "Price vector length must match the number of products."
        
        self.add_dq_objective(self.dq_model, price)
        
        self.dq_model['model'].update()
        self.dq_model['model'].optimize()

        try:
            bundle = np.array([self.dq_model['p'][h].x for h in range(self.n_product)])
        except:
            raise ValueError(f"The prosumer greedy bundle model is {gurobi_status_converter(self.dq_model['model'].status)}.")
        
        return bundle, self.dq_model['model'].ObjVal + np.dot(price, bundle)


    def get_value(self, bundle: np.ndarray) -> float:
        assert len(bundle) == self.n_product, "Bundle length must match the number of products."
        
        [self.value_model['con_bundle'][i].setAttr(GRB.Attr.RHS, bundle[i]) for i in range(self.n_product)]

        self.value_model['model'].update()
        self.value_model['model'].optimize()
        
        if self.value_model['model'].status != 2:
            return np.nan

        return self.value_model['model'].ObjVal


    def get_capacity_generic_goods(self, ) -> list:
        """ returns the list of lower and upper bounds """
        bounds = [[], []]
        for i in range(self.n_product):
            for j, sense in enumerate([-1, 1]):
                self.capacity_model['model'].setObjective(self.capacity_model['p'][i] * sense, GRB.MAXIMIZE)
                self.capacity_model['model'].update()
                self.capacity_model['model'].optimize()
                bounds[j].append(self.capacity_model['p'][i].x)
        bounds[0] = np.array(bounds[0])  # lower bounds
        bounds[1] = np.array(bounds[1])  # upper bounds
        return bounds


    def get_name(self, ):
        return self.name


class ProsumerRenewable(Prosumer):

    def add_asset_params_vars(self, m: dict):
        
        nH = self.n_product

        #sets
        m['H'] = list(range(nH))

        # preference parameters
        m['oppdprice'] = self.get_param_values('oppdprice', nH) # (h)
        m['generation'] = self.get_param_values('generation', nH)

        # variables
        m['p'] = m['model'].addVars(m['H'], name=f'p_{self.name}', lb=float('-inf')) # power for storage

        return
    

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['p'][h] <= 0 for h in m['H']), name=f'cons10u_{self.name}')
        m['model'].addConstrs((m['p'][h] >= -m['generation'][h] for h in m['H']), name=f'cons10l_{self.name}')

        return


    def get_value_obj_expr(self, m: dict):
        return gp.quicksum(m['oppdprice'][h] * m['p'][h] for h in m['H'])


class ProsumerConsumer(Prosumer):

    def add_asset_params_vars(self, m: dict):
        
        nH = self.n_product

        #sets
        m['H'] = list(range(nH))

        # preference parameters
        m['oppcprice'] = self.get_param_values('oppcprice', nH) # (h)
        m['consumption'] = self.get_param_values('consumption', nH)

        # variables
        m['p'] = m['model'].addVars(m['H'], name=f'p_{self.name}', lb=float('-inf'))

        return
    

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['p'][h] >= 0 for h in m['H']), name=f'cons10u_{self.name}')
        m['model'].addConstrs((m['p'][h] <= m['consumption'][h] for h in m['H']), name=f'cons10l_{self.name}')

        return


    def get_value_obj_expr(self, m: dict):
        return gp.quicksum(m['oppcprice'][h] * m['p'][h] for h in m['H'])


class ProsumerStorage(Prosumer):

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['p'][h] <= m['M'] * m['delta'][h] for h in m['H']), name=f'cons1_{self.name}')
        m['model'].addConstrs((m['p'][h] >= -m['M'] * (1 - m['delta'][h]) for h in m['H']), name=f'cons2_{self.name}')
        m['model'].addConstrs((m['s'][h-1] + m['eta'] * m['p'][h] - m['gamma'][h] - m['M'] * (1 - m['delta'][h]) <= m['s'][h] for h in m['H']), name=f'cons3l_{self.name}')
        m['model'].addConstrs((m['s'][h] <= m['s'][h-1] + m['eta'] * m['p'][h] - m['gamma'][h] + m['M'] * (1 - m['delta'][h]) for h in m['H']), name=f'cons3u_{self.name}')
        m['model'].addConstrs((m['s'][h-1] + m['p'][h] / m['eta'] - m['gamma'][h] - m['M'] * m['delta'][h] <= m['s'][h] for h in m['H']), name=f'cons4l_{self.name}')
        m['model'].addConstrs((m['s'][h] <= m['s'][h-1] + m['p'][h] / m['eta'] - m['gamma'][h] + m['M'] * m['delta'][h] for h in m['H']), name=f'cons4u_{self.name}')
        
        m['model'].addConstrs((m['s'][h] <= m['su'][h] for h in m['H']), name=f'cons5_{self.name}')
        m['model'].addConstrs((m['s'][h] >= m['sl'][h] for h in m['H']), name=f'cons6_{self.name}')
        m['model'].addConstr((m['s'][-1] == m['s0']), name=f'cons7a_{self.name}')
        # if m['sn'] >=0:
        #     m['model'].addConstr((m['s'][m['H'][-1]] == m['sn']), name=f'cons7b_{self.name}')

        m['model'].addConstrs((m['P'][h] >= m['p'][h] for h in m['H']), name=f'cons8_{self.name}')
        m['model'].addConstrs((m['P'][h] >= -m['p'][h] for h in m['H']), name=f'cons9_{self.name}')

        m['model'].addConstrs((m['p'][h] <= m['power_limit_up'] for h in m['H']), name=f'cons10u_{self.name}')
        m['model'].addConstrs((m['p'][h] >= m['power_limit_down'] for h in m['H']), name=f'cons10l_{self.name}')

        m['model'].addConstrs((m['P'][h] <= m['M'] * m['available'][h] for h in m['H']), name=f'cons12_{self.name}')

        return


    def get_value_obj_expr(self, m: dict):
        if m['sn'] >= 0:
            return gp.quicksum( - m['P'][h]*m['beta'][h] + m['oppcprice'][h] * m['p'][h] * m['delta'][h] + m['oppdprice'][h] * m['p'][h] * (1 - m['delta'][h]) - (m['s'][h] - m['sn']) * (m['s'][h] - m['sn']) * m['alpha'] for h in m['H'])
        else:
            return gp.quicksum( - m['P'][h]*m['beta'][h] + m['oppcprice'][h] * m['p'][h] * m['delta'][h] + m['oppdprice'][h] * m['p'][h] * (1 - m['delta'][h]) for h in m['H'])


    def add_asset_params_vars(self, m: dict):
        
        nH = self.n_product

        #sets
        m['H'] = list(range(nH))
        m['Hs'] = list(range(-1, nH)) # for storage SoC

        m['M'] = 1000 # Is there a better way to do this - like the one in mvnn mip?

        # preference parameters
        m['oppcprice'] = self.get_param_values('oppcprice', nH) # (h) lower limit violation penalty coeff
        m['oppdprice'] = self.get_param_values('oppdprice', nH) # (h) upper limit violation penalty coeff
        m['available'] = self.get_param_values('available', nH)
        m['su'] = self.get_param_values('su', nH) # (h) SoC upper limit
        m['sl'] = self.get_param_values('sl', nH) # (h) SoC lower limit
        m['s0'] = self.get_param_values('s0') # () initial SoC
        m['sn'] = self.get_param_values('sn') # () terminal SoC
        m['alpha'] = self.get_param_values('alpha')
        m['eta'] = self.get_param_values('eta') # () charging efficiency
        m['gamma'] = self.get_param_values('gamma', nH) # (h) dissipation rate
        m['beta'] = self.get_param_values('beta', nH) # (h) storage degradation coeff
        m['power_limit_up'] = self.get_param_values('power_limit_up') # () ramp limit
        m['power_limit_down'] = self.get_param_values('power_limit_down') # () ramp limit

        # variables
        m['p'] = m['model'].addVars(m['H'], name=f'p_{self.name}', lb=float('-inf')) # power for storage
        #   auxiliary: storage
        m['delta'] = m['model'].addVars(m['H'], vtype=GRB.BINARY, name=f'delta_{self.name}')
        m['s'] = m['model'].addVars( m['Hs'], name=f's_{self.name}')
        m['P'] = m['model'].addVars(m['H'], name=f'P_{self.name}')

        return


class ProsumerSwitch(Prosumer):

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((gp.quicksum( m['pd'][h] for h in range(k,k+m['tr'])) >= m['z'][k] * m['tr'] for k in m['K']), name=f'consw1_{self.name}')
        m['model'].addConstr((gp.quicksum(m['z'][k] for k in m['K']) >= 1 - m['y']), name=f'consw2_{self.name}')
        m['model'].addConstrs((m['p'][h] >= m['capacity'] * m['pd'][h] for h in m['H']), name=f'consw3_{self.name}')
        m['model'].addConstrs((m['p'][h] <= m['capacity'] for h in m['H']), name=f'consw4_{self.name}')

        return


    def get_value_obj_expr(self, m: dict):
        return gp.quicksum( m['oppcprice'][h] * m['pd'][h] * m['capacity'] for h in m['H'])


    def add_asset_params_vars(self, m: dict):
        
        nH = self.n_product

        #sets
        m['H'] = list(range(nH))

        # preference parameters
        m['oppcprice'] = self.get_param_values('oppcprice', nH) # (h) lower limit violation penalty coeff
        m['capacity'] = self.get_param_values('capacity')
        m['tr'] = self.get_param_values('tr')
        m['tl'] = self.get_param_values('tl')
        m['tu'] = self.get_param_values('tu')
        m['K'] = list(range(m['tl']-1, m['tu']-m['tr']+1))

        # variables
        m['p'] = m['model'].addVars(m['H'], name=f'p_{self.name}') # power transaction which is continuous
        #   auxiliary: storage
        m['pd'] = m['model'].addVars(m['H'], vtype=GRB.BINARY, name=f'pd_{self.name}')
        m['z'] = m['model'].addVars(m['K'], vtype=GRB.BINARY, name=f'z_{self.name}')
        m['y'] = m['model'].addVar(vtype=GRB.BINARY, name=f'y_{self.name}')

        return


class ProsumerStorageFlex(Prosumer):

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['s0'] + gp.quicksum(m['pe'][h] + m['pf'][h] for h in range(nh+1)) <= m['su'] for nh in m['H']), name=f'cons1u_{self.name}')
        m['model'].addConstrs((m['s0'] + gp.quicksum(m['pe'][h] + m['pf'][h] for h in range(nh+1)) >= 0 for nh in m['H']), name=f'cons1l_{self.name}')
        m['model'].addConstrs((m['s0'] + gp.quicksum(m['pe'][h] - m['pf'][h] for h in range(nh+1)) <= m['su'] for nh in m['H']), name=f'cons2u_{self.name}')
        m['model'].addConstrs((m['s0'] + gp.quicksum(m['pe'][h] - m['pf'][h] for h in range(nh+1)) >= 0 for nh in m['H']), name=f'cons2l_{self.name}')

        if m['fixflex'] is not None:
            m['model'].addConstrs((m['pf'][h] == m['fixflex'][h] for h in m['H']), name=f'cons5_{self.name}')
        if m['fixener'] is not None:
            m['model'].addConstrs((m['pe'][h] == m['fixener'][h] for h in m['H']), name=f'cons6_{self.name}')

        m['model'].addConstrs((m['p'][h] == m['pe'][h] for h in m['H']), name=f'cons3_{self.name}')
        m['model'].addConstrs((m['p'][h + m['nH']] == m['pf'][h] for h in m['H']), name=f'cons4_{self.name}')

        return


    def get_value_obj_expr(self, m: dict):
        return gp.quicksum(m['alphae'][h]*m['pe'][h]*m['pe'][h] + m['oppeprice'][h] * m['pe'][h] for h in m['H']) + gp.quicksum(m['alphaf'][h]*m['pf'][h]*m['pf'][h] + m['oppfprice'][h] * m['pf'][h] for h in m['H'])


    def add_asset_params_vars(self, m: dict):
        
        assert self.n_product % 2 == 0, "The number of products must be even for ProsumerStorageEnerFlexToy."
        m['nH'] = int(self.n_product/2)

        #sets
        m['H'] = list(range(m['nH']))

        # preference parameters
        m['oppeprice'] = self.get_param_values('oppeprice', m['nH']) # (h)
        m['oppfprice'] = self.get_param_values('oppfprice', m['nH']) # (h)
        m['alphae'] = self.get_param_values('alphae', m['nH']) # (h) 
        m['alphaf'] = self.get_param_values('alphaf', m['nH']) # (h)
        m['su'] = self.get_param_values('su') # () SoC upper limit
        m['s0'] = self.get_param_values('s0') # () initial SoC
        m['fixflex'] = None if not self.config['fixflex'] else self.get_param_values('fixflex', m['nH'])
        m['fixener'] = None if not self.config['fixener'] else self.get_param_values('fixener', m['nH'])

        # variables
        m['pe'] = m['model'].addVars(m['H'], name=f'pe_{self.name}', lb=float('-inf')) # power for storage
        m['pf'] = m['model'].addVars(m['H'], name=f'pf_{self.name}', lb=float('-inf'), ub=0) # flexibility

        m['p'] = m['model'].addVars(list(range(self.n_product)), name=f'p_{self.name}', lb=float('-inf')) # products

        return


class ProsumerDSO(Prosumer):

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['pe'][h] == 0 for h in m['H']), name=f'conD1_{self.name}')
        m['model'].addConstrs((m['pf'][h] <= m['flexlimit'][h] for h in m['H']), name=f'conD2_{self.name}')

        if m['fixflex'] is not None:
            m['model'].addConstrs((m['pf'][h] == m['fixflex'][h] for h in m['H']), name=f'cons5_{self.name}')
        if m['fixener'] is not None:
            m['model'].addConstrs((m['pe'][h] == m['fixener'][h] for h in m['H']), name=f'cons6_{self.name}')

        m['model'].addConstrs((m['p'][h] == m['pe'][h] for h in m['H']), name=f'cons3_{self.name}')
        m['model'].addConstrs((m['p'][h + m['nH']] == m['pf'][h] for h in m['H']), name=f'cons4_{self.name}')

        return


    def get_value_obj_expr(self, m: dict):
        return gp.quicksum(m['alphaf'][h]*m['pf'][h]*m['pf'][h] + m['oppfprice'][h] * m['pf'][h] for h in m['H'])


    def add_asset_params_vars(self, m: dict):
        
        assert self.n_product % 2 == 0, "The number of products must be even for ProsumerStorageEnerFlexToy."
        m['nH'] = int(self.n_product/2)

        #sets
        m['H'] = list(range(m['nH']))

        # preference parameters
        m['oppfprice'] = self.get_param_values('oppfprice', m['nH']) # (h)
        m['alphaf'] = self.get_param_values('alphaf', m['nH']) # (h)
        m['flexlimit'] = self.get_param_values('flexlimit', m['nH']) # (h) flexibility limit
        m['fixflex'] = None if not self.config['fixflex'] else self.get_param_values('fixflex', m['nH'])
        m['fixener'] = None if not self.config['fixener'] else self.get_param_values('fixener', m['nH'])

        # variables
        m['pe'] = m['model'].addVars(m['H'], name=f'pe_{self.name}', lb=float('-inf')) # power
        m['pf'] = m['model'].addVars(m['H'], name=f'pf_{self.name}') # flexibility - can only buy

        m['p'] = m['model'].addVars(list(range(self.n_product)), name=f'p_{self.name}', lb=float('-inf')) # products

        return


class ProsumerConsumerFlex(Prosumer):

    def add_asset_params_vars(self, m: dict):
        assert self.n_product % 2 == 0, "The number of products must be even for ProsumerConsumerFlex."
        m['nH'] = int(self.n_product/2)

        #sets
        m['H'] = list(range(m['nH']))

        # preference parameters
        m['oppcprice'] = self.get_param_values('oppcprice', m['nH']) # (h)
        m['consumption'] = self.get_param_values('consumption', m['nH'])
        m['alphae'] = self.get_param_values('alphae', m['nH']) # (h)
        m['fixener'] = None if not self.config['fixener'] else self.get_param_values('fixener', m['nH'])

        # variables
        m['pe'] = m['model'].addVars(m['H'], name=f'pe_{self.name}', lb=float('-inf'))
        m['p'] = m['model'].addVars(list(range(self.n_product)), name=f'p_{self.name}', lb=float('-inf')) # products

        return
    

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['pe'][h] >= 0 for h in m['H']), name=f'cons10u_{self.name}')
        m['model'].addConstrs((m['pe'][h] <= m['consumption'][h] for h in m['H']), name=f'cons10l_{self.name}')
        if m['fixener'] is not None:
            m['model'].addConstrs((m['pe'][h] == m['fixener'][h] for h in m['H']), name=f'cons6_{self.name}')

        # return product package 'p'
        m['model'].addConstrs((m['p'][h] == m['pe'][h] for h in m['H']), name=f'cons3_{self.name}')
        m['model'].addConstrs((m['p'][h + m['nH']] == 0 for h in m['H']), name=f'cons4_{self.name}')

        return


    def get_value_obj_expr(self, m: dict):
        return gp.quicksum(m['alphae'][h] * m['pe'][h] * m['pe'][h] + m['oppcprice'][h] * m['pe'][h] for h in m['H'])


class ProsumerRenewableFlex(Prosumer):

    def add_asset_params_vars(self, m: dict):
        assert self.n_product % 2 == 0, "The number of products must be even for ProsumerRenewableFlex."
        m['nH'] = int(self.n_product/2)

        #sets
        m['H'] = list(range(m['nH']))

        # preference parameters
        m['oppdprice'] = self.get_param_values('oppdprice', m['nH']) # (h)
        m['generation'] = self.get_param_values('generation', m['nH'])
        m['alphae'] = self.get_param_values('alphae', m['nH']) # (h)
        m['fixener'] = None if not self.config['fixener'] else self.get_param_values('fixener', m['nH'])

        # variables
        m['pe'] = m['model'].addVars(m['H'], name=f'pe_{self.name}', lb=float('-inf')) # power for storage
        m['p'] = m['model'].addVars(list(range(self.n_product)), name=f'p_{self.name}', lb=float('-inf')) # products

        return
    

    def add_asset_constraints(self, m: dict):

        m['model'].addConstrs((m['pe'][h] <= 0 for h in m['H']), name=f'cons10u_{self.name}')
        m['model'].addConstrs((m['pe'][h] >= -m['generation'][h] for h in m['H']), name=f'cons10l_{self.name}')
        if m['fixener'] is not None:
            m['model'].addConstrs((m['pe'][h] == m['fixener'][h] for h in m['H']), name=f'cons6_{self.name}')

        # return product package 'p'
        m['model'].addConstrs((m['p'][h] == m['pe'][h] for h in m['H']), name=f'cons3_{self.name}')
        m['model'].addConstrs((m['p'][h + m['nH']] == 0 for h in m['H']), name=f'cons4_{self.name}')
        return


    def get_value_obj_expr(self, m: dict):
        return gp.quicksum(m['alphae'][h] * m['pe'][h] * m['pe'][h] + m['oppdprice'][h] * m['pe'][h] for h in m['H'])


# outdated
class LogarithmicBidder(Bidder):
    def __init__(self, name: str, horizon: int, config: dict):
        self.name = name
        self.intervals = horizon
        
        # class specific parameters
        self.flowlimitup = config[name]['flowlimitup']
        self.flowlimitdown = config[name]['flowlimitdown']
        self.scale = config[name]['scale']
        self.shift = config[name]['shift']
        self.full_info_imp = True

        # piecewise linear approximation of the logarithmic function
        self.x_pts = np.linspace(self.flowlimitdown, self.flowlimitup, config[name]['granularity'])
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
        x = model.addVars([i for i in range(self.intervals)], name=f'{self.name}_x', vtype=GRB.CONTINUOUS, lb=self.flowlimitdown, ub=self.flowlimitup)

        y = model.addVars([i for i in range(self.intervals)], name=f'{self.name}_y', vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY)
        [[model.addConstr(y[i] <= self.yl[j] + (x[i] - self.xl[j])*self.slope[j]) for j in range(len(self.xl))] for i in range(self.intervals)]

        return x, gp.quicksum(y[i]*self.scale for i in range(self.intervals))


    def bundle_query(self, price):
        assert len(price) == self.intervals, "Price vector length must match the horizon."

        res = opt.minimize(self.__neg_utility_func, x0=np.zeros(self.intervals), args=(price), constraints=[], bounds=[(self.flowlimitdown, self.flowlimitup)]*self.intervals)
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
        return [np.array([self.flowlimitdown]*self.intervals), np.array([self.flowlimitup]*self.intervals)]


    def get_name(self, ):
        return self.name


if __name__ == "__main__":
    bidder1 = ProsumerStorage('EV2', 2)
    # bidder2 = ProsumerGurobi('c', 2)

    bounds1 = bidder1.get_capacity_generic_goods()
    bounds = bounds1

    x1 = np.linspace(-bounds[0], bounds[0], 50)
    x2 = np.linspace(-bounds[1], bounds[1], 50)
    x1, x2 = np.meshgrid(x1, x2)
    X1, X2 = x1.flatten(), x2.flatten()
    X3_1 = []
    for x11, x22 in zip(X1, X2):
        bundle = np.array([x11, x22])
        X3_1.append(bidder1.get_value(bundle))

    X3_1 = np.array(X3_1)
    X1, X2, X3_1 = X1.reshape(x1.shape), X2.reshape(x1.shape), X3_1.reshape(x1.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X1, X2, X3_1, alpha=0.5)
    ax.legend()
    plt.show()
