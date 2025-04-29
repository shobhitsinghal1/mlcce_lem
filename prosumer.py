from utils import prosumer_configs, asset_configs
import numpy as np
import pyomo.environ as pyo

class Prosumer:
    def __init__(self, prosumer: str):
        self.prosumer_config = prosumer_configs[prosumer]
        self.prosumer_demand_query_model = self.__construct_demand_query_model()
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
        
    def __construct_demand_query_model(self, ) -> pyo.AbstractModel:
        """
        Construct the demand query optimization parametric model for the prosumer.
        :return: demand query model.
        """
        model = pyo.AbstractModel()

        nIf = len(self.asset_configs_by_type['fixedload'])
        nIs = len(self.asset_configs_by_type['storage'])
        nH = len(self.asset_configs_by_type['fixedload'][0]['lamb']) if nIf > 0 else len(self.asset_configs_by_type['storage'][0]['lambl'])

        #sets
        model.H = pyo.RangeSet(0, nH-1)
        model.Hs = pyo.RangeSet(-1, nH-1)
        model.If = pyo.RangeSet(0, nIf-1)
        model.Is = pyo.RangeSet(0, nIs-1)

        model.M = pyo.Param(default=1000) # Is there a better way to do this - like the one in mvnn mip?

        #price as parameter
        model.price = pyo.Param(model.H) # price vector

        #preference parameters
        #   fixed load
        model.lamb = pyo.Param(model.If, model.H, default=self.__get_param_values('lamb', 'fixedload')) # linear hourly coefficients
        model.gu = pyo.Param(model.If, model.H, default=self.__get_param_values('gu', 'fixedload')) # upper limit on production
        model.gl = pyo.Param(model.If, model.H, default=self.__get_param_values('gl', 'fixedload')) # lower limit on production
        #   storage
        model.lambl = pyo.Param(model.Is, model.H, default=self.__get_param_values('lambl', 'storage')) # lower limit violation penalty coeff
        model.lambu = pyo.Param(model.Is, model.H, default=self.__get_param_values('lambu', 'storage')) # upper limit violation penalty coeff
        model.su = pyo.Param(model.Is, model.H, default=self.__get_param_values('su', 'storage')) # SoC upper limit
        model.sl = pyo.Param(model.Is, model.H, default=self.__get_param_values('sl', 'storage')) # SoC lower limit
        model.s0 = pyo.Param(model.Is, default=self.__get_param_values('s0', 'storage')) # initial SoC
        model.eta = pyo.Param(model.Is, default=self.__get_param_values('eta', 'storage')) # charging efficiency
        model.gamma = pyo.Param(model.Is, model.H, default=self.__get_param_values('gamma', 'storage')) # dissipation rate
        model.beta = pyo.Param(model.Is, model.H, default=self.__get_param_values('beta', 'storage')) # storage degradation coeff

        #variables
        model.p = pyo.Var(model.H) # net power
        model.pf = pyo.Var(model.If, model.H) # power for fixed load
        model.ps = pyo.Var(model.Is, model.H) # power for storage
        #   auxiliary: storage
        model.deltap = pyo.Var(model.Is, model.H, domain=pyo.Binary)
        model.s = pyo.Var(model.Is, model.Hs, domain=pyo.NonNegativeReals)
        model.tu = pyo.Var(model.Is, model.H, domain=pyo.NonNegativeReals)
        model.tl = pyo.Var(model.Is, model.H, domain=pyo.NonNegativeReals)
        model.P = pyo.Var(model.Is, model.H)

        #objective

        def objf(m, i):
            return sum(m.pf[i, h] * m.lamb[i, h] for h in m.H)

        def objs(m, i):
            return sum(-m.tu[i,h]*m.lambu[i,h] - m.tl[i,h]*m.lambl[i,h] - m.P[i,h]*m.beta[i,h] for h in m.H)

        def obj_expression(m):
            return sum(objf(m, i) for i in m.If) + sum(objs(m, i) for i in m.Is) - sum(m.p[h]*m.price[h] for h in m.H)

        model.obj = pyo.Objective(rule=obj_expression, sense=pyo.maximize)
        # model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

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

        def cons9(m, i, h):
            return m.P[i,h] >= m.ps[i,h]

        def cons10(m, i, h):
            return m.P[i,h] >= -m.ps[i,h]

        def cons11(m, i):
            return m.s[i,-1] == m.s0[i]

        model.conp = pyo.Constraint(model.H, rule=conp)
        model.conf1 = pyo.Constraint(model.If, model.H, rule=conf1)
        model.cons1 = pyo.Constraint(model.Is, model.H, rule=cons1)
        model.cons2 = pyo.Constraint(model.Is, model.H, rule=cons2)
        model.cons3u = pyo.Constraint(model.Is, model.H, rule=cons3u)
        model.cons3l = pyo.Constraint(model.Is, model.H, rule=cons3l)
        model.cons4u = pyo.Constraint(model.Is, model.H, rule=cons4u)
        model.cons4l = pyo.Constraint(model.Is, model.H, rule=cons4l)
        model.cons5 = pyo.Constraint(model.Is, model.H, rule=cons5)
        model.cons6 = pyo.Constraint(model.Is, model.H, rule=cons6)
        model.cons9 = pyo.Constraint(model.Is, model.H, rule=cons9)
        model.cons10 = pyo.Constraint(model.Is, model.H, rule=cons10)
        model.cons11 = pyo.Constraint(model.Is, rule=cons11)

        return model
        
    def bundle_query(self, price: np.ndarray) -> np.ndarray:
        """
        Query the utility maximizing demand of the prosumer for given prices.
        :param prices: price vector.
        :return: demand vector.
        """
        instance = self.dq_model.create_instance({None: {
                                                         'price': {h: price[h] for h in range(len(price))}
                                                        }})

        solver = pyo.SolverFactory('gurobi_direct')
        solution = solver.solve(instance, tee=False, options={"MIPGap": 0})

        if solution.Solver.termination_condition == pyo.TerminationCondition.infeasibleOrUnbounded:
            raise ValueError("The prosumer greedy bundle model is infeasible or unbounded.")

        return np.array(instance.p[h] for h in instance.H)
    
    def calculate_value(self, bundle) -> float:
        # TODO: implement this function
        pass