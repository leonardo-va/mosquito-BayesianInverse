import inspect
import numpy as np
from matplotlib import pyplot as plt
import smoothing
# from functools import wraps
import math
import copy
import pytensor.tensor as pt


class ConstantFunction:

    def __call__(self, T, func_params):
        return self.eval(T, func_params)
    def eval(self, T, func_params):
        val = func_params[0]
        return val
    
    def eval_pt(self, T, func_params):
        # func_params can be a PyTensor vector
        val = func_params[0]
        return val

    def eval_sp(self, T, func_params):
        val = func_params[0]
        return val

    def get_stan_string(self):
        return ""

class BriereFunction:
    scaling = 1
    def __init__(self, scaling=1):
        self.scaling = scaling
    
    def __call__(self, T, func_params):
        return self.eval(T, func_params)

    def eval(self, T, func_params):
        '''
        :param T: temperature
        :param func_params: q, Tmin, Tmax in order
        '''
        q, Tmin, Tmax = func_params
        if((T>np.max([0,Tmin])) and (T <Tmax)):
            retval = q/self.scaling*T*(T-Tmin)*np.sqrt(Tmax-T)
        else: retval = 0
        return retval
    
    def eval_pt(self, T, func_params):
        """
        :param T: scalar or PyTensor scalar (temperature)
        :param func_params: vector [q, Tmin, Tmax] (PyMC RV or PyTensor vector)
        :param scaling: scalar value
        """
        q = func_params[0]
        Tmin = func_params[1]
        Tmax = func_params[2]

        # ensure Tmin >= 0
        Tmin_pos = pt.maximum(0, Tmin)

        # condition: (T > Tmin_pos) & (T < Tmax)
        # cond = pt.logical_and(T > Tmin_pos, T < Tmax)
        cond = (T > Tmin_pos) & (T < Tmax)
        # formula
        retval_formula = q / self.scaling * T * (T - Tmin) * pt.sqrt(Tmax - T)

        # use pt.switch instead of if
        retval = pt.switch(cond, retval_formula, 0.0)

        return retval
    
    def get_stan_string(self):
        stan_string = f"""
        real _func_(real T, vector function_parameters) {{
            real q = function_parameters[1];
            real Tmin = function_parameters[2];
            real Tmax    = function_parameters[3];
            real c    = _scaling_;
            real retval;
            if ( (T > fmax(0, Tmin)) && (T < Tmax) ) {{
            retval = (q / c) * T * (T - Tmin) * sqrt(Tmax - T);
            }} else {{
            retval = 0;
            }}
            return retval;
        }}
        """
        return stan_string
    
class QuadraticFunction:
    scaling = 1
    def __init__(self, scaling = 1):
        self.scaling = scaling
    
    def __call__(self, T, func_params):
        return self.eval(T, func_params)

    def eval(self, T, func_parameters):
        '''
        :param T: temperature
        :param func_parameters: q, Tmin, Tmax
        '''
        q, Tmin, Tmax = func_parameters
        quadratic_expression = q/self.scaling*(T-Tmin)*(Tmax-T)
        if(T>Tmin and T<Tmax):
            return min(quadratic_expression,1)
        else: return 0

    def eval_pt(self, T, func_params):
        """
        :param T: scalar or PyTensor scalar
        :param func_params: vector [q, Tmin, Tmax]
        :param scaling: scalar
        """
        q = func_params[0]
        Tmin = func_params[1]
        Tmax = func_params[2]

        # quadratic expression
        quad_expr = q / self.scaling * (T - Tmin) * (Tmax - T)

        # condition: Tmin < T < Tmax
        # cond = pt.logical_and(T > Tmin, T < Tmax)
        cond = (T > Tmin) & (T < Tmax)

        # apply condition, then cap at 1
        retval = pt.switch(cond, pt.minimum(quad_expr, 1.0), 0.0)

        return retval
    
    def get_stan_string(self):
        stan_string = f"""
        real _func_(real T, vector function_parameters) {{
            real q    = function_parameters[1];
            real Tmin = function_parameters[2];
            real Tmax = function_parameters[3];
            ral c = _scaling_;

            real retval = 0;
            if (T > Tmin && T < Tmax) {{
            retval = fmin((q / c) * (T - Tmin) * (Tmax - T), 1);
            }}
            return retval;
        }}
        """
        return stan_string

class TruncatedLinear:
    scaling = 1
    def __init__(self, scaling = 1):
        self.scaling = scaling

    def __call__(self, T, func_params):
        return self.eval(T, func_params)

    def eval(self, T, func_params):
        '''
        :param T: temperature
        :param func_params: beta, Tmax
        '''
        beta, Tmax = func_params
        if(T<Tmax):
            return -beta/self.scaling*(T-Tmax)
        else: return 0

    def eval_pt(self, T, func_params):
        """
        PyTensor-compatible version
        """
        beta = func_params[0]
        Tmax = func_params[1]

        # main expression
        expr = -beta / self.scaling * (T - Tmax)

        # condition: T < Tmax
        cond = T < Tmax

        # symbolic switch
        return pt.switch(cond, expr, 0.0)

    def get_stan_string(self):
        stan_string = f"""
        real _func_(real T, vector function_parameters) {{
            real beta = function_parameters[1];
            real Tmax = function_parameters[2];
            real c = _scaling_;
            real retval = 0;
            if (T < Tmax) {{
            retval = -beta / c * (T - Tmax);
            }}
            return retval;
        }}
        """
        return stan_string
    
class Sigmoidal:
    n_params = 2
    scaling_param = False

    def __call__(self, T, func_params):
        return self.eval(T, func_params)
    
    def eval(self, T, func_params):
        '''
        :param T: temperature
        :param func_params: alpha, beta
        '''
        alpha, beta = func_params
        return 1/(1+np.exp(-(beta*T+alpha)))
    
    def eval_pt(self, T, func_params):
        """
        PyTensor-compatible
        """
        alpha = func_params[0]
        beta  = func_params[1]

        return 1 / (1 + pt.exp(-(beta * T + alpha)))
    
    def get_stan_string(self):
        stan_string = f"""
        real _func_(real T, vector function_parameters) {{
            real alpha  = function_parameters[1];
            real beta = function_parameters[2];

            real retval = 1 / (1 + exp(-(beta * T + alpha)));
            return retval;
        }}
        """
        return stan_string

class ExponentialDecay:
    n_params = 2
    scaling_param = False

    def __call__(self, T, func_params):
        return self.eval(T, func_params)

    def eval(self, T, func_params):
        '''
        :param T: temperature
        :param func_params: alpha, beta
        '''
        alpha, beta = func_params
        return np.exp(-beta*T+alpha)
    
    
    def eval_pt(self, T, func_params):
        """
        PyTensor-compatible version
        """
        alpha = func_params[0]
        beta  = func_params[1]

        return pt.exp(-beta * T + alpha)
    
    def get_stan_string(self):
        stan_string = f"""
        real _func_(real T, vector function_parameters) {{
            real alpha  = function_parameters[1];
            real beta = function_parameters[2];

            real retval = exp(-beta * T + alpha));
            return retval;
        }}
        """
        return stan_string
    
# def visualize_function_class(function_class):


def create_constant_function(val = None):
    '''
    return (func,derivative, derivative_discontinuities)
    '''
    return (lambda t: val, lambda t: 0, [], "")

def create_briere_function(q=None, c=1, Tmin=None, Tmax=None):
    '''
    parameters to fit: q, Tmin, Tmax
    c is a scaling factor such that q/c is on the same scale as tmin and tmax
    return (func,derivative, derivative_discontinuities) tuple
    '''

    if any(arg is None for arg in (q,Tmin,Tmax)):
        def func(T, ):
            if((T>np.max([0,Tmin])) and (T <Tmax)):
                retval = q/c*T*(T-Tmin)*np.sqrt(Tmax-T)
            else: retval = 0
            return retval

    def func(T):
        if((T>np.max([0,Tmin])) and (T <Tmax)):
            retval = q/c*T*(T-Tmin)*np.sqrt(Tmax-T)
        else: retval = 0
        return retval
    def derivative(T):
        if((T>max(0,Tmin)) and (T < Tmax)):
            retval =  q/c * ((2*T-Tmin) * np.sqrt(Tmax - T) - T/2*(T-Tmin)/np.sqrt(Tmax-T))
        else:
            retval = 0
        return retval
    stan_string = f"""
    real _func_(real T, vector function_parameters) {{
        real q = function_parameters[1];
        real Tmin = function_parameters[2];
        real Tmax    = function_parameters[3];
        real c    = _scaling_;
        real retval;
        if ( (T > fmax(0, Tmin)) && (T < Tmax) ) {{
        retval = (q / c) * T * (T - Tmin) * sqrt(Tmax - T);
        }} else {{
        retval = 0;
        }}
        return retval;
    }}
    """

    derivative_discontinuities = [np.max([Tmin,0]), Tmax]

    return func, derivative, derivative_discontinuities, stan_string

def quadratic_zeros(a,b,c):
    if(math.isclose(a,0)):
        return None
    r_term = b**2-4*a*c
    if(math.isclose(r_term, 0)):
        r_term = 0
    if(r_term<0):
        return None
    sol_1 = (-b+r_term)/2*a
    sol_2 = (-b-r_term)/2*a
    return sol_1, sol_2

def create_quadratic_function(q, c, Tmin, Tmax):
    def quadratic_expression(T):
        return q/c*(T-Tmin)*(Tmax-T)
    q_zeros = quadratic_zeros(-q/c,q/c*(Tmin+Tmax),-(q/c*Tmin*Tmax+1))
    derivative_discontinuities = [Tmin, Tmax]
    if(q_zeros is not None):
        for q_zero in q_zeros:
            if(Tmin < q_zero < Tmax):derivative_discontinuities.append(q_zeros[0])
    def func(T):
        retval = 0
        if(T>Tmin and T<Tmax):
            retval = min(quadratic_expression(T),1)
        return retval
    def derivative(T):
        retval = 0
        if(T>Tmin and T<Tmax):
            if(quadratic_expression(T)<=1):
                retval = q/c*(Tmax+Tmin-2*T)
        return retval
    
    stan_string = f"""
    real _func_(real T, vector function_parameters) {{
        real q    = function_parameters[1];
        real Tmin = function_parameters[2];
        real Tmax = function_parameters[3];
        ral c = _scaling_;

        real retval = 0;
        if (T > Tmin && T < Tmax) {{
        retval = fmin((q / c) * (T - Tmin) * (Tmax - T), 1);
        }}
        return retval;
    }}
    """

    return func, derivative, derivative_discontinuities, stan_string

def create_truncated_linear(beta, Tmax, scale=1):
    #Tmax = alpha/beta
    def func(T):
        retval = 0
        if(T<Tmax):
            retval = -beta/scale*(T-Tmax)
        return retval
    def derivative(T):
        retval = 0
        if(T<Tmax):
            retval = -beta/scale
        return retval
    derivative_discontinuities = [Tmax]

    stan_string = f"""
    real _func_(real T, vector function_parameters) {{
        real beta = function_parameters[1];
        real Tmax = function_parameters[2];
        real c = _scaling_;
        real retval = 0;
        if (T < Tmax) {{
        retval = -beta / c * (T - Tmax);
        }}
        return retval;
    }}
    """
    return func, derivative, derivative_discontinuities, stan_string

def create_sigmoidal(alpha, beta):
    def func(T):
        retval = 1/(1+np.exp(-(beta*T+alpha)))
        return retval
    def derivative(T):
        retval = -1/(func(T)**2)*(-beta*np.exp(-(beta*T+alpha)))
        return retval
    derivative_discontinuities = []

    stan_string = f"""
    real _func_(real T, vector function_parameters) {{
        real alpha  = function_parameters[1];
        real beta = function_parameters[2];

        real retval = 1 / (1 + exp(-(beta * T + alpha)));
        return retval;
    }}
    """
    return func, derivative, derivative_discontinuities, stan_string

def create_exponential_decay(alpha,beta):
    def func(T):
        retval = np.exp(-beta*T+alpha)
        return retval
    def derivative(T):
        retval = -beta*func(T)
        return retval
    derivative_discontinuities = []
    stan_string = f"""
    real _func_(real T, vector function_parameters) {{
        real alpha  = function_parameters[1];
        real beta = function_parameters[2];

        real retval = exp(-beta * T + alpha));
        return retval;
    }}
    """

    return func, derivative, derivative_discontinuities, stan_string

# def add_temperature_dependence(func, parameters_temp_dependent):
#     sig = inspect.signature(func)
#     @wraps(func)
#     def func_temp_dependent(*args, **kwargs):
#         bound_args = sig.bind(*args, **kwargs)
#         bound_args.apply_defaults()
#         if 'parameters' and 't' in bound_args.arguments:
#             current_t = bound_args.arguments['t']
#             bound_args.arguments['parameters'] = [temp_func(current_t) for _,temp_func in parameters_temp_dependent.items()]
#         if 'setup' in bound_args.arguments:
#             bound_args.arguments['setup']['parameters'] = parameters_temp_dependent
#         return func(*bound_args.args, **bound_args.kwargs)
#     return func_temp_dependent


class tempDependentParameterMeta:
    name:str
    long_name:str
    function_type:str
    _function_parameters:list = []
    _type_parameters:list = []
    # smoothing_intervals:list
    function_type_list = {
        "briere":BriereFunction,
        "quadratic":QuadraticFunction,
        "constant":ConstantFunction,
        "truncated_linear":TruncatedLinear,
        "sigmoidal":Sigmoidal,
        "exponential_decay":ExponentialDecay
        }
    function_type_n_params = {
        "briere":3,
        "quadratic":3,
        "constant":1,
        "truncated_linear":2,
        "sigmoidal":2,
        "exponential_decay":2
    }
    # function_type_scaling_arg_pos = {
    #     "briere":1,
    #     "quadratic":1,
    #     "truncated_linear":2,
    #     "constant":None,
    #     "sigmoidal":None,
    #     "exponential_decay":None
    # }
    def __init__(self, name = "", function_type="",function_parameters=[], additional_type_parameters = [],long_name = ""):
        self.name = name
        self.long_name = long_name
        self.function_type = function_type
        self._function_parameters = function_parameters
        # these are passed to the function class cunstructor
        self._type_parameters = additional_type_parameters
        # self.smoothing_intervals = smoothing_intervals
    def stan_entries(self):
        stan_function_string = self.function_type_list[self.function_type]().get_stan_string()
        stan_function_string = stan_function_string.replace("_func_", f"{self.name}_func")
        if(len(self._type_parameters)>0):
            print(f"warning creating stan entries: _scaling_ is associated with type_parameters[0]")
            stan_function_string = stan_function_string.replace("_scaling_", f"{self._type_parameters[0]}")
        param_length = self.function_type_n_params[self.function_type]
        if(self.function_type == "constant"):
            stan_param_name = self.name
        else:
            stan_param_name = f"{self.name}_func_params"
        func_call = f"{self.name}_func(T, {stan_param_name});"
        # if(self.function_type_scaling_arg_pos[self.function_type]) is not None:
        #     func_call += f", {str(self._function_parameters[self.function_type_scaling_arg_pos[self.function_type]])}"
        # func_call += f");"
        stan_entries = {
            "functions":stan_function_string,
            "parameters":f"vector[{str(param_length)}] {stan_param_name}",
            "ode_rhs_access":f"{self.name}_func(T, {stan_param_name})",
            "stan_param_name":stan_param_name
        }
        return stan_entries
    def print_info(self):
        print("Parameter information:", flush=True)
        print(f"\t parameter {self.name} : {self.long_name}", flush = True)
        print(f"\t modeled by {self.function_type} function with parameters {self._function_parameters}")

class tempDependentParameter:
    name:str
    long_name:str
    function_type:str
    n_function_parameters:int
    _function_parameters:list = []
    _type_parameters:list = []
    is_constant:bool
    meta:tempDependentParameterMeta
    func:callable
    def __init__(self, name = "", func = None, function_type="", n_function_parameters = 0,function_parameters=[], additional_type_parameters = [],long_name = "", meta=None):
        self.meta = meta
        self.name = name
        self.long_name = long_name
        self.func = func
        self.function_type = function_type
        self.n_function_parameters = n_function_parameters
        self._function_parameters = function_parameters
        self._type_parameters = additional_type_parameters
        if(len(self._function_parameters)>0):
            self.is_constant = True
        else:
            self.is_constant = False
    
    @classmethod
    def from_metadata(cls, meta:tempDependentParameterMeta, pytensor_compatible = False):
        func_class = meta.function_type_list[meta.function_type](*meta._type_parameters)
        if(pytensor_compatible):
            func = func_class.eval_pt
        else:
            func = func_class.eval
        n_func_params = meta.function_type_n_params[meta.function_type]
        init_args = [meta.name, func, meta.function_type, n_func_params ,meta._function_parameters, meta._type_parameters, meta.long_name, meta]
        return cls(*init_args)
    
    def get_function_parameters(self):
            '''
            return a deepcopy of the internal function parameters
            '''
            return copy.deepcopy(self._function_parameters)
    
    def get_function_parameter_length(self):
        return self.n_function_parameters

    def visualize(self, func_params = None,time_dependent = False):
        if(func_params is None):
            func_params = self.get_function_parameters()
        if(time_dependent):
            xaxis = np.linspace(0,100,500)
            xlabel = "time(days)"
        else:
            xaxis = np.linspace(-10,60,500)
            xlabel = "temp(degrees C)"

        if(time_dependent):
            evals = [self.func(temperature_profile(t), func_params) for t in xaxis]
        else:
            evals = [self.func(t, func_params) for t in xaxis]
        plt.plot(xaxis,evals)
        plt.title(f"{self.name} / {self.long_name}")
        plt.xlabel(xlabel)
        plt.ylabel(self.name)
        plt.show()
    
    def print_info(self):
        self.meta.print_info()
        print(f"\tconstant:{self.is_constant}")

def temperature_profile(time:float):
    time_15_0 = np.arcsin(1/2)*100/np.pi
    time_15_1 = (np.pi - np.arcsin(1/2))*100/np.pi
    
    if(time<0 or time>100): return 15
    return 30*np.sin((time_15_0 + time/100*(time_15_1-time_15_0))*np.pi/100)

def temperature_profile_pt(time):
    """
    time: PyTensor scalar or array variable
    returns: PyTensor expression
    """
    time_15_0 = pt.arcsin(0.5) * 100.0 / pt.pi
    time_15_1 = (pt.pi - pt.arcsin(0.5)) * 100.0 / pt.pi

    core = 30.0 * pt.sin((time_15_0 + time/100.0 * (time_15_1 - time_15_0)) * pt.pi / 100.0)
    return pt.switch(
        pt.or_(pt.lt(time, 0.0), pt.gt(time, 100.0)),
        15.0,
        core
    )

def make_gt_temp_dependent_parameters(setup:dict)->tuple[dict[str,tempDependentParameter], dict[str,tempDependentParameter]]:
    '''
    :param setup: Description
    :type setup: dict
    :return: tuple of temp dependent parameter list: (numpy version, pytensor version)
    :rtype: tuple[dict[str, tempDependentParameter], dict[str, tempDependentParameter]]
    '''
    #some are temporary treated as constants. update this if adding new temp dependent ones
    # _params_in_order = ["delta_E", "p_E", "beta", "alpha", "delta_J", "p_J", "omega", "lf_M", "a", "b_M", "EIP_M", "Lambda", "b_H", "mu_H", "alpha_H", "gamma_H"]
    _params_in_order = setup["parameters"].keys()
    _temp_dependent_indices = [0,1,2,4,5,7,8,9,10]
    # _constant_indices = []
    # for idx,_ in _params_in_order:
    #     if idx not in _temp_dependent_indices:
    #         _constant_indices.append(idx)
    temp_dependent_param_metas = [
    tempDependentParameterMeta("delta_E", "briere", [3.43, 2.53, 51.69], [1.6*10**4],"egg hatching rate"),
    # scaling c is not known for this. Use cx.pallens data (cx.pipiens missing)
    tempDependentParameterMeta("p_E", "quadratic", [5.49, 6.56, 39.52], [1.6*10**4],"egg viability"),
    # beta = a*eggs_per_raft, eggs_per_raft = 140
    tempDependentParameterMeta("beta", "briere", [3.71, 1.93, 45.1], [43000/140],"egg laying rate"),
    tempDependentParameterMeta("delta_J", "briere", [3.96, 1.57, 41.9], [1.6*10**4],"juvenile development rate"),
    tempDependentParameterMeta("p_J", "quadratic", [4.41, 7.38, 36.14], [10**3], "juvenile survival"),
    tempDependentParameterMeta("lf_M", "truncated_linear", [4.42, 34.05], [150],"adult life span"),
    tempDependentParameterMeta("a", "briere", [3.71, 1.93, 45.1], [43000], "biting rate"),
    tempDependentParameterMeta("b_M", "sigmoidal", [-4.64, 0.25], [], "mosquito infection probability"),
    tempDependentParameterMeta("EIP_M", "exponential_decay", [5.36, 0.08], [], "extrinsic incubation period mosquitoes")
    ]

    param_meta_dict = {}
    rel_idx = 0
    for idx, pname in enumerate(_params_in_order):
        if(idx in _temp_dependent_indices):
            param_meta_dict[pname] = temp_dependent_param_metas[rel_idx]
            rel_idx += 1
        else:
            param_meta_dict[pname] = tempDependentParameterMeta(pname, "constant", [setup["parameters"][pname][0]], [], "")
    
    param_dict = {pname:tempDependentParameter.from_metadata(pmeta) for pname,pmeta in param_meta_dict.items()}
    param_dict_pt = {pname:tempDependentParameter.from_metadata(pmeta, pytensor_compatible=True) for pname,pmeta in param_meta_dict.items()}
    return param_dict, param_dict_pt