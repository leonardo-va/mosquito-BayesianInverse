import numpy as np
import temperature_dependent_parameters as tdp
import pytensor as pt

class PyModelCreator:
    temp_dependent_params: list[tdp.tempDependentParameter] 
    offsets = None
    temperature_profile: callable
    temperature_profile_pytensor = None
    _names_in_order_list = ["delta_E", "p_E", "beta", "alpha", "delta_J", "p_J", "omega", "lf_M", "a", 
                            "b_M", "EIP_M", "Lambda", "b_H", "mu_H", "alpha_H", "gamma_H"]
    def __init__(self, temp_dependent_params: list[tdp.tempDependentParameter], temperature_profile):
        self.temp_dependent_params = temp_dependent_params
        offsets = []
        for param in self.temp_dependent_params:
            n_func_params = param.meta.function_type_n_params[param.function_type]
            offsets.append(n_func_params)
        self.offsets = offsets
        self.temperature_profile = temperature_profile
        self._name_check()
    
    def _name_check(self):
        for idx,param in enumerate(self.temp_dependent_params):
            if(not param.name == self._names_in_order_list[idx]):
                print(f"Warning: parameter name mismatch in parameter {idx}: {param.name} is not {self._names_in_order_list[idx]}")

    @classmethod
    def from_setup(cls, setup:dict):
        pass

    def get_parameter_length(self):
        return sum(self.offsets)
    
    
    def temp_dependent_rhs(self, t, u, parameter_vector):
        # unpack parameter vector with offsets:
        current_pos = 0
        func_parameters = []
        for offset in self.offsets:
            func_parameters.append(parameter_vector[current_pos:current_pos+offset])
            current_pos += offset
        
        params_evaluated = []
        for idx, param in enumerate(self.temp_dependent_params):
            temperature = self.temperature_profile(t)
            params_evaluated.append(param.func(temperature, func_parameters[idx]))
        u = u.T
        # print(self.temp_dependent_params[1].function_type)
        # print(params_evaluated[1])
        res = np.zeros((1,len(u))).squeeze()
        res[0] = params_evaluated[2]*(u[2]+u[3]+u[4]) - params_evaluated[0] * u[0] - (params_evaluated[0]/params_evaluated[1] - params_evaluated[0]) * u[0]
        res[1] = params_evaluated[0] * u[0] - params_evaluated[4] * u[1] -params_evaluated[3]*u[1]**2 - (params_evaluated[4]/params_evaluated[5] - params_evaluated[4]) * u[1]
        res[2] = params_evaluated[6]*params_evaluated[4] * u[1] - params_evaluated[8]*params_evaluated[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(params_evaluated[7]*150)*u[2]
        res[3] = params_evaluated[8]*params_evaluated[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/params_evaluated[10]*u[3] - 1/(params_evaluated[7]*150)*u[3]
        res[4] = 1/params_evaluated[10]*u[3] - 1/(params_evaluated[7]*150)*u[4]
        res[5] = params_evaluated[11] - params_evaluated[8]*params_evaluated[12]*u[5]*u[4]/(u[2]+u[3]+u[4]) - params_evaluated[13]*u[5]
        res[6] = params_evaluated[8]*params_evaluated[12]*u[5]*u[4]/(u[2]+u[3]+u[4]) - params_evaluated[14]*u[6] - params_evaluated[13]*u[6]
        res[7] = params_evaluated[14]*u[6] - params_evaluated[15]*u[7] - params_evaluated[13]*u[7]
        res[8] = params_evaluated[15]*u[7] - params_evaluated[13] * u[8]
        return res
    
    def temp_dependent_rhs_pt(self, u, t, parameter_vector):
        '''
        u and t are switched here
        '''
        current_pos = 0
        func_parameters = []
        for offset in self.offsets:
            func_parameters.append(parameter_vector[current_pos:current_pos+offset])
            current_pos += offset
        
        params_evaluated = []
        for idx, param in enumerate(self.temp_dependent_params):
            temperature = self.temperature_profile(t)
            params_evaluated.append(param.func(temperature, func_parameters[idx]))
        # ---- symbolic u ----
        # uT = u.T

        res = pt.tensor.stack([
            # r0
            params_evaluated[2]*(u[2]+u[3]+u[4]) - params_evaluated[0] * u[0] - (params_evaluated[0]/params_evaluated[1] - params_evaluated[0]) * u[0],
            # r1
            params_evaluated[0] * u[0] - params_evaluated[4] * u[1] -params_evaluated[3]*u[1]**2 - (params_evaluated[4]/params_evaluated[5] - params_evaluated[4]) * u[1],
            # r2
            params_evaluated[6]*params_evaluated[4] * u[1] - params_evaluated[8]*params_evaluated[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/(params_evaluated[7]*150)*u[2],
            # r3
            params_evaluated[8]*params_evaluated[9]*u[2]*u[7]/(u[5]+u[6]+u[7]+u[8]) - 1/params_evaluated[10]*u[3] - 1/(params_evaluated[7]*150)*u[3],
            # r4
            1/params_evaluated[10]*u[3] - 1/(params_evaluated[7]*150)*u[4],
            # r5
            params_evaluated[11] - params_evaluated[8]*params_evaluated[12]*u[5]*u[4]/(u[2]+u[3]+u[4]) - params_evaluated[13]*u[5],
            # r6
            params_evaluated[8]*params_evaluated[12]*u[5]*u[4]/(u[2]+u[3]+u[4]) - params_evaluated[14]*u[6] - params_evaluated[13]*u[6],
            # r7
            params_evaluated[14]*u[6] - params_evaluated[15]*u[7] - params_evaluated[13]*u[7],
            # r8
            params_evaluated[15]*u[7] - params_evaluated[13] * u[8],
        ])

        return res.T

    def get_model_function(self, pytensor_version = False):
        if(pytensor_version): return self.temp_dependent_rhs_pt
        else:return self.temp_dependent_rhs