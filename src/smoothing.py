import numpy as np
from matplotlib import pyplot as plt

def cubic_polynomial(x, cubic_coeffs):
    return (np.array([x**3, x**2, x, 1]).dot(cubic_coeffs)).item()

def fit_cubic_poly(f_0, f_1, f_prime_0, f_prime_1, x_0, x_1):
    lhs = np.array([f_0,f_1,f_prime_0,f_prime_1]).reshape((4,1))
    lgs_mat = np.array([[x_0**3, x_0**2, x_0, 1],
                        [x_1**3, x_1**2, x_1, 1],
                        [3*x_0**2,2*x_0, 1, 0],
                        [3*x_1**2,2*x_1, 1, 0]])
    cubic_coeffs = np.linalg.inv(lgs_mat) @ lhs
    return lambda x: cubic_polynomial(x, cubic_coeffs)

def estimate_smoothing_intervals(discontinuity_list):
    discontinuity_list = sorted(list(set(discontinuity_list)))
    # discontinuity_list.insert(0, discontinuity_list[0]-10)
    # discontinuity_list.append(discontinuity_list[-1]+10)
    if(len(discontinuity_list) == 1):
        diffs_next = [np.abs(discontinuity_list[0]+1)]
        diffs_prev = [np.abs(discontinuity_list[0]-1)]
        # return [[discontinuity_list[0]-1, discontinuity_list[0]+1]]
    else:
        diffs_next = [abs(discontinuity_list[ii+1]-discontinuity_list[ii]) for ii in np.arange(len(discontinuity_list))[:-1]]
        diffs_next.append(1)
        diffs_prev = diffs_next.copy()
        diffs_prev.insert(0,1)
    smoothing_intervals = []
    for idx, disc_position in enumerate(discontinuity_list):
        replace_interval = [disc_position - 0.1*diffs_prev[idx], disc_position + 0.1*diffs_next[idx]]
        smoothing_intervals.append(replace_interval)
    return smoothing_intervals


class CubicSmoothing:
    original_function = None
    original_function_derivative = None
    smoothed_function = None
    smooth_intervals = [[0, 0]]
    cubic_polys = []

    def __init__(self, function, derivative, intervals: list[list]):
        self.original_function = function
        self.original_function_derivative = derivative
        self.smooth_intervals = intervals
        self._validate_input()

        for interval in self.smooth_intervals:
            lhs_tuple = self._eval_conditions(interval)
            cubic_poly = fit_cubic_poly(*lhs_tuple, interval[0], interval[1])
            self.cubic_polys.append(cubic_poly)
        self.smoothed_function = self._assemble_function()

    def _validate_input(self):
        for interval in self.smooth_intervals:
            assert (interval[0] < interval[1])

    def test_result(self):
        for idx, interval in enumerate(self.smooth_intervals):
            print(f"interval {idx}: [{interval[0]},{interval[1]}]", flush=True)
            print(
                f"x_0: {interval[0]} , |f(x_0)-poly(x_0)|: {np.abs(self.original_function(interval[0]) - self.cubic_polys[idx](interval[0]))}",flush=True)
            print(
                f"x_1: {interval[1]} , |f(x_1)-poly(x_1)|: {np.abs(self.original_function(interval[1]) - self.cubic_polys[idx](interval[1]))}",flush=True)

    def _eval_conditions(self, interval):
        f_0 = self.original_function(interval[0])
        f_1 = self.original_function(interval[1])
        f_prime_0 = self.original_function_derivative(interval[0])
        f_prime_1 = self.original_function_derivative(interval[1])
        return f_0, f_1, f_prime_0, f_prime_1

    def _assemble_function(self):
        def smooth_function(x):
            retval = None
            for idx, interval in enumerate(self.smooth_intervals):
                if interval[0] <= x <= interval[1]:
                    retval = self.cubic_polys[idx](x)
            if (retval is None):
                retval = self.original_function(x)
            return retval

        return smooth_function

    def get_smoothed_function(self):
        return self.smoothed_function

def test_function(x):
    retval = np.nan
    if x<0:
        retval = 0
    if x>=0:
        retval = x
    return retval

def test_function_derivative(x):
    retval = np.nan
    if x<0: retval = 0
    if x>0: retval = 1
    return retval


def main():
    xs = np.linspace(-5,5,500)
    ys = [test_function(x) for x in xs]
    yder = [test_function_derivative(x) for x in xs]
    plt.plot(xs,ys)
    plt.plot(xs, yder)
    plt.show()
    smoothing = CubicSmoothing(test_function, test_function_derivative, [[-1,1]])
    smoothing.test_result()
    test_smoothed = smoothing.get_smoothed_function()
    ys_smoothed = [test_smoothed(x) for x in xs]
    plt.plot(xs, ys_smoothed)
    plt.plot(xs, ys)
    plt.show()
    print("testing interval estimation:\n",flush=True)
    print([1,5,8,9,12,4,8], "\n",flush=True)
    print(estimate_smoothing_intervals([1,5,8,9,12,4]))

if(__name__ == "__main__"):
    main()
