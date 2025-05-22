import numpy as np

def x_normalizer(X, var_array):
    def max_min_scaler(x, x_max, x_min):
        return (x - x_min) / (x_max - x_min)
    x_norm = []
    for x in X:
        x_norm.append([
            max_min_scaler(x[i], max(var_array[i]), min(var_array[i]))
            for i in range(len(x))
        ])
    return x_norm

def x_denormalizer(x_norm, var_array):
    def max_min_rescaler(x, x_max, x_min):
        return x * (x_max - x_min) + x_min
    x_original = []
    for x in x_norm:
        x_original.append([
            max_min_rescaler(x[i], max(var_array[i]), min(var_array[i]))
            for i in range(len(x))
        ])
    return x_original

def get_closest_value(given_value, array_list):
    return min(array_list, key=lambda v: abs(v - given_value))

def get_closest_array(suggested_x, var_list):
    return np.array([
        [get_closest_value(x[i], var_list[i]) for i in range(len(x))]
        for x in suggested_x
    ])
