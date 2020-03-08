import numpy as np

def compute_approx_temperature_at_depth(depth):
    mean_annual_temp_at_surf = 65
    temp_at_14000 = 290
    gradient = (temp_at_14000 - mean_annual_temp_at_surf) / 14000
    return depth * gradient + mean_annual_temp_at_surf


def resistivity_temperature_correction(value, depth):
    temp_1 = compute_approx_temperature_at_depth(depth)
    temp_2 = 65
    return value * ((temp_1 + 6.77) / (temp_2 + 6.77))


def df_apply_res_temp_corr(series):
    if series.sum( ) == 0:
        return np.nan
    return resistivity_temperature_correction(value=series.values, depth=series.index)
