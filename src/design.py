import numpy as np
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs.latin_design import LatinDesign

def make_linspace(start, stop, step):
    num_points = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, num_points), 4)

def get_variable_space():
    return [
        make_linspace(0.25, 0.55, 0.01),   # speed_inorg
        make_linspace(0.25, 0.55, 0.01),   # speed_org
        make_linspace(80, 200, 1),         # inkFL_inorg
        make_linspace(100, 280, 1),        # inkFL_org
        make_linspace(0.8, 1.6, 0.05),     # conc_inorg
        make_linspace(0.4, 1.2, 0.05),     # conc_org
        make_linspace(11, 44, 1),          # humidity
        make_linspace(20, 65, 5)           # temperature
    ]

def get_parameter_space():
    return ParameterSpace([
        ContinuousParameter("speed_inorg", 0.25, 0.55),
        ContinuousParameter("speed_org", 0.25, 0.55),
        ContinuousParameter("inkFL_inorg", 80, 200),
        ContinuousParameter("inkFL_org", 100, 280),
        ContinuousParameter("conc_inorg", 0.8, 1.6),
        ContinuousParameter("conc_org", 0.4, 1.2),
        ContinuousParameter("humidity", 11, 44),
        ContinuousParameter("temp", 20, 65),
    ])

def generate_initial_design(n_samples=10):
    space = get_parameter_space()
    design = LatinDesign(space)
    return design.get_samples(n_samples)