import numpy as np

def safe_div(a, b, default=0):
    return np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b != 0)
