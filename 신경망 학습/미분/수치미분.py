def numerical_ciff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h) / (2*h))