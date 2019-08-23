
import sys
import numpy as np
import pandas as pd


def linear_regression(df, alpha, max_it):
    mat = df.values.astype(np.float)
    (n, d) = mat.shape
    x, y = mat[:, 0: d-1], mat[:, d-1]
    x = (x - np.mean(x, axis=0))/np.std(x, axis=0)
    x = np.column_stack((np.ones(n), x))
    w = np.zeros(d)

    for t in range(max_it):
        fx = w.dot(x.T)
        w = w - x.T.dot((fx - y)) * alpha / n
        w = np.array(w, dtype=np.float64)
    loss = np.sum((w.dot(x.T) - y)**2) / (2*n)

    return w, loss


def main():
    file_in = sys.argv[1]
    file_ot = sys.argv[2]
    dsn = pd.read_csv(file_in, header=None)
    result = []
    for a in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
        beta, loss = linear_regression(dsn, a, 100)
        result.append([a, 100] + list(beta))
    beta, loss = linear_regression(dsn, 0.2, 500)
    result.append([0.2, 500] + list(beta))

    """
    best = float('inf')   
    for a in np.arange(0.1, 1.2, 0.1):
        for it in np.arange(100, 1100, 100):
            beta, loss = linear_regression(dsn, a, it)
            result.append([a, it, loss] + list(beta))
            if loss < best:
                best = loss
                print([a, it, loss] + list(beta))
    """

    res = pd.DataFrame(result)
    res.to_csv(file_ot, header=False, index=False)


if __name__ == '__main__':
    main()