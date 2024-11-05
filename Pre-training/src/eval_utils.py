import numpy as np


def root_mean_sq_err(src, tgt):
    '''
    Root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : root mean squared error
    '''

    return np.sqrt(np.mean((tgt - src) ** 2))

def mean_abs_err(src, tgt):
    '''
    Mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(tgt - src))

def inv_root_mean_sq_err(src, tgt):
    '''
    Inverse root mean squared error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse root mean squared error
    '''

    return np.sqrt(np.mean(((1.0 / tgt) - (1.0 / src)) ** 2))

def inv_mean_abs_err(src, tgt):
    '''
    Inverse mean absolute error

    Arg(s):
        src : numpy[float32]
            source array
        tgt : numpy[float32]
            target array
    Returns:
        float : inverse mean absolute error
    '''

    return np.mean(np.abs((1.0 / tgt) - (1.0 / src)))

def log_root_mean_sq_err(src, tgt):
    return np.sqrt(np.mean((np.log(src)-np.log(tgt))**2))

def abs_rel_err(src, tgt):
    return np.mean(np.abs(src-tgt)/tgt)

def sq_rel_err(src, tgt):
    return np.mean(((src-tgt)**2)/tgt)

def thresh_ratio_err(src, tgt, thresh=1.25):
    ratio = np.maximum((tgt/src), (src/tgt))
    return np.mean(ratio < thresh)
