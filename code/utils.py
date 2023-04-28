import copy
import multiprocessing
import warnings
from functools import wraps
from multiprocessing.pool import ThreadPool
from time import time

import numpy as np
import scipy.integrate as integrate
import scipy.stats as sps

warnings.simplefilter('ignore')
np.random.seed(0)


##################################################################
# Decorators
##################################################################

def timer_func(func):
    """
    decorator of function to calculate its running time
    """
    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def esperance_CIR(times):
    """
    decorator of function to rerun it times time
    calculate exp(-X) of output X of the last time step of simulation
    compatible with CIR model (compute_CIR or path_CIR) which returns result of shape [M,T]
    M is number of simulation per function
    return shape [M*times,]
    """
    def esperance_wrap(func):
        @wraps(func)
        def wrapper(kwargss):
            last_X = []
            for result in func(kwargss * times):
                last_X.append(result[:, -1])
            last_X = np.array(last_X)
            return np.exp(-last_X).flatten()

        return wrapper

    return esperance_wrap


def esperance_Heston(times, r, K):
    """
    decorator of function to rerun it times time
    calculate exp(-r)*(K-exp(X1))+ of output [X1,X2,X3,X4] of the last time step of simulation
    compatible with Heston model (compute_Heston or path_Heston) which returns result of shape [4,M,T]
    M is number of simulation per function
    return shape [M*times,]
    """
    def esperance_wrap(func):
        @wraps(func)
        def wrapper(kwargss):
            last_X1 = []
            for result in func(kwargss * times):
                last_X1.append(result[0, :, -1])
            last_X1 = np.concatenate(last_X1)
            gain = K - np.exp(last_X1)
            gain[gain < 0] = 0
            return (np.exp(-r) * gain).flatten()

        return wrapper

    return esperance_wrap


def multi_thread_wrapper(func):
    """
    should be combined with multi_input_wrapper
    divide a list of parameters into groups and run these groups of parameters simultaneously
    """
    @wraps(func)
    def wrapper(kwargss):
        result_list = []
        numberOfThreads = min(multiprocessing.cpu_count(), len(kwargss))
        pool = ThreadPool(processes=numberOfThreads)
        Chunks = np.array_split(kwargss, numberOfThreads)
        results = pool.map_async(func, Chunks)
        pool.close()
        pool.join()
        for result in results.get():
            result_list.extend(result)
        return result_list

    return wrapper


def multi_input_wrapper(func):
    """
    decorator of function to enable it to run with several sets of parameters sequentially
    """
    @wraps(func)
    def wrapper(kwargss):
        result_list = []
        for kwargs in kwargss:
            result_list.append(np.concatenate(list(func(kwargs)), axis=-1))
        return result_list

    return wrapper


def compute_CIR(func):
    """
    decorator of function to wrap it as a generator
    iterate over n time steps of CIR or similar model
    return the result of the last time step of shape [M,1]
    M is number of simulation
    """
    @wraps(func)
    def wrapper(kwargs):
        kwargs = copy.deepcopy(kwargs)
        x0 = func(**kwargs)
        for i in range(kwargs["n"]):
            x0 = func(**kwargs)
            kwargs["x0"] = x0
        yield copy.deepcopy(x0[:, None])

    return wrapper


def path_CIR(func):
    """
    decorator of function to wrap it as a generator
    iterate over n time steps of CIR or similar model
    for each time step, yield the result of shape [M,1]
    M is number of simulation
    """
    @wraps(func)
    def wrapper(kwargs):
        kwargs = copy.deepcopy(kwargs)
        for i in range(kwargs["n"]):
            x0 = func(**kwargs)
            kwargs["x0"] = x0
            yield copy.deepcopy(x0[:, None])

    return wrapper

def compute_Heston(func):
    """
    decorator of function to wrap it as a generator
    iterate over n time steps of Heston or similar model
    return the result of the last time step of shape [4,M,1]
    M is number of simulation
    """
    @wraps(func)
    def wrapper(kwargs):
        kwargs = copy.deepcopy(kwargs)
        x0 = func(**kwargs)
        for i in range(kwargs["n"]):
            x0 = func(**kwargs)
            kwargs["x0"] = x0
        yield copy.deepcopy(x0[:, :, None])

    return wrapper


def path_Heston(func):
    """
    decorator of function to wrap it as a generator
    iterate over n time steps of Heston or similar model
    for each time step, yield the result of shape [4,M,1]
    M is number of simulation
    """
    @wraps(func)
    def wrapper(kwargs):
        kwargs = copy.deepcopy(kwargs)
        for i in range(kwargs["n"]):
            x0 = func(**kwargs)
            kwargs["x0"] = x0
            yield copy.deepcopy(x0[:, :, None])

    return wrapper

def compute_vanilla(func):
    """
    like compute_CIR but over n*T time steps
    """
    @wraps(func)
    def wrapper(kwargs):
        kwargs = copy.deepcopy(kwargs)
        x0 = func(0, **kwargs)
        for i in range(kwargs["n"] * kwargs["T"]):
            x0 = func(i, **kwargs)
            kwargs["x0"] = x0
        yield copy.deepcopy(x0[:, None])

    return wrapper


def path_vanilla(func):
    """
    like path_CIR but over n*T time steps
    """
    @wraps(func)
    def wrapper(kwargs):
        kwargs = copy.deepcopy(kwargs)
        for i in range(kwargs["n"] * kwargs["T"]):
            x0 = func(i, **kwargs)
            kwargs["x0"] = x0
            yield copy.deepcopy(x0[:, None])

    return wrapper


def esperance_vanilla(times, r, K, T):
    """
    decorator of function to rerun it times time
    calculate exp(-rT)*(K-X)+ of output X of the last time step of simulation
    K is an iterable object, meaning that we will evaluate the above value for each strike on the same simulation
    M is number of simulation per function
    return shape [num of strikes, M*times]
    """
    def esperance_wrap(func):
        @wraps(func)
        def wrapper(kwargss):
            last_X1 = []
            for result in func(kwargss * times):
                last_X1.append(result[:, -1])
            last_X1 = np.concatenate(last_X1)
            gain = K[:, None] - last_X1[None, :]
            gain[gain < 0] = 0
            return (np.exp(-r * T) * gain)

        return wrapper

    return esperance_wrap

##################################################################
# Euler
##################################################################

def Euler_DD(sigma, a, k, n, x0, N):
    t = 1 / n
    W = np.random.randn(N)
    return x0 + (a - k * x0) * t + sigma * np.sqrt(x0 * (x0 >= 0) * t) * W


def Euler_HM(sigma, a, k, n, x0, N):
    t = 1 / n
    W = np.random.randn(N)
    return x0 + (a - k * x0) * t + sigma * np.sqrt(np.abs(x0) * t) * W

def Euler_L(sigma, a, k, n, x0, N):
    t = 1 / n
    W = np.random.randn(N)
    return x0 + (a - k * x0 * (x0 >= 0)) * t + sigma * np.sqrt(x0 * (x0 >= 0) * t) * W


def Euler_B(sigma, a, k, n, x0, N):
    t = 1 / n
    W = np.random.randn(N)
    return np.abs(x0 + (a - k * x0) * t + sigma * np.sqrt(x0 * t) * W)


##################################################################
# Exact CIR
##################################################################

def z(k, t):
    return (1 - np.exp(-k * t)) / k


def c(sigma, k, t):
    return 4 / (sigma ** 2 * z(k, t))


def d(sigma, k, t):
    return c(sigma, k, t) * np.exp(-k * t)


def Exact(sigma, a, k, n, x0):
    t = 1 / n
    N = np.random.poisson(d(sigma, k, t) * x0 / 2)
    return np.random.gamma(N + 2 * a / sigma ** 2, 2 / c(sigma, k, t))


def exact_expectation(x0, a, sigma, k, t):
    return np.exp(-d(sigma, k, t) * x0 * t / (c(sigma, k, t) + 2 * t)) * (1 + 2 * t / c(sigma, k, t)) ** (
                -2 * a / sigma ** 2)


##################################################################
# Random Variables
##################################################################

def u_va(N):
    return np.random.uniform(size=(N))


def y1_va(N):
    return np.random.choice([np.sqrt(3), -np.sqrt(3), 0, 0, 0, 0], (N))


def y2_va(N):
    thresh = (np.sqrt(6) - 2) / (4 * np.sqrt(6))
    U = u_va(N)
    Y = np.zeros((N))
    Y = np.where(U <= thresh, np.sqrt(3 + np.sqrt(6)), Y)
    Y = np.where((thresh < U) * (U <= 2 * thresh), - np.sqrt(3 + np.sqrt(6)), Y)
    Y = np.where((2 * thresh < U) * (U <= thresh + 1 / 2), np.sqrt(3 - np.sqrt(6)), Y)
    Y = np.where(thresh + 1 / 2 < U, -np.sqrt(3 - np.sqrt(6)), Y)
    return Y


def epsilon_va(N):
    return np.random.choice([-1, 1], (N))


def zeta_va(N):
    return np.random.choice([1, 2, 3], (N))


##################################################################
# CIR
##################################################################

def zeta(k, t):
    if k == 0:
        return t
    else:
        return 1 / k * (1 - np.exp(-k * t))


def delta(u1, u2):
    return 1 - u1 ** 2 / u2


def pi(delta):
    return (1 - np.sqrt(delta)) / 2


def u1(a, k, t, x):
    return x * np.exp(-k * t) + a * zeta(k, t)


def u2(sigma, a, k, t, x):
    return u1(a, k, t, x) ** 2 + sigma ** 2 * zeta(k, t) * (a * zeta(k, t) / 2 + x * np.exp(-k * t))


def u3(sigma, a, k, t, x):
    return u1(a, k, t, x) * u2(sigma, a, k, t, x) + \
           sigma ** 2 * zeta(k, t) * (2 * x ** 2 * np.exp(-2 * k * t) +
                                      zeta(k, t) * (a + sigma ** 2 / 2) *
                                      (3 * x * np.exp(-k * t) + a * zeta(k, t)))


def k2(sigma, a, k, t):
    if sigma ** 2 <= 4 * a:
        return 0
    else:
        return np.exp(k * t / 2) * ((sigma ** 2 / 4 - a) * zeta(k, t / 2) +
                                    (np.sqrt(np.exp(k * t / 2) * (sigma ** 2 / 4 - a) * zeta(k,
                                                                                             t / 2)) + sigma / 2 * np.sqrt(
                                        3 * t)) ** 2)


def k3(sigma, a, k, t):
    if sigma ** 2 < 4 * a and sigma ** 2 > 4 / 3 * a:
        return (np.sqrt(np.sqrt(sigma ** 2 / 4 - a + sigma / np.sqrt(2) * np.sqrt(a - sigma ** 2 / 4)))
                + sigma / 2 * np.sqrt(3 + np.sqrt(6))
                ) ** 2 * zeta(-k, t)
    elif sigma ** 2 <= 4 / 3 * a:
        return (sigma / np.sqrt(2) * np.sqrt(a - sigma ** 2 / 4)) * zeta(-k, t)
    else:
        return (sigma ** 2 / 4 - a +
                (np.sqrt(sigma / np.sqrt(2) * np.sqrt(sigma ** 2 / 4 - a))
                 + sigma / 2 * np.sqrt(3 + np.sqrt(6))
                 ) ** 2) * zeta(-k, t)


def X0(x, a, sigma, k, t, N, contidion):
    return x + (a - sigma ** 2 / 4) * zeta(-k, t)


def X1(x, a, sigma, k, t, N, contidion):
    return (np.sqrt(x * contidion) + sigma * np.sqrt(zeta(-k, t)) * y2_va(N) / 2) ** 2


def X_tilde(x, a, sigma, k, t, N, contidion):
    return x + sigma / np.sqrt(2) * np.sqrt(np.abs(a - sigma ** 2 / 4)) * epsilon_va(N) * zeta(-k, t)


def CIR_2nd(sigma, a, k, n, x0, N):
    t = 1 / n
    U = u_va(N)
    Y = y1_va(N)

    condition1 = x0 >= k2(sigma, a, k, t)

    uu1 = u1(a, k, t, x0)
    uu2 = u2(sigma, a, k, t, x0)
    ddelta = delta(uu1, uu2)
    ppi = pi(ddelta)
    condition2 = U < ppi

    x0 = np.where(condition1,
                  np.exp(-k * t / 2) * (
                          np.sqrt(((a - sigma ** 2 / 4) * zeta(k, t / 2) + np.exp(-k * t / 2) * x0) * condition1) +
                          sigma / 2 * np.sqrt(t) * Y
                  ) ** 2 + \
                  (a - sigma ** 2 / 4) * zeta(k, t / 2),
                  x0)

    x0 = np.where(np.logical_not(condition1) * condition2,
                  uu1 / (2 * ppi),
                  x0)
    x0 = np.where(np.logical_not(condition1) * np.logical_not(condition2),
                  uu1 / (2 * (1 - ppi)),
                  x0)
    return x0


def CIR_3rd(sigma, a, k, n, x0, N):
    t = 1 / n
    zzeta = zeta_va(N)
    U = u_va(N)

    condition1 = x0 >= k3(sigma, a, k, t)
    condition2 = condition1 * (zzeta == 1) * (sigma ** 2 <= 4 * a)
    condition3 = condition1 * (zzeta == 1) * (sigma ** 2 > 4 * a)
    condition4 = condition1 * (zzeta == 2) * (sigma ** 2 <= 4 * a)
    condition5 = condition1 * (zzeta == 2) * (sigma ** 2 > 4 * a)
    condition6 = condition1 * (zzeta == 3) * (sigma ** 2 <= 4 * a)
    condition7 = condition1 * (zzeta == 3) * (sigma ** 2 > 4 * a)

    assert np.all(condition2 + condition3 + condition4 + condition5 + condition6 + condition7 == condition1)

    x0_tmp_1 = X_tilde(X0(X1(x0, a, sigma, k, t, N, condition2), a, sigma, k, t, N, condition2), a, sigma, k, t, N,
                       condition2)
    x0_tmp_2 = X_tilde(X1(X0(x0, a, sigma, k, t, N, condition3), a, sigma, k, t, N, condition3), a, sigma, k, t, N,
                       condition3)
    x0_tmp_3 = X0(X_tilde(X1(x0, a, sigma, k, t, N, condition4), a, sigma, k, t, N, condition4), a, sigma, k, t, N,
                  condition4)
    x0_tmp_4 = X1(X_tilde(X0(x0, a, sigma, k, t, N, condition5), a, sigma, k, t, N, condition5), a, sigma, k, t, N,
                  condition5)
    x0_tmp_5 = X0(X1(X_tilde(x0, a, sigma, k, t, N, condition6), a, sigma, k, t, N, condition6), a, sigma, k, t, N,
                  condition6)
    x0_tmp_6 = X1(X0(X_tilde(x0, a, sigma, k, t, N, condition7), a, sigma, k, t, N, condition7), a, sigma, k, t, N,
                  condition7)

    uu1 = u1(a, k, t, x0)
    uu2 = u2(sigma, a, k, t, x0)
    uu3 = u3(sigma, a, k, t, x0)
    s = (uu3 - uu1 * uu2) / (uu2 - uu1 ** 2)
    p = (uu1 * uu3 - uu2 ** 2) / (uu2 - uu1 ** 2)
    delta = np.sqrt((s ** 2 - 4 * p) * np.logical_not(condition1))
    ppi = (uu1 - (s - delta) / 2) / (delta + condition1)

    x0 = np.where(condition2, x0_tmp_1, x0)
    x0 = np.where(condition3, x0_tmp_2, x0)
    x0 = np.where(condition4, x0_tmp_3, x0)
    x0 = np.where(condition5, x0_tmp_4, x0)
    x0 = np.where(condition6, x0_tmp_5, x0)
    x0 = np.where(condition7, x0_tmp_6, x0)
    x0 = np.where(condition1,
                  x0 * np.exp(-k * t), x0)

    condition8 = U < ppi
    x0 = np.where(np.logical_not(condition1) * condition8,
                  (s + delta) / 2,
                  x0)
    x0 = np.where(np.logical_not(condition1) * np.logical_not(condition8),
                  (s - delta) / 2,
                  x0)
    return x0


##################################################################
# Heston
##################################################################

def Sch1(rou, r, sigma, a, k, n, x0, N, level):
    t = 1 / n
    if level == 1:
        x2 = Euler_DD(sigma, a, k, n, x0[1], N)
    elif level == 2:
        x2 = CIR_2nd(sigma, a, k, n, x0[1], N)
    elif level == 3:
        x2 = CIR_3rd(sigma, a, k, n, x0[1], N)
    else:
        raise
    x3 = x0[2] + 1 / 2 * (x2 + x0[1]) * t
    x1 = x0[0] + (r - a * rou / sigma) * t + (k * rou / sigma - 1 / 2) * (x3 - x0[2]) + rou / sigma * (x2 - x0[1])
    x4 = x0[3] + 1 / 2 * (np.exp(x0[0]) + np.exp(x1)) * t
    return np.array((x1, x2, x3, x4))


def Sch2(rou, n, x0, N):
    t = 1 / n
    NN = np.random.normal(size=N)
    return np.array((x0[0] + np.sqrt(x0[1]) * np.sqrt(1 - rou ** 2) * np.sqrt(t) * NN, x0[1], x0[2], x0[3]))


def Heston_1st(rou, r, sigma, a, k, n, x0, N):
    t = 1 / n
    W1 = np.random.normal(scale=np.sqrt(t), size=(N))
    W2 = np.random.normal(scale=np.sqrt(t), size=(N))
    x0[2] += x0[1] * t
    x0[3] += np.exp(x0[0]) * t
    x0[0] += (r - x0[1]*(x0[1] >= 0) / 2) * t + np.sqrt(x0[1]*(x0[1] >= 0)) * (rou * W1 + np.sqrt(1 - rou ** 2) * W2)
    x0[1] = Euler_DD(sigma, a, k, n, x0[1], N)
    return x0


def Heston_2nd(rou, r, sigma, a, k, n, x0, N):
    B = np.random.choice((0, 1), size=(N))
    x1 = Sch1(rou, r, sigma, a, k, n, Sch2(rou, n, x0, N), N, 2)
    x2 = Sch2(rou, n, Sch1(rou, r, sigma, a, k, n, x0, N, 2), N)
    x0 = np.where(B == 1, x1, x2)
    return x0


def Heston_3rd(rou, r, sigma, a, k, n, x0, N):
    B = np.random.choice((0, 1), size=(N))
    x1 = Sch1(rou, r, sigma, a, k, n, Sch2(rou, n, x0, N), N, 3)
    x2 = Sch2(rou, n, Sch1(rou, r, sigma, a, k, n, x0, N, 3), N)
    x0 = np.where(B == 1, x1, x2)
    return x0

##################################################################
# Exact Heston
##################################################################

def delta2(rho, sigma, u1, k, lamda):
    return (rho * sigma * u1 - k) ** 2 - sigma ** 2 * (u1 ** 2 - 2 * lamda * u1)


def Psi(rho, sigma, u1, k, _delta):
    return (k - rho * sigma * u1 + np.sqrt(_delta)) / sigma ** 2


def g2(rho, sigma, u1, u2, k, _delta):
    return (k - rho * sigma * u1 + np.sqrt(_delta) - sigma ** 2 * u2) / (
                k - rho * sigma * u1 - np.sqrt(_delta) - sigma ** 2 * u2)


def phi(r, a, t, sigma, u1, u2, _delta, _Psi, _g):
    if _delta == 0:
        return (r * u1 + a * _Psi) * t - 2 * a / sigma ** 2 * np.log(1 + sigma ** 2 / 2 * t * (_Psi - u2))
    else:
        return (r * u1 + a * (_Psi - 2 * np.sqrt(_delta) / sigma ** 2)) * t - 2 * a / sigma ** 2 * np.log(
            (np.exp(-np.sqrt(_delta) * t) - _g) / (1 - _g))


def psi1(u1):
    return u1


def psi2(sigma, u2, k, t, _delta, _Psi, _g):
    if _delta == 0:
        return u2 + (_Psi - u2) ** 2 * sigma ** 2 * t / (2 + sigma ** 2 / 2 * t * (_Psi - u2))
    else:
        return u2 + (_Psi - u2) * (1 - np.exp(np.sqrt(_delta) * t)) / (1 - _g * np.exp(np.sqrt(_delta) * t))


def Phi(r, a, t, rho, sigma, v, k, x1, x2, lamda):
    _delta = delta2(rho, sigma, v * 1j, k, lamda)
    _Psi = Psi(rho, sigma, v * 1j, k, _delta)
    _g = g2(rho, sigma, v * 1j, 0, k, _delta)
    return np.exp(
        phi(r, a, t, sigma, v * 1j, 0, _delta, _Psi, _g) + psi1(v * 1j) * x1 + psi2(sigma, 0, k, t, _delta, _Psi,
                                                                                    _g) * x2)


def Heston_Call(T, K, r, a, k, rho, sigma, x1, x2):
    S0 = np.exp(x1)
    return S0 * (1 / 2 + 1 / np.pi * integrate.quad(lambda v: np.real(
        np.exp(-v * 1j * np.log(K)) * Phi(r, a, T, rho, sigma, v, k - rho * sigma, x1, x2, -1 / 2) / (v * 1j)), 0, 500,
                                                    limit=100)[0]) - \
           K * np.exp(-r * T) * (1 / 2 + 1 / np.pi * integrate.quad(
        lambda v: np.real(np.exp(-v * 1j * np.log(K)) * Phi(r, a, T, rho, sigma, v, k, x1, x2, 1 / 2) / (v * 1j)), 0,
        500, limit=100)[0])


def Heston_Put(T, K, r, a, k, rho, sigma, x1, x2):
    S0 = np.exp(x1)
    return Heston_Call(T, K, r, a, k, rho, sigma, x1, x2) - S0 + K * np.exp(-r * T)

##################################################################
# BSM
##################################################################

def d_p(s, k, v):
    return np.log(s / k) / np.sqrt(v) + 1 / 2 * np.sqrt(v)


def d_m(s, k, v):
    return np.log(s / k) / np.sqrt(v) - 1 / 2 * np.sqrt(v)


def BS_Call(x0, K, T, r, sigma):
    return x0 * sps.norm.cdf(d_p(x0, K * np.exp(-r * T), sigma ** 2 * T)) - K * np.exp(-r * T) * sps.norm.cdf(
        d_m(x0, K * np.exp(-r * T), sigma ** 2 * T))


def BS_Put(x0, K, T, r, sigma):
    return BS_Call(x0, K, T, r, sigma) - x0 + K * np.exp(-r * T)


##################################################################
# Dichotomy
##################################################################

def Dichotomic(value, bound, num_iter):
    """
    Dichotomy search of value for num_iter iterations
    bound of format (low,high)
    decorator of functions of format f(*args,x) where x is the parameter to search
    return x, f(*args,x), error of estimation
    """

    def wrapper_Dichotomic(func):
        @wraps(func)
        def wrapper(*args):
            lower, upper = bound
            mid = (upper + lower) / 2
            v = func(*args, mid)
            epsilon = np.abs(func(*args, lower) - func(*args, upper))
            for i in range(num_iter):
                mid = (upper + lower) / 2
                v = func(*args, mid)
                epsilon = np.abs(func(*args, lower) - func(*args, upper))
                upper = np.where(value < v, mid, upper)
                lower = np.where(value > v, mid, lower)
            return mid, v, epsilon / 2

        return wrapper

    return wrapper_Dichotomic
