#!/usr/bin/env python3.9.5
import numpy as np
from scipy.special import xlogy


def Entropy(px: np.array) -> float:
    """Compute the entropy of a discrete random variable given its 
       probability mass function px = [p1, p2, ..., pN]
    Arg:
        px (np.array): p.m.f.
    Return:
        float: entropy (nats units)
    """
    return - np.sum(xlogy(px, px))
    

def Joint_entropy(pxy: np.ndarray) -> float:
    """Compute the joint entropy of two generic discrete random 
       variables given their joint p.m.f.
    Arg:
        pxy (np.ndarray): joint p.m.f.
    Return:
        float: joint entropy (nats units)
    """
    return - np.sum(xlogy(pxy, pxy))


def Mutual_information(pxy: np.ndarray, px: np.array, py: np.array) -> float:
    """Compute the mutual information of two generic discrete random 
       variables given their joint and marginal p.m.f., using the 
       relation:
                I(X;Y) = H(X) + H(Y) - H(X,Y)
    Args:
        pxy (np.ndarray): joint p.m.f.
        px (np.array): marginal p.m.f. of r.v. X
        py (np.array): marginal p.m.f. of r.v. Y
    Return:
        float: mutual information (nats units)
    """
    return Entropy(px) + Entropy(py) - Joint_entropy(pxy)
