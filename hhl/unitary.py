from xmlrpc.client import Boolean
import numpy as np
from scipy.linalg import expm
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control

import ipdb


def Unitary(A, t):
    #허수 단위 i를 정의
    i = complex(0,1)
    #scipy의 행렬의 exponential을 계산하는 함수를 사용
    U = expm(i*A*t)

    #unitary를 기반으로 양자 gate를 만들고 반환함
    ipdb.set_trace()

    U_gate = UnitaryGate(U)
    return U_gate

# check whether U is unitary
# np.allclose(np.eye(len(U)), U.dot(U.T.conj()))