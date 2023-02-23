import qiskit
from scipy.linalg import expm
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control
from qiskit import IBMQ, Aer, transpile, assemble

from qiskit.circuit.library import PhaseEstimation
from hhl.unitary import Unitary

#QPE에 사용될 Quantum Fourier Transformation을 정의한다.
def qft(n, inverse = False):
    #사용될 양자 회로를 정의한다.
    qc = QuantumCircuit(n)
    #QFT의 swap gate에 대한 효과를 reversed함수를 통해서 gate를 반대로 적용하여 사용하였다.
    #QFT는 각 qubit에 대해서 Hadamard gate를 걸고 그 qubit다음에 존재하는 qubit들에 대해서 
    #두 qubit이 떨어져 있는 정도에 따라서 회전을 시키는 과정을 모든 qubit에 적용하여 작동한다.
    #회전하는 정도는 pi를 두 qubit이 떨어진 정도를 2의 지수로 표현한 것으로 나눈것이다. 
    for j in reversed(range(n)):
        qc.h(j)
        for m in reversed(range(j)):
            #m은 항상 j보다 작거나 같기 때문에 2의 지수는 항상 음수이다.
            qc.cp(np.pi * (2.0 ** (m - j)), m, j)
    #inverse qft를 얻고 싶을 경우 inverse 함수를 사용
    if inverse:
        qc = qc.inverse()
    return qc

#Quantum Phase Estimation을 정의하는 함수
def QPE(n_l, n_b, A,t):
    #주어진 Hermition 행렬을 t를 통해서 unitary 행렬로 바꾸어 양자 gate의 형태로 바꾼다.
    U = Unitary(A, t)
    #U를 통해 만든 gate를 controlled gate의 형태로 바꾼다. (1은 control qubit의 수가 1이라는 의미)
    CU = add_control(U,1,ctrl_state=None, label="CU")
    #이름 부여
    CU.name = "CU"
    #QPE를 적용할 양자 register를 만든다.
    nl_rg = QuantumRegister(n_l, "state")
    nb_rg = QuantumRegister(n_b, "q")
    #양자 register들을 합친 회로를 만든다. 
    qc = QuantumCircuit(nl_rg,nb_rg)
    #회로의 이름
    qc.name = "QPE"

    #아래부터 QPE의 정의부분
    #nl register의 모든 qubit에 Hadamard gate를 걸어주어 모든 상태가 중첩된 상태를 만든다.
    qc.h(nl_rg[:]) 
    qc.barrier()
    #controlled unitary gate를 각각 2^l만큼 제곱한 gate를 걸어주어 행렬 A의 eigenvalue를 Fourier basis에서 표현한다.
    for l in range(n_l):
        qc.append(CU.power(2**l), [nl_rg[l]]+[nb_rg[i] for i in range(n_b)]) 
    qc.barrier()
    #QFT를 적용하여 eigenvalue를 nl register의 computational basis 상태로 표현
    qc = qc.compose(qft(n_l, inverse = True).reverse_bits(), nl_rg[:])
    qc.barrier()
    return qc

#qiskit에서 제공한 PhaseEstimation 함수를 사용해서 QPE를 구현
def qpe_qiskit(nl, A, t, adjoint=False):
    U = Unitary(A, t, adjoint=adjoint)
    qpe = PhaseEstimation(nl, U)
    return qpe


if __name__ == "__main__":
    qc_qft = qft(4).reverse_bits()
    qc_qft.draw("mpl").savefig("image/QFT.png")

    A = np.array([[2,-1],[1,4]])
    A = np.vstack((np.hstack((np.zeros_like(A),A)),np.hstack((A.conj().T, np.zeros_like(A)))))
    qc_qpe = QPE(3, 2, A, np.pi/16)
    qc_qpe.draw("mpl").savefig("image/QPE.png")