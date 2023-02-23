from xmlrpc.client import Boolean
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit import IBMQ, Aer, transpile, assemble
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal

#PiecewiseChebyshev를 사용해서 eigenvalue의 역수만큼 amplitude를 회전시키는 방법, 근사적인 방법을 사용해서 정확도가 떨어져서 결과가 잘 나오지 않는다.
def rotation(nl):
    #c는 A의 가장 작은 eigenvalue보다 작아야하지만 (확률은 최대 1이기 때문) 최대한 커야할 필요가 있다. (flag가 1로 측정될 확률을 늘리기 위해서)
    #c를 결정하는 방법이 존재 Haener, T., Roetteler, M., & Svore, K. M. (2018). Optimizing Quantum Circuits for Arithmetic. `arXiv:1805.12445 <http://arxiv.org/abs/1805.12445>`
    c = 1 
    #degree는 근사하는 polynomial의 최대 차수, breakpoint는 구간을 나누는 점들, num_state_qubits는 상태의 수. 이 값들 역시 최적값이 위의 논문에서 계산됨
    f_x, degree, breakpoints, num_state_qubits = lambda x: np.arcsin(c / x), 2, [1,2,3,4], nl
    #PiecewiseChebyshev를 사용한 회로를 만듬
    pw_approximation = PiecewiseChebyshev(f_x, degree, breakpoints, num_state_qubits)
    pw_approximation._build()

    #필요한 양자 register들을 만든다.
    nl_rg = QuantumRegister(nl, "eval")
    na_rg = QuantumRegister(nl, "a")
    nf_rg = QuantumRegister(1, "f")
    #양자 register들을 합쳐 양자 회로를 구성한다.
    qc = QuantumCircuit(nl_rg, nf_rg, na_rg)
    #양자 회로에 PiecewiseChebyshev를 적용해서 회로를 반환한다.
    qc.append(pw_approximation,nl_rg[:]+[nf_rg[0]]+na_rg[:])
    return qc

#qiskit에서 제공하는 ExactReciprocal함수를 사용해서 양자상태를 eignevalue의 역수만큼 정확하게 돌린다.
#arcsin을 통한 각도에 대한 계산을 고전 컴퓨터에서 시행하고 그 각도만큼 회전을 적용하기 때문에 정확하게 계산이 가능하다.
#delta는 scaling을 위한 값으로 들어가며 nl*scaling을 eigenvalue의 분자 부분으로 사용하게 된다. delta의 정확한 값은 eignevalue의 최소값을 사용해서 계산되어야 한다.
def Reciprocal(nl, delta, neg_vals):
    reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=neg_vals)
    return reciprocal_circuit

#RY gate, gate의 이름을 지정하기 위해서 새로 정의 
def RY(angle, eigen):
    #register 설정, 1개의 qubit에 대해서만 적용한다.
    nf_rg = QuantumRegister(1, "f")
    qc = QuantumCircuit(nf_rg)
    #rotation Y gate
    qc.ry(angle,0)
    #이름 설정
    if eigen == 0:
        qc.name ="RY(0)"
    else:
        qc.name = f"RY(2arcsin(C/{eigen}))"
    return qc.to_gate()

#Multi controlled Rotation Y gate
#가능한 eigenvalue들에 따라서 상태를 회전시켜주는 함수
def MCRY(eigen, nl, scaling, neg_vals=True):
    """
    eigen: given eigen value
    nl: number of state qubit
    scaling: scaling factor
    neg_vals: if there is a negative eigenvalue, True
    """
    #초기 회로 설정, 음의 eigenvalue가 있는 경우 qubit 하나를 추가한다.
    nl_rg = QuantumRegister(nl+neg_vals, "state")
    nf_rg = QuantumRegister(1, "f")
    qc = QuantumCircuit(nl_rg, nf_rg)

    #전체 회로에 rotation이 들어갈 경우 gate를 뒤집어서 연결하기 때문에 마지막 qubit이 0번 qubit이다.
    #음의 eigenvalue를 가지는 경우 
    #(주어진 eigenvalue가 음수가 아니더라도 neg_vals가 True일 경우 0번 qubit을 음수와 양수를 구별하기 위해서 사용하기 때문에 이를 구별해야 한다.)
    if neg_vals:
        #주어진 eigenvalue가 음수일 때 
        if eigen < 0:
            #음의 eigenvalue를 해당하는 이진 bitstring에 맞추어 다시 연산
            bitstring = format(eigen + 2**nl, "b").zfill(nl)
        else:
            #해당하는 수를 이진 bitstring으로 변환
            bitstring = format(eigen, "b").zfill(nl)
            #0번 qubit이 0인 경우(양수인 경우)에만 gate가 적용되도록 설계
            qc.x(nl_rg[-1])
    #해당하는 수를 이진 bitstring으로 변환
    else:
        bitstring = format(eigen, "b").zfill(nl)
    
    #eigenvalue가 0일 경우에는 회전을 진행하지 않는다.
    if eigen == 0:
        Gate = RY(2*np.arcsin(scaling/(2**nl)), eigen)
    else:
        #1상태의 amplitude가 C/eigenvalue의 형태가 되도록 각도를 계산하여 회전시킨다.
        Gate = RY(2*np.arcsin(scaling/eigen), eigen)
    #회전 gate를 다중 control gate로 변화시킨다.
    CRY = Gate.control(nl+neg_vals)
    
    #이진 bitstring에 맞추어 해당되는 eigenvalue에서만 회전이 적용되도록 한다.
    #전체 회로가 뒤집어져서 적용되기 때문에 문자열을 뒤집어서 반환한다.
    for i, string in enumerate(reversed(bitstring)):
        #해당 bit가 0인 경우에 회전이 적용되게 하기 위해서 x gate를 추가한다.
        if int(string) == 0:
            qc.x(nl_rg[i])
        else:
            pass
    #controlled rotation gate를 적용한다.
    qc.append(CRY,range(nl+neg_vals+1))
    
    #앞선 과정을 다시 적용해 flag qubit을 제외한 qubit들이 원래 상태로 돌아오도록 한다. 
    for i, string in enumerate(reversed(bitstring)):
        if int(string) == 0:
            qc.x(nl_rg[i])
        else:
            pass
    
    if neg_vals:
        if eigen >= 0:
            qc.x(nl_rg[-1])
            
    #회로의 이름 설정    
    qc.name = "MCRY"
    return qc

#앞서 정의한 gate들을 모든 가능한 eigenvalue들에 대해서 적용한다.
def my_rotation(nl, nf, scaling, neg_vals: Boolean):
    #기본 회로 정의
    nl_rg = QuantumRegister(nl, "state")
    nf_rg = QuantumRegister(nf, "f")
    qc = QuantumCircuit(nl_rg, nf_rg)
    #음의 eigenvalue를 포함하는 경우
    if neg_vals:
        #부호를 의미하는 첫 번째 qubit을 제외하고 가능한 경우들에 대해 loop 
        for eigen in range(2**(nl-1)):
            #가능한 양수 부분에 대한 회전을 적용
            qc = qc.compose(MCRY(eigen, nl-1, scaling, neg_vals=neg_vals))
            #가능한 음수 부분에 대한 회전을 적용
            qc = qc.compose(MCRY(eigen-2**(nl-1), nl-1, scaling, neg_vals=neg_vals))
            #gate간의 구별을 위한 barrier
            qc.barrier()
    #모든 eigenvalue가 양수인 경우
    else:
        #eigenvalue가 모두 양수인 경우 모든 가능한 경우에 대해서 loop
        for eigen in range(2**nl):
            #가능한 gate들을 모두 적용
            qc = qc.compose(MCRY(eigen, nl, scaling, neg_vals=neg_vals))
            #gate간의 구별을 위한 barrier
            qc.barrier()
    #회로의 이름 정의
    qc.name = "C/x"
    return qc

if __name__ == "__main__":
    qc = my_rotation(3, 1, 1, True)
    qc.draw("mpl").savefig("image/rotation.png")
