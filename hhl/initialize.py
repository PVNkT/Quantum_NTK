import numpy as np
from qiskit import QuantumCircuit

#b 벡터를 양자 상태로 바꿔주는 함수
def make_b_state(b):
    #벡터를 통해서 필요한 qubit의 수를 계산
    nb = int(np.log2(len(b)))
    #기초 회로를 설정
    vector_circuit = QuantumCircuit(nb)
    #입력한 벡터와 동일한 상태를 만드는 회로를 만듬
    vector_circuit.isometry(b / np.linalg.norm(b), list(range(nb)), None)
    #회로와 qubit의 수를 반환
    return vector_circuit, nb