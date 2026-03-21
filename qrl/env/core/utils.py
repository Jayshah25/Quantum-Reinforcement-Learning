import numpy as np

# Define gates as numpy matrices
GATES = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "H": (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),
    "S": np.array([[1, 0], [0, 1j]], dtype=complex),
    "SDG": np.array([[1, 0], [0, -1j]], dtype=complex),
    "T": np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex),
    "TDG": np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]], dtype=complex),
}

def RX(theta): return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                                [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
def RY(theta): return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                                [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
def RZ(theta): return np.array([[np.exp(-1j*theta/2), 0],
                                [0, np.exp(1j*theta/2)]], dtype=complex)

STEP_PENALTY  = 0.01
SUCCESS_BONUS = 1.0
 
# State index → human label
STATE_LABELS = {
    0: r"$|0\rangle$",
    1: r"$|1\rangle$",
    2: r"$|+\rangle$",
    3: r"$|-\rangle$",
    4: r"$|{+i}\rangle$",
    5: r"$|{-i}\rangle$",
}
 
# State index → Bloch vector (x, y, z)
STATE_BLOCH = np.array([
    [ 0,  0,  1],   # |0⟩
    [ 0,  0, -1],   # |1⟩
    [ 1,  0,  0],   # |+⟩
    [-1,  0,  0],   # |-⟩
    [ 0,  1,  0],   # |+i⟩
    [ 0, -1,  0],   # |-i⟩
], dtype=np.float32)
 
# State index → statevector
STATE_VECTORS = {
    0: np.array([1, 0],                          dtype=complex),
    1: np.array([0, 1],                          dtype=complex),
    2: np.array([1, 1],                          dtype=complex) / np.sqrt(2),
    3: np.array([1, -1],                         dtype=complex) / np.sqrt(2),
    4: np.array([1, 1j],                         dtype=complex) / np.sqrt(2),
    5: np.array([1, -1j],                        dtype=complex) / np.sqrt(2),
}
 
# Gate unitaries
_s2 = 1 / np.sqrt(2)
# GATE_UNITARIES = {
#     "H": np.array([[_s2,  _s2], [ _s2, -_s2]], dtype=complex),
#     "X": np.array([[0,    1  ], [ 1,    0  ]], dtype=complex),
#     "Z": np.array([[1,    0  ], [ 0,   -1  ]], dtype=complex),
#     "S": np.array([[1,    0  ], [ 0,    1j ]], dtype=complex),
# }
ACTION_NAMES = ["H", "X", "Z", "S"]
 
# Deterministic transition table T[s, a] → s'
# Precomputed from verified quantum simulation (see module docstring)
_TRANSITIONS = np.array([
    #  H  X  Z  S
    [  2, 1, 0, 0],   # |0⟩
    [  3, 0, 1, 1],   # |1⟩
    [  0, 2, 3, 4],   # |+⟩
    [  1, 3, 2, 5],   # |-⟩
    [  5, 5, 5, 3],   # |+i⟩
    [  4, 4, 4, 2],   # |-i⟩
], dtype=np.int32)
 
# Graph layout: hexagonal arrangement mirroring the Bloch sphere
# |0⟩ at top, |1⟩ at bottom, equatorial band in middle
_GRAPH_POS = {
    0: ( 0.0,  1.6),   # |0⟩  top
    2: ( 1.4,  0.4),   # |+⟩  right-upper
    4: ( 1.4, -0.8),   # |+i⟩ right-lower
    1: ( 0.0, -1.6),   # |1⟩  bottom
    5: (-1.4, -0.8),   # |-i⟩ left-lower
    3: (-1.4,  0.4),   # |-⟩  left-upper
}

