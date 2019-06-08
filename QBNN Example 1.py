
import sys
import numpy as np
import cmath
import math
import copy
from projectq import MainEngine
from projectq.ops import Ph,ControlledGate,NOT,CNOT,H, Z, All, R, Swap, Measure, X, get_inverse, QFT, Tensor, BasicGate

from projectq.meta import Loop, Compute, Uncompute, Control
from projectq.backends import CircuitDrawer,Simulator

theta = math.pi/4
  
def qbn(eng): 

    CNOT | (layer1_weight_reg[0],layer1_input_reg[0])
    CNOT | (layer1_weight_reg[1],layer1_input_reg[1])  
    

    ControlledGate(NOT,2) | (layer1_input_reg[0],layer1_input_reg[1],output)  


def oracle(eng):
    
    ControlledGate(Ph(theta), 2) | (output,des_output,ancilla_qubit2)
    X|output
    X|des_output
    ControlledGate(Ph(theta), 2) | (output,des_output,ancilla_qubit2)
    X|output
    X|des_output    
    
def qnn(eng):

    with Compute(eng):
        qbn(eng)

    oracle(eng)
     
    Uncompute(eng)  
    
def run_qnn(eng):

    qnn(eng)

    X|layer1_input_reg[0]
    qnn(eng)
    X|layer1_input_reg[0]


    X|layer1_input_reg[1]
    #X|des_output
    qnn(eng)
    X|layer1_input_reg[1]
    #X|des_output
    
    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|des_output
    qnn(eng)
    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|des_output

   
    
def quanutm_phase_estimation(eng):
    
    All(H) | phase_reg

    with Control(eng, phase_reg[0]):
            run_qnn(eng)                

    with Control(eng, phase_reg[1]):
        with Loop(eng,2):
            run_qnn(eng)
            
    with Control(eng, phase_reg[2]):
        with Loop(eng,4):
            run_qnn(eng)
     
    Swap | (phase_reg[0], phase_reg[2])
            
    get_inverse(QFT) | phase_reg
       
def add_minus_sign(eng):

    with Compute(eng):
          quanutm_phase_estimation(eng)
          
    
    X|phase_reg[0]
    X|phase_reg[1]
    ControlledGate(NOT, 3)|(phase_reg[0],phase_reg[1],phase_reg[2],ancilla_qubit)
    X|phase_reg[0]
    X|phase_reg[1]

    Uncompute(eng)

    
def diffusion(eng):

    with Compute(eng):
            All(H) | layer1_weight_reg
            
            All(X) | layer1_weight_reg
           
    
    ControlledGate(Z, 1) | (layer1_weight_reg[0],layer1_weight_reg[1])

    Uncompute(eng)
    
def run_qbnn(eng):    
    
    add_minus_sign(eng)
    
    diffusion(eng)
    

if __name__ == "__main__":
    
    eng = MainEngine(backend = Simulator(rnd_seed = 1))
    
    layer1_weight_reg = eng.allocate_qureg(2)
    layer1_input_reg = eng.allocate_qureg(2)

    output= eng.allocate_qubit()
    des_output = eng.allocate_qubit()
    ancilla_qubit2 = eng.allocate_qubit()
    ancilla_qubit = eng.allocate_qubit()
    phase_reg = eng.allocate_qureg(3)
    
    X | ancilla_qubit
    H | ancilla_qubit

    All(H) | layer1_weight_reg
    
    with Loop(eng, 3):
         run_qbnn(eng)
    #add_minus_sign(eng)
    #quanutm_phase_estimation(eng)
    #qnn(eng)
    #run_qnn(eng)
    H | ancilla_qubit
    X | ancilla_qubit
    
    eng.flush()
    
    #mapping, wavefunction = copy.deepcopy(eng.backend.cheat())

    #print("The full wavefunction is: {}".format(wavefunction))
 
    
    #a=eng.backend.get_amplitude('1100000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #b=eng.backend.get_amplitude('1000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #c=eng.backend.get_amplitude('0100000001',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    
    #d1=eng.backend.get_amplitude('00000000110',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #d2=eng.backend.get_amplitude('10000000110',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #d3=eng.backend.get_amplitude('01000000100',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #d4=eng.backend.get_amplitude('11000000100',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #dd=eng.backend.get_amplitude('00000000001',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #ddd=eng.backend.get_amplitude('000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #dddd=eng.backend.get_amplitude('0000000011',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    d1=eng.backend.get_amplitude('00000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    d2=eng.backend.get_amplitude('10000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    d3=eng.backend.get_amplitude('01000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    d4=eng.backend.get_amplitude('11000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    
    
    #p1=eng.backend.get_probability('00000000110',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #p2=eng.backend.get_probability('10000000110',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    #p2=eng.backend.get_probability('00000000001',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    
    
    
    e=eng.backend.get_probability('0',ancilla_qubit)
    r=eng.backend.get_probability('000',layer1_input_reg)
    g1=eng.backend.get_probability('001',phase_reg)
    g2=eng.backend.get_probability('010',phase_reg)
    g3=eng.backend.get_probability('011',phase_reg)
    g4=eng.backend.get_probability('100',phase_reg)
    g5=eng.backend.get_probability('000',phase_reg)
    
    w1=eng.backend.get_probability('00',layer1_weight_reg)
    w2=eng.backend.get_probability('01',layer1_weight_reg)
    w3=eng.backend.get_probability('10',layer1_weight_reg)
    w4=eng.backend.get_probability('11',layer1_weight_reg)
    
    
    
    #print("Measured: {}".format(a))
    #print("Measured: {}".format(b))
    #print("Measured: {}".format(c))
    print("Measured: {}".format(d1))
    print("Measured: {}".format(d2))
    print("Measured: {}".format(d3))
    print("Measured: {}".format(d4))
    #print("Measured: {}".format(p1))
    #print("Measured: {}".format(p2))
    
    
    print("Measured: {}".format(e))
    print("Measured: {}".format(r))
    print("Measured: {}".format(g1))
    print("Measured: {}".format(g2))
    print("Measured: {}".format(g3))
    print("Measured: {}".format(g4))
    print("Measured: {}".format(g5))
    
    print("Measured: {}".format(w1))
    print("Measured: {}".format(w2))
    print("Measured: {}".format(w3))
    print("Measured: {}".format(w4))
    
    
    #print(int(layer1_weight_reg[0]),int(layer1_weight_reg[1]),int(layer1_weight_reg[2]))
    #print(drawing_engine.get_latex())




