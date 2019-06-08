
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:57:20 2019

@author: Yidong Liao 
"""

import sys
import numpy as np
import cmath
import math
import copy
from projectq import MainEngine
from projectq.ops import Ph,ControlledGate,NOT,CNOT,H, Z, All, R, Swap, Measure, X, get_inverse, QFT, Tensor, BasicGate

from projectq.meta import Loop, Compute, Uncompute, Control
from projectq.backends import CircuitDrawer,Simulator

theta = math.pi/8
  
def qbn(eng): 

    CNOT | (layer1_weight_reg[0],layer1_input_reg[0])
    CNOT | (layer1_weight_reg[1],layer1_input_reg[1])  
    CNOT | (layer1_weight_reg[2],layer1_input_reg[2])
    

    ControlledGate(NOT,3) | (layer1_input_reg[0],layer1_input_reg[1],layer1_input_reg[2],output)  
    X|layer1_input_reg[0]
    ControlledGate(NOT,3) | (layer1_input_reg[0],layer1_input_reg[1],layer1_input_reg[2],output) 
    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    ControlledGate(NOT,3) | (layer1_input_reg[0],layer1_input_reg[1],layer1_input_reg[2],output) 
    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
    ControlledGate(NOT,3) | (layer1_input_reg[0],layer1_input_reg[1],layer1_input_reg[2],output)  
    X|layer1_input_reg[2]

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
    
    #X|des_output
    qnn(eng)
   # X|des_output
    
    X|layer1_input_reg[0]
    #X|des_output
    qnn(eng)
    X|layer1_input_reg[0]
    #X|des_output


    X|layer1_input_reg[1]
    #X|des_output
    qnn(eng)
    X|layer1_input_reg[1]
    #X|des_output

    X|layer1_input_reg[2]
    #X|des_output
    qnn(eng)
    X|layer1_input_reg[2]
    #X|des_output

    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|des_output
    qnn(eng)
    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|des_output

    X|layer1_input_reg[0]
    X|layer1_input_reg[2]
    #X|des_output
    qnn(eng)
    X|layer1_input_reg[0]
    X|layer1_input_reg[2]
    #X|des_output

    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
    X|des_output
    qnn(eng)
    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
    X|des_output

    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
    #X|des_output
    qnn(eng)
    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
    #X|des_output
    
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
    
    #X|phase_reg[1]
    #X|phase_reg[0]
    X|phase_reg[2]
    ControlledGate(NOT, 3)|(phase_reg[0],phase_reg[1],phase_reg[2],ancilla_qubit)
    #X|phase_reg[0]
    X|phase_reg[2]
    #X|phase_reg[1]
    
    Uncompute(eng)
    
def diffusion(eng):

    with Compute(eng):
            All(H) | layer1_weight_reg
            
            All(X) | layer1_weight_reg
           
    
    ControlledGate(Z, 2) | (layer1_weight_reg[0],layer1_weight_reg[1],layer1_weight_reg[2])

    Uncompute(eng)
    
def run_qbnn(eng):    
    
    add_minus_sign(eng)
    
    diffusion(eng)
    

if __name__ == "__main__":
    
    eng = MainEngine(backend = Simulator(rnd_seed = 1))
    
    layer1_weight_reg = eng.allocate_qureg(3)
    layer1_input_reg = eng.allocate_qureg(3)

    output= eng.allocate_qubit()
    des_output = eng.allocate_qubit()
    ancilla_qubit2 = eng.allocate_qubit()
    ancilla_qubit = eng.allocate_qubit()
    phase_reg = eng.allocate_qureg(3)
    
    X | ancilla_qubit
    H | ancilla_qubit

    All(H) | layer1_weight_reg
    
    with Loop(eng,2):
         run_qbnn(eng)
   # add_minus_sign(eng)
    
    #run_qbnn(eng)
    
    H | ancilla_qubit
    X | ancilla_qubit
    
    eng.flush()
    
    #mapping, wavefunction = copy.deepcopy(eng.backend.cheat())

    #print("The full wavefunction is: {}".format(wavefunction))
    
    a=eng.backend.get_amplitude('1110000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    b=eng.backend.get_amplitude('1100000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    c=eng.backend.get_amplitude('1010000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    d=eng.backend.get_amplitude('0110000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    h=eng.backend.get_amplitude('0010000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    i=eng.backend.get_amplitude('0100000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    j=eng.backend.get_amplitude('1000000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )
    k=eng.backend.get_amplitude('0000000000000',layer1_weight_reg+layer1_input_reg+output+des_output+ancilla_qubit2+ancilla_qubit+phase_reg )


    d1=eng.backend.get_probability('111',phase_reg)
    d2=eng.backend.get_probability('110',phase_reg)
    d3=eng.backend.get_probability('101',phase_reg)
    d4=eng.backend.get_probability('011',phase_reg)
    d5=eng.backend.get_probability('001',phase_reg)
    d6=eng.backend.get_probability('010',phase_reg)
    d7=eng.backend.get_probability('100',phase_reg)
    d8=eng.backend.get_probability('000',phase_reg)
    
    #d9=eng.backend.get_probability('1',phase_reg[0])
    
    
    e1=eng.backend.get_probability('0',ancilla_qubit2)
    e2=eng.backend.get_probability('0',ancilla_qubit)
    e3=eng.backend.get_probability('0',des_output)
    e4=eng.backend.get_probability('0',output)
    e5=eng.backend.get_probability('000',layer1_input_reg)
    
    
    p1=eng.backend.get_probability('111',layer1_weight_reg)
    p2=eng.backend.get_probability('110',layer1_weight_reg)
    p3=eng.backend.get_probability('101',layer1_weight_reg)
    p4=eng.backend.get_probability('011',layer1_weight_reg)
    p5=eng.backend.get_probability('001',layer1_weight_reg)
    p6=eng.backend.get_probability('010',layer1_weight_reg)
    p7=eng.backend.get_probability('100',layer1_weight_reg)
    p8=eng.backend.get_probability('000',layer1_weight_reg)
    
    print("Measured: {}".format(a))
    print("Measured: {}".format(b))
    print("Measured: {}".format(c))
    print("Measured: {}".format(d))
    print("Measured: {}".format(h))
    print("Measured: {}".format(i))
    print("Measured: {}".format(j))
    print("Measured: {}".format(k))
    
    print("Measured: {}".format(d1))
    print("Measured: {}".format(d2))
    print("Measured: {}".format(d3))
    print("Measured: {}".format(d4))
    print("Measured: {}".format(d5))
    print("Measured: {}".format(d6))
    print("Measured: {}".format(d7))
    print("Measured: {}".format(d8))
    
    print("Measured: {}".format(p1))
    print("Measured: {}".format(p2))
    print("Measured: {}".format(p3))
    print("Measured: {}".format(p4))
    print("Measured: {}".format(p5))
    print("Measured: {}".format(p6))
    print("Measured: {}".format(p7))
    print("Measured: {}".format(p8))
    
    print("Measured: {}".format(e1))
    print("Measured: {}".format(e2))
    print("Measured: {}".format(e3))
    print("Measured: {}".format(e4))
    print("Measured: {}".format(e5))
    
    #print("Measured: {}".format(d9))
    
    #print(int(layer1_weight_reg[0]),int(layer1_weight_reg[1]),int(layer1_weight_reg[2]))
    #print(drawing_engine.get_latex())



