
#author: Yidong Liao         yidong.liao@uq.net.au

import cmath
import math

from projectq.ops import Ph,ControlledGate,NOT,CNOT,H, Z, All, R, Swap, Measure, X, get_inverse, QFT, Tensor, BasicGate
from projectq.meta import Loop, Compute, Uncompute, Control

from projectq.cengines import (MainEngine,
                               AutoReplacer,
                               LocalOptimizer,
                               TagRemover,
                               DecompositionRuleSet)
from hiq.projectq.cengines import GreedyScheduler, HiQMainEngine
from hiq.projectq.backends import SimulatorMPI
import projectq.setups.decompositions
from mpi4py import MPI

theta = math.pi/8    #the incremental in phase accumulation

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
    #X|des_output
    qnn(eng)
    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
    #X|des_output

    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
    X|des_output
    qnn(eng)
    X|layer1_input_reg[0]
    X|layer1_input_reg[1]
    X|layer1_input_reg[2]
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

    backend = SimulatorMPI(gate_fusion=True, num_local_qubits=20)

    cache_depth = 10
    rule_set = DecompositionRuleSet(modules=[projectq.setups.decompositions])
    engines = [TagRemover()
               , LocalOptimizer(cache_depth)
               , AutoReplacer(rule_set)
               , TagRemover()
               , LocalOptimizer(cache_depth)
               , GreedyScheduler()
               ]

    eng = HiQMainEngine(backend, engines)

    if MPI.COMM_WORLD.Get_rank() == 0:

    
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
   
      H | ancilla_qubit
      X | ancilla_qubit

      All(Measure) | layer1_weight_reg

      eng.flush()
   
      print("===========================================================================")
      print("This is the QBNN 3-1, task 2 demo")
      print("The measured weight string is:")
      print(int(layer1_weight_reg[0]),int(layer1_weight_reg[1]),int(layer1_weight_reg[2]))
    
    








