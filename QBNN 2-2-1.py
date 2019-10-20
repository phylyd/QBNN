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

theta = math.pi/4
  
def qbn(eng): 

    CNOT | (layer1_weight_reg[0],layer1_input_reg[0])
    CNOT | (layer1_weight_reg[1],layer1_input_reg[1])  
    
    ControlledGate(NOT,2) | (layer1_input_reg[0],layer1_input_reg[1],output_reg[0]) 
       
    CNOT | (layer1_weight_reg[2],layer1_input_reg[2])
    CNOT | (layer1_weight_reg[3],layer1_input_reg[3])  
    
    ControlledGate(NOT,2) | (layer1_input_reg[2],layer1_input_reg[3],output_reg[1]) 
    
    CNOT | (layer2_weight_reg[0],output_reg[0])
    CNOT | (layer2_weight_reg[1],output_reg[1]) 

    ControlledGate(NOT,2) | (output_reg[0],output_reg[1],output_reg[2]) 
    
def oracle(eng):
    
    ControlledGate(Ph(theta), 2) | (output_reg[2],des_output,ancilla2)
    X|output_reg[2]
    X|des_output
    ControlledGate(Ph(theta), 2) | (output_reg[2],des_output,ancilla2)
    X|output_reg[2]
    X|des_output    
      
    
def qnn(eng):

    with Compute(eng):
        qbn(eng)

    oracle(eng)
     
    Uncompute(eng) 
    
    
def run_qnn(eng):

    qnn(eng)

    X|layer1_input_reg[0]
    X|layer1_input_reg[2]
    qnn(eng)
    X|layer1_input_reg[0]
    X|layer1_input_reg[2]

    X|layer1_input_reg[1]
    X|layer1_input_reg[3]
   # X|des_output
    qnn(eng)
    X|layer1_input_reg[1]
    X|layer1_input_reg[3]
    #X|des_output

    X|layer1_input_reg[0]
    X|layer1_input_reg[2]
    X|layer1_input_reg[1]
    X|layer1_input_reg[3]
    X|des_output
    qnn(eng)
    X|layer1_input_reg[0]
    X|layer1_input_reg[2]
    X|layer1_input_reg[1]
    X|layer1_input_reg[3]
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
            All(H) | layer2_weight_reg
            All(X) | layer1_weight_reg
            All(X) | layer2_weight_reg

    ControlledGate(Z, 5) | (layer1_weight_reg[0],layer1_weight_reg[1],layer1_weight_reg[2],layer1_weight_reg[3],layer2_weight_reg[0],layer2_weight_reg[1])

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

       layer1_weight_reg = eng.allocate_qureg(4)
       layer1_input_reg = eng.allocate_qureg(4)

       layer2_weight_reg = eng.allocate_qureg(2)


       output_reg = eng.allocate_qureg(3)
       des_output = eng.allocate_qubit()
       ancilla_qubit = eng.allocate_qubit()
       ancilla2 = eng.allocate_qubit()
       phase_reg = eng.allocate_qureg(3)
    
       X | ancilla_qubit
       H | ancilla_qubit

       All(H) | layer1_weight_reg
       All(H) | layer2_weight_reg

       with Loop(eng, 2):
           run_qbnn(eng)
    
       H | ancilla_qubit
       X | ancilla_qubit

       eng.flush()
     
       w1=eng.backend.get_probability('000000',layer1_weight_reg+layer2_weight_reg)
       w2=eng.backend.get_probability('000001',layer1_weight_reg+layer2_weight_reg)
       w3=eng.backend.get_probability('000010',layer1_weight_reg+layer2_weight_reg)
       w4=eng.backend.get_probability('000011',layer1_weight_reg+layer2_weight_reg)
       w5=eng.backend.get_probability('000101',layer1_weight_reg+layer2_weight_reg)
      
       print("==========================================================================")
       print("This is the QBN 2-2-1 demo")
       print("With the highest N_t, the probabilities of obtaining the weight strings after 2 iterations are:")
    
       print("Measured probabilty of weight string 0000 00 : {}".format(w1))
       print("Measured probabilty of weight string 0000 01 : {}".format(w2))
       print("Measured probabilty of weight string 0000 10 : {}".format(w3))
       print("Measured probabilty of weight string 0000 11 : {}".format(w4))
       print("Measured probabilty of weight string 0001 01 : {}".format(w5))
      
