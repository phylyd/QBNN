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

theta = math.pi/4     #the incremental in phase accumulation
  
def qbn(eng): 
  
  """The operations in Quantum Binary Neurons (QBNs): multiplications between weights and inputs are performed by CNOTs, 
 addition and activation are done by a series multi-controlled gates"""

    CNOT | (layer1_weight_reg[0],layer1_input_reg[0])
    CNOT | (layer1_weight_reg[1],layer1_input_reg[1])  
    

    ControlledGate(NOT,2) | (layer1_input_reg[0],layer1_input_reg[1],output)  


def oracle(eng):
    
  """The Oracle that compares output of the network and the desired ouput with respect to corresponding inputs, 
  adds a incremental phase if they are identical."""
  
    ControlledGate(Ph(theta), 2) | (output,des_output,ancilla_qubit2)
    X|output
    X|des_output
    ControlledGate(Ph(theta), 2) | (output,des_output,ancilla_qubit2)
    X|output
    X|des_output    
    
def qnn(eng):
  
  """ The marking process--for one input: Computation with QBNN, Oracle adding phase followed by Uncomputation of QBNN"""

    with Compute(eng):
        qbn(eng)

    oracle(eng)
     
    Uncompute(eng)  
    
def run_qnn(eng):
  
  """ The marking process--accumulation for many inputs in the training set: 
  We chagne the inputs for each QBNN run by applying X gates on the input register, adopting Task 2 for this instance."""

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
  
  """ Phase estimation process to evaluate the accumulated phase, namely the goodness of each weight """
    
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
  
  """ Binarize the marking--According to the evaluated phase(goodness) of each weight from PE, add a minus sign on those ones that match
the critia of optimal weights--For this instance, weights with goodness of 100% accuracy (a weight is good for all the 
input-desired ouput pairs) are marked with a minus sign in front of them. """

    with Compute(eng):
          quanutm_phase_estimation(eng)
          
    
    X|phase_reg[0]
    X|phase_reg[1]
    ControlledGate(NOT, 3)|(phase_reg[0],phase_reg[1],phase_reg[2],ancilla_qubit)
    X|phase_reg[0]
    X|phase_reg[1]

    Uncompute(eng)

    
def diffusion(eng):
  
  """ Standard Grover's Diffusion to amplified the marked weights """

    with Compute(eng):
            All(H) | layer1_weight_reg
            
            All(X) | layer1_weight_reg
           
    
    ControlledGate(Z, 1) | (layer1_weight_reg[0],layer1_weight_reg[1])

    Uncompute(eng)
    
def run_qbnn(eng):   
  
  """ The whole training cycle: binarized marking + diffusion """
    
    add_minus_sign(eng)
    
    diffusion(eng)
    

if __name__ == "__main__":
    
    eng = MainEngine(backend = Simulator(rnd_seed = 1))
    
    #allocate all the qubits
    layer1_weight_reg = eng.allocate_qureg(2)
    layer1_input_reg = eng.allocate_qureg(2)

    output= eng.allocate_qubit()
    des_output = eng.allocate_qubit()
    ancilla_qubit2 = eng.allocate_qubit()
    ancilla_qubit = eng.allocate_qubit()
    phase_reg = eng.allocate_qureg(3)
    
    #initialize the ancilla and weight qubits
    X | ancilla_qubit
    H | ancilla_qubit

    All(H) | layer1_weight_reg
   
    #run the training cycles
    with Loop(eng, 3):
         run_qbnn(eng)
    #add_minus_sign(eng)
    #quanutm_phase_estimation(eng)
    #qnn(eng)
    #run_qnn(eng)
    H | ancilla_qubit
    X | ancilla_qubit
    
    eng.flush()

    
    w1=eng.backend.get_probability('00',layer1_weight_reg)
    w2=eng.backend.get_probability('01',layer1_weight_reg)
    w3=eng.backend.get_probability('10',layer1_weight_reg)
    w4=eng.backend.get_probability('11',layer1_weight_reg)
    
    
    print("===========================================================================")
    print("This is the QBNN demo")
    print("The probabilities of obtaining the weight strings are:")
    
    print("Measured probabilty of weight string 00: {}".format(w1))
    print("Measured probabilty of weight string 01: {}".format(w2))
    print("Measured probabilty of weight string 10: {}".format(w3))
    print("Measured probabilty of weight string 11: {}".format(w4))
    




