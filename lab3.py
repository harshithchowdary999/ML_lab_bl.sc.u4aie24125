import math

import numpy as np
def evealuatevectors(A,B):
    while(len(A) != len(B)):
        print("length of the both vectars are not equal")
        val = input("Enter elements separated by space: ").split()
        A = np.array(val, dtype=int)
        val1 = input("Enter elements separated by space: ").split()
        B = np.array(val1, dtype=int)

    dotans=0
    for i in range(len(A)):
        dotans+=A[i]*B[i]
    normofA=0
    normofB=0
    for i in range(len(A)):
        normofA+=A[i]**2
        normofB+=B[i]**2
    normofA=math.sqrt(normofA)
    normofB=math.sqrt(normofB)
    inbuiltdot = np.dot(A,B)
    inbuiltnormA = np.linalg.norm(A)
    inbuiltnormB = np.linalg.norm(B)
    return dotans,normofA,normofB,inbuiltdot,inbuiltnormA,inbuiltnormB
def interclassspread():
    def mean(data):
        return sum(data)/len(data)
    def varianc(data):
        mu =  mean(data)
        return (sum(mu-x)**2 for x in data)/len(data)
    def std(data):
        return math.sqrt(varianc(data))
    
def main():
    #A1)
    val = input("Enter elements separated by space: ").split()
    A = np.array(val, dtype=int)
    val1 = input("Enter elements separated by space: ").split()
    B = np.array(val1, dtype=int)
    dotans, normofA, normofB, inbuiltdot, inbuiltnormA, inbuiltnormB = evealuatevectors(A, B)





