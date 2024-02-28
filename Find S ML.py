
import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/HP/Documents/python(ML)/breastcancer.csv")
print(data,"n")
d = np.array(data)[:,:-1]
print(" attribte is",d)
target=np.array(data)[:,:-1]
print("targer att is ",target)

def train(c,t):
    sphy=None
    for i, val in enumerate(t):
        if val == "yes":
            sphy=c[i].copy()
            break
    if sphy is None:
        return sphy

    for i, val in enumerate(c):
        if t[i]== "yes":
            for x in range(len(sphy)):
                sphy[x]='?'
        else:
             pass

    return sphy
