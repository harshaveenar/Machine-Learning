import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/HP/Documents/python(ML)/enjoysport.csv")
print(data, "n")
d = np.array(data)[:,:-1]
print("The attributes are: ", d)
target = np.array(data)[:,-1]
print("The target is: ", target)

def train(c, t):
    S = [c[i] for i, val in enumerate(t)
         if val == "yes"][0]
    G = [['?' for _ in range(len(c[0]))]
         for _ in range(len(c[0]))]

    for i, val in enumerate(t):
        if val == "yes":
            for x in range(len(S)):
                if S[x] != c[i][x]:
                    S[x] = '?'  
                    G[x][x] = '?'  
        else:
            for x in range(len(S)):
                if S[x] == c[i][x]:
                    G[x][x] = S[x]  
    return S, G
S, G = train(d, target)
print("The most specific hypothesis is:", S)
print("The most general hypothesis is:", G)
