import numpy as np
import pandas as pd

data= pd.read_csv("data.csv")
print(data)

concepts=np.array(data.iloc[:,:-1])
print(concepts)

target=np.array(data.iloc[:,-1])
print(target)

def learn(concepts, target):
    specific_h=concepts[0].copy()
    general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    for i, h in enumerate(concepts):
        if target[i]=="yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x]=specific_h[x]

        if target[i]=="no":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'

        print(general_h)
        print(specific_h)
        print()

    indices=[i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]
    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])
    print()

    print(general_h)
    return specific_h, general_h

s_final, f_final = learn(concepts,target)

print(s_final)
print(f_final)