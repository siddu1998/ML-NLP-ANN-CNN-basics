#Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Market.csv',header=None)
transaction=[]
for i in range(0,7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0,20)])


#training the apriori on the dataset
from apyori import apriori
#pass the 3 params mentioned in theory
rules=apriori(transaction,min_support=0.003,min_confidence=0.2,min_lift=3,min_lenght=2)

#Visualize stuff we will get better picture of what to do
result=list(rules)
print(*result,sep='\n')
