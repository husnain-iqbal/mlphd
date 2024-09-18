#!/usr/bin/env python
# coding: utf-8

# In[11]:


#1.1
nums = [3, 5, 7, 8, 12]
print(nums)

cubes = []
for num in nums:
    cube = num ** 3
    cubes.append(cube)
    
print(cubes)

#1.2
dict = {}
dict['parrot'] = 2
dict['goat'] = 4
dict['spider'] = 8
dict['crab'] = 10

print(dict)

#1.3
legCount = 0
for animal, legs in dict.items():
    print(f'{animal}: {legs} legs')
    legCount += legs

print(f'Total legs count: {legCount}')

#1.4
A = (3, 9, 4, [5, 6])
print(A)
A[3][0] = 8
print(A)

#1.5
del A

#1.6
B = ('a', 'p', 'p', 'l', 'e')
freqP = B.count('p')
print(f"The character 'p' occured {freqP} times in the tuple.")

#1.7
indexOf_l = B.index('l')

print(f"The index of 'l': {indexOf_l}")


# In[37]:


import numpy as np
#2.1
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
z = np.array([1, 0, 1])
#2.2
b = A[:2, 1:3] 
#2.3
C = np.empty_like(A)
#2.4
for i in range(A.shape[1]):
    C[:, i] = A[:, i] + z

X = np.array([[1,2],[3,4]])
Y = np.array([[5,6],[7,8]])
v = np.array([9,10])
#2.5
print(f"X+Y={X+Y}")
#2.6
print(f"X*Y={X*Y}")
#2.7
sqrtY = np.sqrt(Y)
print(f"Element-wise sqaure-root of Matrix Y: {sqrtY}")
#2.8
dotProd_Xv = np.dot(X, v)
print(f"X.v = {dotProd_Xv}")
#2.9
sum_cols_X = np.sum(X, axis=0)
print(f"Sum of column elements of X: {sum_cols_X}")


# In[4]:


#3.1
def calcVelocity(distance, time):
    if time == 0:
        raise ValueError("Time cannot be zero.")
    velocity = distance / time
    return velocity

distance = 1000  # in meters
time = 60    # in seconds
velocity = calcVelocity(distance, time)
print(f"Velocity = {velocity} meters per seconds")

#3.2
even_num = [2, 4, 6, 8, 10, 12] # ignoring 0 as it will result 0 after multiplication

def mult(numbers):
    result = 1  
    for num in numbers:
        result *= num 
    return result

# Example usage:
result = mult(even_num)
print(f"Product of even numbers up to 12: {result}")


# In[14]:


import pandas as pd

table = {
    'C1': [1, 2, 3, 5, 5],
    'C2': [6, 7, 5, 4, 8],
    'C3': [7, 9, 8, 6, 5],
    'C4': [7, 5, 2, 8, 8]
}
pdf = pd.DataFrame(table)
# 4.1
print(pdf.head(2))
# 4.2 
print(pdf['C2'])
# 4.3 
pdf.rename(columns={'C3': 'B3'}, inplace=True)
# 4.4
pdf['Sum'] = pdf.sum(axis=1)
# 4.5
pdf['Sum'] = pdf[['C1', 'C2', 'B3', 'C4']].sum(axis=1)
print(pdf)

# 4.6
df = pd.read_csv('hello_sample.csv')
# 4.7
print(f"Complete DataFrame: {df}")
# 4.8
print(f"Bottom 2 records of the DataFrame: {df.tail(2)}")
# 4.9
print(f"DataFrame Info: {df.info()}")
# 4.10
print(f"Share (Rows X Columns) of DataFrame: {df.shape}")
# 4.11
df_sorted = df.sort_values(by='Weight')
print(f"Sorted data of DataFrame using column Weight {df_sorted}")
# 4.12
print(df.isnull().sum())
df_dropped = df.dropna()
print(f"DataFrame after using isNull() & dropNa() methods: {df_dropped}")


# In[ ]:




