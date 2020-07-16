import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

p_train = pd.read_csv('train.txt', header=None, sep=' ', dtype='float64')
train_arr = p_train.values
len_train = train_arr[:, 0].size

class_1 = []
class_2 = []


for i in range(len_train):
    if train_arr[i, 2] == 1:
        class_1.extend([train_arr[i, 0:2]])
    else:
        class_2.extend([train_arr[i, 0:2]])

class_1 = np.array(class_1)
class_2 = np.array(class_2)

x1 = class_1[:, 0]
y1 = class_1[:, 1]
x2 = class_2[:, 0]
y2 = class_2[:, 1]

plt.scatter(x1, y1, color='red', marker='+',label='class_1')
plt.scatter(x2, y2, color='green', marker='*',label='class_2')
plt.legend()
plt.show()


len_1 = class_1[:, 0].size
len_2 = class_2[:, 0].size

phi_1 = []
phi_2 = []



for i in range(len_1):
    x = class_1[i, 0]
    y = class_1[i, 1]
    phi1  = np.array([x ** 2, y ** 2, x * y, x, y, 1])
    phi_1.extend([phi1])

phi_1 = np.array(phi_1)


for i in range(len_2):
    x = class_2[i, 0]
    y = class_2[i, 1]
    phi1  = np.array([x ** 2, y ** 2, x * y, x, y, 1])
    phi_2.extend([phi1])
    

phi_2 = np.array(phi_2)


phi_2 = np.dot(phi_2, -1)

phi = np.concatenate((phi_1, phi_2))
length = phi[:, 0].size

learning_rate = []
iteration = []
iteration_single = []
weight_batch = []

def batch_process():
     a = 0.1
     while True:
         initial_weight = [1, 1, 1, 1, 1, 1]

         for i in range(700):
             count = 0
             sum_weight = [0, 0, 0, 0, 0, 0]
             for j in range(length):
                 if np.dot(phi[j, :], initial_weight) > 0:
                     count = count+1
                 elif np.dot(phi[j, :], initial_weight) <= 0:
                     sum_weight += a*phi[j, :]
             initial_weight = sum_weight + initial_weight
             if count == length:
                  break
           

    
         iteration.append(i + 1)
         a = a + 0.1
         if(a>1):
            break
     return iteration
iteration = batch_process()
print(iteration)


def single_process():
    weight_single = []
    a = 0.1
    
    while True:
        initial_weight = [1, 1, 1, 1, 1, 1]
        count = 0
        for i in range(700):
             sum_weight = [0, 0, 0, 0, 0, 0]
             for j in range(length):
                 if np.dot(phi[j, :], initial_weight) > 0:
                     count = count+1
                     if count == length:
                         break
                 elif np.dot(phi[j, :], initial_weight) <= 0:
                    sum_weight = initial_weight + a*phi[j, :]
                    count = 0
                    initial_weight = sum_weight
             if count == length:
                   break
        iteration_single.append(i + 1)
        a = a + 0.1
        if(a>1):
            break
    return iteration_single    
iteration_single = single_process()
print(iteration_single)

labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']

x = np.arange(len(labels))  
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, iteration, width, label='Batch processing')
rects2 = ax.bar(x + width/2, iteration_single, width, label='Single processing')
ax.set_ylabel('Iteration')
ax.set_title('Comparison between Single processing vs Batch processing')
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()






