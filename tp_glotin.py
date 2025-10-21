from copy import deepcopy
import soundfile as sf
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
# drive glotin init recherche
data1, Fe1 = sf.read("glotin init recherche/ONECAT_20200114_150201_247.wav")
data2, Fe2 = sf.read("glotin init recherche/S70_20200114_150211_982.wav")


print(data1[0][4]) #temps = 0, piste = 4

print()
i1 = 0
while data1[i1][4] < 0.5 :
    i1+=1

print(i1)

j1 = i1

while data1[j1][4] > -0.5 :
    j1+=1


print(j1)
print(j1-i1)
i2 = 0
while data2[i2][4] < 0.5 :
    i2+=1

print(i2)

j2 = i2

while data2[j2][4] > -0.5 :
    j2+=1

print(j2)
print(j2-i2)

max1 = 0
imax1 = 0
min1 = 0
imin1 = 0
for k in range(i1-2000,j1+2000) :
    if data1[k][4] > max1 :
        max1 = data1[k][4]
        imax1 = k
    elif data1[k][4] < min1 :
        min1 = data1[k][4]
        imin1 = k
print(min1,max1)
print(imin1,imax1)

M1 = []
tmp = 0
pre = data1[0][4]
newdata1 = []
i1p = 0
for k in range(0,len(data1)) :
    if data1[k][4] > 0.5 and pre < 0.5 :
        M1 +=[tmp]
        tmp = 0
        newdata1.append(deepcopy(data1[i1p:k]))
        i1p = k
    else :
        tmp += 1
    pre = data1[k][4]


# M1 = list(filter(lambda x: x != 0, M1))
# print(M1)
# print(len(M1))
# print(newdata1)
# print(len(newdata1))
print()
print()
print()
moyenne_len = 0
temps = 0
plt.title("Nombre d'échantillons pas seconde")
for m in newdata1 :
    print(len(m))
    plt.plot(temps,len(m), marker='.')
    temps+=1
    moyenne_len += len(m)
plt.show()
moyenne_len = moyenne_len / len(newdata1)

print("Fréquence Fe : ",Fe1, Fe2)
print("moyenne fréquence réelle : ",moyenne_len)
exit(0)
ntmp = Fe1
print()
for k in range(0,len(data1), ntmp//10) :
    if k + ntmp > len(data1) :
        break
    val = sum(data1[k:k+ntmp,0] * data1[i1:i1+ntmp,4])/(np.linalg.norm(data1[i1:i1+ntmp,4]) * np.linalg.norm(data1[k:k+ntmp,0]))
    print(val)
    plt.plot(k,val, marker='.', linestyle='-', color='blue')
plt.show()
print(j1-i1)
# hdata1 = []
# pre = data1[i1][4]
# it = i1
# for k in range(i1,len(data1)) :
#     if (k >= it + Fe1) :
#         hdata1.append(deepcopy(data1[it:k]))
#         it = k
# #hdata1.append(deepcopy(data1[it::]))
# print()
# print()
# print()
# for m in hdata1 :
#     print(len(m))
# print(len(hdata1))
# # data1f = sp.resample(data1,Fe1)
# # print(data1f)
# print()
# print()
print()
