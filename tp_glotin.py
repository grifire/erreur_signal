from copy import deepcopy
import soundfile as sf
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
# drive glotin init recherche
# data1, Fe1 = sf.read("glotin init recherche/ONECAT_20200114_150201_247.wav")
# data2, Fe2 = sf.read("glotin init recherche/S70_20200114_150211_982.wav")
data1, Fe1 = sf.read("/scratch/yroblin156/donnee_glotin/d1/filtre/ONECAT_20200114_150201_247.wav")
data2, Fe2 = sf.read("/scratch/yroblin156/donnee_glotin/d1/meute/S70_20200114_150211_982.wav")


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

newdata1 = newdata1[1:-1]

# M1 = list(filter(lambda x: x != 0, M1))
# print(M1)
# print(len(M1))
# print(newdata1)
# print(len(newdata1))
print()
print()
print()
# moyenne_len = 0
# temps = 0
# plt.title("Nombre d'échantillons pas seconde")
# for m in newdata1 :
#     print(len(m))
#     plt.plot(temps,len(m), marker='.')
#     temps+=1
#     moyenne_len += len(m)
# plt.show()
# moyenne_len = moyenne_len / len(newdata1)

# print("Fréquence Fe : ",Fe1, Fe2)
# print("moyenne fréquence réelle : ",moyenne_len)

len_max = 365596
# for m in newdata1 :
#     if len(m) > len_max :
#         len_max = len(m)

resamp_data1 = []
print("max = ",len_max)
for m in newdata1 :
    resamp_data1.append(sp.signal.resample(m, int(len_max)))

# print(resamp_data1[0])
print("len = ",len(resamp_data1))
data1_final = np.concatenate(resamp_data1, axis=0)

print("len final = ",len(data1_final))
print(data1_final[0])
#exit(0)

# Calcule d'énergie pour trouver les peaks de la piste 0
ntmp = Fe1//1000
window = ntmp*5
peaks = []
for i in range(0,len(data1_final),ntmp) :
    
    # print("Piste ",i,": ", peak)
    peaks += [sum(data1_final[i:i+window,0]**2)]

peaks = np.array(peaks)
print(len(peaks))
# Recherge des 1000 plus gros peaks
idx0 = np.argsort(peaks)[-1000:]
print("1000 plus gros peaks : ", peaks[idx0])
print("index 1000 plus gros peaks : ", idx0)

# Recherche des peaks correspondant dans les autres pistes
marge = ntmp * 20
idx1 = []
for i in idx0 :
    max_piste1 = 0
    id_max_p1 = 0
    max_piste2 = 0
    id_max_p2 = 0
    max_piste3 = 0
    id_max_p3 = 0
    for j in (-marge,marge,ntmp) :
        print(sp.signal.correlate(data1_final[i:i+window,0],data1_final[i+marge:i+marge+window,1]))
        if max_piste1 < sp.signal.correlate(data1_final[i:i+window,0],data1_final[i+marge:i+marge+window,1]) :
            id_max_p1 = i+j
    idx1 += [id_max_p1]

print(idx1)
print("taille de idx1 : ",len(idx1))
exit(0)

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
