import soundfile as sf
import numpy as np
# drive glotin init recherche
data, Fe = sf.read("/scratch/yroblin156/donnee_glotin/d1/meute/ONECAT_20200114_150201_247.wav")


print(data[0][4]) #temps = 0, piste = 4

i = 0
while data[i][4] < 0.5 :
    i+=1

print(i)

j = 0 

while data[j][4] > -0.5 :
    j+=1

print(j)
print(j-i)

max = 0
imax = 0
min = 0
imin = 0
for k in range(i-2000,j+2000) :
    if data[k][4] > max :
        max = data[k][4]
        imax = k
    elif data[k][4] < min :
        min = data[k][4]
        imin = k
print(min,max)
print(imin,imax)