from copy import deepcopy
import soundfile as sf
import numpy as np
import scipy as sp
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data1, Fe1 = sf.read("/content/erreur_signal/ONECAT_20200114_150201_247.wav") # A changer en fonction de l'emplacement des fichiers
data2, Fe2 = sf.read("/content/erreur_signal/S70_20200114_150211_982.wav")

"""
GetPPS permet de trouver le début et la fin du signal PPS
Entrée :
  - data : données du signal
Sortie :
  - startPPS : début du signal PPS
  - endPPS : fin du signal PPS
"""
def GetPPS(data) :
  startPPS = 0
  while(data[startPPS][4]) < 0.5 :
    startPPS+=1
  endPPS = startPPS
  while data[endPPS][4] > -0.5 :
    endPPS+=1
  return startPPS, endPPS

print(f"Premier PPS de data1 : {GetPPS(data1)[0]}")

"""
AveragePPS permet de calculer la moyenne du signal PPS
Entrée :
  - data : données du signal
  - no_extrem : si vrai garde les élément au extrimité qui sont imcomplètes
Sortie :
  - average_bit : tableau du nombre de bit par seconde
  - average_data : tableau de data séparé par seconde
"""
def AveragePPS(data, no_extrem=False) :
  average_bit = []
  tmp = 0
  pre = data[0][4]
  average_data = []
  i1p = 0
  for k in range(0,len(data)) :
      if data[k][4] > 0.6 and pre < 0.5 :
          average_bit +=[tmp]
          tmp = 0
          average_data.append(deepcopy(data[i1p:k]))
          i1p = k
      else :
          tmp += 1
      pre = data[k][4]
  if no_extrem :
    return average_data, average_bit
  # Ignore le premier et dernier élément qui sont probablement incomplé
  return average_data[1:-1], average_bit[1:-1]

#Bit moyen par PPS
_, av_bit_data1 = AveragePPS(data1, True)
_, av_bit_data2 = AveragePPS(data2, True)
plt.figure(figsize=(10, 6))
plt.title("Moyenne du nombre de bit par PPS")
plt.xlabel("Numéro de PPS")
plt.ylabel("Nombre de bit")
plt.scatter(range(len(av_bit_data1)), av_bit_data1, marker='o', color='blue', )
plt.scatter(range(len(av_bit_data2)), av_bit_data2, marker='o', color='red')
#plt.scatter(av_bit_data2,len(av_bit_data2))
print(f"Fréquence moyenne data1 obesrver / Fréquence indiqué: {sum(av_bit_data1) // len(av_bit_data1)} / {Fe1}")
print(f"Fréquence moyenne data2 obesrver/ Fréquence indiqué: {sum(av_bit_data2) // len(av_bit_data2)} / {Fe2}")
plt.show()

"""
ReconstructSignal permet de reconstruire le signal.
Il rééchantillonne les données et les concatène.
Entrée :
  - average_data : tableau de data séparé par seconde
  - nb_bit : nombre de bit par seconde
Sortie :
  - data_final : signal reconstruit
"""
def ReconstructSignal(average_data, nb_bit) :
  resamp_data = []
  # Rééchantillonnage du signal
  for m in average_data :
      resamp_data.append(sp.signal.resample(m, int(nb_bit)))
  # Concatenate la data pour pouvoir l'exploiter
  data_final = np.concatenate(resamp_data, axis=0)
  return data_final

def Traitement(file1, start1, end1, file2, start2, end2) :
  # Vérification de la cohérence des tailles d'échantillon
  if end1-start1 != end2-start2 :
    print("Erreur dans la fonciton Traitement:\n\tLa taille des données de sortie ne sont pas égales!")
    return -1
  print("Loading files...")
  data1, Fe1 = sf.read(file1)
  data2, Fe2 = sf.read(file2)

  print("Reconstructing data...")
  startPPS1, endPPS1 = GetPPS(data1) # Récupération du premier PPS de data1
  startPPS2, endPPS2 = GetPPS(data2) # Récupération du premier PPS de data2

  # Calcul des moyennes de bit par PPS
  average_data1, average_bit1 = AveragePPS(data1)
  average_data2, average_bit2 = AveragePPS(data2)

  # Début et fin de chaque signal
  average_data1, average_bit1 = average_data1[start1:end1], average_bit1[start1:end1]
  average_data2, average_bit2 = average_data2[start2:end2], average_bit2[start2:end2]
  
  # Reconstruction du signal avec concatennation basé sur la moyenne du nombre de bit par PPS des deux signaux
  Fe = (sum(average_bit1) // len(average_bit1) + sum(average_bit2) // len(average_bit2)) // 2
  data1 = ReconstructSignal(average_data1, Fe)
  data2 = ReconstructSignal(average_data2, Fe)
  return data1, data2, Fe

"""
CalcEnergie calcul l'énergie du signal et retourne les nb_peaks peaks les plus
élevés.
Entrée :
  - data : données du signal
  - piste : numéro de la piste
  - Freq : fréquence du signal
  - ratio : facteur de conversion de fréquence
  - window_size : taille de la fenêtre de calcul de l'énergie
  - nb_peaks : nombre de peaks à retourner
Sortie :
  - peaks : tableau des nb_peaks peaks les plus élevés
  - index : tableau des index des nb_peaks peaks les plus élevés
"""
def CalcEnergie(data, piste, Freq, ratio, window_size, nb_peaks) :
  interval = Freq // ratio
  window = interval * window_size
  peaks = []
  for i in range(0,len(data),interval) :
    peaks += [sum(data[i:i+window,piste]**2)]
  peaks = np.array(peaks)
  index = np.argsort(peaks)[-nb_peaks:]
  return peaks[index], index, interval, window

"""
PeaksSimilitude permet de calculer les distances entre les peaks de deux pistes.
Entrée :
  - data1 : données du signal 1
  - piste1 : numéro de la piste
  - data2 : données du signal 2
  - piste2 : numéro de la piste
  - index : tableau des index des nb_peaks peaks les plus élevés
  - interval : intervalle entre deux fenêtres
  - window : taille de la fenêtre de calcul de l'énergie
  - marge : marge de recherche
  Sortie :
  - deltas : tableau des distances entre les peaks
"""
def PeaksSimilitude(data1, piste1, data2, piste2, index, interval, window, marge) :
  deltas = [] #Liste de distances
  for i in index :
    max_corr = 0
    max_corr_idx = 0
    idx = i * interval
    for j in range(max(0, idx - marge),min(idx + marge, len(data2)), interval) :
      corr = sum(sp.signal.correlate(data1[idx:idx + window, piste1],data2[j:j + window, piste2]))
      # print(idx, j)
      if max_corr < corr :
        max_corr = corr
        max_corr_idx = j
    deltas += [max_corr_idx - idx]

  return deltas

"""
BuildDeltas permet de construire la liste des deltas.
"""
def BuildDeltas(data1, data2, Freq, ratio, window_size, nb_peaks) :
  delta = []
  for i in range(4) :
    peaks1, index1, interval1, window1 = CalcEnergie(data1, i, Freq, ratio, window_size, nb_peaks)
    peaks2, index2, interval2, window2 = CalcEnergie(data1, i, Freq, ratio, window_size, nb_peaks)
    for j in range(4) :
      print(f"{i} -- {j}")
      if i != j :
        delta += [PeaksSimilitude(data1, i, data1, j, index1, interval1, window1, 100000)]
        delta += [PeaksSimilitude(data2, i, data2, j, index2, interval2, window2, 100000)]
      delta += [PeaksSimilitude(data1, i, data2, j, index1, interval1, window1, 100000)]
      delta += [PeaksSimilitude(data2, i, data1, j, index2, interval2, window2, 100000)]
  return delta

data1, data2, Fe = Traitement("/content/erreur_signal/ONECAT_20200114_150201_247.wav", 10, 110,  "/content/erreur_signal/S70_20200114_150211_982.wav", 0, 100)

print(data1.shape)
print(data2.shape)
print(Fe)

delta = BuildDeltas(data1, data2, Fe, 5, 5, 500)

#Pour sauvegarder les deltas

# Define a filename for saving the delta data
output_filename = 'delta_data1.npy'

# Save the delta_np array to the file
np.save(output_filename, delta)

print(f"Delta data saved to {output_filename}")

delta=np.load(output_filename)
print(delta.shape)

"""
Fonction pour calculer l'entropie
"""
def entropie(p) :
  return - np.sum(p * np.log2(p))

row_entropies = []
for d in delta :
  hist_counts = np.histogram(d, bins=200, density=False)[0]
  total_elements_in_row = hist_counts.sum()
  probabilities = hist_counts / total_elements_in_row
  row_entropies.append(entropie(probabilities))
plt.figure(figsize=(10, 6))
plt.hist(row_entropies, bins=10, edgecolor='black', color='lightcoral')
plt.title("Histogramme d'entropie")
plt.xlabel("Entropie")
plt.ylabel("Fréquence")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#TSNE
tsne = TSNE(n_components=2, random_state=0, perplexity=30)
tsne = tsne.set_params()
delta_2d = tsne.fit_transform(delta)

k = 3 #Nombre de clusters
codebook, _ = sp.cluster.vq.kmeans(delta_2d, k)
labels, _ = sp.cluster.vq.vq(delta_2d, codebook)

codebook_np = np.array(codebook)
unique_labels, counts = np.unique(labels, return_counts=True)
# Nombre d'élément par cluster
for label, count in zip(unique_labels, counts):
    print(f"  Cluster {label}: {count} elements")
plt.figure(figsize=(10, 8))
plt.title('Visualisation des clusters')
plt.scatter(delta_2d[:, 0], delta_2d[:, 1], c=labels, cmap='viridis')
plt.scatter(codebook_np[:,0], codebook_np[:,1], marker='x', color='red')
# Legende des cluster
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                               markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                   for i in range(len(set(labels)))]

# Add an element for the cluster centers to the legend
legend_elements.append(plt.Line2D([0], [0], marker='x', color='red', label='Centers',
                                  linestyle='None', markersize=10))

plt.legend(handles=legend_elements, title='Clusters')
