import funciones
import json
import FiltrosECG
import FiltrosPCG
import segmentaciónseñales

"""## Importar señal"""

dict_signals = {}

"""Faltan los ECG de los archivos 41, 117, 220 y 233. Simplemente no los agregue al dict, asi que el len del dict es 405 en vez de 409."""

for xxx in range(408, 408):
    xxx += 1
    # Formatear el número xxx para obtener la representación de tres dígitos
    xxx_formatted = f"{xxx:03d}"
    print(xxx_formatted)
    dat_file = '/content/drive/MyDrive/ECG+PCG | Proyecto Final /Dataset/archive/training-a/a0'+xxx_formatted+'.dat'
    hea_file = '/content/drive/MyDrive/ECG+PCG | Proyecto Final /Dataset/archive/training-a/a0'+xxx_formatted+'.hea'
    wav_file = '/content/drive/MyDrive/ECG+PCG | Proyecto Final /Dataset/archive/training-a/a0'+xxx_formatted+'.wav'
    print(dat_file, '\n', hea_file)
    raw_ecg, tiempo_ecg, fs_ecg, condicion = funciones.importarECG(dat_file, hea_file)
    raw_pcg, tiempo_pcg, fs_pcg = funciones.importarPCG(wav_file, hea_file)
    dict_signals[xxx] = {'id': xxx_formatted,
                'condicion': condicion,
                'Raw ECG': raw_ecg.tolist(),
                'fs ECG': fs_ecg,
                'Tiempo ECG': tiempo_ecg.tolist(),
                'Raw PCG': raw_pcg.tolist(),
                'Tiempo PCG': tiempo_pcg,
                'fs PCG': fs_pcg}


dict_signals = FiltrosECG.filtrarECG(dict_signals)
dict_signals = FiltrosPCG.filtrarPCG(dict_signals)


normal = []
anormal = []
for i in range(len(dict_signals)):
  if (dict_signals[i]['condicion'] == 'Normal'): normal.append(i)
  else: anormal.append(i)

#Cortamos las señales:
"""En ventanas de 8  segundos"""
fs = 2000
longitud = fs*8 #Longitud de 8 segundos
stride_anormal = fs*8 
stride_normal = fs*3

ventanas = segmentaciónseñales.CortarSeñal(dict_signals, longitud, stride_normal, stride_anormal, fs)

"""Espectrograma"""
for i in range(len(ventanas)):
  f, t, Sxx = funciones.generar_espectrograma(ventanas[i]['pcg'], 2000)
  ventanas[i]['Espectrograma'] = {
     'f': f,
     't': t,
     'Sxx': Sxx
  }

#Cortamos las señales:
"""En latidos, de R a R"""
ecg_a_invertir = [6, 78, 84, 126, 159, 178, 187, 204, 241, 242, 246, 248, 271, 276, 287, 295, 314,316, 321, 328, 345]

for i in range(len(ecg_a_invertir)):
  dict_signals[ecg_a_invertir[i]]['Filtrado ECG'] = funciones.InvertirSeñal(dict_signals[i]['Filtrado ECG'], ecg_a_invertir[i])

"""Busco picos R"""
picos_R, indices_to_delete = funciones.EncontrarPicosR(dict_signals)
# Sort indices in descending order to avoid index shifting
indices_to_delete.sort(reverse=True)
for index in indices_to_delete:
    del dict_signals[index]

#Borro las señales en las que no se encontraron picos
indices_to_delete2 = []
for i in range(len(picos_R)):
  if(len(picos_R[i]) == 0):
    indices_to_delete2.append(i)
    print(f'Para {i} no se encontraron picos R')
indices_to_delete2.sort(reverse=True)
for ind in indices_to_delete2:
    del dict_signals[ind]
    del picos_R[ind]

latidos = segmentaciónseñales.CortarLatidos(dict_signals, picos_R, normal, anormal, 2000)

"""Espectrograma"""
for i in range(len(latidos)):
  f, t, Sxx = funciones.generar_espectrograma(latidos[i]['pcg'], 2000)
  latidos[i]['Espectrograma'] = {
     'f': f,
     't': t,
     'Sxx': Sxx
  }



"""# Guardar datos"""

filtrados = {}

filtrados

for i in range(410):
  if(i!=0 and i!=41 and i!=117 and i!=220 and i!=233):
    filtrados[i] = {'id': dict_signals[i]['id'],
                    'condicion': dict_signals[i]['condicion'],
                    'ECG': dict_signals[i]['Filtrado ECG'].tolist(),
                    'fs ECG': dict_signals[i]['fs ECG'],
                    'PCG': dict_signals[i]['Filtrado PCG'].tolist(),
                    'fs PCG': dict_signals[i]['fs PCG']}

# Ruta del archivo 
drive_file_path = 'C:\Users\lourd\Documents\Proyecto Final\pfc.venv\datos_filtrados.json'

# Guardar el diccionario en un archivo JSON
with open(drive_file_path, 'w') as file:
    json.dump(filtrados, file, indent=2)

print(f"Diccionario guardado en {drive_file_path}")

