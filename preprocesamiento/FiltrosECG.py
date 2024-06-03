import funciones

"""## Funciones

Para graficar

https://python-charts.com/es/colores/ --> para elegir más colores
"""
def filtrarECG(dict_ecg):
  """
      Función para filtrar las señales de ECG.
      
      Args:
          dict_ecg (dict): Diccionario que contiene las señales de ECG.

      Returns:
          dict: Diccionario que contiene las señales de ECG filtradas.
      """
  print('Inicio del filtrado para ECG')

  print('\n Normalización')
  """## Normalización"""

  for i in range(len(dict_ecg)):
    try:
      if(i!=0 and i!=41 and i!=117 and i!=220 and i!=233):
        raw_ecg = dict_ecg[i]['Raw ECG']
        print(raw_ecg)
        dict_ecg[i]['Normal ECG'] = funciones.normalize_signal(raw_ecg)
    except IndexError as e:
        print(f"Error para normalizar ECG en la iteración {i}: {e}")

  """## Pasa banda 1-60Hz"""
  print('\n Pasa Banda 1-60Hz')
  fc_low = 60
  fc_high = 1
  for i in range(405, 410):
    try:
      if(i!=0 and i!=41 and i!=117 and i!=220 and i!=233):
        fs = dict_ecg[i]['fs ECG']
        normal = dict_ecg[i]['Normal ECG']
        ecg_LP = funciones.butter_lowpass_filter(normal, fc_low, fs, 5)
        ecg_HP = funciones.butter_highpass_filter(ecg_LP, fc_high, fs, 5)
        dict_ecg[i]['Filtrado ECG'] = ecg_HP
    except IndexError as e:
        print(f"Error para el pasa banda en ECG en la iteración {i}: {e}")


  print('Finalización del filtrado para las señales de ECG')

  return dict_ecg