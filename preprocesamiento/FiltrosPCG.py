import funciones
import numpy as np
"""#PCG"""
def filtrarPCG(dict_pcg):
    print('Inicio del filtrado de señales de PCG')
    """## Normalización"""
    print('\n Normalización')
    for i in range(len(dict_pcg)):
        try:
            if(i!=0 and i!=41 and i!=117 and i!=220 and i!=233):
                raw_pcg = dict_pcg[i]['Raw PCG']
                dict_pcg[i]['Normal PCG'] = funciones.normalize_signal(raw_pcg)
        except IndexError as e:
            print(f"Error para normalizar PCG en la iteración {i}: {e}")

    """## Pasa banda 25 - 400Hz"""
    print('\n Pasa banda 25-400Hz')
    fc_LP = 400
    fc_HP = 25

    for i in range(len(dict_pcg)):
        try:
            if(i!=0 and i!=41 and i!=117 and i!=220 and i!=233):
                fs_pcg = dict_pcg[i]['fs PCG']
                pcg_normal = dict_pcg[i]['Normal PCG']
                pcg_LP = funciones.butter_lowpass_filter(pcg_normal, fc_LP, fs_pcg, 4)
                pcg_HP = funciones.butter_highpass_filter(pcg_LP, fc_HP, fs_pcg, 4)
                dict_pcg[i]['Filtrado PCG'] = pcg_HP
        except IndexError as e:
            print(f"Error para el pasa banda en PCG en la iteración {i}: {e}")



    """# Sacar picos"""
    print('\n Saco picos')
    for i in range(len(dict_pcg)):
        try:
            if i not in [0, 41, 117, 220, 233]:
                pcg = np.array(dict_pcg[i]['Filtrado PCG'])
                fs = dict_pcg[i]['fs PCG']
                pcg_picos = funciones.schmidt_spike_removal(pcg, fs)
                dict_pcg[i]['Filtrado PCG'] = pcg_picos
        except IndexError as e:
            print(f"Error para sacar picos en PCG en la iteración {i}: {e}")


    print('Finalización del filtrado para las señales de PCG')
    return dict_pcg
