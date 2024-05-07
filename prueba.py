def load_ecg(record):
    print(record)
    ecg = sio.loadmat(record)['ecg'].squeeze()
    ecg = signal.resample(ecg, 2048)
    return ecg

ecgs = []
labels = []
with open(os.path.join(data_json, 'ecg_data.json'), 'r') as fid:
    data = [json.loads(l) for l in fid]
for d in tqdm.tqdm(data):
    archivo = d['ecg'].split('Proyecto')[-1]
    ecg_path = '/home/lougonzalez/Proyecto-Final'+archivo
    print(f'Cargando el archivo ecg_path')
    labels.append(d['label'])
    ecgs.append(load_ecg(ecg_path))

longitudes = []
for i in range(len(ecgs)):
  longitudes.append(len(ecgs[i]))

print(min(longitudes))
print(max(longitudes))
