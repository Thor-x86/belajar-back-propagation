# Ambil library yang diperlukan
from random import random
import csv
import math

#################### Parameter Uji ######################

NAMA_FILE = 'data.csv'
JUMLAH_ITERASI = 5
LEARNING_RATE = 0.3
JUMLAH_GENERASI = 500
JUMLAH_HIDDEN_NODE = 5

################# Tahap Proses Dataset #################

print('Memproses berkas %s...' % NAMA_FILE)

# Ambil data dari file CSV
dataset = list()
with open(NAMA_FILE, 'r') as file:
    csv_reader = csv.reader(file)
    for tiap_baris in csv_reader:
        # Hiraukan baris kosong
        if not tiap_baris:
            continue
        dataset.append(tiap_baris)

# Dataset di sini ibaratkan seperti pada MS
# Excel, memiliki baris, kolom, dan sel.
# Untuk membaca dataset, kita menggunakan
# format ini
# dataset[indeks_baris][indeks_kolom]

jumlah_kolom = len(dataset[0])

# Sel masih berupa string, kita perlu konversi
# sel tersebut menjadi float (atau int khusus label)
for baris in dataset:
    for indeks_kolom in range(jumlah_kolom):
        sel = baris[indeks_kolom].strip()
        if indeks_kolom < jumlah_kolom - 1:
            # Selain label
            baris[indeks_kolom] = float(sel)
        else:
            # Untuk label
            baris[indeks_kolom] = int(sel)

# Kumpulkan semua ID label yang terdapat di dataset
labelset = set()
for baris in dataset:
    label = baris[jumlah_kolom-1]
    labelset.add(label)

jumlah_label = len(labelset)

# Cari nilai minimum dan maksimum
nilai_minimum = 0.0
nilai_maksimum = 0.0
for baris in dataset:
    for indeks_kolom in range(jumlah_kolom-1):
        sel = baris[indeks_kolom]
        nilai_minimum = min(nilai_minimum, sel)
        nilai_maksimum = max(nilai_maksimum, sel)

selisih = nilai_maksimum - nilai_minimum

# Ubah rentang min-max menjadi 0.0-1.0 pada dataset
# Rumus: f(x) = (x - x_min) / (x_max - x_min)
for baris in dataset:
    for indeks_kolom in range(jumlah_kolom-1):
        baris[indeks_kolom] = (baris[indeks_kolom] - nilai_minimum) / selisih

################ Inisiasi Neural Network ###############

print('Melatih neural network...')

# Buat list kosong untuk menyimpan semua neural layer
neural_network = list()

# Buat hidden layer dengan nilai weight acak
hidden_layer = list()
for _ in range(JUMLAH_HIDDEN_NODE):
    neuron = {'weight': list()}
    for _ in range(jumlah_kolom-1):
        neuron['weight'].append(random())
    hidden_layer.append(neuron)
neural_network.append(hidden_layer)

# Buat output layer dengan nilai weight acak
output_layer = list()
for _ in range(jumlah_label):
    neuron = {'weight': list()}
    for _ in range(JUMLAH_HIDDEN_NODE):
        neuron['weight'].append(random())
    output_layer.append(neuron)
neural_network.append(output_layer)


################### Forward Propagate ##################


def aktivasi_neuron(list_weight, list_input):
    '''
    Rumus: z = sum(weight_i * input_i) + bias
    '''
    z = list_weight[-1]
    n = len(list_weight)
    for i in range(n - 1):
        z += list_weight[i] * list_input[i]
    return z


def sigmoid(z):
    '''
    Rumus:  σ(z) = 1 / (1 + e^(-z))
    '''
    return 1.0 / (1.0 + math.exp(-z))


def forward_propagate(neural_network, baris):
    baris_sekarang = baris
    for layer in neural_network:
        baris_baru = []
        for neuron in layer:
            z = aktivasi_neuron(neuron['weight'], baris_sekarang)
            neuron['output'] = sigmoid(z)
            baris_baru.append(neuron['output'])
        baris_sekarang = baris_baru
    return baris_sekarang


################# Back Propagate Error ##################


def derivatif_sigmoid(z):
    return z * (1.0 - z)


def error_back_propagation(neural_network, nilai_yang_diharapkan):
    for i in reversed(range(len(neural_network))):
        tiap_lapisan = neural_network[i]
        list_error = list()
        if i != len(neural_network)-1:
            for j in range(len(tiap_lapisan)):
                error = 0.0
                for neuron in neural_network[i + 1]:
                    error += (neuron['weight'][j] * neuron['delta'])
                list_error.append(error)
        else:
            for j in range(len(tiap_lapisan)):
                neuron = tiap_lapisan[j]
                list_error.append(neuron['output'] - nilai_yang_diharapkan[j])
        for j in range(len(tiap_lapisan)):
            neuron = tiap_lapisan[j]
            neuron['delta'] = list_error[j] * \
                derivatif_sigmoid(neuron['output'])


################# Latih neural network ##################


def update_weight_per_baris(neural_network, baris, l_rate):
    for i in range(len(neural_network)):
        list_input = baris[:-1]
        if i != 0:
            list_input = [neuron['output'] for neuron in neural_network[i - 1]]
        for neuron in neural_network[i]:
            for j in range(len(list_input)):
                neuron['weight'][j] -= l_rate * neuron['delta'] * list_input[j]
            neuron['weight'][-1] -= l_rate * neuron['delta']


# Melatih neural network, dengan jumlah generasi yang
# ditentukan pada parameter uji
for _ in range(JUMLAH_GENERASI):
    for baris in dataset:
        list_output = forward_propagate(neural_network, baris)
        list_diharapkan = [0 for i in range(jumlah_label)]
        # list_diharapkan[baris[-1]] = 1
        error_back_propagation(neural_network, list_diharapkan)
        update_weight_per_baris(neural_network, baris, LEARNING_RATE)


##################### Tes Prediksi ######################

while True:
    print('')

    # Minta data ke pengguna
    masukan = input('Masukkan %i angka dibatasi spasi: ' % (jumlah_kolom-1))

    # Masukan kosong artinya keluar
    if len(masukan) == 0:
        print('Berhenti.')
        print('')
        input('[Tekan ENTER]')
        break

    list_angka = masukan.split(' ')

    # Pastikan jumlah data sesuai dengan data training
    if len(list_angka) != jumlah_kolom-1:
        print('Harus %i angka!' % (jumlah_kolom-1))
        continue

    # Konversi data ke float, peringati jika terdeteksi
    # bukan angka
    try:
        for indeks in range(len(list_angka)):
            list_angka[indeks] = float(list_angka[indeks])
    except ValueError:
        print('Mohon masukkan angka saja!')
        continue

    # Normalisasi input
    minimum = 0
    maksimum = 0
    for angka in list_angka:
        minimum = min(minimum, angka)
        maksimum = max(maksimum, angka)
    for indeks in range(len(list_angka)):
        angka = list_angka[indeks]
        list_angka[indeks] = (angka - minimum) / (maksimum - minimum)

    print('Memproses...')
    list_output = forward_propagate(neural_network, list_angka)
    hasil = list_output.index(max(list_output)) + 1
    print('')
    print('ID Label adalah %i' % hasil)
