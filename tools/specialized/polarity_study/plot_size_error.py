import numpy as np
import matplotlib.pyplot as plt
import os

txt_file_path = input('Enter txt file path: ').strip()

### load file
size_error_data = np.loadtxt(txt_file_path, dtype=str, delimiter=' ')
print(size_error_data.shape)
print()


### output dir path
output_dir = os.path.dirname(txt_file_path)


### set constants
c = 299792458  # [m/s]
epsilon_r = 9.0


### load data and make arrays for plot
LPR_circle = []
LPR_square = []
Bipolar_circle = []
Bipolar_square = []
list4save = []
for i in range(size_error_data.shape[0]):
    true_size = float(size_error_data[i, 2])
    estimated_size = (float(size_error_data[i, 4]) - float(size_error_data[i, 3])) * 1e-9 * c / np.sqrt(epsilon_r) / 2 * 100 # [cm]
    error = (estimated_size - true_size) / true_size * 100 # [%]

    if size_error_data[i, 0] == 'LPR-like':
        if size_error_data[i, 1] == 'circle':
            LPR_circle.append([true_size, estimated_size, error])
        elif size_error_data[i, 1] == 'square':
            LPR_square.append([true_size, estimated_size, error])
        else:
            print(f'Invarid type in rock shape: row {i}')
    elif size_error_data[i, 0] == 'Bipolar':
        if size_error_data[i, 1] == 'circle':
            Bipolar_circle.append([true_size, estimated_size, error])
        elif size_error_data[i, 1] == 'square':
            Bipolar_square.append([true_size, estimated_size, error])
        else:
            print(f'Invarid type in rock shape: row {i}')
    else:
        print(f'Invarid type in waveform: row {i}')
    list4save.append([true_size, estimated_size, error])


### save list4save
header = ['true_size [cm]', 'estimated_size [cm]', 'error [%]']
np.savetxt(os.path.join(output_dir, 'size_error_output.txt'), list4save, fmt='%f', delimiter=' ', header=' '.join(header))

LPR_circle = np.array(LPR_circle)
LPR_square = np.array(LPR_square)
Bipolar_circle = np.array(Bipolar_circle)
Bipolar_square = np.array(Bipolar_square)
print('Unipolar_circle: ', LPR_circle.shape)
print('Unipolar_square', LPR_square.shape)
print('Bipolar-circle: ', Bipolar_circle.shape)
print('Bipolar_square: ', Bipolar_square.shape)


### y = x
y = x = np.linspace(6, 15, 100)
### plot
fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

size_estimations = [LPR_circle, LPR_square, Bipolar_circle, Bipolar_square]
names = ['LPR_circle', 'LPR_square', 'Bipolar_circle', 'Bipolar_square']
colors = ['r', 'r', 'b', 'b']
linestyles = ['-', '--', '-', '--']
# true size VS estimated size
for i, data in enumerate(size_estimations):
    if data.shape[0] == 0:
        print('No data for one of the configurations. Please check the input txt file.')
    else:
        ax[0].plot(data[:, 0], data[:, 1], label=names[i], marker='o')
# ax[0].plot(LPR_circle[:, 0], LPR_circle[:, 1], label='Bipolar-circle', color='r', linestyle='-', marker='o')
# ax[0].plot(LPR_square[:, 0], LPR_square[:, 1], label='Bipolar-square', color='r', linestyle='--', marker='o')
# ax[0].plot(Bipolar_circle[:, 0], Bipolar_circle[:, 1], label='Unipolar-circle', color='b', linestyle='-', marker='o')
# ax[0].plot(Bipolar_square[:, 0], Bipolar_square[:, 1], label='Unipolar-square', color='b', linestyle='--', marker='o')
ax[0].plot(x, y, label='y = x', color='k', linestyle='-')

ax[0].set_xlabel('True size [cm]', fontsize=20)
ax[0].set_ylabel('Estimated size [cm]', fontsize=20)
ax[0].legend(fontsize=16)
ax[0].tick_params(labelsize=16)
ax[0].grid()

# True size VS error
for i, data in enumerate(size_estimations):
    if data.shape[0] == 0:
        print('No data for one of the configurations. Please check the input txt file.')
    else:
        ax[1].plot(data[:, 0], np.abs(data[:, 2]), label=names[i], marker='o')
# ax[1].plot(LPR_circle[:, 0], np.abs(LPR_circle[:, 2]), label='Bipolar-circle', color='r', linestyle='-', marker='o')
# ax[1].plot(LPR_square[:, 0], np.abs(LPR_square[:, 2]), label='Bipolar-square', color='r', linestyle='--', marker='o')
# ax[1].plot(Bipolar_circle[:, 0], np.abs(Bipolar_circle[:, 2]), label='Unipolar-circle', color='b', linestyle='-', marker='o')
# ax[1].plot(Bipolar_square[:, 0], np.abs(Bipolar_square[:, 2]), label='Unipolar-square', color='b', linestyle='--', marker='o')

ax[1].set_xlabel('True size [cm]', fontsize=20)
ax[1].set_ylabel('Error [%]', fontsize=20)
# ax[1].legend(fontsize=16)
ax[1].tick_params(labelsize=16)
ax[1].grid()
# save
plt.savefig(os.path.join(output_dir, 'size_error.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'size_error.pdf'))

plt.show()
