import numpy as np
import matplotlib.pyplot as plt
import os

txt_file_path = input('Enter txt file path: ').strip()

### load file
size_error_data = np.loadtxt(txt_file_path, dtype=str, delimiter=' ')
print("Input data shape:", size_error_data.shape)
print()


### output dir path
output_dir = os.path.dirname(txt_file_path)
# EfieldとEnvelopeの比較用ディレクトリを1つ作成
output_dir_comparison = os.path.join(output_dir, 'error_comparison')
os.makedirs(output_dir_comparison, exist_ok=True)


### set constants
c = 299792458  # [m/s]
epsilon_r = 9.0


### load data and make arrays for plot
Efield_data = []
Envelope_data = []
list4save = []

for i in range(size_error_data.shape[0]):
    true_size = float(size_error_data[i, 2])
    estimated_size = (float(size_error_data[i, 4]) - float(size_error_data[i, 3])) * 1e-9 * c / np.sqrt(epsilon_r) / 2 * 100 # [cm]
    error = (estimated_size - true_size) / true_size * 100 # [%]

    # 第2列（インデックス1）で Efield か Envelope かを判定
    method = size_error_data[i, 1]
    if method == 'Efield':
        Efield_data.append([true_size, estimated_size, error])
    elif method == 'Envelope':
        Envelope_data.append([true_size, estimated_size, error])
    else:
        print(f'Invalid analysis method: row {i}, value: {method}')
        
    list4save.append([true_size, estimated_size, error])


### save list4save
header = ['true_size [cm]', 'estimated_size [cm]', 'error [%]']
np.savetxt(os.path.join(output_dir, 'size_error_output.txt'), list4save, fmt='%f', delimiter=' ', header=' '.join(header))

# リストをNumPy配列に変換
Efield_data = np.array(Efield_data)
Envelope_data = np.array(Envelope_data)

print('Efield data: ', Efield_data.shape)
print('Envelope data: ', Envelope_data.shape)


### y = x
x = np.linspace(5, 16, 100) # データ範囲（6〜15）に合わせて少し余裕を持たせました
y = x

### plot settings
names = ['Efield', 'Envelope']
colors = ['b', 'r'] # Efieldを青、Envelopeを赤に設定
linestyles = ['-', '--']
size_estimations = [Efield_data, Envelope_data]

# ==========================================
# --- plot for absolute error (絶対誤差) ---
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

# 1. True size VS Estimated size
for i, data in enumerate(size_estimations):
    if data.shape[0] == 0:
        print(f'No data for {names[i]}.')
    else:
        ax[0].plot(data[:, 0], data[:, 1], label=names[i], marker='o', color=colors[i], linestyle=linestyles[i])

ax[0].plot(x, y, label='y = x', color='k', linestyle='-')
ax[0].set_xlabel('True size [cm]', fontsize=20)
ax[0].set_ylabel('Estimated size [cm]', fontsize=20)
ax[0].legend(fontsize=16)
ax[0].tick_params(labelsize=16)
ax[0].grid()

# 2. True size VS Error (abs)
for i, data in enumerate(size_estimations):
    if data.shape[0] > 0:
        ax[1].plot(data[:, 0], np.abs(data[:, 2]), label=names[i], marker='o', color=colors[i], linestyle=linestyles[i])

ax[1].set_ylim(0, 15)  # y軸の下限を0に設定
ax[1].set_xlabel('True size [cm]', fontsize=20)
ax[1].set_ylabel('Absolute Error [%]', fontsize=20)
ax[1].legend(fontsize=16)
ax[1].tick_params(labelsize=16)
ax[1].grid()

# save
plt.savefig(os.path.join(output_dir_comparison, 'size_error_abs.png'), dpi=300)
plt.savefig(os.path.join(output_dir_comparison, 'size_error_abs.pdf'))


# ==============================================
# --- plot for non-absolute error (生の誤差) ---
# ==============================================
fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

# 1. True size VS Estimated size
for i, data in enumerate(size_estimations):
    if data.shape[0] > 0:
        ax[0].plot(data[:, 0], data[:, 1], label=names[i], marker='o', color=colors[i], linestyle=linestyles[i])

ax[0].plot(x, y, label='y = x', color='k', linestyle='-')
ax[0].set_xlabel('True size [cm]', fontsize=20)
ax[0].set_ylabel('Estimated size [cm]', fontsize=20)
ax[0].legend(fontsize=16)
ax[0].tick_params(labelsize=16)
ax[0].grid()

# 2. True size VS Error (non-abs)
for i, data in enumerate(size_estimations):
    if data.shape[0] > 0:
        ax[1].plot(data[:, 0], data[:, 2], label=names[i], marker='o', color=colors[i], linestyle=linestyles[i])

ax[1].set_ylim(-14, 14)
ax[1].set_xlabel('True size [cm]', fontsize=20)
ax[1].set_ylabel('Error [%]', fontsize=20)
ax[1].legend(fontsize=16)
ax[1].tick_params(labelsize=16)
ax[1].grid()

# save
plt.savefig(os.path.join(output_dir_comparison, 'size_error.png'), dpi=300)
plt.savefig(os.path.join(output_dir_comparison, 'size_error.pdf'))

plt.show()