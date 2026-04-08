import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

txt_file_path = input('Enter txt file path: ').strip()

### load file
size_error_data = np.loadtxt(txt_file_path, dtype=str, delimiter=' ')
print("Input data shape:", size_error_data.shape)
print()


### output dir path
# 比較用ディレクトリと、個別プロット用のディレクトリを作成
output_dir = os.path.join(os.path.dirname(txt_file_path), 'size_error_plots')
output_dir_Efield = os.path.join(output_dir, 'error_Efield')
output_dir_Envelope = os.path.join(output_dir, 'error_Envelope')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_Efield, exist_ok=True)
os.makedirs(output_dir_Envelope, exist_ok=True)


### set constants
c = 299792458  # [m/s]
epsilon_r = 9.0

# --- 追加: 予想遅延時間計算用の定数 ---
c_cm_ns = c * 100 * 1e-9  # [cm/ns] (約29.98 cm/ns)
epsilon_r_air = 1.0
epsilon_r_soil = 3.0

v_air = c_cm_ns / np.sqrt(epsilon_r_air)
v_soil = c_cm_ns / np.sqrt(epsilon_r_soil)
v_rock = c_cm_ns / np.sqrt(epsilon_r)

# 空間・土壌の往復伝搬時間 (ns)
t_air_roundtrip = (2 * 30.0) / v_air
t_soil_roundtrip = (2 * 200.0) / v_soil


### load data and make arrays for plot
Efield_data = []
Envelope_data = []
list4save = []

for i in range(size_error_data.shape[0]):
    true_size = float(size_error_data[i, 2])
    measured_top = float(size_error_data[i, 3])
    measured_bottom = float(size_error_data[i, 4])
    
    estimated_size = (measured_bottom - measured_top) * 1e-9 * c / np.sqrt(epsilon_r) / 2 * 100 # [cm]
    error = (estimated_size - true_size) / true_size * 100 # [%]

    # 第2列（インデックス1）で Efield か Envelope かを判定
    method = size_error_data[i, 1]
    
    # --- 追加: 予想遅延時間と差分の計算 ---
    if method == 'Efield':
        tx_delay = 2.06
    elif method == 'Envelope':
        tx_delay = 2.01
    else:
        print(f'Invalid analysis method: row {i}, value: {method}')
        tx_delay = 0

    expected_top = t_air_roundtrip + t_soil_roundtrip + tx_delay
    expected_bottom = expected_top + (2 * true_size) / v_rock
    
    diff_top = measured_top - expected_top
    diff_bottom = measured_bottom - expected_bottom
    # ------------------------------------

    if method == 'Efield':
        # diff_top と diff_bottom を配列に追加
        Efield_data.append([true_size, estimated_size, error, diff_top, diff_bottom])
    elif method == 'Envelope':
        Envelope_data.append([true_size, estimated_size, error, diff_top, diff_bottom])
        
    list4save.append([true_size, estimated_size, error, diff_top, diff_bottom])


### save list4save (差分の列を追加)
header = ['true_size [cm]', 'estimated_size [cm]', 'error [%]', 'diff_top [ns]', 'diff_bottom [ns]']
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

# 縦軸のフォーマッタ（+OO ns, -XX ns）
ns_formatter = ticker.FuncFormatter(lambda val, pos: f"{val:+.2f} ns")


# ========================================================
# --- plot for absolute error (絶対誤差) : Comparison ---
# ========================================================
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

plt.savefig(os.path.join(output_dir, 'size_error_abs.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'size_error_abs.pdf'))


# ============================================================
# --- plot for non-absolute error (生の誤差) : Comparison ---
# ============================================================
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

plt.savefig(os.path.join(output_dir, 'size_error.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'size_error.pdf'))


# ============================================================
# --- 追加: plot for time difference (時間差) : Comparison ---
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

# 1. True size VS Top Difference
for i, data in enumerate(size_estimations):
    if data.shape[0] > 0:
        # インデックス 3 = diff_top
        ax[0].plot(data[:, 0], data[:, 3], label=names[i], marker='o', color=colors[i], linestyle=linestyles[i])

ax[0].axhline(0, color='k', linestyle='-', linewidth=1)
ax[0].set_ylim(-0.15, 0.15)
ax[0].set_xlabel('True size [cm]', fontsize=20)
ax[0].set_ylabel('Top Delay Difference', fontsize=20)
ax[0].yaxis.set_major_formatter(ns_formatter)
ax[0].legend(fontsize=16)
ax[0].tick_params(labelsize=16)
ax[0].grid()

# 2. True size VS Bottom Difference
for i, data in enumerate(size_estimations):
    if data.shape[0] > 0:
        # インデックス 4 = diff_bottom
        ax[1].plot(data[:, 0], data[:, 4], label=names[i], marker='o', color=colors[i], linestyle=linestyles[i])

ax[1].axhline(0, color='k', linestyle='-', linewidth=1)
ax[0].set_ylim(-0.15, 0.15)
ax[1].set_xlabel('True size [cm]', fontsize=20)
ax[1].set_ylabel('Bottom Delay Difference', fontsize=20)
ax[1].yaxis.set_major_formatter(ns_formatter)
ax[1].legend(fontsize=16)
ax[1].tick_params(labelsize=16)
ax[1].grid()

plt.savefig(os.path.join(output_dir, 'time_diff.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'time_diff.pdf'))


# ========================================================
# --- plots for Individual methods (Efield / Envelope) ---
# ========================================================
individual_configs = [
    ('Estimated', Efield_data, 'r', '--', output_dir_Efield),
    ('Estimated', Envelope_data, 'b', '-.', output_dir_Envelope)
]

for name, data, color, linestyle, out_dir in individual_configs:
    if data.shape[0] == 0:
        continue

    # --- Individual absolute error (絶対誤差) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

    # 1. True size VS Estimated size
    ax[0].plot(data[:, 0], data[:, 1], label=name, marker='o', color=color, linestyle=linestyle)
    ax[0].plot(x, y, label='y = x', color='k', linestyle='-')
    ax[0].set_xlabel('True size [cm]', fontsize=20)
    ax[0].set_ylabel('Estimated size [cm]', fontsize=20)
    ax[0].legend(fontsize=16)
    ax[0].tick_params(labelsize=16)
    ax[0].grid()

    # 2. True size VS Error (abs)
    ax[1].plot(data[:, 0], np.abs(data[:, 2]), label=name, marker='o', color=color, linestyle=linestyle)
    ax[1].set_ylim(0, 15)
    ax[1].set_xlabel('True size [cm]', fontsize=20)
    ax[1].set_ylabel('Absolute Error [%]', fontsize=20)
    ax[1].legend(fontsize=16)
    ax[1].tick_params(labelsize=16)
    ax[1].grid()

    plt.savefig(os.path.join(out_dir, f'size_error_abs.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'size_error_abs.pdf'))

    # --- Individual non-absolute error (生の誤差) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

    # 1. True size VS Estimated size
    ax[0].plot(data[:, 0], data[:, 1], label=name, marker='o', color=color, linestyle=linestyle)
    ax[0].plot(x, y, label='y = x', color='k', linestyle='-')
    ax[0].set_xlabel('True size [cm]', fontsize=20)
    ax[0].set_ylabel('Estimated size [cm]', fontsize=20)
    ax[0].legend(fontsize=16)
    ax[0].tick_params(labelsize=16)
    ax[0].grid()

    # 2. True size VS Error (non-abs)
    ax[1].plot(data[:, 0], data[:, 2], label=name, marker='o', color=color, linestyle=linestyle)
    ax[1].set_ylim(-14, 14)
    ax[1].set_xlabel('True size [cm]', fontsize=20)
    ax[1].set_ylabel('Error [%]', fontsize=20)
    ax[1].legend(fontsize=16)
    ax[1].tick_params(labelsize=16)
    ax[1].grid()

    plt.savefig(os.path.join(out_dir, f'size_error.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'size_error.pdf'))


    # --- 追加: Individual time difference (時間差) ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

    # 1. True size VS Top Difference
    ax[0].plot(data[:, 0], data[:, 3], label=name, marker='o', color=color, linestyle=linestyle)
    ax[0].axhline(0, color='k', linestyle='-', linewidth=1)
    ax[0].set_xlabel('True size [cm]', fontsize=20)
    ax[0].set_ylabel('Top Delay Difference', fontsize=20)
    ax[0].yaxis.set_major_formatter(ns_formatter)
    ax[0].legend(fontsize=16)
    ax[0].tick_params(labelsize=16)
    ax[0].grid()

    # 2. True size VS Bottom Difference
    ax[1].plot(data[:, 0], data[:, 4], label=name, marker='o', color=color, linestyle=linestyle)
    ax[1].axhline(0, color='k', linestyle='-', linewidth=1)
    ax[1].set_xlabel('True size [cm]', fontsize=20)
    ax[1].set_ylabel('Bottom Delay Difference', fontsize=20)
    ax[1].yaxis.set_major_formatter(ns_formatter)
    ax[1].legend(fontsize=16)
    ax[1].tick_params(labelsize=16)
    ax[1].grid()

    plt.savefig(os.path.join(out_dir, f'time_diff.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'time_diff.pdf'))

plt.show()