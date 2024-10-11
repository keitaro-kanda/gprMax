import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
import mpl_toolkits.axes_grid1 as axgrid1
import os
import argparse
from tqdm import tqdm
from outputfiles_merge import get_output_data
from scipy import signal
from numpy.linalg import svd, eig, inv
import scipy.linalg as linalg



#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='k_fitting.py',
    description='Process hyperbola fitting',
    epilog='End of help message',
    usage='python -m tools.k_fk_migration [json_path] [-fix]',
)
parser.add_argument('json_path', help='Path to the json file')
parser.add_argument('-fix', choices=['er', 'R'], help='Fix the epsilon_r or R', default=None)
args = parser.parse_args()



#* Load json file
with open(args.json_path) as f:
    params = json.load(f)
#* Load antenna settings
src_step = params['antenna_settings']['src_step']
rx_step = params['antenna_settings']['rx_step']
src_start = params['antenna_settings']['src_start']
rx_start = params['antenna_settings']['rx_start']
#* Check antenna step
if src_step == rx_step:
    antenna_step = src_step
    antenna_start = (src_start + rx_start) / 2



#* Load output file
data_path = params['data']
if args.fix == 'er':
    output_dir = os.path.join(os.path.dirname(data_path), 'fitting_er_fixed')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
elif args.fix == 'R':
    output_dir = os.path.join(os.path.dirname(data_path), 'fitting_R_fixed')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
else:
    output_dir = os.path.join(os.path.dirname(data_path), 'fitting_no_fixed')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
f = h5py.File(data_path, 'r')
nrx = f.attrs['nrx']
for rx in range(nrx):
    data, dt = get_output_data(data_path, (rx+1), params['data_name'])
print('data shape: ', data.shape)




#* Define function to extract byperbola by detecting the peak
def extract_peak(data, trac_num):
    """
    data: Ascan data
    i: trace number
    """

    skip_time = 20 # [ns]
    data = data[int(skip_time*1e-9/dt):]

    #* Detect the peak in the envelope
    threshold = np.max(np.abs(data)) * 0.1

    envelope = np.abs(signal.hilbert(data))

    i = 0
    while i < len(envelope):
        if envelope[i] > threshold:
            start = i
            while i < len(envelope) and envelope[i] > threshold:
                i += 1
            end = i
            extracted_time = np.argmax(np.abs(data[start:end])) + start + int(skip_time*1e-9/dt)
            peak_indeces.append([trac_num, extracted_time])
        i += 1



def fit_hyperbola_shihab(x, y):
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y


    #* Define design matrix and scatter matrix
    D = np.matrix(np.column_stack([x2, xy, y2, x, y, np.ones_like(x)])) # Design matrix
    S = np.matrix(D.T @ D) # Scatter matrix


    #* Define constraint matrix
    C = np.zeros((6, 6))
    C[0, 2] = -2
    C[1, 1] = 1
    C[2, 0] = -2
    #print('C: \n', C)


    #* Solve the generalized eigenvalue problem
    eigval, eigvec = linalg.eig(S, C, right=True, left=False)
    # Normalize eigenvectors to satisfy the constraint
    for i in range(eigvec.shape[1]):
        a_i = eigvec[:, i]
        constraint_value = a_i.T @ C @ a_i

        # Check if constraint_value is not zero to avoid division by zero
        if np.abs(constraint_value) > 1e-12:
            # Normalize the eigenvector
            a_i_normalized = a_i / np.sqrt(np.abs(constraint_value))

            # Ensure the sign of the constraint is positive
            if constraint_value < 0:
                a_i_normalized = -a_i_normalized

        # Verify the constraint
            constraint_check = a_i_normalized.T @ C @ a_i_normalized
            print(f'Eigenvector {i}:')
            print('Normalized eigenvector:', a_i_normalized)
            print('Constraint value:', constraint_check)
        else:
            print(f'Eigenvector {i} has a constraint value too small for reliable normalization.')

    # Select the eigenvector corresponding to the smallest positive eigenvalue
    # (You may need to adjust this selection based on your specific problem)
    valid_indices = np.where((eigval.real > 0) & np.isfinite(eigval))[0]
    if valid_indices.size > 0:
        min_index = valid_indices[np.argmin(eigval[valid_indices].real)]
        best_eigvec = eigvec[:, min_index]
        # Normalize the selected eigenvector
        constraint_value = best_eigvec.T @ C @ best_eigvec
        best_eigvec_normalized = best_eigvec / np.sqrt(np.abs(constraint_value))
        # Ensure the sign of the constraint is positive
        if constraint_value < 0:
            best_eigvec_normalized = -best_eigvec_normalized
        print('Best eigenvector after normalization:', best_eigvec_normalized)
        print(' ')
    else:
        print('No valid eigenvector found.')
        print(' ')

    # Return the coefficients of the fitted hyperbola
    return best_eigvec_normalized.real


def fit_hyperbola(x, y):
    # xとyはカラムベクトルであることが仮定される
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y

    #* Define design matrix and scatter matrix
    D1 = np.column_stack([x2, xy, y2])
    D2 = np.column_stack([x, y, np.ones_like(x)])

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    #* Test the rank of S3
    Us3, Ss3, Vs3 = svd(S3)
    condNrs = Ss3 / Ss3[0]

    epsilon = 1e-10
    if condNrs[2] < epsilon:
        print('Warning: S3 is degenerate')
        return None, None

    #* Define constraint matrix and its inverse matrix
    C = np.array([[0, 0, -2], [0, 1, 0], [-2, 0, 0]])
    Ci = inv(C)

    #* Solve the generalized eigenvalue problem
    T = -inv(S3) @ S2.T
    S = Ci @ (S1 - S2 @ T)

    evals, evec = eig(S)
    evec = evec/np.linalg.norm(evec, axis=0)

    #* Evaluate the constraint values and sort them
    cond = evec[1, :] ** 2 - 4 * evec[0, :] * evec[2, :]
    condVals = np.sort(cond)


    #* Get the hyperbolic solution
    possibleHs = condVals[1:2] + condVals[0]
    possibleHs = possibleHs[possibleHs > 0]
    minDiffAt = np.argmin(np.abs(possibleHs))
    minDiffAt = np.argmin(possibleHs)
    alpha1 = evec[:, minDiffAt + 1]
    alpha2 = T @ alpha1
    hyperbola = np.concatenate([alpha1, alpha2])


    return hyperbola



def extract_hyperbola_parameters_vertical(coeffs):
    A, B, C, D, E, F = coeffs

    # 双曲線の判別式を計算
    discriminant = B**2 - 4 * A * C
    if discriminant <= 0:
        print("This conic is not a hyperbola.")
        return None, None, None, None

    else:
        # 行列形式での計算
        M = np.array([[A, B/2], [B/2, C]])

        # Mの固有値，固有ベクトルを計算
        eigvals, eigvecs = np.linalg.eig(M)
        # 正規化
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
        #print('eigvecs:', eigvecs)

        # x, yをX, Yに変換
        new_2nd_order_coeffs = eigvecs.T @ M @ eigvecs
        A_new = new_2nd_order_coeffs[0, 0]
        B_new = new_2nd_order_coeffs[0, 1]
        C_new = new_2nd_order_coeffs[1, 1]

        # 1次項の係数を計算
        new_1st_order_coeffs = [D, E] @ eigvecs
        D_new = new_1st_order_coeffs[0]
        E_new = new_1st_order_coeffs[1]


        # 平方完成
        F_new = (C_new * D_new**2 + A_new * E_new**2 - 4 * A_new * C_new * F) / (4 * A_new * C_new)
        print('A_new:', A_new)
        print('B_new:', B_new)
        print('C_new:', C_new)
        print('D_new:', D_new)
        print('E_new:', E_new)
        print('F_new:', F_new)
        print(' ')
        a = np.sqrt(F_new / C_new)
        b = np.sqrt(np.abs(F_new / A_new))

        x0 = (D_new / (2 * A_new))
        y0 = (E_new / (2 * C_new))


        return a, b, -x0, y0



#* Extract peaks
peak_indeces = []

for i in range(data.shape[1]):
    extract_peak(data[:, i], i)


#* １つのトレースに複数のピークがある場合，最初のピークのみを抽出
#unique_trace = np.unique(np.array(idx_trace))
#overlap = int(len(idx_trace) / len(unique_trace))
#idx_trace = idx_trace[::overlap] # index, not in m
#idx_time = idx_time[::overlap] # index, not in sec

# 37.5 ns以内のピークのみを抽出
max_time_idx = 37.5 * 1e-9 / dt
peak_indeces = np.array(peak_indeces)
peak_indeces = peak_indeces[peak_indeces[:, 1] < max_time_idx]

hyperbola_x = np.array(peak_indeces[:, 0]) * antenna_step + antenna_start # [m]
hyperbola_t = np.array(peak_indeces[:, 1]) * dt / 1e-9 # [ns]


#* Fit the hyperbola
c = 0.299792458 # [m/ns]
if args.fix == 'er':
    epsilon_r = [3]
    #R = np.arange(0, 0.5, 0.01)
    R = np.arange(0, 1.81, 0.15) # [m]
elif args.fix == 'R':
    epsilon_r = np.arange(1, 10, 0.5)
    R = [1.5] # [m]
else:
    epsilon_r = np.arange(1, 10, 1)
    R = np.arange(0, 1.5, 0.15) # [m]

t0 = np.min(hyperbola_t)
x0 = hyperbola_x[np.argmin(hyperbola_t)+1] # [cm]
print(f't0: {t0} ns, x0: {x0} m')

#* Calculate the hyperbola
hyperbola_list = []
er_R_list = []
for i in tqdm(range(len(epsilon_r)), desc='Calculating hyperbola'):
    for j in range(len(R)):
        v = c / np.sqrt(epsilon_r[i]) # [m/ns]
        hyperbola = 2 / v * (np.sqrt((v * t0 / 2 + R[j])**2 + (hyperbola_x - x0)**2) - R[j])
        hyperbola_list.append(hyperbola)
        er_R_list.append([epsilon_r[i], R[j]])
er_R_list = np.array(er_R_list)


#* Calculate the most similar hyperbola
min_diff = np.inf
min_idx = 0
for i in range(len(hyperbola_list)):
    diff = np.sum(np.abs(hyperbola_list[i] - hyperbola_t))
    if diff < min_diff:
        min_diff = diff
        min_idx = i
print('min_idx:', min_idx)


"""
x = np.array(idx_trace) * antenna_step + antenna_start # [m]
t = np.array(idx_time) * dt / 1e-9 # [ns]
fit_coefficients = fit_hyperbola_shihab(x, t)
print('fit_coefficients:', fit_coefficients)
a_fit, b_fit, x0_fit, y0_fit = extract_hyperbola_parameters_vertical(fit_coefficients)
print("a_fit:", a_fit)
print("b_fit:", b_fit)
print("x0_fit:", x0_fit)
print("y0_fit:", y0_fit)

t0_fit = a_fit + y0_fit # [ns]
v_estimated = 2 * b_fit / a_fit # [m/ns]
R_estimated = (a_fit - t0_fit) * b_fit / a_fit # [m]

#* Estimate the epsilon_r
c = 299792458 # [m/s]
epsilon_r = (c * 1e-9 / v_estimated)**2

#* Save estimated parameters to txt file
param_names = ['a', 'b', 'x0', 'y0', 't0 [ns]', 'v [m/ns]', 'R [m]', 'epsilon_r']
params = [a_fit, b_fit, x0_fit, y0_fit, t0_fit, v_estimated, R_estimated, epsilon_r]
save_params = np.array([param_names, params]).T
np.savetxt(os.path.join(output_dir, 'fitted_params.txt'), save_params, fmt='%s')

print(f'estimated t0 = {t0_fit} ns')
print(f'estimated v = {v_estimated} m/ns')
print(f'estimated R = {R_estimated} m')
print(f'estimated epsilon_r = {epsilon_r}')

hyperbola_fit = 2 / v_estimated * (np.sqrt((v_estimated * t0_fit / 2 + R_estimated)**2 + (x - x0_fit)**2) - R_estimated)
"""





#* Plot the peak on the B-scan
plt.figure(figsize=(20, 15), tight_layout=True)
im = plt.imshow(data, cmap='gray', aspect='auto',
                extent=[antenna_start,  antenna_start + data.shape[1] * antenna_step,
                data.shape[0] * dt / 1e-9, 0],
                vmin=-np.amax(np.abs(data)/100), vmax=np.amax(np.abs(data)/100)
                )
plt.scatter(np.array(peak_indeces[:, 0]) * antenna_step + antenna_start, np.array(peak_indeces[:, 1]) * dt / 1e-9, c='r', s=20, marker='x', label='Peak')
#plt.plot(x, hyperbola_fit, c='b', lw=2, linestyle='--', label='Fitting')

#* Plot fitting hyperbola
# cmapに従って色を変える
colors = plt.cm.jet(np.linspace(0, 1, len(hyperbola_list)))
for i in range(len(hyperbola_list)):
    plt.plot(hyperbola_x, hyperbola_list[i], lw=2, linestyle='--',
                    label=f'er={er_R_list[i, 0]:.3}, R={er_R_list[i, 1]:.3}', c=colors[i])

plt.xlabel('x [m]', fontsize=20)
plt.ylabel('Time [ns]', fontsize=20)
plt.ylim(40, 20)
if args.fix == 'er':
    plt.title(f'er={epsilon_r[0]}, best fit R={er_R_list[min_idx, 1]:.3}', fontsize=20)
elif args.fix == 'R':
    plt.title(f'best fit er={er_R_list[min_idx, 0]}, R={R[0]}', fontsize=20)
plt.tick_params(labelsize=18)
plt.grid(which='both', axis='both', linestyle='-.')
plt.legend(fontsize=16)

delvider = axgrid1.make_axes_locatable(plt.gca())
cax1 = delvider.append_axes('right', size='5%', pad=0.2)
cbar = plt.colorbar(im, cax=cax1)
cbar.ax.tick_params(labelsize=18)
cbar.set_label('Amplitude', fontsize=20)

#* Colorbar for colors
"""
cax2 = delvider.append_axes('bottom', size='5%', pad=1)
cbar2 = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), cax=cax2, orientation='horizontal')
if args.fix == 'er':
    cbar2.set_label('R [m]', fontsize=18)
elif args.fix == 'R':
    cbar2.set_label('er', fontsize=18)
cbar2.ax.tick_params(labelsize=16)
cbar2.set_ticks(np.linspace(0, np.max(epsilon_r), len(hyperbola_list)))
cbar2.set_ticklabels(np.round(er_R_list[:, 1], 2))
"""


if args.fix == 'er':
    plt.savefig(os.path.join(output_dir, f'er{epsilon_r[0]}.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, f'er{epsilon_r[0]}.pdf'), format='pdf', dpi=600)
elif args.fix == 'R':
    plt.savefig(os.path.join(output_dir, f'R{R[0]}.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, f'R{R[0]}.pdf'), format='pdf', dpi=600)
else:
    plt.savefig(os.path.join(output_dir, 'fitting.png'), format='png', dpi=120)
    plt.savefig(os.path.join(output_dir, 'fitting.pdf'), format='pdf', dpi=600)
plt.show()