import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import os



# データ生成（テスト用）
t0_true = 110  # [ns]
x0_true = 3   # [m]
c = 0.299792458 # [m/s]
epsilon_r_true = 3.0
v_true = c / np.sqrt(epsilon_r_true) # [m/ns]
R_true = 0.15 # [m], 半径


def hyperbola_model(x, t0, x0, v, R):
    return 2 / v * (np.sqrt((v * t0 / 2 + R)**2 + (x - x0)**2) - R)


x_data = np.arange(x0_true-2.50, x0_true+2.50, 0.036) # CE-4, [cm]
t_data = hyperbola_model(x_data, t0_true, x0_true, v_true, R_true)
noise_level = 0.0
noise = np.random.normal(0, noise_level, size=t_data.shape)  # ノイズレベルを調整
t_data = t_data + noise


output_dir = os.path.join('kanda_test_programs/fitting_shihab2004' + f'/t0{t0_true}_x0{x0_true}_er{epsilon_r_true}_R{R_true}_noise{noise_level}')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    eigvec_normalized = []
    for i in range(eigvec.shape[1]):
        a_i = eigvec[:, i]
        constraint_value = a_i.T @ C @ a_i # a^T C a

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
            print('normalized eigenvector:', a_i_normalized)
            print('constraint_check:', constraint_check)
            print(' ')
            eigvec_normalized.append(a_i_normalized)

        else:
            print(f'Eigenvector {i} has a constraint value too small for reliable normalization.')

    """
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
        print('Valid eigenvector found.')
        print(' ')
    else:
        print('No valid eigenvector found.')
        print(' ')
    """
    eigvec_normalized = np.array(eigvec_normalized).T
    print('eigvec_normalized: \n', eigvec_normalized)
    index_Bmin = np.argmin(np.abs(eigvec_normalized[1, :]))
    print('index_Bmin:', index_Bmin)
    best_eigvec_normalized = eigvec_normalized[:, index_Bmin]

    # Return the coefficients of the fitted hyperbola
    return best_eigvec_normalized



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
        a = np.sqrt(np.abs(F_new / C_new))
        b = np.sqrt(np.abs(F_new / A_new))

        x0 = (D_new / (2 * A_new))
        y0 = (E_new / (2 * C_new))


        return a, b, np.abs(x0), y0




#* Fitting
fit_coefficients = fit_hyperbola_shihab(x_data, t_data)
print('fit_coefficients:', fit_coefficients)
print(' ')


#* Extract hyperbola parameters
a_fit, b_fit, x0_fit, y0_fit = extract_hyperbola_parameters_vertical(fit_coefficients)
print("a_fit:", a_fit)
print("b_fit:", b_fit)
print("x0_fit:", x0_fit)
print("y0_fit:", y0_fit)


#* Esimate the velocity and the radius
if y0_fit < 0:
    t0_fit = a_fit + y0_fit # [ns]
elif y0_fit > 0 and y0_fit > a_fit:
    t0_fit = a_fit + y0_fit # [ns]
else:
    t0_fit = a_fit - y0_fit
v_estimated = 2 * b_fit / a_fit # [m/ns]
R_estimated = (a_fit - t0_fit) * b_fit / a_fit # [m]
epsilon_r_estimated = (c / v_estimated)**2
print('t0_fit:', t0_fit)
print('v_estimated:', v_estimated)
print('R_estimated:', R_estimated)
print('epsilon_r_estimated:', epsilon_r_estimated)



#* save parameters to txt file
params_name = ['t0_true', 'x0_true', 'epsilon_r_true', 'v_true', 'R_true', 'noise_level','t0_fit', 'v_estimated', 'R_estimated', 'epsilon_r_estimated']
params = [t0_true, x0_true, epsilon_r_true, v_true, R_true, 'noise_level', t0_fit, v_estimated, R_estimated, epsilon_r_estimated]
with open(output_dir + '/parameters.txt', 'w') as f:
    for name, param in zip(params_name, params):
        f.write(f'{name}: {param}\n')


#* Calculate the fitted hyperbola
hyperbola_fit = 2 / v_estimated * (np.sqrt((v_estimated * t0_fit / 2 + R_estimated)**2 + (x_data - x0_fit)**2) - R_estimated)



#* Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x_data, t_data , color='black', s=10, marker='x', label='Data')

ax.plot(x_data, hyperbola_fit, color='red', label='Fitting', linestyle='--')

ax.grid(True)
ax.set_title(f'setting: er={epsilon_r_true}, R={R_true}, estimated: er={epsilon_r_estimated:.4}, R={R_estimated:.4}', fontsize=20)
ax.set_xlabel('x [m]', fontsize=20)
ax.set_ylabel('t [ns]', fontsize=20)
ax.legend(fontsize=18)

ax.set_ylim(np.max([np.max(t_data), np.max(hyperbola_fit)]) + 3, np.min([np.min(t_data), np.min(hyperbola_fit)]) - 3)

plt.tight_layout()
plt.savefig(output_dir + '/fitting_result.png', format='png', dpi=120)
plt.savefig(output_dir + '/fitting_result.pdf', format='pdf', dpi=600)
plt.show()