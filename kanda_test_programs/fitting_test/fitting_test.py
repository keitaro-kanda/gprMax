import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, eig, inv



# データ生成（テスト用）
t0_true = 70  # [ns]
x0_true = 0   # [m]
v_true = 0.299792458 / np.sqrt(4.0) # [m/ns]
R_true = 0.15 # [m], 半径

def hyperbola_model(x, t0, x0, v, R):
    return 2 / v * (np.sqrt((v * t0 / 2 + R)**2 + (x - x0)**2) - R)


x_data = np.arange(x0_true-2.50, x0_true+2.50, 0.036) # CE-4, [cm]
t_data = hyperbola_model(x_data, t0_true, x0_true, v_true, R_true)
noise = np.random.normal(0, 0.1, size=t_data.shape)  # ノイズレベルを調整
#t_data = t_data + noise



def fit_ellipse_and_hyperbola(x, y):
    # xとyはカラムベクトルであることが仮定される
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y

    # 設計行列と散布行列を設定
    D1 = np.column_stack([x2, xy, y2])
    D2 = np.column_stack([x, y, np.ones_like(x)])

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # S3のランクをテスト
    Us3, Ss3, Vs3 = svd(S3)
    condNrs = Ss3 / Ss3[0]

    epsilon = 1e-10
    if condNrs[2] < epsilon:
        print('Warning: S3 is degenerate')
        return None, None

    # 制約行列とその逆行列を定義
    C = np.array([[0, 0, -2], [0, 1, 0], [-2, 0, 0]])
    Ci = inv(C)

    # 一般化固有ベクトル問題を設定して解く
    T = -inv(S3) @ S2.T
    S = Ci @ (S1 - S2 @ T)

    evals, evec = eig(S)
    print('evec: \n', evec)

    # 制約値を評価してソート
    cond = evec[1, :] ** 2 - 4 * evec[0, :] * evec[2, :]
    condVals = np.sort(cond)
    print('condVals: \n', condVals)

    """
    # Compute the discriminant for each eigenvector
    cond = evec[1, :] ** 2 - 4 * evec[0, :] * evec[2, :]

    # Find indices where the conic is a hyperbola (cond > 0)
    hyperbola_indices = np.where(cond > 0)[0]

    if len(hyperbola_indices) == 0:
        print("No hyperbola solution found.")
        return None, None

    # Select the index with the largest positive discriminant
    idx = hyperbola_indices[np.argmax(cond[hyperbola_indices])]

    # Get the corresponding eigenvector
    alpha1 = evec[:, idx]

    # Compute the first-order coefficients
    alpha2 = T @ alpha1

    # Combine to get the full set of coefficients
    hyperbola = np.concatenate([alpha1, alpha2])
    """



    """
    # 双曲線解を取得
    idx = np.argmax(cond)  # 最大の cond を持つものを選ぶ
    alpha1 = evec[:, idx] # 双曲線の係数，固有ベクトルを取得
    alpha2 = T @ alpha1 # 1次項の係数を計算
    hyperbola = np.concatenate([alpha1, alpha2])
    """

    # 双曲線解を取得
    possibleHs = condVals[1:2] + condVals[0]
    possibleHs = possibleHs[possibleHs > 0]
    minDiffAt = np.argmin(np.abs(possibleHs))
    minDiffAt = np.argmin(possibleHs)
    print('minDiffAt:', minDiffAt)
    alpha1 = evec[:, minDiffAt + 1]
    print('alpha1: \n', alpha1)
    alpha2 = T @ alpha1
    hyperbola = np.concatenate([alpha1, alpha2])

    alpha01 = evec[:, 0]
    alpha02 = T @ alpha01
    hyperbola0 = np.concatenate([alpha01, alpha02])

    alpha11 = evec[:, 1]
    alpha12 = T @ alpha11
    hyperbola1 = np.concatenate([alpha11, alpha12])

    alpha21 = evec[:, 2]
    alpha22 = T @ alpha21
    hyperbola2 = np.concatenate([alpha21, alpha22])


    return hyperbola0, hyperbola1, hyperbola2



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
        print('eigvals:', eigvals)
        print('eigvecs:', eigvecs)


        # x, yをX, Yに変換
        new_2nd_order_coeffs = eigvecs.T @ M @ eigvecs
        A_new = new_2nd_order_coeffs[0, 0]
        B_new = new_2nd_order_coeffs[0, 1]
        C_new = new_2nd_order_coeffs[1, 1]
        print('A_new:', A_new)
        print('B_new:', B_new)
        print('C_new:', C_new)
        print(' ')

        # 1次項の係数を計算
        new_1st_order_coeffs = [D, E] @ eigvecs
        D_new = new_1st_order_coeffs[0]
        E_new = new_1st_order_coeffs[1]
        print('D_new:', D_new)
        print('E_new:', E_new)

        # 平方完成
        F_new = (C_new * D_new**2 + A_new * E_new**2 - 4 * A_new * C_new * F) / (4 * A_new * C_new)
        print('F_new:', F_new)
        a = np.sqrt(np.abs(F_new / C_new))
        b = np.sqrt(np.abs(F_new / A_new))
        print('a:', a)
        print('b:', b)

        x0 = - (D_new / (2 * A_new))
        y0 = - (E_new / (2 * C_new))


        return a, b, x0, y0



def solve_y(parameter_array, x_val):
    a, b, c, d, e, f = parameter_array
    # 二次方程式 a y^2 + (bx + e)y + (cx^2 + dx + f) = 0 を解く
    A = c
    B = b * x_val + e
    C = a * x_val**2 + d * x_val + f

    # 判別式を計算
    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        return None  # 実数解が存在しない場合

    # 二次方程式の解
    y1 = (-B + np.sqrt(discriminant)) / (2 * A)
    y2 = (-B - np.sqrt(discriminant)) / (2 * A)

    return y1, y2


# 解を求める
def calc_y(parameter_array, x_val):
    y_pos = []
    y_neg = []

    for x_val in x_data:
        solution = solve_y(parameter_array, x_val)
        if solution:
            y1, y2 = solution
            y_pos.append(y1)
            y_neg.append(y2)

    return  y_pos, y_neg




# フィッティング
hyperbola_fit0, hyperbola_fit1, hyperbola_fit2 = fit_ellipse_and_hyperbola(x_data, t_data)

print('------------------------------------')
a_fit0, b_fit0, x0_fit0, y0_fit0 = extract_hyperbola_parameters_vertical(hyperbola_fit0)
print(f'Fitted parameters: a={a_fit0}, b={b_fit0}, x0={x0_fit0}, y0={y0_fit0}')
if not a_fit0 is None:
    t0_fit0 = a_fit0 - y0_fit0
    print('t0_fit0:', t0_fit0)

    v_estimated0 = 2 * b_fit0 / a_fit0
    R_estimated0 = (a_fit0 - t0_fit0) * b_fit0 / a_fit0
    print(' ')

    y_pos0, y_neg0 = calc_y(hyperbola_fit0, x_data)

print('------------------------------------')
a_fit1, b_fit1, x0_fit1, y0_fit1 = extract_hyperbola_parameters_vertical(hyperbola_fit1)
print(f'Fitted parameters: a={a_fit1}, b={b_fit1}, x0={x0_fit1}, y0={y0_fit1}')
if not a_fit1 is None:
    t0_fit1 = a_fit1 - y0_fit1
    print('t0_fit1:', t0_fit1)

    v_estimated1 = 2 * b_fit1 / a_fit1
    R_estimated1 = (a_fit1 - t0_fit1) * b_fit1 / a_fit1
    print(f'Estimated parameters: R={R_estimated1} [m], v={v_estimated1} [m/ns]')
    print(' ')

    t_fit1 = hyperbola_model(x_data, t0_fit1, x0_fit1, v_estimated1, R_estimated1)

    y_pos1, y_neg1 = calc_y(hyperbola_fit1, x_data)

print('------------------------------------')
a_fit2, b_fit2, x0_fit2, y0_fit2 = extract_hyperbola_parameters_vertical(hyperbola_fit2)
print(f'Fitted parameters: a={a_fit2}, b={b_fit2}, x0={x0_fit2}, y0={y0_fit2}')
if not a_fit2 is None:
    t0_fit2 = a_fit2 - y0_fit2
    print('t0_fit2:', t0_fit2)

    v_estimated2 = 2 * b_fit2 / a_fit2
    R_estimated2 = (a_fit2 - t0_fit2) * b_fit2 / a_fit2
    print(f'Estimated parameters: R={R_estimated2} [m], v={v_estimated2} [m/ns]')
    print(' ')

    t_fit2 = hyperbola_model(x_data, t0_fit2, x0_fit2, v_estimated2, R_estimated2)

    y_pos2, y_neg2 = calc_y(hyperbola_fit2, x_data)




#* Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x_data, t_data , color='black', s=10, marker='x', label='Data')

ax.plot(x_data, t_fit1, color='red', label='fit 1')
ax.plot(x_data, y_pos1, color='green', label='y_pos', linestyle='--')

ax.plot(x_data, t_fit2, color='blue', label='fit 2')
ax.plot(x_data, y_pos2, color='purple', label='y_pos', linestyle='--')

ax.grid(True)
ax.set_xlabel('x [m]')
ax.set_ylabel('t [ns]')
ax.legend()

ax.set_ylim(t0_true + 10, 0)
plt.show()