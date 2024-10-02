import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, eig, inv



# データ生成（テスト用）
t0_true = 70  # [ns]
x0_true = 0   # [m]
v_true = 0.299792458 / np.sqrt(4.0) # [m/ns]
R_true = 0.30 # [m]

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
    cond = evec[1, :] ** 2 - 4 * (evec[0, :] * evec[2, :])
    print('cond:', cond)
    condVals = np.sort(cond)
    #condVals = cond
    print('condVals:', condVals)


    """
    # 双曲線解を取得
    idx = np.argmax(cond)  # 最大の cond を持つものを選ぶ
    alpha1 = evec[:, idx] # 双曲線の係数，固有ベクトルを取得
    alpha2 = T @ alpha1 # 1次項の係数を計算
    hyperbola = np.concatenate([alpha1, alpha2])
    """

    # 双曲線解を取得
    possibleHs = condVals[1:3] + condVals[0]
    possibleHs = possibleHs[possibleHs > 0]
    minDiffAt = np.argmin(np.abs(possibleHs))
    minDiffAt = np.argmin(possibleHs)
    print('minDiffAt:', minDiffAt)
    alpha1 = evec[:, minDiffAt + 1]
    print('alpha1: \n', alpha1)
    alpha2 = T @ alpha1
    hyperbola = np.concatenate([alpha1, alpha2])


    return hyperbola



def extract_hyperbola_parameters_vertical(coeffs):
    A, B, C, D, E, F = coeffs

    # 双曲線の判別式を計算
    discriminant = B**2 - 4 * A * C
    if discriminant <= 0:
        print("This conic is not a hyperbola.")
        return None


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



# フィッティング
hyperbola_fit = fit_ellipse_and_hyperbola(x_data, t_data)

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
y_pos = []
y_neg = []
x_vals = []

x_fit = np.arange(-5, 5, 0.01)

for x_val in x_data:
    solution = solve_y(hyperbola_fit, x_val)
    if solution:
        y1, y2 = solution
        y_pos.append(y1)
        y_neg.append(y2)
        x_vals.append(x_val)
if hyperbola_fit is None:
    print("Hyperbola fitting failed.")
else:
    print('Fitted coefficients:', hyperbola_fit)
    print(' ')


    a_fit, b_fit, x0_fit, y0_fit = extract_hyperbola_parameters_vertical(hyperbola_fit)
    print(f'Fitted parameters: a={a_fit}, b={b_fit}, x0={x0_fit}, t0={y0_fit}')
    print(' ')

    t0_fit = a_fit - y0_fit
    print('t0_fit:', t0_fit)
    print('x0_fit:', x0_fit)

    #* Asymptotes of hyperbola
    y_asymptote_pos = (b_fit / a_fit) * (x_data - x0_fit) + y0_fit
    y_asymptote_neg = -(b_fit / a_fit) * (x_data - x0_fit) + y0_fit

    #* Calculate v and R
    v_estimated = 2 * b_fit / a_fit
    R_estimated = (a_fit - t0_fit) * b_fit / a_fit
    print(f'Estimated parameters: R={R_estimated} [m], v={v_estimated} [m/ns]')

    #* Make best fit hyperbola

    t_fit = hyperbola_model(x_data, t0_fit, x0_fit, v_estimated, R_estimated)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_data, t_data , color='black', s=10, marker='x', label='Data')
    ax.plot(x_data, t_fit, color='red', label='fit')
    #ax.plot(x_data, y_neg, color='blue', label='y_neg')
    ax.plot(x_data, y_pos, color='green', label='y_pos', linestyle='--')

    ax.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('t [ns]')
    ax.legend()
    plt.show()