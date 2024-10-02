import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, eig, inv

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

    # 制約値を評価してソート
    cond = evec[1, :] ** 2 - 4 * (evec[0, :] * evec[2, :])
    condVals = np.sort(cond)


    # 双曲線解を取得
    #idx = np.argmax(cond)  # 最大の cond を持つものを選ぶ
    #alpha1 = evec[:, idx] # 双曲線の係数，固有ベクトルを取得
    #alpha2 = T @ alpha1 # 1次項の係数を計算
    #hyperbola = np.concatenate([alpha1, alpha2])

    # 双曲線解を取得
    possibleHs = condVals[1:3] + condVals[0]
    minDiffAt = np.argmin(np.abs(possibleHs))
    alpha1 = evec[:, minDiffAt + 1]
    alpha2 = T @ alpha1
    hyperbola = np.concatenate([alpha1, alpha2])

    return hyperbola



def extract_hyperbola_parameters_vertical(coeffs):
    A, B, C, D, E, F = coeffs

    # 回転角の計算
    theta = 0.5 * np.arctan2(B, A - C)

    # 回転角が小さい場合はゼロと仮定
    if np.isclose(theta, 0):
        theta = 0
        print('theta is close to zero.')

    # 行列形式での計算
    M = np.array([[A, B/2], [B/2, C]])
    offset = np.array([D/2, E/2])

    # 中心座標の計算
    center = -np.linalg.inv(M) @ offset
    x0, y0 = center

    # 係数の再計算
    F_center = A*x0**2 + B*x0*y0 + C*y0**2 + D*x0 + E*y0 + F
    print('F_center:', F_center)

    # 軸長の計算（開口が y 軸方向の場合）
    numerator = 2 * F_center
    denom_y = A * np.sin(theta)**2 - B * np.sin(theta) * np.cos(theta) + C * np.cos(theta)**2
    denom_x = A * np.cos(theta)**2 + B * np.sin(theta) * np.cos(theta) + C * np.sin(theta)**2
    print('denom_y:', denom_y)

    a = np.sqrt(np.abs(numerator / denom_y))
    b = np.sqrt(np.abs(numerator / denom_x))

    return a, b, x0, y0, theta

# データ生成（テスト用）
t0_true = 50  # [ns]
x0_true = 0   # [m]
v_true = 0.15 # [m/ns]
v_true = 0.299792458 / np.sqrt(4.0) # [m/ns]
R_true = 0.30 # [m]

def hyperbola_model(x, t0, x0, v, R):
    return 2 / v * (np.sqrt((v * t0 / 2 + R)**2 + (x - x0)**2) - R)

x_data = np.linspace(-2.5, 2.5, 100)
x_data = np.arange(-2.5, 2.5, 0.036) # CE-4
t_data = hyperbola_model(x_data, t0_true, x0_true, v_true, R_true)
noise = np.random.normal(0, 0.1, size=t_data.shape)  # ノイズレベルを調整
t_data_noisy = t_data + noise

# フィッティング
hyperbola_fit = fit_ellipse_and_hyperbola(x_data, t_data_noisy)

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

    a_fit, b_fit, x0_fit, y0_fit, theta_fit = extract_hyperbola_parameters_vertical(hyperbola_fit)
    print(f'Fitted parameters: a={a_fit}, b={b_fit}, x0={x0_fit}, t0={y0_fit}, theta={theta_fit}')
    print(' ')

    t0_fit = y0_fit - a_fit

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
    fig, ax = plt.subplots()
    ax.scatter(x_data, t_data_noisy , color='black', s=10, marker='x', label='Data')
    ax.plot(x_data, t_fit, color='red', label='Best fit')
    ax.plot(x_data, y_pos, color='blue', label='Hyperbola')
    ax.plot(x_data, y_asymptote_pos, color='green', label='Asymptote')
    ax.plot(x_data, y_asymptote_neg, color='green')

    ax.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('t [ns]')
    ax.legend()
    plt.show()