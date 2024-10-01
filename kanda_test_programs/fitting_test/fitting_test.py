import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, inv

def fit_ort_hyperbola(x, y):
    """
    x と y は同じ長さのベクトルであると仮定
    MATLABのコードをPythonに翻訳した関数で、データに双曲線をフィッティングします。
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    x2 = x ** 2
    y2 = y ** 2
    xy = x * y

    # デザイン行列の構築
    D = np.column_stack([y2 - x2, xy, x, y, np.ones_like(x)])

    # 特異値分解
    U, S, Vt = svd(D, full_matrices=False)

    # 最適な解を選択
    result = Vt[-1, :]

    # 円錐曲線の係数への変換
    # orthConic = [A, B, C, D, E, F]
    #orthConic = np.concatenate(([-result[0], result[1], result[0], result[2:]]))
    orthConic = np.concatenate(([-result[0]], [result[1]], [result[0]], result[2:]))


    # 残差の標準偏差の計算（必要に応じて使用）
    fitStd = S[-1] / np.sqrt(len(x))

    return orthConic, fitStd

def extract_hyperbola_parameters_vertical(coeffs):
    """
    フィッティングで得られた係数から双曲線のパラメータを抽出します。
    双曲線が縦軸（y軸、t軸）方向に開いていると仮定しています。
    """
    A, B, C, D, E, F = coeffs

    # 回転角をゼロと仮定（B = 0）
    B = 0

    # 中心座標の計算
    x0 = -D / (2 * A)
    y0 = -E / (2 * C)

    # 定数項の再計算
    F_center = A * x0**2 + C * y0**2 + D * x0 + E * y0 + F

    # 軸長の計算（開口が y 軸方向の場合）
    a = np.sqrt(np.abs(F_center / C))
    b = np.sqrt(np.abs(-F_center / A))

    # 回転角はゼロ
    theta = 0

    return a, b, x0, y0, theta

def estimate_v_R_from_hyperbola_params(a, b, t0):
    """
    双曲線のパラメータから物理パラメータ v（速度）と R（距離）を推定します。
    """
    # v の計算
    v_estimated = 2 * b / a

    # R の計算
    R_estimated = b - (v_estimated * t0) / 2

    return v_estimated, R_estimated

# データ生成（テスト用）
t0_true = 70  # [ns]
x0_true = 0   # [m]
v_true = 0.15 # [m/ns]
R_true = 0.30 # [m]

def hyperbola_model(x, t0, x0, v, R):
    return 2 / v * (np.sqrt((v * t0 / 2 + R)**2 + (x - x0)**2) - R)

x_data = np.linspace(-2.5, 2.5, 100)
x_data = np.arange(x0_true - 2, x0_true + 2, 0.036)
t_data = hyperbola_model(x_data, t0_true, x0_true, v_true, R_true)

# ノイズの追加
noise = np.random.normal(0, 0.1, size=t_data.shape)  # ノイズレベルを調整
#t_data_noisy = t_data + noise

# フィッティング
coeffs, fitStd = fit_ort_hyperbola(x_data, t_data)
print('Fitted coefficients:', coeffs)

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
    solution = solve_y(coeffs, x_val)
    if solution:
        y1, y2 = solution
        y_pos.append(y1)
        y_neg.append(y2)
        x_vals.append(x_val)

# パラメータの抽出
a_fit, b_fit, x0_fit, t0_fit, theta_fit = extract_hyperbola_parameters_vertical(coeffs)
print(f'Fitted parameters: a={a_fit}, b={b_fit}, x0={x0_fit}, t0={t0_fit}, theta={theta_fit}')

# 物理パラメータの推定
v_estimated, R_estimated = estimate_v_R_from_hyperbola_params(a_fit, b_fit, t0_fit)
print(f'Estimated parameters: v={v_estimated} [m/ns], R={R_estimated} [m]')

# フィッティング結果のプロット
t_fit = hyperbola_model(x_data, t0_fit, x0_fit, v_estimated, R_estimated)

# プロット
plt.scatter(x_data, t_data, color='black', s=10, marker='x', label='Data')
plt.plot(x_data, t_fit, color='red', label='Fitted Model')
plt.plot(x_vals, y_neg, color='blue')

plt.grid(True)
plt.xlabel('x [m]')
plt.ylabel('t [ns]')
plt.ylim(np.max(t_data) + 10, 0)
plt.legend()
plt.show()
