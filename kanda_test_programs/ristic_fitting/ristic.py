import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# サンプルデータ（実際のデータを使用する場合は、ここを置き換えてください）
# x_data: レーダー位置（距離）
# t_data: 反射時間（時間）
x_data = np.linspace(-10, 10, 100)
t_data = np.sqrt((x_data - 0)**2 + 5**2)  # 仮想的なハイパーボラデータ

# データ生成（テスト用）
t0_true = 50  # [ns]
x0_true = 0   # [m]
v_true = 0.15 # [m/ns]
v_true = 0.299792458 / np.sqrt(4.0) # [m/ns]
R_true = 0.30 # [m]

# ステップ 5.1: ハイパーボラの頂点座標 (x0, t0) の推定

def model_step1(x0, t0, x):
    return np.sqrt((x - x0)**2 + t0**2)

def cost_function_step1(params):
    x0, t0 = params
    residuals = model_step1(x0, t0, x_data) - t_data
    cost = 0.5 * np.sum(residuals**2)
    return cost

def gradient_step1(params):
    x0, t0 = params
    residuals = model_step1(x0, t0, x_data) - t_data
    dC_dx0 = np.sum(residuals * ((x_data - x0) / model_step1(x0, t0, x_data)))
    dC_dt0 = np.sum(residuals * (t0 / model_step1(x0, t0, x_data)))
    gradient = np.array([dC_dx0, dC_dt0])
    return gradient

# 勾配降下法の実装
def gradient_descent_step1(initial_params, learning_rate=0.001, max_iter=1000, tol=1e-6):
    params = np.array(initial_params)
    for i in range(max_iter):
        grad = gradient_step1(params)
        params_new = params - learning_rate * grad
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

# 初期推定値
initial_guess_step1 = [0, np.min(t_data)]

# 最適化の実行
x0_est, t0_est = gradient_descent_step1(initial_guess_step1)

print("ステップ 5.1 の推定結果:")
print(f"x0 = {x0_est}")
print(f"t0 = {t0_est}")

# ステップ 5.2: 境界速度 v0 の推定

def model_step2(v0, R0, x):
    term = (v0 * t0_est / 2 + R0)
    return (2 / v0) * np.sqrt(term**2 + (x - x0_est)**2) - R0

def cost_function_step2(params):
    v0, R0 = params
    residuals = model_step2(v0, R0, x_data) - t_data
    cost = 0.5 * np.sum(residuals**2)
    return cost

def gradient_step2(params):
    v0, R0 = params
    residuals = model_step2(v0, R0, x_data) - t_data
    term = (v0 * t0_est / 2 + R0)
    denom = np.sqrt(term**2 + (x_data - x0_est)**2)
    
    # 勾配の計算
    dC_dv0 = np.sum(residuals * (-2 / v0**2) * denom + (2 / v0) * ((v0 * t0_est / 2 + R0) * (t0_est / 2)) / denom)
    dC_dR0 = np.sum(residuals * ((2 / v0) * (term / denom) - 1))
    gradient = np.array([dC_dv0, dC_dR0])
    return gradient

# 勾配降下法の実装
def gradient_descent_step2(initial_params, learning_rate=1e-9, max_iter=1000, tol=1e-6):
    params = np.array(initial_params)
    for i in range(max_iter):
        grad = gradient_step2(params)
        params_new = params - learning_rate * grad
        if np.linalg.norm(params_new - params) < tol:
            break
        params = params_new
    return params

# 初期推定値
initial_guess_step2 = [0.01, 0.1]  # v0 の初期値は適切な値に調整してください

# 最適化の実行
v0_est, R0_est = gradient_descent_step2(initial_guess_step2)

print("\nステップ 5.2 の推定結果:")
print(f"v0 = {v0_est}")
print(f"R0 = {R0_est}")

# ステップ 5.3: v と R の同時推定

if v0_est < 0.3:
    v_max = v0_est
else:
    v_max = 0.3
v_min = v0_est * 0.5  # v の下限（適切な値に調整してください）
v_values = np.linspace(v_max, v_min, 50)
foo_values = []
R_values = []

for v in v_values:
    def model_step3(R, x):
        term = (v * t0_est / 2 + R)
        return (2 / v) * np.sqrt(term**2 + (x - x0_est)**2) - R

    def cost_function_step3(R):
        residuals = model_step3(R, x_data) - t_data
        cost = 0.5 * np.sum(residuals**2)
        return cost

    def gradient_step3(R):
        residuals = model_step3(R, x_data) - t_data
        term = (v * t0_est / 2 + R)
        denom = np.sqrt(term**2 + (x_data - x0_est)**2)
        dC_dR = np.sum(residuals * ((2 / v) * (1 + term / denom) - 1))
        return dC_dR

    # 勾配降下法の実装
    def gradient_descent_step3(initial_R, learning_rate=0.001, max_iter=1000, tol=1e-6):
        R = initial_R
        for i in range(max_iter):
            grad = gradient_step3(R)
            R_new = R - learning_rate * grad
            if abs(R_new - R) < tol:
                break
            R = R_new
        return R

    # R の初期推定値
    initial_R = R0_est

    # 最適化の実行
    R_est = gradient_descent_step3(initial_R)
    R_values.append(R_est)

    # 勾配の計算と foo の評価
    residuals = model_step3(R_est, x_data) - t_data
    grad = gradient_step3(R_est)
    foo = np.linalg.norm(grad, ord=np.inf)
    foo_values.append(foo)

# 最適な foo を持つ v を選択
min_foo_index = np.argmin(foo_values)
v_opt = v_values[min_foo_index]
R_opt = R_values[min_foo_index]

print("\nステップ 5.3 の推定結果:")
print(f"最適な v = {v_opt}")
print(f"対応する R = {R_opt}")

# 結果のプロット
plt.figure(figsize=(10, 6))
plt.scatter(x_data, t_data, label='観測データ', color='blue')
t_fit = model_step2(v_opt, R_opt, x_data)
plt.plot(x_data, t_fit, label='フィッティング結果', color='red')
plt.xlabel('x')
plt.ylabel('t')
plt.legend()
plt.title('地中レーダーデータの双曲線フィッティング')
plt.show()