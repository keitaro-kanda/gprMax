import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# サンプルデータ（実際のデータを使用する場合は、ここを置き換えてください）
# x_data: レーダー位置（距離）
# t_data: 反射時間（時間）
#x_data = np.linspace(-10, 10, 100)
#t_data = np.sqrt((x_data - 0)**2 + 5**2)  # 仮想的なハイパーボラデータ

# データ生成（テスト用）
t0_true = 50  # [ns]
x0_true = 0   # [m]
c = 29.9792458 # [cm/ns]
epsilon_r = 4.0
v_true = c / np.sqrt(epsilon_r) # [cm/ns]
R_true = 30 # [cm]

x_data = np.arange(-250, 250, 3.6) # CE-4 [cm]
t_data = 2 / v_true * (np.sqrt((v_true * t0_true / 2 + R_true)**2 + (x_data - x0_true)**2) - R_true)
#t_data += np.random.normal(0, 0.1, size=t_data.shape)


# ステップ 5.1: ハイパーボラの頂点座標 (x0, t0) の推定
def model_step1(params, x):
    x0, t0 = params
    return np.sqrt((x - x0)**2 + t0**2)

def residuals_step1(params, x, t):
    return model_step1(params, x) - t

# 初期推定値
initial_guess_step1 = [0, np.min(t_data)]

# 最適化の実行
result_step1 = least_squares(residuals_step1, initial_guess_step1, args=(x_data, t_data))
x0_est, t0_est = result_step1.x
x0_est = 0
t0_est = 50

print("ステップ 5.1 の推定結果:")
print(f"x0 = {x0_est}")
print(f"t0 = {t0_est}")



# ステップ 5.2: 境界速度 v0 の推定
def model_step2(params, x):
    v0, R0 = params
    term = (v0 * t0_est / 2 + R0)
    return (2 / v0) * (np.sqrt(term**2 + (x - x0_est)**2) - R0)

def residuals_step2(params, x, t):
    return model_step2(params, x) - t

# 初期推定値
initial_guess_step2 = [1, 1] # v [cm/ns], R [cm]

# パラメータに制約を設定（速度は正の値、光速以下、R0 は正の値）
bounds = ([0, -np.inf], [c, np.inf])


# 最適化の実行
result_step2 = least_squares(residuals_step2, initial_guess_step2, args=(x_data, t_data), bounds=bounds)
v0_est, R0_est = result_step2.x

print("\nステップ 5.2 の推定結果:")
print(f"v0 = {v0_est}")
print(f"R0 = {R0_est}")

# ステップ 5.3: v と R の同時推定
v_min = c/1000  # v の下限（適切な値に調整してください）
v_values = np.linspace(v0_est, v_min, 1000)
foo_values = []
R_values = []

for v in v_values:
    def model_step3(R, x):
        term = (v * t0_est / 2 + R)
        return (2 / v) * (np.sqrt(term**2 + (x - x0_est)**2) - R)

    def residuals_step3(R, x, t):
        return model_step3(R, x) - t

    # R の初期推定値
    initial_R = [20] # [cm]

    # R の制約を設定（R は正の値）
    bounds_R = (0, np.inf)

    # 最適化の実行
    result_step3 = least_squares(residuals_step3, initial_R, args=(x_data, t_data), bounds=bounds_R)
    R_est = result_step3.x[0]
    R_values.append(R_est)

    # 勾配の計算と foo の評価
    jacobian = result_step3.jac
    gradient = 2 * jacobian.T @ residuals_step3(R_est, x_data, t_data)
    foo = np.linalg.norm(gradient, ord=np.inf)
    foo_values.append(foo)


# plot v_valuse and r_values
plt.figure()
plt.plot(v_values, R_values)
plt.xlabel('v')
plt.ylabel('R')
plt.grid(True)
plt.show()


# plot v_values and foo_values
plt.figure()
plt.plot(v_values, foo_values)
plt.xlabel('v')
plt.ylabel('foo')
plt.grid(True)
plt.show()

# 最適な foo を持つ v を選択
min_foo_index = np.argmin(foo_values)
print(min_foo_index)
v_opt = v_values[min_foo_index]
R_opt = R_values[min_foo_index]

print("\nステップ 5.3 の推定結果:")
print(f"最適な v = {v_opt}")
print(f"対応する R = {R_opt}")

# 結果のプロット
plt.figure(figsize=(10, 6))
plt.scatter(x_data, t_data, label='data', color='blue')
t_fit = model_step2([v_opt, R_opt], x_data)
plt.plot(x_data, t_fit, label='fit', color='red')
plt.xlabel('x')
plt.ylabel('t')
plt.legend()
plt.grid(True)
plt.show()
