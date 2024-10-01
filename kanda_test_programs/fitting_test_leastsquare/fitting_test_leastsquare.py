import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 双曲線のモデル関数
def hyperbola_model(x, a, b, x0, y0):
    return y0 - a * np.sqrt(1 + ((x - x0) / b)**2)

# データ生成（テスト用）
t0_true = 50e-9
x0_true = 0
v_true = 1.5e8
R_true = 0.30

def hyperbola_model(x, t0, x0, v, R):
    return  - 2 / v * (np.sqrt((v * t0 / 2 + R)**2 + (x - x0)**2) - R)

#a_true = 3  # 真の値
#b_true = 1
#x0_true = 0
#y0_true = 0

x_data = np.linspace(-2.5, 2.5, 100)
#y_data = hyperbola_model(x_data, a_true, b_true, x0_true, y0_true)
y_data = hyperbola_model(x_data, t0_true, x0_true, v_true, R_true)

# ノイズの追加
noise = np.random.normal(0, 1e-9, size=y_data.shape)
y_data_noisy = y_data + noise

# フィッティング
initial_guess = [1e-10, 1e-10, 0, 0]  # パラメータの初期推定値
params_opt, params_cov = curve_fit(hyperbola_model, x_data, y_data_noisy, p0=initial_guess)

# フィッティング結果の取得
b_fit, a_fit, x0_fit, y0_fit = params_opt
print(f"推定されたパラメータ: a={a_fit}, b={b_fit}, x0={x0_fit}, y0={y0_fit}")

R_estimated = (a_fit - t0_true) * b_fit / a_fit
v_estimated = 2 * b_fit / a_fit
print(f"推定されたパラメータ: R={R_estimated}, v={v_estimated}")

# フィッティング曲線の生成
y_fit = hyperbola_model(x_data, a_fit, b_fit, x0_fit, y0_fit)

# プロット
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data_noisy, label='data', color='black', s=10, marker='x')
plt.plot(x_data, y_fit, label='best fit', color='red')
plt.xlabel('x')
plt.ylabel('t')
#plt.ylim(np.min(y_data_noisy) - 10e-9, 0)
plt.legend()
plt.grid(True)
plt.show()
