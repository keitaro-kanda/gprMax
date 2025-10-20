import numpy as np
from scipy.optimize import brentq

# 既知の値を設定（例）
W = 1.8 # 岩石の幅 [m]
epsilon_r = 3.0  # 比誘電率 (例: 水や湿った土)
h = 0.3 # アンテナ高さ [m]
d = 2.0 # 岩石の深さ [m]
transmission_delay = 2.57e-9  # 伝送路遅延 [s]

# 1. 解くべき関数 f(theta_2) を定義
def f(theta_2, W, epsilon_r, h, d):
    # エラーを防ぐため、theta_2が0やpi/2に近すぎないようにする
    if theta_2 <= 0 or theta_2 >= np.pi/2:
        return -np.inf # 範囲外は考慮しない

    sin_theta_2 = np.sin(theta_2)
    cos_theta_2 = np.cos(theta_2)
    
    # 全反射の条件チェック
    sqrt_term_1_val = 1 - epsilon_r * (sin_theta_2**2)
    if sqrt_term_1_val <= 0:
        # 臨界角を超えた場合（物理的に無効）、大きな値を返して探索範囲外とする
        return np.inf 
        
    term1 = h / np.sqrt(sqrt_term_1_val)
    term2 = d / cos_theta_2
    
    val = np.sqrt(epsilon_r) * sin_theta_2 * (term1 + term2) - (W / 2)
    return val

# 2. 探索範囲を決定
# 臨界角 (radian)
theta_critical = np.arcsin(1 / np.sqrt(epsilon_r))

# 探索範囲は 0 より少し大きく、臨界角より少し小さい範囲
# (0 や臨界角ちょうどでは計算が発散(inf)するため)
search_min = 1e-9
search_max = theta_critical - 1e-9

try:
    # 3. brentq を使って根を探索
    # f(search_min) と f(search_max) の符号が異なる必要がある
    theta_2_rad = brentq(f, search_min, search_max, args=(W, epsilon_r, h, d))
    
    theta_2_deg = np.degrees(theta_2_rad)
    
    print(f"数値解 (ラジアン): {theta_2_rad:.6f} rad")
    print(f"数値解 (度): {theta_2_deg:.6f} °")
    print(" -----------------------------------")

except ValueError as e:
    print(f"エラー: 指定された範囲 [{search_min}, {search_max}] で解が見つからないか、")
    print(f"範囲の両端で f(theta_2) が同符号です。{e}")
    # f(search_min) と f(search_max) の値を確認してみる
    # print(f"f(search_min) = {f(search_min, W, epsilon_r, h, d)}")
    # print(f"f(search_max) = {f(search_max, W, epsilon_r, h, d)}")

# 4. 遅れ時間の計算
L = h / np.sqrt(1 -  epsilon_r * np.sin(theta_2_rad)**2) + d * np.sqrt(epsilon_r) / np.cos(theta_2_rad)
delay_time = L * 2 / 3e8 + transmission_delay  # [s]
print("伝搬距離 L: {:.3f} m".format(L))
print(f"遅れ時間: {delay_time*1e9:.3f} ns")