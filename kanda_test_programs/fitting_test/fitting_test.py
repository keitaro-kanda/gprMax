import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, eig, inv


"""
def fit_ellipse(x, y):
    # Build design matrix
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones(len(x))]).T
    
    # Build scatter matrix
    S = np.dot(D.T, D)
    
    # Build 6x6 constraint matrix
    C = np.zeros((6, 6))
    C[2, 2] = 1
    C[0, 2] = -2
    C[2, 0] = -2
    
    # Solve generalized eigenvalue problem
    geval, gevec = eig(np.dot(np.linalg.inv(S), C))
    
    # Find the only negative eigenvalue
    neg_idx = np.where((geval < 0) & ~np.isinf(geval))[0][0]
    
    # Get fitted parameters
    a = gevec[:, neg_idx]
    
    return a
"""



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
    print("Condition numbers:", condNrs)

    #epsilon = np.finfo(float).eps
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
    
    # 楕円解を取得
    alpha1 = evec[:, np.argmin(condVals)]
    alpha2 = T @ alpha1
    ellipse = np.concatenate([alpha1, alpha2])
    
    # 双曲線解を取得
    possibleHs = condVals[1:3] + condVals[0]
    minDiffAt = np.argmin(np.abs(possibleHs))
    alpha1 = evec[:, minDiffAt + 1]
    alpha2 = T @ alpha1
    hyperbola = np.concatenate([alpha1, alpha2])
    
    return ellipse, hyperbola



def extract_hyperbola_parameters(coeffs):
    A, B, C, D, E, F = coeffs

    # 回転角の計算
    theta = 0.5 * np.arctan2(B, A - C)

    # 回転行列
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 行列形式での計算
    M = np.array([[A, B/2], [B/2, C]])
    offset = np.array([D/2, E/2])

    # 中心座標の計算
    center = -np.linalg.inv(M) @ offset

    # 座標の移動
    x0, y0 = center

    # 係数の再計算
    F_center = A*x0**2 + B*x0*y0 + C*y0**2 + D*x0 + E*y0 + F

    # 軸長の計算
    numerator = 2 * F_center
    denom_x = A * cos_t**2 + B * cos_t * sin_t + C * sin_t**2
    denom_y = A * sin_t**2 - B * cos_t * sin_t + C * cos_t**2

    # 軸長の計算
    a = np.sqrt(np.abs(numerator / denom_y))
    b = np.sqrt(np.abs(numerator / denom_x))


    return a, b, x0, y0, theta

def extract_hyperbola_parameters_no_rotation(coeffs):
    A, B, C, D, E, F = coeffs

    # B をゼロと仮定
    B = 0

    # 中心座標の計算
    x0 = -D / (2 * A)
    y0 = -E / (2 * C)

    # 定数項の再計算
    F_center = A*x0**2 + C*y0**2 + D*x0 + E*y0 + F

    # 軸長の計算
    a = np.sqrt(np.abs(-F_center / A))
    b = np.sqrt(np.abs(-F_center / C))

    # 回転角はゼロ
    theta = 0

    return a, b, x0, y0, theta





# データ生成（テスト用）
t0_true = 80 # [ns]
x0_true = 0 # [m]
c = 0.299792458 # [m/ns]
epsilon_r = 3.0
v_true = c / np.sqrt(epsilon_r) # [m/ns]
R_true = 0.15 # [m]

def hyperbola_model(x, t0, x0, epsilon_r, R):
    """
    x: [m], array-like
    t0: [ns], float
    x0: [m], float
    epsilon_r: dimensionless, float
    R: [m], float
    """

    c = 0.299792458 # [m/ns]
    v = c / np.sqrt(epsilon_r)
    return   2 / v * (np.sqrt((v * t0 / 2 + R)**2 + (x - x0)**2) - R)


#x_data = np.linspace(-2.5, 2.5, 100)
x_data = np.arange(-2.5, 2.5, 0.036)
#t_data = hyperbola_model(x_data, t0_true, x0_true, epsilon_r, R_true)
t_data = hyperbola_model(x_data, 100, 0, 3.0, 0.15)
noise = np.random.normal(0, 1, size=t_data.shape) # [ns]
t_data += noise



# フィッティング
ellipse_fit, hyperbola_fit = fit_ellipse_and_hyperbola(x_data, t_data)
print('Fitted coefficeints:', hyperbola_fit)

a_fit, b_fit, x0_fit, t0_fit, theta_fit = extract_hyperbola_parameters(hyperbola_fit)
#a_fit, b_fit, x0_fit, t0_fit, theta_fit = extract_hyperbola_parameters_no_rotation(hyperbola_fit)
print(f'Fitted parameters: a={a_fit}, b={b_fit}, x0={x0_fit}, t0={t0_fit}, theta={theta_fit}')
print(' ')


#* Calculate v and R
R_estimated = (a_fit - t0_fit) * b_fit / a_fit
v_estimated = 2 * b_fit / a_fit
print(f'Estimated parameters: R={R_estimated} [m], v={v_estimated} [m/ns]')

#* Make best fit hyperbola
t_fit = hyperbola_model(x_data, t0_fit, x0_fit, v_estimated, R_estimated)


# Plot
fig, ax = plt.subplots()
ax.scatter(x_data, t_data , color='black', s=10, marker='x', label='Data')
ax.plot(x_data, t_fit, color='red', label='Best fit')

ax.grid(True)
ax.set_xlabel('x [m]')
ax.set_ylabel('t [ns]')
#ax.set_ylim(np.max(t_data) + 10, 0)
ax.legend()

output_dir = 'kanda_test_programs/fitting_test'
plt.show()