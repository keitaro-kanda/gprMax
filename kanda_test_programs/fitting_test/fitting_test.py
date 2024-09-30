import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, eig, inv



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
    condNrs = np.diag(Ss3) / Ss3[0]
    print(condNrs)

        # 対角成分を抽出
    diag_condNrs = np.diag(condNrs)

    epsilon = np.finfo(float).eps
    if diag_condNrs[2] < epsilon:
        print('Warning: S3 is degenerate')
        return None, None

    # 制約行列とその逆行列を定義
    C = np.array([[0, 0, -2], [0, 1, 0], [-2, 0, 0]])
    Ci = inv(C)
    
    # 一般化固有ベクトル問題を設定して解く
    T = -inv(S3) @ S2.T
    S = Ci @ (S1 - S2 @ T)
    
    evals, evec = eig(S)
    print(evec)
    print(evals)
    
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



#* Hyperbola for test
a = 1
b = 3
center_x = 0
center_y = 0
rotation = 0
x = np.linspace(center_x-a, center_x+a, 100, endpoint=True)
y = - b * np.sqrt((x - center_x)**2 / a**2 + 1) + center_y

#* Add noise
y += np.random.normal(0, 0.1, len(y))

# フィッティング
ellipse_fit, hyperbola_fit = fit_ellipse_and_hyperbola(x, y)
print('ellipse: ', ellipse_fit)
print('hyperbola: ', hyperbola_fit)


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

for x_val in x:
    solution = solve_y(ellipse_fit, x_val)
    if solution:
        y1, y2 = solution
        y_pos.append(y1)
        y_neg.append(y2)
        x_vals.append(x_val)


# Plot
fig, ax = plt.subplots()
ax.scatter(x, y, color='black', s=10, marker='x')
t = np.linspace(0, 2*np.pi, 100)
ax.plot(x_vals, y_neg, color='blue')

ax.set_aspect('equal')
ax.set_xlim(center_x-a-1, center_x+a+1)
#ax.set_ylim(center_y-1, center_y+b+1)
ax.grid(True)


output_dir = 'kanda_test_programs/fitting_test'
output_name = f'a_{a}_b_{b}.png'
output_path = f'{output_dir}/{output_name}'
plt.savefig(output_path)
plt.show()
