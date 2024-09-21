import matplotlib.pyplot as plt
import numpy as np


#* Define function to calculate the intersection of a ray and a circle
def compute_circle_intersection(p0, d, center, radius, tol=1e-8):
    """
    Parameters:
    - p0: np.array([x, y]), point on the circle, starting point of the ray
    - d: np.array([dx, dy]), normalized direction vector
    - center: tuple (cx, cy), center of the circle
    - radius: float, radius of the circle
    - tol: float, tolerance for numerical errors

    Returns:
    - intersection: np.array([x, y]), point of intersection
    - None: if no intersection
    """
    # 円の方程式: ||p - center||^2 = radius^2
    # レイの方程式: p = p0 + t*d
    # 代入して解く: ||p0 + t*d - center||^2 = radius^2
    p0 = np.array(p0)
    d = np.array(d)
    center = np.array(center)
    
    # 定数項
    a = np.dot(d, d)
    b = 2 * np.dot(d, p0 - center)
    c = np.dot(p0 - center, p0 - center) - radius**2
    
    # 判別式
    discriminant = b**2 - 4*a*c
    
    if discriminant < -tol:
        # 交点なし
        return None
    elif np.abs(discriminant) < tol:
        # 接線の場合、交点が1つ
        t = -b / (2*a)
        if t > tol:
            return p0 + t*d
        else:
            return None
    else:
        # 交点が2つ
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)
        # t=0 は開始点なので、t > tol の解を選ぶ
        t_values = [t for t in [t1, t2] if t > tol]
        if not t_values:
            return None
        # 最小の正の t を選ぶ
        t_min = min(t_values)
        return p0 + t_min * d, t_min



#* Defin function to calculate the reflection and refraction of light
def calc_vec(incident, normal, n1, n2):
    """
    Parameters:
    - incident: vector of incident light
    - normal: vector of normal to the surface
    - n1: refractive index of incident medium
    - n2: refractive index of transmission medium

    Returns:
    - R: vector of reflected light
    - T: vector of refracted light
    """


    #* Calculate the reflection vector
    R = incident - 2 * np.dot(incident, normal) * normal
    R = R / np.linalg.norm(R)

    #* Calculate the refraction vector
    eta = n2 / n1
    cos_theta_i = -np.dot(normal, incident)
    sin_theta_i_sq = 1 - cos_theta_i**2
    sin_theta_t_sq = (1 / eta**2) * sin_theta_i_sq
    if sin_theta_t_sq > 1:
        # Total reflection
        T = None
    else:
        cos_theta_t = np.sqrt(1 - sin_theta_t_sq)
        T = (1 / eta) * incident + ((1 / eta) * cos_theta_i - cos_theta_t) * normal
        T = T / np.linalg.norm(T)

    return R, T


#* 円のパラメータ
center = (1.5, 3)  # 円の中心座標
radius = 0.30       # 円の半径
surface = 5

# 円を描くための角度の範囲
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = center[0] + radius * np.cos(theta)
y_circle = center[1] + radius * np.sin(theta)

# 法線ベクトルを描画する点の数
num_vectors = 12  # 例: 12本の法線ベクトルを描画
delta_angle = 2 * np.pi / num_vectors
angles = np.linspace(np.pi / 2, np.pi, num_vectors, endpoint=True)

# 入射波
I = np.array([0, -1])
I = I / np.linalg.norm(I)

# 屈折率
regolith = 3.0
rock = 9.0

# 屈折波の交点を保存するリスト
refraction_intersections = []



#* プロットの設定
fig, ax = plt.subplots(figsize=(8,8))

# 円をプロット
ax.plot(x_circle, y_circle, c='k')

# 表面をプロット
ax.axhline(y=surface, color='k', linestyle='-')

# 法線ベクトル、入射波、反射波、屈折波をプロット
for i, angle in enumerate(angles):
    # 法線ベクトルの始点（円周上の点）
    x_start = center[0] + radius * np.cos(angle)
    y_start = center[1] + radius * np.sin(angle)

    # 法線ベクトルの方向（外向き）
    n = np.array([np.cos(angle), np.sin(angle)])

    #* 外部からの入射波に対する挙動
    # 反射波の計算
    reflected_1, refracted_1 = calc_vec(I, n, regolith, rock)

    # 屈折波が存在する場合、交点を計算
    if refracted_1 is not None:
        intersection, distance = compute_circle_intersection(
            p0=[x_start, y_start],
            d=refracted_1,
            center=center,
            radius=radius
        )

        # 交点の法線ベクトルを計算
        normal_new = - (intersection - center) # minus: to the inside
        normal_new = normal_new / np.linalg.norm(normal_new)


    #* 屈折波の挙動計算
    reflected_2, refracted_2 = calc_vec(refracted_1, normal_new, rock, regolith)

    # ベクトルを描画（矢印）
    if i == 0:
        # 最初のベクトルにのみラベルを付ける
        ax.arrow(x_start - I[0]/10, y_start - I[1]/10, I[0]/10, I[1]/10,
                    head_width=0.01, head_length=0.01,  fc='r', ec='r', alpha=0.5, label='Incident wave')
        ax.arrow(x_start, y_start, reflected_1[0]/10, reflected_1[1]/10,
                    head_width=0.01, head_length=0.01,  fc='g', ec='g', label='Reflected wave')
        if refracted_1 is not None:
            ax.arrow(x_start, y_start, refracted_1[0]/10, refracted_1[1]/10,
                        head_width=0.01, head_length=0.01,  fc='b', ec='b', label='Refracted wave')
    else:
        # 他のベクトルにはラベルを付けない
        ax.arrow(x_start - I[0]/10, y_start - I[1]/10, I[0]/10, I[1]/10,
                    head_width=0.01, head_length=0.01, fc='r', ec='r', alpha=0.5)
        ax.arrow(x_start, y_start, reflected_1[0]/10, reflected_1[1]/10,
                    head_width=0.01, head_length=0.01, fc='g', ec='g')
        if refracted_1 is not None:
            ax.arrow(x_start, y_start, refracted_1[0]/10, refracted_1[1]/10,
                        head_width=0.01, head_length=0.01, fc='b', ec='b')
        ax.arrow(intersection[0] - refracted_1[0]/10, intersection[1] - refracted_1[1]/10, refracted_1[0]/10, refracted_1[1]/10,
                    head_width=0.01, head_length=0.01,  fc='r', ec='r', alpha=0.5)
        ax.arrow(intersection[0], intersection[1], reflected_2[0]/10, reflected_2[1]/10,
                    head_width=0.01, head_length=0.01,  fc='g', ec='g')
        if refracted_2 is not None:
            ax.arrow(intersection[0], intersection[1], refracted_2[0]/10, refracted_2[1]/10,
                        head_width=0.01, head_length=0.01,  fc='b', ec='b')


# グラフの見た目を整える
ax.set_xlim(0, 3)
ax.set_ylim(2.5, 6.5)
ax.set_aspect('equal')  # アスペクト比を等しくする
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)
ax.legend()

# プロットを表示
plt.savefig('kanda_test_programs/circle_normal.png')
plt.show()
