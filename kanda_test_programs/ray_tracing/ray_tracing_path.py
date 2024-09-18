import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1

#* Simulation parameters
x_size, y_size = 3, 5  # [m]
dx, dy = 0.005, 0.005  # [m]
dt = 0.01e-9  # [s]
time_window = 30e-9  # [s]
Nt = int(time_window / dt)
c = 299792458  # [m/s]
intensity_threshold = 1e-6  # 閾値を小さく設定

#* Define the dielectric constant distribution
def create_default_dielectric_constant(Nx, Ny):
    epsilon_r = np.ones((Nx, Ny))
    #* Background regolith
    epsilon_r[:, int(2/dy):] = 3.0
    #* Rock
    x = np.linspace(0, x_size, Nx)
    y = np.linspace(0, y_size, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt((X - 1.5)**2 + (Y - 4)**2)
    epsilon_r[r < 0.15] = 9.0

    return epsilon_r

#* Initialize rays
def initialize_rays(num_rays, source_position):
    rays = []
    #* Define beam width in radians
    angles = np.linspace(np.pi/3, np.pi*2/3, num_rays, endpoint=False)
    for idx, angle in enumerate(angles):
        ray = {
            'id': idx,
            'position': np.array([source_position[0], source_position[1]], dtype=float),
            'direction': np.array([np.cos(angle), np.sin(angle)], dtype=float),
            'intensity': 1.0,
            'terminated': False,
            'path': [np.array([source_position[0], source_position[1]], dtype=float)],
        }
        rays.append(ray)
    return rays

#* Compute the refractive index from the dielectric constant
def compute_refractive_index(epsilon_r):
    return np.sqrt(epsilon_r)

#* Ray tracing simulation
def ray_tracing_simulation(rays, epsilon_r, dt, Nt, dx, dy):
    Nx, Ny = epsilon_r.shape
    n_map = compute_refractive_index(epsilon_r)
    all_rays = rays.copy()  # すべての光線を管理するリスト

    for step in tqdm(range(Nt), desc='Simulating'):
        new_rays = []
        rays_to_process = all_rays  # 現在のタイムステップで処理する光線
        all_rays = []  # 次のタイムステップ用にリセット

        for ray in rays_to_process:
            if ray['terminated']:
                continue
            x, y = ray['position']
            ix, iy = int(x / dx), int(y / dy)
            if ix < 0 or ix >= Nx or iy < 0 or iy >= Ny:
                ray['terminated'] = True
                continue
            n = n_map[ix, iy]
            v = c / n

            # 位置を更新
            ray['position'] += ray['direction'] * v * dt
            ray['path'].append(ray['position'].copy())

            # インターフェースの検出
            ix_new, iy_new = int(ray['position'][0] / dx), int(ray['position'][1] / dy)
            if ix_new != ix or iy_new != iy:
                if ix_new < 0 or ix_new >= Nx or iy_new < 0 or iy_new >= Ny:
                    ray['terminated'] = True
                    continue
                n_new = n_map[ix_new, iy_new]
                if n_new != n:
                    # インターフェースの法線ベクトルを計算
                    normal = compute_interface_normal(ix, iy, n_map, dx, dy)

                    # 入射角を計算
                    cos_theta_i = np.dot(normal, ray['direction'])
                    cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0)
                    sin_theta_i = np.sqrt(1 - cos_theta_i**2)

                    # スネルの法則を適用
                    n1, n2 = n, n_new
                    eta = n1 / n2
                    sin_theta_t = eta * sin_theta_i

                    if abs(sin_theta_t) > 1.0:
                        # 全反射
                        R = 1.0
                        T = 0.0
                        # 反射方向を計算
                        reflected_direction = ray['direction'] - 2 * cos_theta_i * normal
                        reflected_direction /= np.linalg.norm(reflected_direction)
                        ray['direction'] = reflected_direction
                        ray['intensity'] *= R
                    else:
                        # 部分的な反射と透過
                        cos_theta_t = np.sqrt(1 - sin_theta_t**2)

                        # フレネルの式を用いて反射率と透過率を計算
                        Rs = ((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)) ** 2
                        Rp = ((n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i)) ** 2
                        R = 0.5 * (Rs + Rp)
                        T = 1 - R

                        # 数値的不安定性のチェック
                        R = np.clip(R, 0.0, 1.0)
                        T = np.clip(T, 0.0, 1.0)

                        # 反射光線を生成
                        if R * ray['intensity'] > intensity_threshold:
                            reflected_direction = ray['direction'] - 2 * cos_theta_i * normal
                            reflected_direction /= np.linalg.norm(reflected_direction)
                            reflected_ray = {
                                'id': ray['id'],
                                'position': ray['position'].copy(),
                                'direction': reflected_direction,
                                'intensity': ray['intensity'] * R,
                                'terminated': False,
                                'path': ray['path'][:-1] + [ray['position'].copy()],
                            }
                            new_rays.append(reflected_ray)

                        # 屈折光線を更新
                        if T * ray['intensity'] > intensity_threshold:
                            transmitted_direction = eta * ray['direction'] + (eta * cos_theta_i - cos_theta_t) * normal
                            transmitted_direction /= np.linalg.norm(transmitted_direction)
                            ray['direction'] = transmitted_direction
                            ray['intensity'] *= T
                        else:
                            ray['terminated'] = True
                            continue
                # 光線の強度が閾値以下の場合、終了
                if ray['intensity'] < intensity_threshold:
                    ray['terminated'] = True
                    continue
            all_rays.append(ray)  # 次のタイムステップのために保存
        # 新たに生成された反射光線を次のタイムステップで処理
        all_rays.extend(new_rays)
    return all_rays

# インターフェースの法線ベクトルを計算
def compute_interface_normal(ix, iy, n_map, dx, dy):
    grad_nx = (n_map[min(ix + 1, n_map.shape[0] - 1), iy] - n_map[max(ix - 1, 0), iy]) / (2 * dx)
    grad_ny = (n_map[ix, min(iy + 1, n_map.shape[1] - 1)] - n_map[ix, max(iy - 1, 0)]) / (2 * dy)
    normal = np.array([grad_nx, grad_ny])
    if np.linalg.norm(normal) == 0:
        normal = np.array([0.0, 1.0])  # 法線ベクトルがゼロの場合のデフォルト
    else:
        normal = normal / np.linalg.norm(normal)
    return normal

# 可視化
def plot_ray_paths(rays, epsilon_r, source_position):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    extent = [0, x_size, y_size, 0]

    # 誘電率構造をプロット
    dielectric_img = ax.imshow(
        epsilon_r.T,
        cmap='binary',
        extent=extent,
        interpolation='nearest',
        alpha=0.5,
        origin='upper'
    )
    ax.plot(source_position[0], source_position[1], 'ro')
    ax.set_title('Ray Tracing Simulation')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(dielectric_img, cax=cax)
    cbar.set_label(r'$\epsilon_r$')

    # 各光線の軌跡をプロット
    for ray in rays:
        path = np.array(ray['path'])
        ax.plot(path[:, 0], path[:, 1], linewidth=0.5)

    plt.tight_layout()
    plt.savefig('kanda_test_programs/ray_tracing/ray_tracing_paths.png')
    plt.show()

# 誘電率分布と反射率分布をプロットする関数
def plot_dielectric_and_reflection(epsilon_r, dx, dy):
    Nx, Ny = epsilon_r.shape
    x = np.linspace(0, x_size, Nx)
    y = np.linspace(0, y_size, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 誘電率分布のプロット
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    extent = [0, x_size, y_size, 0]

    ax = axes[0]
    dielectric_img = ax.imshow(
        epsilon_r.T,
        cmap='binary',
        extent=extent,
        interpolation='nearest',
        alpha=0.5,
        origin='upper'
    )
    ax.set_title('Dielectric Constant Distribution')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    divider = axgrid1.make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(dielectric_img, cax=cax)
    cbar.set_label(r'$\epsilon_r$')

    # 反射率分布の計算
    reflection_map = compute_reflection_map(epsilon_r, dx, dy)

    # 反射率分布のプロット
    ax = axes[1]
    reflection_img = ax.imshow(
        reflection_map.T,
        cmap='viridis',
        extent=extent,
        interpolation='nearest',
        origin='upper'
    )
    ax.set_title('Reflection Coefficient Distribution')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    divider = axgrid1.make_axes_locatable(axes[1])
    cax_scat = divider.append_axes('right', size='5%', pad=0.1)
    cbar_scat = plt.colorbar(reflection_img, cax=cax_scat)
    cbar_scat.set_label('Reflection Coefficient')

    fig.tight_layout()

    plt.tight_layout()
    plt.savefig('kanda_test_programs/ray_tracing/ray_tracing_path_model.png')
    plt.show()

def compute_reflection_map(epsilon_r, dx, dy):
    Nx, Ny = epsilon_r.shape
    n_map = np.sqrt(epsilon_r)
    R_map = np.zeros((Nx, Ny))

    for ix in range(Nx):
        for iy in range(Ny):
            n1 = n_map[ix, iy]
            # x方向の隣接セルとの反射率
            if ix + 1 < Nx:
                n2_x = n_map[ix + 1, iy]
                R_x = fresnel_reflectance(n1, n2_x)
            else:
                R_x = 0
            # y方向の隣接セルとの反射率
            if iy + 1 < Ny:
                n2_y = n_map[ix, iy + 1]
                R_y = fresnel_reflectance(n1, n2_y)
            else:
                R_y = 0
            # 反射率の最大値をマップに保存
            R_map[ix, iy] = max(R_x, R_y)
    return R_map

def fresnel_reflectance(n1, n2):
    # 正入射（入射角0度）での反射率を計算
    if n1 == n2:
        return 0.0
    else:
        R = ((n1 - n2) / (n1 + n2)) ** 2
        return R

# メイン関数
def main():
    #* Prepare model
    Nx, Ny = int(x_size / dx), int(y_size / dy)
    epsilon_r = create_default_dielectric_constant(Nx, Ny)
    plot_dielectric_and_reflection(epsilon_r, dx, dy)

    #* Settting about rays
    source_position = (1.5, 1)  # [m]
    num_rays = 100  # 光線の数
    rays = initialize_rays(num_rays, source_position)

    rays = ray_tracing_simulation(rays, epsilon_r, dt, Nt, dx, dy)
    plot_ray_paths(rays, epsilon_r, source_position)

if __name__ == '__main__':
    main()
