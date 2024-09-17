import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1

# シミュレーションパラメータ
x_size, y_size = 10, 10  # [m]
dx, dy = 0.01, 0.01  # [m]
dt = 0.01e-9  # [s]
time_window = 10e-9  # [s]
Nt = int(time_window / dt)
c0 = 299792458  # [m/s]

# 誘電率の分布を定義
def create_default_dielectric_constant(Nx, Ny):
    epsilon_r = np.ones((Nx, Ny))
    # 誘電体の境界を設定
    epsilon_r[:, int(3/dy):] = 4.0  # 右半分の誘電率を高く設定
    return epsilon_r

# 光線を初期化
def initialize_rays(num_rays, source_position):
    rays = []
    angles = np.linspace(np.pi/4, np.pi*3/4, num_rays, endpoint=False)
    for angle in angles:
        ray = {
            'position': np.array([source_position[0], source_position[1]], dtype=float),
            'direction': np.array([np.cos(angle), np.sin(angle)]),
            'intensity': 1.0,
            'terminated': False
        }
        rays.append(ray)
    return rays

# 誘電率から屈折率を計算
def compute_refractive_index(epsilon_r):
    return np.sqrt(epsilon_r)

# 光線の伝搬をシミュレート
def ray_tracing_simulation(rays, epsilon_r, dt, Nt, dx, dy):
    frames = []
    Nx, Ny = epsilon_r.shape
    n_map = compute_refractive_index(epsilon_r)
    for _ in tqdm(range(Nt), desc='Simulating'):
        positions = []
        for ray in rays:
            if ray['terminated']:
                continue
            x, y = ray['position']
            ix, iy = int(x / dx), int(y / dy)
            if ix < 0 or ix >= Nx or iy < 0 or iy >= Ny:
                ray['terminated'] = True
                continue
            n = n_map[ix, iy]
            v = c0 / n

            # 位置を更新
            ray['position'] += ray['direction'] * v * dt

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
                    normal = -normal  # 法線の向きを修正

                    # 入射角を計算
                    cos_theta_i = -np.dot(normal, ray['direction'])
                    theta_i = np.arccos(cos_theta_i)

                    # スネルの法則を適用
                    sin_theta_t = n / n_new * np.sin(theta_i)
                    if sin_theta_t > 1.0:
                        # 全反射
                        ray['direction'] = ray['direction'] - 2 * np.dot(ray['direction'], normal) * normal
                    else:
                        # 屈折方向を計算
                        theta_t = np.arcsin(sin_theta_t)
                        ray_dir_perp = (ray['direction'] + cos_theta_i * normal)
                        ray_dir_perp = ray_dir_perp / np.linalg.norm(ray_dir_perp)
                        ray['direction'] = n / n_new * ray_dir_perp - np.sqrt(1 - (n / n_new)**2 * (1 - cos_theta_i**2)) * normal
                        ray['direction'] = ray['direction'] / np.linalg.norm(ray['direction'])
            positions.append(ray['position'].copy())
        frames.append(np.array(positions))
    return frames

# インターフェースの法線ベクトルを計算
def compute_interface_normal(ix, iy, n_map, dx, dy):
    n_center = n_map[ix, iy]
    grad_nx = (n_map[min(ix + 1, n_map.shape[0] - 1), iy] - n_map[max(ix - 1, 0), iy]) / (2 * dx)
    grad_ny = (n_map[ix, min(iy + 1, n_map.shape[1] - 1)] - n_map[ix, max(iy - 1, 0)]) / (2 * dy)
    normal = np.array([grad_nx, grad_ny])
    if np.linalg.norm(normal) == 0:
        normal = np.array([0.0, 1.0])  # 法線ベクトルがゼロの場合のデフォルト
    else:
        normal = normal / np.linalg.norm(normal)
    return normal

# 可視化
def animate_rays(frames, epsilon_r, source_position):
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, x_size, y_size, 0]

    # 誘電率構造をプロット
    dielectric_img = ax.imshow(
        epsilon_r.T,
        cmap='binary',
        extent=extent,
        interpolation='nearest',
        alpha=0.3
    )
    ax.plot(source_position[0], source_position[1], 'ro')
    ax.set_title('Ray Tracing Simulation')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    delvider = axgrid1.make_axes_locatable(ax)
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(dielectric_img, cax=cax)
    cbar.set_label(r'$\epsilon_r$')

    # アニメーションの更新関数
    scat = ax.scatter([], [], s=1, c='b')

    def update(i):
        positions = frames[i]
        scat.set_offsets(positions)
        time_in_ns = i * dt / 1e-9
        ax.set_title(f'Ray Tracing at t = {time_in_ns:.2f} ns')
        return scat,

    print('Animating...')
    print('Number of frames:', len(frames))
    fps = 30
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True, repeat=False
    )

    plt.tight_layout()
    ani.save('kanda_test_programs/ray_tracing_animation.mp4', writer='ffmpeg', fps=fps)
    plt.show()

# メイン関数
def main():
    Nx, Ny = int(x_size / dx), int(y_size / dy)
    epsilon_r = create_default_dielectric_constant(Nx, Ny)
    print('Completed creating dielectric structure')
    source_position = (5.0, 2.0)  # [m]
    num_rays = 361  # 光線の数

    rays = initialize_rays(num_rays, source_position)
    print('Completed initializing rays')

    frames = ray_tracing_simulation(rays, epsilon_r, dt, Nt, dx, dy)
    animate_rays(frames, epsilon_r, source_position)

if __name__ == '__main__':
    main()
