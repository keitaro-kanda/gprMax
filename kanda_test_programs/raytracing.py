import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1
import matplotlib.cm as cm

#* Simulation parameters
x_size, y_size = 3, 5  # [m]
dx, dy = 0.005, 0.005  # [m]
dt = 0.01e-9  # [s]
time_window = 30e-9  # [s]
Nt = int(time_window / dt)
c = 299792458  # [m/s]
intensity_threshold = 0.0001



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
        }
        rays.append(ray)
    return rays



#* Compute the refractive index from the dielectric constant
def compute_refractive_index(epsilon_r):
    return np.sqrt(epsilon_r)



#* Ray tracing simulation
def ray_tracing_simulation(rays, epsilon_r, dt, Nt, dx, dy):
    frames_positions = []
    frames_ids = []

    Nx, Ny = epsilon_r.shape
    n_map = compute_refractive_index(epsilon_r)
    all_rays = rays.copy()  # すべての光線を管理するリスト

    for step in tqdm(range(Nt), desc='Simulating'):
        positions = []
        ids = []
        new_rays = []
        for ray in all_rays:
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
                    cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0)
                    sin_theta_i = np.sqrt(1 - cos_theta_i**2)

                    # スネルの法則を適用
                    n1, n2 = n, n_new
                    sin_theta_t = n1 / n2 * sin_theta_i

                    if abs(sin_theta_t) > 1.0:
                        # 全反射
                        R = 1.0
                        T = 0.0
                        # 反射方向のみを計算
                        reflected_direction = ray['direction'] - 2 * np.dot(ray['direction'], normal) * normal
                        reflected_direction /= np.linalg.norm(reflected_direction)
                        ray['direction'] = reflected_direction
                        ray['intensity'] *= R
                    else:
                        # 部分的な反射と透過
                        cos_theta_t = np.sqrt(1 - sin_theta_t**2)

                        # フレネルの式を用いて反射率と透過率を計算
                        epsilon = 1e-8  # 微小量
                        denominator_s = n1 * cos_theta_i + n2 * cos_theta_t + epsilon
                        denominator_p = n1 * cos_theta_t + n2 * cos_theta_i + epsilon
                        Rs = ((n1 * cos_theta_i - n2 * cos_theta_t) / denominator_s) ** 2 # reflectance for s-polarized light
                        Rp = ((n1 * cos_theta_t - n2 * cos_theta_i) / denominator_p) ** 2 # reflectance for p-polarized light
                        R = 0.5 * (Rs + Rp)
                        R = np.clip(R, 0.0, 1.0)
                        T = 1 - R

                        if np.isnan(R) or np.isinf(R):
                            R = 0.0
                        if np.isnan(T) or np.isinf(T):
                            T = 1.0

                        # 反射光線を生成
                        if R * ray['intensity'] > intensity_threshold:
                            reflected_direction = ray['direction'] - 2 * np.dot(ray['direction'], normal) * normal
                            reflected_direction /= np.linalg.norm(reflected_direction)
                            reflected_ray = {
                                'id': ray['id'],  # 親光線のIDを引き継ぐ
                                'position': ray['position'].copy(),
                                'direction': reflected_direction,
                                'intensity': ray['intensity'] * R,
                                'terminated': False,
                            }
                            new_rays.append(reflected_ray)

                        # 屈折光線を更新
                        if T * ray['intensity'] > intensity_threshold:
                            transmitted_direction = (n1 / n2) * (ray['direction'] + cos_theta_i * normal) - cos_theta_t * normal
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
            positions.append(ray['position'].copy())
            ids.append(ray['id'])
        all_rays.extend(new_rays)
        frames_positions.append(np.array(positions))
        frames_ids.append(np.array(ids))
    return frames_positions, frames_ids

# インターフェースの法線ベクトルを計算
def compute_interface_normal(ix, iy, n_map, dx, dy):
    #n_center = n_map[ix, iy]
    grad_nx = (n_map[min(ix + 1, n_map.shape[0] - 1), iy] - n_map[max(ix - 1, 0), iy]) / (2 * dx)
    grad_ny = (n_map[ix, min(iy + 1, n_map.shape[1] - 1)] - n_map[ix, max(iy - 1, 0)]) / (2 * dy)
    normal = np.array([grad_nx, grad_ny])
    if np.linalg.norm(normal) == 0:
        normal = np.array([0.0, 1.0])  # 法線ベクトルがゼロの場合のデフォルト
    else:
        normal = normal / np.linalg.norm(normal)
    return normal

# 可視化
def animate_rays(frames_positions, frames_ids, epsilon_r, source_position, num_rays):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    extent = [0, x_size, y_size, 0]

    # 誘電率構造をプロット
    dielectric_img = ax.imshow(
        epsilon_r.T,
        cmap='binary',
        extent=extent,
        interpolation='nearest',
        alpha=0.5
    )
    ax.plot(source_position[0], source_position[1], 'ro')
    ax.set_title('Ray Tracing Simulation')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(dielectric_img, cax=cax)
    cbar.set_label(r'$\epsilon_r$')

    # カラーマップを作成
    cmap = plt.get_cmap('jet')
    # 空でない ids のみを考慮して num_unique_ids を計算
    max_ids_in_frames = [max(ids) for ids in frames_ids if len(ids) > 0]
    if max_ids_in_frames:
        num_unique_ids = max(max_ids_in_frames) + 1
    else:
        num_unique_ids = num_rays  # または適切なデフォルト値

    colors = cmap(np.linspace(0, 1, num_unique_ids))

    # スキャッタープロットを初期化
    scat = ax.scatter([], [], s=1)

    # ブリッティングのための初期化関数
    def init():
        scat.set_offsets(np.empty((0, 2)))  # 修正
        return scat,

    # 更新関数
    def update(i):
        positions = frames_positions[i]
        ids = frames_ids[i]
        if len(ids) > 0:
            point_colors = [colors[id % num_unique_ids] for id in ids]
            scat.set_offsets(positions)
            scat.set_color(point_colors)
        else:
            scat.set_offsets(np.empty((0, 2)))
        time_in_ns = i * dt / 1e-9
        ax.set_title(f'Ray Tracing at t = {time_in_ns:.2f} ns')
        return scat,

    print('Animating...')
    print('Number of frames:', len(frames_positions))
    fps = 30

    # tqdmを用いたフレームジェネレータ
    def frame_generator():
        for i in tqdm(range(len(frames_positions)), desc='Creating Animation'):
            yield i

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_generator(),  # プログレスバー付きのフレームジェネレータ
        init_func=init,
        interval=1000 / fps,
        blit=True,
        repeat=False,
        save_count=len(frames_positions),  # キャッシュサイズを指定
        # cache_frame_data=False,  # またはこれを使用
    )

    plt.tight_layout()
    ani.save('kanda_test_programs/ray_tracing_animation.mp4', writer='ffmpeg', fps=fps)
    plt.close()

# メイン関数
def main():
    Nx, Ny = int(x_size / dx), int(y_size / dy)
    epsilon_r = create_default_dielectric_constant(Nx, Ny)
    source_position = (1.5, 1)  # [m]
    num_rays = 100  # 光線の数

    rays = initialize_rays(num_rays, source_position)
    frames_positions, frames_ids = ray_tracing_simulation(rays, epsilon_r, dt, Nt, dx, dy)
    animate_rays(frames_positions, frames_ids, epsilon_r, source_position, num_rays)

if __name__ == '__main__':
    main()
