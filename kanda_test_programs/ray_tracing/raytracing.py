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
intensity_threshold = 1e-12  # 閾値を小さく設定

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
    print(f"Initialized {len(rays)} rays.")
    return rays

#* Compute the refractive index from the dielectric constant
def compute_refractive_index(epsilon_r):
    return np.sqrt(epsilon_r)

#* Ray tracing simulation
def ray_tracing_simulation(epsilon_r, dt, Nt, dx, dy, source_position, num_rays):
    Nx, Ny = epsilon_r.shape
    n_map = compute_refractive_index(epsilon_r)

    # 光線を配列として初期化
    angles = np.linspace(np.pi/3, np.pi*2/3, num_rays, endpoint=False)
    positions = np.full((num_rays, 2), source_position)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    intensities = np.ones(num_rays)

    frames_positions = []
    frames_intensities = []

    for step in tqdm(range(Nt), desc='Simulating'):
        new_positions = []
        new_directions = []
        new_intensities = []

        # 閾値以上の強度を持つ光線のみを処理
        active_indices = intensities >= intensity_threshold
        positions = positions[active_indices]
        directions = directions[active_indices]
        intensities = intensities[active_indices]

        if len(positions) == 0:
            break  # 処理する光線がなくなったら終了

        # 光線の位置を更新
        ix = (positions[:, 0] / dx).astype(int)
        iy = (positions[:, 1] / dy).astype(int)

        # 領域外の光線を削除
        valid = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)
        positions = positions[valid]
        directions = directions[valid]
        intensities = intensities[valid]
        ix = ix[valid]
        iy = iy[valid]

        n = n_map[ix, iy]
        v = c / n
        positions += directions * v[:, np.newaxis] * dt

        # インターフェースの検出
        ix_new = (positions[:, 0] / dx).astype(int)
        iy_new = (positions[:, 1] / dy).astype(int)

        # 領域外の光線を削除
        valid = (ix_new >= 0) & (ix_new < Nx) & (iy_new >= 0) & (iy_new < Ny)
        positions = positions[valid]
        directions = directions[valid]
        intensities = intensities[valid]
        ix_new = ix_new[valid]
        iy_new = iy_new[valid]
        ix = ix[valid]
        iy = iy[valid]
        n = n[valid]  # 追加

        # 新しい屈折率を取得
        n_new = n_map[ix_new, iy_new]

        # 屈折率の変化を計算
        delta_n = np.abs(n_new - n)

        interface_indices = delta_n > 1e-6

        # インターフェースを通過する光線の処理
        for i in np.where(interface_indices)[0]:
            # インターフェースの法線ベクトルを計算
            normal = compute_interface_normal(ix[i], iy[i], n_map, dx, dy)

            # 入射角の計算
            cos_theta_i = -np.dot(normal, directions[i])
            cos_theta_i = np.clip(cos_theta_i, -1.0, 1.0)
            sin_theta_i = np.sqrt(1 - cos_theta_i**2)

            n1, n2 = n[i], n_new[i]
            eta = n1 / n2
            sin_theta_t = eta * sin_theta_i

            if sin_theta_t > 1.0:
                # 全反射
                R = 1.0
                T = 0.0
            else:
                cos_theta_t = np.sqrt(1 - sin_theta_t**2)
                Rs = ((n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)) ** 2
                Rp = ((n1 * cos_theta_t - n2 * cos_theta_i) / (n1 * cos_theta_t + n2 * cos_theta_i)) ** 2
                R = 0.5 * (Rs + Rp)
                T = 1 - R

            R = np.clip(R, 0.0, 1.0)
            T = np.clip(T, 0.0, 1.0)

            # 反射光線の生成
            if intensities[i] * R > intensity_threshold:
                reflected_direction = directions[i] - 2 * cos_theta_i * normal
                reflected_direction /= np.linalg.norm(reflected_direction)
                new_positions.append(positions[i].copy())
                new_directions.append(reflected_direction)
                new_intensities.append(intensities[i] * R)
                print(f"Ray {i} is reflected at ({positions[i][0]:.2f}, {positions[i][1]:.2f})")

            # 屈折光線の更新
            if intensities[i] * T > intensity_threshold:
                transmitted_direction = eta * directions[i] + (eta * cos_theta_i - np.sqrt(1 - (eta * sin_theta_i) ** 2)) * normal
                transmitted_direction /= np.linalg.norm(transmitted_direction)
                directions[i] = transmitted_direction
                intensities[i] *= T
            else:
                # 光線を削除（強度をゼロに設定）
                intensities[i] = 0.0

        # 強度が閾値以下の光線を削除
        active_indices = intensities >= intensity_threshold
        positions = positions[active_indices]
        directions = directions[active_indices]
        intensities = intensities[active_indices]

        # 新たな光線を追加
        if new_positions:
            positions = np.vstack((positions, new_positions))
            directions = np.vstack((directions, new_directions))
            intensities = np.concatenate((intensities, new_intensities))

        # フレームデータを記録
        frames_positions.append(positions.copy())
        frames_intensities.append(intensities.copy())

        if step % 100 == 0:
            print(f"At {step * dt / 1e-9:.2f} ns: Number of rays = {len(positions)}")
        if len(positions) == 0:
            print(f"Simulation terminated at {step * dt / 1e-9:.2f} ns.")

    return frames_positions, frames_intensities


# インターフェースの法線ベクトルを計算
def compute_interface_normal(ix, iy, n_map, dx, dy):
    grad_nx = (n_map[min(ix + 1, n_map.shape[0] - 1), iy] - n_map[max(ix - 1, 0), iy]) / 2
    grad_ny = (n_map[ix, min(iy + 1, n_map.shape[1] - 1)] - n_map[ix, max(iy - 1, 0)]) / 2
    normal = np.array([grad_nx, grad_ny])
    if np.linalg.norm(normal) == 0:
        normal = np.array([0.0, 1.0])  # 法線ベクトルがゼロの場合のデフォルト
    else:
        normal = normal / np.linalg.norm(normal)
    return normal

# 可視化
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

# 可視化
def animate_rays(frames_positions, frames_intensities, epsilon_r, source_position):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)  # dpiを調整しました
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
    ax.set_xlabel('X [m]', fontsize=20)
    ax.set_ylabel('Y [m]', fontsize=20)
    ax.tick_params(labelsize=18)

    # 時刻表示用のテキストを追加
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # カラーバー1（誘電率のカラーバー）
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes('right', size='5%', pad=0.1)  # カラーバー1用のスペースを作成
    cbar1 = plt.colorbar(dielectric_img, cax=cax1)
    cbar1.set_label(r'$\epsilon_r$', fontsize=18)
    cbar1.ax.tick_params(labelsize=16)

    # 光線の強度に基づくカラーマップ
    all_intensities = np.concatenate(frames_intensities)
    vmin = np.min(all_intensities)
    vmax = np.max(all_intensities)
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # スキャッタープロットを初期化
    scat = ax.scatter([], [], s=1, c=[], cmap=cmap, norm=norm)

    # カラーバー2（光線強度のカラーバー）
    cax2 = divider.append_axes('right', size='5%', pad=1.0)  # padを調整して2本目が重ならないようにする
    cbar2 = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax2)
    cbar2.set_label('Intensity of rays', fontsize=18)
    cbar2.ax.tick_params(labelsize=16)

    # 初期化関数
    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        time_text.set_text('')
        return scat, time_text

    # 更新関数
    def update(i):
        positions = frames_positions[i]
        intensities = frames_intensities[i]

        if positions.size > 0:
            scat.set_offsets(positions)
            scat.set_array(intensities)
        else:
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array(np.array([]))
        time_in_ns = i * dt / 1e-9
        time_text.set_text(f't = {time_in_ns:.2f} ns', fontsize=20)
        return scat, time_text

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
        frames=frame_generator(),
        init_func=init,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    ani.save('kanda_test_programs/ray_tracing/ray_tracing_animation.mp4', writer='ffmpeg', fps=fps)
    plt.show()


# メイン関数
def main():
    #* Prepare model
    Nx, Ny = int(x_size / dx), int(y_size / dy)
    epsilon_r = create_default_dielectric_constant(Nx, Ny)

    #* Settting about rays
    source_position = (1.5, 1)  # [m]
    num_rays = 100  # 光線の数
    rays = initialize_rays(num_rays, source_position)

    frames_positions, frames_intensities = ray_tracing_simulation(epsilon_r, dt, Nt, dx, dy, source_position, num_rays)
    print('frames_positions:', len(frames_positions))
    print('frames_intensities:', len(frames_intensities))
    animate_rays(frames_positions, frames_intensities, epsilon_r, source_position)

if __name__ == '__main__':
    main()