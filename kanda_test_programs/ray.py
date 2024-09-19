import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 基本定数
c = 299792458  # 光速 [m/s]
intensity_threshold = 1e-12  # 強度閾値

# 誘電率分布を生成
def create_default_dielectric_constant(Nx, Ny, dx, dy):
    epsilon_r = np.ones((Nx, Ny))
    # 背景レゴリス
    epsilon_r[:, int(2/dy):] = 3.0
    # 岩石
    x = np.linspace(0, dx * Nx, Nx)
    y = np.linspace(0, dy * Ny, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.sqrt((X - 1.5)**2 + (Y - 4)**2)
    epsilon_r[r < 0.15] = 9.0
    return epsilon_r

# 屈折率を計算
def compute_refractive_index(epsilon_r):
    return np.sqrt(epsilon_r)

# 法線ベクトルの計算
# インターフェースの法線ベクトルを計算
def compute_interface_normal(ix, iy, n_map):
    if ix <= 0 or iy <= 0 or ix >= n_map.shape[0] - 1 or iy >= n_map.shape[1] - 1:
        return np.array([0.0, 0.0])
    
    # 屈折率の勾配（法線）を計算
    grad_nx = (n_map[min(ix + 1, n_map.shape[0] - 1), iy] - n_map[max(ix - 1, 0), iy]) / 2
    grad_ny = (n_map[ix, min(iy + 1, n_map.shape[1] - 1)] - n_map[ix, max(iy - 1, 0)]) / 2

    normal = np.array([grad_nx, grad_ny])
    norm = np.linalg.norm(normal)
    
    if norm > 0:
        return normal / norm
    else:
        return np.array([0.0, 0.0])


# 光線の初期化
def initialize_rays(num_rays, source_position):
    angles = np.linspace(np.pi*9/20, np.pi*11/20, num_rays, endpoint=False)
    positions = np.full((num_rays, 2), source_position)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    intensities = np.ones(num_rays)
    return positions, directions, intensities

# 幾何光学ベースのレイトレーシング
# 幾何光学ベースのレイトレーシング
def ray_tracing_simulation(epsilon_r, dx, dy, dt, Nt, source_position, num_rays):
    Nx, Ny = epsilon_r.shape
    n_map = compute_refractive_index(epsilon_r)
    positions, directions, intensities = initialize_rays(num_rays, source_position)

    frames_positions = []
    frames_intensities = []

    for step in tqdm(range(Nt), desc='Simulating'):
        # 強度が閾値以上の光線のみをフィルタリング
        active_indices = intensities >= intensity_threshold

        # デバッグ用：フィルタリング前のサイズを確認
        print(f'{step * dt / 1e-9:.2f} ns: ')
        print(f"Before filtering: positions={len(positions)}, directions={len(directions)}, intensities={len(intensities)}, active_indices={len(active_indices)}")

        # 各配列に対して `active_indices` を適用してフィルタリング
        positions = positions[active_indices]
        directions = directions[active_indices]
        intensities = intensities[active_indices]

        # デバッグ用：フィルタリング後のサイズを確認
        print(f"After filtering: positions={len(positions)}, directions={len(directions)}, intensities={len(intensities)}")

        # 光線が無くなった場合に終了
        if len(positions) == 0:
            print("All rays terminated.")
            break

        # 光線の位置インデックスを取得
        ix = (positions[:, 0] / dx).astype(int)
        iy = (positions[:, 1] / dy).astype(int)

        # 領域外の光線を削除
        valid = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)
        positions = positions[valid]
        directions = directions[valid]
        intensities = intensities[valid]
        ix = ix[valid]
        iy = iy[valid]

        # 領域外の光線が無くなったか確認
        if len(positions) == 0:
            print("All rays out of bounds.")
            break

        # 現在位置の屈折率を取得
        n = n_map[ix, iy]
        v = c / n
        positions_new = positions + directions * v[:, np.newaxis] * dt

        # 新しい位置のインデックスを取得
        ix_new = (positions_new[:, 0] / dx).astype(int)
        iy_new = (positions_new[:, 1] / dy).astype(int)

        # 領域外の光線を削除
        valid = (ix_new >= 0) & (ix_new < Nx) & (iy_new >= 0) & (iy_new < Ny)
        positions_new = positions_new[valid]
        positions = positions[valid]
        directions = directions[valid]
        intensities = intensities[valid]
        ix_new = ix_new[valid]
        iy_new = iy_new[valid]
        ix = ix[valid]
        iy = iy[valid]

        # 新しい位置での屈折率を取得
        n_new = n_map[ix_new, iy_new]

        # 屈折率の変化を計算
        delta_n = np.abs(n_new - n)
        interface_indices = delta_n > 1e-6

        new_positions = []
        new_directions = []
        new_intensities = []

        # インターフェースを跨ぐ光線の処理
        for i in np.where(interface_indices)[0]:
            # 法線ベクトルを計算
            normal = compute_interface_normal(ix[i], iy[i], n_map)
            cos_theta_i = -np.dot(normal, directions[i])
            sin_theta_i = np.sqrt(1 - cos_theta_i**2)

            n1, n2 = n[i], n_new[i]
            eta = n2 / n1
            sin_theta_t = eta * sin_theta_i

            # 全反射か屈折の計算
            if sin_theta_t > 1.0:  # 全反射の場合
                reflected_direction = directions[i] - 2 * np.dot(directions[i], normal) * normal
                directions[i] = reflected_direction / np.linalg.norm(reflected_direction)
            else:  # 屈折
                cos_theta_t = np.sqrt(1 - sin_theta_t**2)
                transmitted_direction = eta * directions[i] + (eta * cos_theta_i - cos_theta_t) * normal
                transmitted_direction /= np.linalg.norm(transmitted_direction)

                # 透過波を生成
                if intensities[i] * (1 - cos_theta_t) > intensity_threshold:
                    new_positions.append(positions[i].copy())
                    new_directions.append(transmitted_direction)
                    new_intensities.append(intensities[i] * (1 - cos_theta_t))

            # 反射波を生成
            reflected_direction = directions[i] - 2 * np.dot(directions[i], normal) * normal
            reflected_direction /= np.linalg.norm(reflected_direction)

            if intensities[i] * cos_theta_i > intensity_threshold:
                directions[i] = reflected_direction
                intensities[i] *= cos_theta_i

        # 新たな光線（透過波）を追加
        if len(new_positions) > 0:
            new_positions = np.array(new_positions)
            new_directions = np.array(new_directions)
            new_intensities = np.array(new_intensities)

            positions = np.concatenate((positions, new_positions), axis=0)
            directions = np.concatenate((directions, new_directions), axis=0)
            intensities = np.concatenate((intensities, new_intensities))

        # 位置を更新
        if len(positions) is not len(intensities):
            print(f"Length of positions and intensities are different: {len(positions)} vs. {len(intensities)}")
            break
        # フレームデータを記録
        frames_positions.append(positions.copy())
        frames_intensities.append(intensities.copy())

    return frames_positions, frames_intensities







# 結果のプロット
def plot_ray_paths(frames_positions, epsilon_r, source_position, dx, dy):
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, epsilon_r.shape[0]*dx, epsilon_r.shape[1]*dy, 0]

    # 誘電率分布をプロット
    dielectric_img = ax.imshow(epsilon_r.T, cmap='binary', extent=extent, origin='upper', alpha=0.5)
    ax.plot(source_position[0], source_position[1], 'ro')

    # 各フレームの光線の経路をプロット
    for positions in frames_positions:
        ax.scatter(positions[:, 0], positions[:, 1], s=1, c='b')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Ray Paths')
    plt.colorbar(dielectric_img, label=r'$\epsilon_r$')
    plt.show()

# メイン関数
def main():
    # モデルの準備
    x_size, y_size = 3, 5  # [m]
    dx, dy = 0.005, 0.005  # [m]
    dt = 0.01e-9  # [s]
    time_window = 30e-9  # [s]
    Nt = int(time_window / dt)

    Nx, Ny = int(x_size / dx), int(y_size / dy)
    epsilon_r = create_default_dielectric_constant(Nx, Ny, dx, dy)

    # 光線の初期位置と数
    source_position = (1.5, 1.0)  # [m]
    num_rays = 10  # 光線の数

    # レイトレーシングシミュレーションの実行
    frames_positions, frames_intensities = ray_tracing_simulation(epsilon_r, dx, dy, dt, Nt, source_position, num_rays)

    # 結果をプロット
    plot_ray_paths(frames_positions, epsilon_r, source_position, dx, dy)

if __name__ == '__main__':
    main()
