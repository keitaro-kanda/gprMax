import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os
import json
import argparse

# 定数
c0 = 3e8  # 真空中の光速 [m/s]

# コマンドライン引数の解析
parser = argparse.ArgumentParser(
    prog='huygens_wave_simulation.py',
    description='Simulate wave propagation using Huygens principle with secondary sources at media boundaries.',
    epilog='End of help message',
    usage='python kanda_test_programs/huygens.py [json]',
)
parser.add_argument('json', help='Path to the parameter json file')
args = parser.parse_args()

# 媒質の追加関数
def add_circle(epsilon_grid, center, radius, epsilon, x, y):
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    epsilon_grid[mask] = epsilon
    return epsilon_grid

def add_square(epsilon_grid, bottom_left, size, epsilon, x, y):
    mask = (
        (x >= bottom_left[0]) & (x <= bottom_left[0] + size[0]) &
        (y >= bottom_left[1]) & (y <= bottom_left[1] + size[1])
    )
    epsilon_grid[mask] = epsilon
    return epsilon_grid

# 初期波面の設定関数（円周上の点として設定）
def init_wavefront(position_x, position_y, source_radius, num_points):
    angles = np.linspace(np.pi, 2 * np.pi, num_points, endpoint=False)
    wave_points = np.array([
        position_x + source_radius * np.cos(angles),
        position_y + source_radius * np.sin(angles)
    ]).T
    return wave_points

# メイン
if __name__ == '__main__':
    # パラメータの読み込み
    with open(args.json, 'r') as f:
        params = json.load(f)
    output_dir = os.path.dirname(args.json)

    # シミュレーション空間の設定
    size_x, size_y = params['simulation_space']['size']
    dx = params['simulation_space']['grid_spacing']
    nx, ny = int(size_x / dx), int(size_y / dx)
    x = np.linspace(0, size_x, nx)
    y = np.linspace(0, size_y, ny)
    X, Y = np.meshgrid(x, y)

    # 時間の設定
    dt = params['time']['time_step']
    time_window = params['time']['time_window']
    num_steps = int(time_window / dt)

    # 媒質の設定（誘電率マップ）
    epsilon_grid = np.ones((ny, nx))
    media = params['media']
    for medium in media:
        if medium['type'] == 'circle':
            epsilon_grid = add_circle(
                epsilon_grid,
                medium['center'],
                medium['radius'],
                medium['epsilon'],
                X, Y
            )
        elif medium['type'] == 'square':
            epsilon_grid = add_square(
                epsilon_grid,
                medium['bottom_left'],
                medium['size'],
                medium['epsilon'],
                X, Y
            )

    # 波速マップの計算
    v_grid = c0 / np.sqrt(epsilon_grid)

    # 初期波面の設定
    source_position = params['source']['position']
    source_radius = params['source']['radius']  # 波源の半径 [m]（必要に応じて調整）
    num_points = 180  # 波面上の点の数（必要に応じて調整）
    wave_points = init_wavefront(
        source_position[0],
        source_position[1],
        source_radius,
        num_points
    )

    # 二次波源のリスト
    secondary_sources = []

    # 波面の位置を保存するリスト
    wavefronts = []

    # 結果を保存するディレクトリを作成
    output_dir_frames = os.path.join(output_dir, 'wave_simulation_frames')
    if not os.path.exists(output_dir_frames):
        os.makedirs(output_dir_frames)

    # シミュレーションの実行
    for step in tqdm(range(num_steps), desc='Calculating...'):
        new_wave_points = []
        # 一次波面の更新
        for point in wave_points:
            x, y = point
            # 位置がシミュレーション領域内か確認
            if 0 <= x < size_x and 0 <= y < size_y:
                i = int((y / size_y) * ny)
                j = int((x / size_x) * nx)
                # 局所的な波速を計算
                epsilon = epsilon_grid[i, j]
                c_medium = c0 / np.sqrt(epsilon)
                # 波の進行方向は放射状
                direction = point - np.array(source_position)
                if np.linalg.norm(direction) != 0:
                    direction = direction / np.linalg.norm(direction)
                    # 位置の更新
                    new_point = point + c_medium * dt * direction
                    # 境界のチェック
                    i_new = int((new_point[1] / size_y) * ny)
                    j_new = int((new_point[0] / size_x) * nx)
                    if 0 <= i_new < ny and 0 <= j_new < nx:
                        epsilon_new = epsilon_grid[i_new, j_new]
                        if epsilon != epsilon_new:
                            # 境界に到達した場合、二次波源を追加
                            secondary_sources.append({
                                'position': point.copy(),
                                'time': step * dt
                            })
                        else:
                            # 境界に到達していない場合、位置を更新
                            new_wave_points.append(new_point)
        wave_points = np.array(new_wave_points)
        wavefronts.append({
            'primary': wave_points.copy(),
            'secondary': []
        })

        # 二次波源からの波面を計算
        secondary_wave_points = []
        for source in secondary_sources:
            time_elapsed = (step * dt) - source['time']
            if time_elapsed >= 0:
                # 二次波源からの波面を計算
                i_src = int((source['position'][1] / size_y) * ny)
                j_src = int((source['position'][0] / size_x) * nx)
                epsilon = epsilon_grid[i_src, j_src]
                c_medium = c0 / np.sqrt(epsilon)
                radius = c_medium * time_elapsed
                # 二次波面上の点を計算
                angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
                points = np.array([
                    source['position'][0] + radius * np.cos(angles),
                    source['position'][1] + radius * np.sin(angles)
                ]).T
                # シミュレーション領域内の点のみを追加
                for point in points:
                    x, y = point
                    if 0 <= x < size_x and 0 <= y < size_y:
                        secondary_wave_points.append(point)
        wavefronts[-1]['secondary'] = np.array(secondary_wave_points)

    print('Calculation done.')
    print(' ')
    frame_indices = [i for i in range(len(wavefronts)) if i % 50 == 0]

    # 可視化とフレームの保存
    for idx, i in tqdm(enumerate(frame_indices), desc='Saving frames...', total=len(frame_indices)):
        wavefront = wavefronts[i]
        plt.figure(figsize=(10, 10 * size_y / size_x))
        plt.imshow(epsilon_grid, extent=(0, size_x, 0, size_y), cmap='Greys', origin='lower', alpha=0.7)
        # 一次波面のプロット
        if wavefront['primary'].size > 0:
            plt.scatter(wavefront['primary'][:, 0], wavefront['primary'][:, 1], s=1, c='blue', label='Primary Wavefront')
        # 二次波面のプロット
        if wavefront['secondary'].size > 0:
            plt.scatter(wavefront['secondary'][:, 0], wavefront['secondary'][:, 1], s=1, c='red', label='Secondary Wavefront')
        current_time = i * dt * 1e9  # 時間を正しく計算
        plt.title(f'Time: {current_time:.2f} ns', fontsize=24)
        plt.xlabel('x (m)', fontsize=24)
        plt.ylabel('y (m)', fontsize=24)
        plt.xlim(0, size_x)
        plt.ylim(0, size_y)
        plt.legend(fontsize=16)
        plt.tick_params(labelsize=20)
        plt.savefig(f'{output_dir_frames}/frame_{idx:04d}.png')
        plt.close()
    print('Frames saved.')
    print(' ')

    # アニメーションの作成
    print('Creating animation...')
    animation_wavefronts = [wavefronts[i] for i in frame_indices]


    figsize_ratio = size_y / size_x
    fig, ax = plt.subplots(figsize=(10, 10 * figsize_ratio))

    class Animator:
        def __init__(self, ax, wavefronts, epsilon_grid, frame_indices, dt, size_x, size_y):
            self.ax = ax
            self.wavefronts = wavefronts
            self.epsilon_grid = epsilon_grid
            self.frame_indices = frame_indices
            self.dt = dt
            self.size_x = size_x
            self.size_y = size_y
            self.pbar = tqdm(total=len(wavefronts), desc='Animating...')
            self.counter = 0  # プログレスバーのカウンター

        def __call__(self, i):
            self.ax.clear()
            self.ax.imshow(self.epsilon_grid, extent=(0, self.size_x, 0, self.size_y), cmap='Greys', origin='lower', alpha=0.7)
            wavefront = self.wavefronts[i]
            # 一次波面のプロット
            if wavefront['primary'].size > 0:
                self.ax.scatter(wavefront['primary'][:, 0], wavefront['primary'][:, 1], s=1, c='blue', label='Primary Wavefront')
            # 二次波面のプロット
            if wavefront['secondary'].size > 0:
                self.ax.scatter(wavefront['secondary'][:, 0], wavefront['secondary'][:, 1], s=1, c='red', label='Secondary Wavefront')
            current_time = self.frame_indices[i] * dt * 1e9  # 正しい時間を計算
            self.ax.set_title(f'Time: {current_time:.2f} ns', fontsize=24)
            self.ax.set_xlabel('x (m)', fontsize=24)
            self.ax.set_ylabel('y (m)', fontsize=24)
            self.ax.set_xlim(0, self.size_x)
            self.ax.set_ylim(0, self.size_y)
            self.ax.legend(fontsize=16)
            self.ax.tick_params(labelsize=20)
            self.pbar.update(1)
            self.counter += 1
            return self.ax

        def close(self):
            self.pbar.close()

    animator = Animator(ax, animation_wavefronts, epsilon_grid, frame_indices, dt, size_x, size_y)
    ani = animation.FuncAnimation(fig, animator, frames=len(animation_wavefronts), interval=50)
    ani.save(os.path.join(output_dir, 'wave_simulation.gif'), writer='pillow')
    animator.close()
    print('Animation saved.')
