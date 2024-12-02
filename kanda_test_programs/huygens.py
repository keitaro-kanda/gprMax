import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
import argparse

# 定数
c0 = 3e8  # 真空中の光速 [m/s]

# コマンドライン引数の解析
parser = argparse.ArgumentParser(
    prog='huygens_wave_simulation.py',
    description='Simulate wave propagation using Huygens principle with recursive secondary sources at media boundaries.',
    epilog='End of help message',
    usage='python huygens_wave_simulation.py [json]',
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

# 波面クラスの定義
class Wavefront:
    def __init__(self, positions, time_created, source_positions):
        self.positions = positions
        self.time_created = time_created
        self.source_positions = source_positions
        self.active = True

    def update(self, dt, c_grid, epsilon_grid, size_x, size_y, nx, ny, dx, dy):
        new_positions = []
        new_wavefronts = []
        for idx, point in enumerate(self.positions):
            x, y = point
            source_position = self.source_positions[idx]
            if 0 <= x < size_x and 0 <= y < size_y:
                i = int(y / dy)
                j = int(x / dx)
                epsilon = epsilon_grid[i, j]
                c = c0 / np.sqrt(epsilon)
                direction = point - source_position
                if np.linalg.norm(direction) != 0:
                    direction = direction / np.linalg.norm(direction)
                    new_point = point + c * dt * direction
                    # 媒質の境界チェック
                    i_new = int(new_point[1] / dy)
                    j_new = int(new_point[0] / dx)
                    if 0 <= i_new < ny and 0 <= j_new < nx:
                        epsilon_new = epsilon_grid[i_new, j_new]
                        if epsilon != epsilon_new:
                            # 境界に到達した場合、新たな波面を生成
                            new_wavefront = Wavefront(
                                positions=[point.copy()],
                                time_created=self.time_created + dt,
                                source_positions=[point.copy()]
                            )
                            new_wavefronts.append(new_wavefront)
                        else:
                            new_positions.append(new_point)
                    else:
                        self.active = False  # シミュレーション領域外
                else:
                    self.active = False  # 方向が定義できない
            else:
                self.active = False  # シミュレーション領域外
        self.positions = new_positions
        self.source_positions = [self.source_positions[idx] for idx in range(len(self.positions))]
        if not self.positions:
            self.active = False  # 波面が消滅
        return new_wavefronts  # 新たに生成された波面を返す

# メイン
if __name__ == '__main__':
    # パラメータの読み込み
    with open(args.json, 'r') as f:
        params = json.load(f)
    output_dir = os.path.dirname(args.json)

    # シミュレーション空間の設定（左半分）
    size_x, size_y = params['simulation_space']['size']
    size_x = size_x / 2  # 左半分のみ
    dx = params['simulation_space']['grid_spacing']
    dy = dx  # 正方格子を想定
    nx, ny = int(size_x / dx), int(size_y / dy)
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
            center = medium['center']
            if center[0] <= size_x:  # 左半分にある媒質のみ追加
                epsilon_grid = add_circle(
                    epsilon_grid,
                    medium['center'],
                    medium['radius'],
                    medium['epsilon'],
                    X, Y
                )
        elif medium['type'] == 'square':
            bottom_left = medium['bottom_left']
            if bottom_left[0] + medium['size'][0] <= size_x:
                epsilon_grid = add_square(
                    epsilon_grid,
                    medium['bottom_left'],
                    medium['size'],
                    medium['epsilon'],
                    X, Y
                )

    # 波速マップの計算
    c_grid = c0 / np.sqrt(epsilon_grid)

    # 波面の初期化
    wavefronts = []

    # 初期波面（左半分のみ）
    source_position = params['source']['position']
    if source_position[0] > size_x:
        print("Source position is outside the simulation domain.")
        exit()
    source_radius = 0.5  # 波源の半径 [m]
    num_points = 180  # 波面上の点の数
    angles = np.linspace(np.pi / 2, np.pi * 3 / 2, num_points)  # 左半分
    initial_positions = np.array([
        source_position[0] + source_radius * np.cos(angles),
        source_position[1] + source_radius * np.sin(angles)
    ]).T
    initial_source_positions = np.array([source_position] * num_points)

    initial_wavefront = Wavefront(
        positions=initial_positions.tolist(),
        time_created=0.0,
        source_positions=initial_source_positions.tolist()
    )
    wavefronts.append(initial_wavefront)

    # 波面の履歴を保存するリスト
    wavefronts_history = []

    # フレームを保存するディレクトリを作成
    output_dir_frames = os.path.join(output_dir, 'wave_simulation_frames')
    if not os.path.exists(output_dir_frames):
        os.makedirs(output_dir_frames)

    # シミュレーションの実行
    for step in tqdm(range(num_steps), desc='Calculating...'):
        current_time = step * dt
        new_wavefronts = []
        for wavefront in wavefronts:
            if wavefront.active:
                generated_wavefronts = wavefront.update(
                    dt, c_grid, epsilon_grid,
                    size_x, size_y, nx, ny, dx, dy
                )
                new_wavefronts.extend(generated_wavefronts)
        # 新たに生成された波面を追加
        wavefronts.extend(new_wavefronts)
        # アクティブでない波面を削除
        wavefronts = [w for w in wavefronts if w.active]
        # 可視化のために波面の位置を保存
        wavefronts_history.append([w.positions for w in wavefronts])

    print('Calculation done.')
    print(' ')

    # 可視化とフレームの保存
    for i, wavefronts in tqdm(enumerate(wavefronts_history), desc='Saving frames...', total=len(wavefronts_history)):
        plt.figure(figsize=(10, 10 * size_y / size_x))
        plt.imshow(epsilon_grid, extent=(0, size_x, 0, size_y), cmap='Greys', origin='lower', alpha=0.3)
        # 全ての波面をプロット
        for positions in wavefronts:
            if positions:
                positions = np.array(positions)
                plt.scatter(positions[:, 0], positions[:, 1], s=1, c='blue')
        plt.title(f'Time: {i * dt * 1e9:.2f} ns', fontsize=24)
        plt.xlabel('x (m)', fontsize=24)
        plt.ylabel('y (m)', fontsize=24)
        plt.xlim(0, size_x)
        plt.ylim(0, size_y)
        plt.tick_params(labelsize=20)
        plt.savefig(f'{output_dir_frames}/frame_{i:04d}.png')
        plt.close()
    print('Frames saved.')
    print(' ')

    # アニメーションの作成
    import matplotlib.animation as animation

    figsize_ratio = size_y / size_x
    fig, ax = plt.subplots(figsize=(10, 10 * figsize_ratio))

    def animate_func(i):
        ax.clear()
        ax.imshow(epsilon_grid, extent=(0, size_x, 0, size_y), cmap='Greys', origin='lower', alpha=0.3)
        wavefronts = wavefronts_history[i]
        for positions in wavefronts:
            if positions:
                positions = np.array(positions)
                ax.scatter(positions[:, 0], positions[:, 1], s=1, c='blue')
        ax.set_title(f'Time: {i * dt * 1e9:.2f} ns', fontsize=24)
        ax.set_xlabel('x (m)', fontsize=24)
        ax.set_ylabel('y (m)', fontsize=24)
        ax.set_xlim(0, size_x)
        ax.set_ylim(0, size_y)
        ax.tick_params(labelsize=20)
        return ax

    ani = animation.FuncAnimation(fig, animate_func, frames=len(wavefronts_history), interval=50)
    ani.save(os.path.join(output_dir, 'wave_simulation.gif'), writer='pillow')
    print('Animation saved.')
