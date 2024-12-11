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
def init_wavefront(position_x, position_y, source_radius):
    num_points = 9
    angles = np.linspace(5/4 * np.pi, 3/2 * np.pi, num_points, endpoint=True) # 45 degree
    wave_points = np.array([
        position_x + source_radius * np.cos(angles),
        position_y + source_radius * np.sin(angles)
    ]).T
    return wave_points

def create_new_wavefronts(position_x, position_y, source_radius):
    num_points = 36
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False) # 360 degree
    wave_points = np.array([
        position_x + source_radius * np.cos(angles),
        position_y + source_radius * np.sin(angles)
    ]).T
    return wave_points


# 波面のクラス
class Wavefront:
    def __init__(self, positions, time_created, source_position, dt, c0, epsilon_grid, size_x, size_y, nx, ny):
        self.positions = positions
        self.time_created = time_created
        self.source_position = source_position
        self.active = True
        self.dt = dt
        self.c0 = c0
        self.epsilon_grid = epsilon_grid
        self.size_x = size_x
        self.size_y = size_y
        self.nx = nx
        self.ny = ny

    def update(self):
        new_positions = []
        new_wavefronts = []
        dx = self.size_x / self.nx
        dy = self.size_y / self.ny

        for point in self.positions:
            x, y = point
            if 0 <= x < self.size_x and 0 <= y < self.size_y:
                i = int(y / dy)
                j = int(x / dx)
                epsilon_at_point = self.epsilon_grid[i, j]
                v_at_point = self.c0 / np.sqrt(epsilon_at_point)

                direction = point - np.array(self.source_position)
                norm = np.linalg.norm(direction)
                if norm != 0:
                    direction = direction / norm
                    new_point = point + v_at_point * self.dt * direction
                    i_new = int(new_point[1] / dy)
                    j_new = int(new_point[0] / dx)

                    if 0 <= i_new < self.ny and 0 <= j_new < self.nx:
                        epsilon_new = self.epsilon_grid[i_new, j_new]
                        if epsilon_at_point != epsilon_new:
                            # 境界に達した点から新たな球面波を生成
                            # 小さな半径の円周上に点群を作る
                            secondary_radius = dx  # 極小半径
                            new_wave_points = create_new_wavefronts(
                                new_point[0],  # new_pointのx座標
                                new_point[1],  # new_pointのy座標
                                secondary_radius
                            )

                            new_wavefront = Wavefront(
                                positions=new_wave_points.tolist(),
                                time_created=self.time_created + self.dt,
                                source_position=new_point.copy(),
                                dt=self.dt,
                                c0=self.c0,
                                epsilon_grid=self.epsilon_grid,
                                size_x=self.size_x,
                                size_y=self.size_y,
                                nx=self.nx,
                                ny=self.ny
                            )
                            new_wavefronts.append(new_wavefront)
                            # この点は新たな波面として独立したため、元のwavefrontからは削除
                            # （つまり、new_positionsには追加しない）
                        else:
                            # 境界に達していない場合、継続
                            new_positions.append(new_point)
                    else:
                        # シミュレーション領域外の場合はこの点は無効
                        # 何も追加しない
                        pass
                else:
                    # 方向がない場合、この点は残せない
                    pass
            else:
                # シミュレーション領域外の場合、この点は無効
                pass

        self.positions = new_positions
        if len(self.positions) == 0 and len(new_wavefronts) == 0:
            # すべての点が境界か領域外で消えた場合
            self.active = False

        return new_wavefronts

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
    #v_grid = c0 / np.sqrt(epsilon_grid)

    # 初期波面の設定
    source_position = params['source']['position']
    source_radius = params['source']['radius']  # 波源の半径 [m]（必要に応じて調整）
    initial_positions = init_wavefront(source_position[0], source_position[1], source_radius)

    # 初期波面をWavefrontとして生成
    wavefronts = [
        Wavefront(
            positions=initial_positions.tolist(),
            time_created=0.0,
            source_position=source_position,
            dt=dt,
            c0=c0,
            epsilon_grid=epsilon_grid,
            size_x=size_x,
            size_y=size_y,
            nx=nx,
            ny=ny
        )
    ]

    wavefronts_history = []

    for step in tqdm(range(num_steps), desc='Calculating...'):
        new_wavefronts = []
        for w in tqdm(wavefronts, desc=f'Calculate wavefron: step {step}', leave=False):
            if w.active:
                gen_wfs = w.update()
                new_wavefronts.extend(gen_wfs)
        # 新たに生成された波面を追加
        wavefronts.extend(new_wavefronts)
        # アクティブでない波面を削除
        wavefronts = [w for w in wavefronts if w.active]
        # 記録
        wavefronts_history.append([w.positions for w in wavefronts])

    print('Calculation done.')
    print(' ')

    # 50タイムステップごとにフレームを保存
    frame_indices = [i for i in range(len(wavefronts_history )+1) if i % 50 == 0]
    print(f"Total frames to save: {len(frame_indices)}")

    output_dir_frames = os.path.join(output_dir, 'wave_simulation_frames')
    if not os.path.exists(output_dir_frames):
        os.makedirs(output_dir_frames)

    for idx, i in tqdm(enumerate(frame_indices), desc='Saving frames...', total=len(frame_indices)):
        wfs = wavefronts_history[i]
        plt.figure(figsize=(10, 10 * size_y / size_x))
        plt.imshow(epsilon_grid, extent=(0, size_x, 0, size_y), cmap='Greys', origin='lower', alpha=0.7)
        # 全ての波面を同じようにプロット(世代を区別しない)
        colors = ['blue', 'red', 'green', 'magenta', 'cyan']  # 必要なら世代ごとに色変えても良いが、ここでは不要
        for j, pos in enumerate(wfs):
            if len(pos) > 0:
                pos = np.array(pos)
                plt.scatter(pos[:, 0], pos[:, 1], s=1, c='blue')
        current_time = i * dt * 1e9  # 正しい時間を計算
        plt.title(f'Time: {current_time:.2f} ns', fontsize=24)
        plt.xlabel('x (m)', fontsize=24)
        plt.ylabel('y (m)', fontsize=24)
        plt.xlim(0, size_x)
        plt.ylim(0, size_y)
        plt.tick_params(labelsize=20)
        plt.savefig(f'{output_dir_frames}/frame_{idx:04d}.png')
        plt.close()
    print('Frames saved.')
    print(' ')

    # アニメーションの作成
    print('Creating animation...')

    figsize_ratio = size_y / size_x
    fig, ax = plt.subplots(figsize=(10, 10 * figsize_ratio))

    animation_wavefronts = [wavefronts_history[i] for i in frame_indices]

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
            wfs = self.wavefronts[i]  # wfsは posのnumpy配列を複数格納したリスト
            for pos in wfs:
                pos = np.array(pos)
                if pos.size > 0:
                    # posが1点のみの場合、shapeは(2,)となる
                    # reshapeして常に(N,2)形式にする
                    pos = pos.reshape(-1, 2)
                    # これでpos[:,0], pos[:,1]が安全に使える
                    self.ax.scatter(pos[:, 0], pos[:, 1], s=1, c='blue')
            current_time = self.frame_indices[i] * self.dt * 1e9
            self.ax.set_title(f'Time: {current_time:.2f} ns', fontsize=24)
            self.ax.set_xlabel('x (m)', fontsize=24)
            self.ax.set_ylabel('y (m)', fontsize=24)
            self.ax.set_xlim(0, self.size_x)
            self.ax.set_ylim(0, self.size_y)
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
