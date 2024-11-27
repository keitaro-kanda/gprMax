import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1
import json
from scipy.signal import fftconvolve
import os
import argparse


#* Parse command line arguments
parser = argparse.ArgumentParser(
    prog='huygens.py',
    description='Simulate wave propagation using Huygens principle',
    epilog='End of help message',
    usage='python kanda_test_programs/huygens/huygens.py [json]',
)
parser.add_argument('json', help='Path to the parameter json file')
args = parser.parse_args()



#* Define the function to make circular medium
def add_circle(grid, center, radius, epsilon, x, y):
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    grid[mask] = epsilon
    return grid


#* Define the function to make square medium
def add_square(grid, bottom_left, size, epsilon, x, y):
    mask = ( (x >= bottom_left[0]) & (x <= bottom_left[0] + size[0]) &
            (y >= bottom_left[1]) & (y <= bottom_left[1] + size[1]) )
    grid[mask] = epsilon
    return grid


#* Constants
c0 = 3e8  # 真空中の光速


#* Load json file
with open(args.json, 'r') as f:
    params = json.load(f)
output_dir = os.path.dirname(args.json)



#* Get the parameters
size_x, size_y = params['simulation_space']['size']
dx = params['simulation_space']['grid_spacing']
dt = params['time']['time_step']
time_window = params['time']['time_window']
media = params['media']



#* Make the grid
nx = int(size_x / dx)
ny = int(size_y / dx)
x = np.linspace(0, size_x, nx)
y = np.linspace(0, size_y, ny)
X, Y = np.meshgrid(x, y)

#* Initialize the epsilon grid
epsilon_grid = np.ones((ny, nx))

#* Add the media
for medium in media:
    if medium['type'] == 'circle':
        epsilon_grid = add_circle(epsilon_grid, medium['center'], medium['radius'], medium['epsilon'], X, Y)
    elif medium['type'] == 'square':
        epsilon_grid = add_square(epsilon_grid, medium['bottom_left'], medium['size'], medium['epsilon'], X, Y)



# 媒質モデルの保存
plt.figure()
plt.imshow(epsilon_grid, extent=(0, size_x, 0, size_y), cmap='binary', origin='lower')
plt.colorbar(label='Epsilon (Permittivity)')
plt.title('Medium Model (Epsilon Distribution)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig(output_dir + '/medium_model.png')
plt.close()


#* Initialize the wavefront
wavefront = np.zeros((ny, nx))
source_position = params['source']['position']
source_position = (int(source_position[0] / dx), int(source_position[1] / dx))
wavefront[source_position] = 1.0  # 強度を1とする



#* Define the function
def create_huygens_kernel(radius_in_pixels):
    size = radius_in_pixels * 2 + 1
    kernel = np.zeros((size, size))
    center = radius_in_pixels
    for y in range(size):
        for x in range(size):
            distance = np.sqrt((x - center)**2 + (y - center)**2)
            if distance <= radius_in_pixels:
                # 二次波源の影響を距離の逆数で表現
                kernel[y, x] = 1 / (distance + 1e-6)  # ゼロ割り防止
    # カーネルを正規化
    kernel /= np.sum(kernel)
    return kernel



#* Save as image
# シミュレーションの時間ステップ数
num_steps = int(time_window / dt)

# 二次波源のカーネルを作成
radius_in_m = 5 # [m]
radius_in_pixels = int(radius_in_m / dx) # pixel
kernel = create_huygens_kernel(radius_in_pixels)



# 結果を保存するリスト
frames = []

for step in tqdm(range(num_steps), desc='Calculating...'):
    # 波面の畳み込みによる次の波面の計算
    new_wavefront = fftconvolve(wavefront, kernel, mode='same')

    # 媒質による速度の影響を考慮
    c_grid = c0 / np.sqrt(epsilon_grid)
    propagation_factor = dt * c_grid

    # 新しい波面の更新
    wavefront = new_wavefront * propagation_factor

    # 境界条件の適用（シミュレーション領域外への波の伝播を防ぐ）
    wavefront[0, :] = 0
    wavefront[-1, :] = 0
    wavefront[:, 0] = 0
    wavefront[:, -1] = 0

    # 結果の保存
    if step % 2  == 0:
        frames.append(wavefront.copy())
print('Calculation done.')
print(' ')



# 出力ディレクトリの作成
output_dir_frames = os.path.join(output_dir, 'wave_simulation_frames')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# フレームの保存
for i, frame in tqdm(enumerate(frames), desc='Saving frames...'):
    plt.imshow(frame, extent=(0, size_x, 0, size_y), cmap='viridis', origin='lower')
    plt.colorbar(label='Wave Intensity')
    plt.title(f'Time step: {i * 2}')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.savefig(f'{output_dir_frames}/frame_{i:04d}.png')
    plt.close()
print('Frames saved.')
print(' ')



#* Make animation
fig, ax = plt.subplots()
cax = ax.imshow(frames[0], extent=(0, size_x, 0, size_y), cmap='viridis', origin='lower')
fig.colorbar(cax, label='Wave Intensity')

def animate(i):
    cax.set_array(frames[i])
    ax.set_title(f'Time step: {i * 10}')
    return cax,

ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)

# アニメーションの保存
ani.save(output_dir + '/wave_simulation.gif', writer='pillow')
print('Animation saved.')
