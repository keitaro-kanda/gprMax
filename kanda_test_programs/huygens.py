import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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



#* Constants
c0 = 3e8  # 真空中の光速



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



#* Define the function to initialize the wavefront
def init_wavefront(position_x, position_y):
    wavefront = np.zeros((ny, nx))
    source_position = (int(position_x / dx), int(position_y / dx)) # pixel
    #print('Source position:', source_position)
    source_radius_m = 0.15 # [m]
    source_radius = int(source_radius_m / dx) # pixel
    #print('Source radius:', source_radius)
    for y in range(source_position[1] - source_radius, source_position[1] + source_radius + 1):
        for x in range(source_position[0] - source_radius, source_position[0] + source_radius + 1):
            #if np.sqrt((x - source_position[0])**2 + (y - source_position[1])**2) <= source_radius:
            #    wavefront[y, x] = 1e6
            if 0 <= x < nx and 0 <= y < ny:
                distance = np.sqrt((x - source_position[0])**2 + (y - source_position[1])**2)
                if abs(distance - source_radius) <= 1:  # 円周上の厚さ1ピクセルの領域
                    wavefront[y, x] = 1e3

    return wavefront



#* Define the function to create Huygens kernel
"""
def create_huygens_kernel(radius_in_pixels):
    size = radius_in_pixels * 2 + 1
    #kernel = np.zeros((int(size_y/dx), int(size_x/dx)))
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
"""
def create_huygens_kernel(radius_in_pixels):
    size = radius_in_pixels * 2 + 1
    kernel = np.ones((size, size))
    kernel /= np.sum(kernel)
    return kernel


#* Main
if __name__ == '__main__':
    #* Load json file
    with open(args.json, 'r') as f:
        params = json.load(f)
    output_dir = os.path.dirname(args.json)

    #* Get the parameters
    size_x, size_y = params['simulation_space']['size']
    dx = params['simulation_space']['grid_spacing']

    dt = params['time']['time_step']
    time_window = params['time']['time_window']
    num_steps = int(time_window / dt)

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



    #* Plot the medium model
    figsize_ratio = size_y / size_x
    plt.figure(figsize=(10, 10 * figsize_ratio))
    plt.imshow(epsilon_grid, extent=(0, size_x, 0, size_y), cmap='binary', origin='lower')

    plt.xlabel('x (m)', fontsize=24)
    plt.ylabel('y (m)', fontsize=24)
    plt.tick_params(labelsize=20)

    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='5%', pad=0.5)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label(r'$\varepsilon_r$', fontsize=24)
    cbar.ax.tick_params(labelsize=20)

    plt.savefig(output_dir + '/medium_model.png')
    plt.close()
    print('Medium model saved.')
    print(' ')



    #* Initialize the wavefront
    #wavefront = init_wavefront(params['source']['position'][0], params['source']['position'][1])


    #* Create the Huygens kernel
    #radius_in_m = dx * 2  # [m]
    #radius_in_pixels = int(radius_in_m / dx) # pixel
    radius_in_pixels = 3
    kernel = create_huygens_kernel(radius_in_pixels)
    #print('Kernel shape:', kernel.shape)

    # 初期波面の設定
    wavefront_present = init_wavefront(params['source']['position'][0], params['source']['position'][1])
    wavefront_past = wavefront_present.copy()
    wavefront_future = np.zeros_like(wavefront_present)

    # 結果を保存するリスト
    frames = []
    c_grid = c0 / np.sqrt(epsilon_grid)

    for step in tqdm(range(num_steps + 1), desc='Calculating...'):
        if step == 0:
            initial_wavefront = init_wavefront(params['source']['position'][0], params['source']['position'][1])
            wavefront = initial_wavefront.copy()
        else:
            #new_wavefront = fftconvolve(wavefront, kernel, mode='same')
            #wavefront = new_wavefront.copy()

            # 波面の更新
            c_squared = (c_grid * dt / dx) ** 2
            
            # ラプラシアンの計算
            laplacian = (
                wavefront_present[:-2, 1:-1] + wavefront_present[2:, 1:-1] +
                wavefront_present[1:-1, :-2] + wavefront_present[1:-1, 2:] -
                4 * wavefront_present[1:-1, 1:-1]
            )
            
            # 波面の更新式
            wavefront_future[1:-1, 1:-1] = (
                2 * wavefront_present[1:-1, 1:-1] - wavefront_past[1:-1, 1:-1] +
                c_squared[1:-1, 1:-1] * laplacian
            )

            # 境界条件の適用（シミュレーション領域外への波の伝播を防ぐ）
            wavefront[0, :] = 0
            wavefront[-1, :] = 0
            wavefront[:, 0] = 0
            wavefront[:, -1] = 0

            # 時間ステップの更新
            wavefront_past = wavefront_present.copy()
            wavefront_present = wavefront_future.copy()

        # 結果の保存
        saving_interval_ns = 1e-9  # [s]
        saving_interval = int(saving_interval_ns / dt) # step
        #print('saving_interval:', saving_interval)
        if step % saving_interval  == 0:
            frames.append(wavefront.copy())
    print('Calculation done.')
    print(' ')



    # 出力ディレクトリの作成
    output_dir_frames = os.path.join(output_dir, 'wave_simulation_frames')
    if not os.path.exists(output_dir_frames):
        os.makedirs(output_dir_frames)

    # フレームの保存
    for i, frame in tqdm(enumerate(frames), desc='Saving frames...'):
        fig = plt.figure(figsize=(10, 10 * size_y / size_x), tight_layout=True)
        plt.imshow(frame, extent=(0, size_x, 0, size_y), cmap='viridis', origin='lower',
                    vmin=0, vmax=1
                    )

        plt.title(f'{i * saving_interval_ns / 1e-9} ns', fontsize=24)
        plt.xlabel('x (m)', fontsize=24)
        plt.ylabel('y (m)', fontsize=24)
        plt.tick_params(labelsize=20)

        #* Add the source position
        source_position = params['source']['position']
        source = patches.Circle((source_position[0], source_position[1]), radius=0.05, color='k', fill=False)
        plt.gca().add_patch(source)

        #* Add boundaries of the medium
        for medium in media:
            if medium['type'] == 'circle':
                circle = patches.Circle(medium['center'], radius=medium['radius'], color='r', fill=False, linestyle='dashed')
                plt.gca().add_patch(circle)
            elif medium['type'] == 'square':
                square = patches.Rectangle(medium['bottom_left'], medium['size'][0], medium['size'][1], color='r', fill=False, linestyle='dashed')
                plt.gca().add_patch(square)


        delvider = axgrid1.make_axes_locatable(plt.gca())
        cax = delvider.append_axes('right', size='5%', pad=0.5)
        cbar = plt.colorbar(cax=cax)
        cbar.set_label('Wave intensyty', fontsize=24)
        cbar.ax.tick_params(labelsize=20)

        plt.savefig(f'{output_dir_frames}/frame_{i:04d}.png')
        plt.close()
    print('Frames saved.')
    print(' ')



    #* Make animation
    figsize_ratio = size_y / size_x
    fig,ax = plt.subplots(figsize=(10, 10*figsize_ratio), tight_layout=True)
    im = ax.imshow(frames[0], extent=(0, size_x, 0, size_y), cmap='viridis', origin='lower',
                    vmin=0, vmax=1
                    )

    plt.xlabel('x (m)', fontsize=24)
    plt.ylabel('y (m)', fontsize=24)
    plt.tick_params(labelsize=20)

    #* Add the source position
    source_position = params['source']['position']
    source = patches.Circle((source_position[0], source_position[1]), radius=0.05, color='k', fill=False)
    plt.gca().add_patch(source)

    #* Add boundaries of the medium
    for medium in media:
        if medium['type'] == 'circle':
            circle = patches.Circle(medium['center'], radius=medium['radius'], color='r', fill=False, linestyle='dashed')
            plt.gca().add_patch(circle)
        elif medium['type'] == 'square':
            square = patches.Rectangle(medium['bottom_left'], medium['size'][0], medium['size'][1], color='r', fill=False, linestyle='dashed')
            plt.gca().add_patch(square)

    delvider = axgrid1.make_axes_locatable(plt.gca())
    cax = delvider.append_axes('right', size='5%', pad=0.5)
    cbar = plt.colorbar(cax=cax, mappable=im)
    cbar.set_label('Wave intensyty', fontsize=24)
    cbar.ax.tick_params(labelsize=20)

    def animate(i):
        im.set_array(frames[i])
        ax.set_title(f'{i * saving_interval_ns / 1e-9} ns', fontsize=24)
        return im,

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, blit=True)

    # アニメーションの保存
    ani.save(output_dir + '/wave_simulation.gif', writer='pillow')
    print('Animation saved.')
