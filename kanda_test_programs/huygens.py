import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from scipy.ndimage import distance_transform_edt, binary_dilation
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1
import heapq

# Simulation parameters
x_size, y_size = 10, 10  # [m]
dx, dy = 0.005, 0.005  # [m]
Nx, Ny = int(x_size / dx), int(y_size / dy)
dt = 0.1e-9  # [s]
time_window = 3e-9  # [s]
Nt = int(time_window / dt)
c0 = 299792458  # [m/s]

# Load dielectric constant grid from h5 file
def load_dielectric_constant(filename):
    with h5py.File(filename, 'r') as f:
        epsilon_r = f['/epsilon_r'][:]
    return epsilon_r

# For demonstration, create a default dielectric grid if no file is provided
def create_default_dielectric_constant(Nx, Ny):
    epsilon_r = np.ones((Nx, Ny))
    # Introduce a dielectric interface
    epsilon_r[:, Ny // 2 : Ny // 4 * 3] = 4.0  # Right half has higher dielectric constant
    return epsilon_r.T

# Compute arrival times using Dijkstra's algorithm with 8-connectivity
def compute_arrival_times(v, source_position, dx, dy):
    Nx, Ny = v.shape
    arrival_time = np.full((Nx, Ny), np.inf)
    x0, y0 = int(source_position[0] / dx), int(source_position[1] / dy)
    arrival_time[x0, y0] = 0

    # Priority queue: (arrival_time, (x, y))
    heap = []
    heapq.heappush(heap, (0, (x0, y0)))

    visited = np.zeros((Nx, Ny), dtype=bool)

    # Define neighbor offsets for 8-connectivity
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]

    while heap:
        current_time, (x, y) = heapq.heappop(heap)

        if visited[x, y]:
            continue
        visited[x, y] = True

        for dx_offset, dy_offset in neighbor_offsets:
            nx, ny = x + dx_offset, y + dy_offset

            if 0 <= nx < Nx and 0 <= ny < Ny:
                if visited[nx, ny]:
                    continue

                # Compute the travel distance to neighbor
                if dx_offset == 0 or dy_offset == 0:
                    ds = dx  # Horizontal or vertical neighbor
                else:
                    ds = np.sqrt(dx**2 + dy**2)  # Diagonal neighbor

                # Average speed between current and neighbor cell
                v_avg = 0.5 * (v[x, y] + v[nx, ny])
                travel_time = ds / v_avg

                tentative_arrival_time = current_time + travel_time

                if tentative_arrival_time < arrival_time[nx, ny]:
                    arrival_time[nx, ny] = tentative_arrival_time
                    heapq.heappush(heap, (arrival_time[nx, ny], (nx, ny)))

    return arrival_time

# Visualization
def animate_wavefront(frames, epsilon_r, source_position):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    extent = [0, epsilon_r.shape[1]*dx, epsilon_r.shape[0]*dy, 0]

    # 左のプロット：誘電体構造と波源
    dielectric_img = axs[0].imshow(
        epsilon_r,
        cmap='binary',
        extent=extent,
        interpolation='nearest',
        alpha=1
    )
    axs[0].plot(source_position[0], source_position[1], 'rx')
    axs[0].set_title(r'$\epsilon_r$' + ' structure')
    axs[0].set_xlabel('X [m]')
    axs[0].set_ylabel('Y [m]')

    divider = axgrid1.make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(dielectric_img, cax=cax)
    cbar.set_label(r'$\epsilon_r$')

    # 右のプロット：波面のみ
    wavefront_img_right = axs[1].imshow(
        frames[0],
        cmap='viridis',
        extent=extent,
        interpolation='nearest',
        alpha=1
    )
    axs[1].plot(source_position[0], source_position[1], 'rx')
    axs[1].set_title('Wavefront')
    axs[1].set_xlabel('X [m]')
    axs[1].set_ylabel('Y [m]')

    divider = axgrid1.make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(wavefront_img_right, cax=cax)
    cbar.set_label('Wavefront')

    # アニメーションの更新関数
    def update(i):
        wavefront_img_right.set_data(frames[i])
        time_in_ns = i * dt / 1e-9
        axs[1].set_title(f'Wavefront at t = {time_in_ns:.2f} ns')
        return [wavefront_img_right]

    print('Animating...')
    print('Number of frames:', len(frames))
    fps = 10
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True, repeat=False
    )

    plt.tight_layout()
    ani.save('kanda_test_programs/wavefront_animation.mp4', writer='ffmpeg', fps=fps)
    plt.show()

# Main function
def main():
    # Uncomment the following line to load dielectric constant from a file
    # epsilon_r = load_dielectric_constant('dielectric_constant.h5')

    # For this example, create a default dielectric grid
    epsilon_r = create_default_dielectric_constant(Nx, Ny)
    v = c0 / np.sqrt(epsilon_r)

    source_position = (5, 2)  # [m]

    arrival_time = compute_arrival_times(v, source_position, dx, dy)

    # Generate frames
    frames = []
    tol = dt  # Tolerance for wavefront thickness

    Nt = int(time_window / dt)
    for n in range(Nt):
        t = n * dt
        wavefront = np.abs(arrival_time - t) <= tol
        frames.append(wavefront)

    animate_wavefront(frames, epsilon_r, source_position)

if __name__ == '__main__':
    main()
