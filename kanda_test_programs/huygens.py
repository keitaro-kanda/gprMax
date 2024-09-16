import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from scipy.ndimage import distance_transform_edt, binary_dilation
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1

# Simulation parameters
x_size, y_size = 10, 10 # [m]
dx, dy = 0.005, 0.005 # [m]
Nx, Ny = int(x_size / dx), int(y_size / dy)
dt = 0.1e-9 # [s]
time_window = 3e-9 # [s]
Nt = int(time_window / dt)
c0 = 299792458 # [m/s]

# Load dielectric constant grid from h5 file
def load_dielectric_constant(filename):
    with h5py.File(filename, 'r') as f:
        epsilon_r = f['/epsilon_r'][:]
    return epsilon_r

# For demonstration, create a default dielectric grid if no file is provided
def create_default_dielectric_constant(Nx, Ny):
    epsilon_r = np.ones((Nx, Ny))
    # Introduce a dielectric interface
    epsilon_r[:, Ny//2: Ny//4*3] = 4.0  # Right half has higher dielectric constant
    return np.flipud(epsilon_r.T)

# Initialize wavefront
def initialize_wavefront(Nx, Ny):
    wavefront = np.zeros((Nx, Ny), dtype=bool)
    # Set initial wavefront (e.g., a circle at the center)
    source_position = 5, 2  # [m]
    x0, y0 = int(source_position[0] / dx), int(source_position[1] / dy)
    radius = 0.2  # [m]
    x, y = np.ogrid[-x0:Nx - x0, -y0:Ny - y0]
    mask = x**2 + y**2 <= (radius / dx)**2
    wavefront[mask] = True
    return source_position, wavefront.T

# Main simulation loop
def huygens_simulation(epsilon_r, wavefront, c0, dx, dy, dt, T):
    v = c0 / np.sqrt(epsilon_r)
    frames = []
    for n in tqdm(range(int(T/dt)), desc='Simulating'):
        # Calculate the propagation distance for this time step
        s = v * dt * n
        # Use distance transform to find the front of the wavefront
        distance = distance_transform_edt(~wavefront) * dx
        # Generate new wavefront by expanding the current wavefront
        #new_wavefront = distance <= s
        new_wavefront = np.logical_and(distance <= s, distance >= s - c0 * dt)
        #new_wavefront = distance > s - c * dt
        # Handle reflection and refraction at interfaces
        # For simplicity, we'll approximate reflection by keeping the wavefront within bounds
        # This can be improved by more advanced methods

        # Update the wavefront
        wavefront = new_wavefront.copy()
        frames.append(wavefront.copy())
    return frames

# Visualization
def animate_wavefront(frames, epsilon_r, source_position):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    extent = [0, epsilon_r.shape[1]*dx, epsilon_r.shape[0]*dy, 0]

    # 左のプロット：誘電体構造と波面
    dielectric_img = axs[0].imshow(
        np.flipud(epsilon_r),
        cmap='binary',
        extent=extent,
        interpolation='nearest',
        alpha=1
    )
    axs[0].plot(source_position[0], source_position[1], 'rx')
    axs[0].set_title(r'$\epsilon_r$' + ' structure')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')

    delvider = axgrid1.make_axes_locatable(axs[0])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
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
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    delvider = axgrid1.make_axes_locatable(axs[1])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
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
        fig, update, frames=len(frames), interval=1000/fps, blit=True, repeat=False
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

    source_position, wavefront = initialize_wavefront(Nx, Ny)
    frames = huygens_simulation(
        epsilon_r, wavefront, c0, dx, dy, dt, time_window
    )
    animate_wavefront(frames, epsilon_r, source_position)

if __name__ == '__main__':
    main()
