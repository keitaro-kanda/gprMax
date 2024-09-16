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
dt = 1e-11 # [s]
time_window = 1e-9 # [s]
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
    return epsilon_r

# Initialize wavefront
def initialize_wavefront(Nx, Ny):
    wavefront = np.zeros((Nx, Ny), dtype=bool)
    # Set initial wavefront (e.g., a circle at the center)
    x0, y0 = Nx // 2 , Ny // 2 - Ny // 5
    radius = 0.5  # [m]
    x, y = np.ogrid[-x0:Nx - x0, -y0:Ny - y0]
    mask = x**2 + y**2 <= (radius / dx)**2
    wavefront[mask] = True
    return wavefront

# Main simulation loop
def huygens_simulation(epsilon_r, wavefront, c0, dx, dy, dt, T):
    c = c0 / np.sqrt(epsilon_r)
    frames = []
    for n in tqdm(range(int(T/dt)), desc='Simulating'):
        # Calculate the propagation distance for this time step
        s = c * dt * n
        # Use distance transform to find the front of the wavefront
        distance = distance_transform_edt(~wavefront) * dx
        # Generate new wavefront by expanding the current wavefront
        new_wavefront = distance <= s
        # Handle reflection and refraction at interfaces
        # For simplicity, we'll approximate reflection by keeping the wavefront within bounds
        # This can be improved by more advanced methods

        # Update the wavefront
        wavefront = new_wavefront.copy()
        frames.append(wavefront.copy())
    return frames

# Visualization
def animate_wavefront(frames, epsilon_r):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    extent = [0, epsilon_r.shape[1]*dx, 0, epsilon_r.shape[0]*dy]

    # 左のプロット：誘電体構造と波面
    dielectric_img = axs[0].imshow(
        np.flipud(epsilon_r),
        cmap='binary',
        extent=extent,
        interpolation='nearest',
        alpha=0.5
    )
    axs[0].set_title(r'$\epsilon_r$' + ' structure')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')

    delvider = axgrid1.make_axes_locatable(axs[0])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(dielectric_img, cax=cax)
    cbar.set_label(r'$\epsilon_r$')

    # 右のプロット：波面のみ
    wavefront_img_right = axs[1].imshow(
        np.flipud(frames[0]),
        cmap='viridis',
        extent=extent,
        interpolation='nearest',
        alpha=0.9
    )
    axs[1].set_title('Wavefront')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')

    delvider = axgrid1.make_axes_locatable(axs[1])
    cax = delvider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(wavefront_img_right, cax=cax)
    cbar.set_label('Wavefront')

    # アニメーションの更新関数
    def update(i):
        wavefront_img_right.set_data(np.flipud(frames[i]))
        return [wavefront_img_right]

    print('Animating...')
    print('Number of frames:', len(frames))
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=50, blit=True, repeat=False
    )

    plt.tight_layout()
    plt.show()


# Main function
def main():
    # Uncomment the following line to load dielectric constant from a file
    # epsilon_r = load_dielectric_constant('dielectric_constant.h5')

    # For this example, create a default dielectric grid
    epsilon_r = create_default_dielectric_constant(Nx, Ny)

    wavefront = initialize_wavefront(Nx, Ny)
    frames = huygens_simulation(
        epsilon_r, wavefront, c0, dx, dy, dt, time_window
    )
    animate_wavefront(frames, epsilon_r)

if __name__ == '__main__':
    main()
