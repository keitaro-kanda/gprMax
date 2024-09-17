import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import mpl_toolkits.axes_grid1 as axgrid1

# Simulation parameters
x_size, y_size = 10, 10  # [m]
dx, dy = 0.02, 0.02  # [m]
Nx, Ny = int(x_size / dx), int(y_size / dy)
dt = 0.005e-9  # [s]
time_window = 3e-9  # [s]
Nt = int(time_window / dt)
c0 = 299792458  # [m/s]

# Create default dielectric grid
def create_default_dielectric_constant(Nx, Ny):
    epsilon_r = np.ones((Nx, Ny))
    # Introduce a dielectric interface
    epsilon_r[:, Ny // 2:] = 4.0  # Right half has higher dielectric constant
    return epsilon_r

# Initialize level set function (phi)
def initialize_level_set(Nx, Ny, source_position):
    x = np.linspace(0, x_size, Nx)
    y = np.linspace(0, y_size, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    phi = np.sqrt((X - source_position[0])**2 + (Y - source_position[1])**2) - 0.1  # Initial radius
    return phi

# Reflective boundary conditions
def apply_reflective_boundary(phi):
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]
    phi[:, 0] = phi[:, 1]
    phi[:, -1] = phi[:, -2]
    return phi

# Eikonal equation solver using upwind scheme
def eikonal_solver(phi, v, dt, dx, dy):
    phi_new = phi.copy()

    # Compute gradients using upwind scheme
    phi_x_forward = (np.roll(phi, -1, axis=0) - phi) / dx
    phi_x_backward = (phi - np.roll(phi, 1, axis=0)) / dx
    phi_y_forward = (np.roll(phi, -1, axis=1) - phi) / dy
    phi_y_backward = (phi - np.roll(phi, 1, axis=1)) / dy

    phi_x_positive = np.maximum(phi_x_backward, 0)
    phi_x_negative = np.minimum(phi_x_forward, 0)
    phi_y_positive = np.maximum(phi_y_backward, 0)
    phi_y_negative = np.minimum(phi_y_forward, 0)

    grad_phi = np.sqrt(
        np.maximum(phi_x_positive, -phi_x_negative)**2 +
        np.maximum(phi_y_positive, -phi_y_negative)**2
    )

    phi_new -= v * dt * grad_phi
    return phi_new

# Handle wavefront splitting at interfaces
def handle_interfaces(phi, v, epsilon_r, dt, dx, dy):
    # Identify interface cells
    grad_epsilon_x = np.abs(np.roll(epsilon_r, -1, axis=0) - epsilon_r) > 0
    grad_epsilon_y = np.abs(np.roll(epsilon_r, -1, axis=1) - epsilon_r) > 0
    interface_cells = np.logical_or(grad_epsilon_x, grad_epsilon_y)

    # Find cells where wavefront reaches the interface
    wavefront = np.abs(phi) < (dx + dy)
    interface_wavefront = np.logical_and(wavefront, interface_cells)

    # Create reflected wave
    phi_reflected = phi.copy()
    phi_reflected[~interface_wavefront] = np.inf  # Only consider interface wavefront cells

    # Update reflected wave using reflection coefficient
    # For simplicity, assume total reflection (reflection coefficient = 1)
    phi_reflected = eikonal_solver(phi_reflected, v, dt, dx, dy)
    phi_reflected = apply_reflective_boundary(phi_reflected)

    return phi_reflected

# Main function
def main():
    epsilon_r = create_default_dielectric_constant(Nx, Ny)
    v = c0 / np.sqrt(epsilon_r)

    source_position = (5, 2)  # [m]

    phi = initialize_level_set(Nx, Ny, source_position)
    phi_reflected = np.full_like(phi, np.inf)  # Initialize reflected wave level set

    frames = []

    for n in tqdm(range(Nt), desc='Simulating'):
        phi = eikonal_solver(phi, v, dt, dx, dy)
        phi = apply_reflective_boundary(phi)

        # Handle reflection at interfaces
        phi_reflected_new = handle_interfaces(phi, v, epsilon_r, dt, dx, dy)
        phi_reflected = np.minimum(phi_reflected, phi_reflected_new)

        # Update reflected wave
        phi_reflected = eikonal_solver(phi_reflected, v, dt, dx, dy)
        phi_reflected = apply_reflective_boundary(phi_reflected)

        # Combine original and reflected waves
        total_phi = np.minimum(phi, phi_reflected)

        # Capture zero level set as wavefront
        wavefront = np.abs(total_phi) < (dx + dy)
        frames.append(wavefront)

    animate_wavefront(frames, epsilon_r, source_position)

# Visualization
def animate_wavefront(frames, epsilon_r, source_position):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    extent = [0, x_size, y_size, 0]

    # Left plot: Dielectric structure and source
    dielectric_img = axs[0].imshow(
        epsilon_r.T,
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

    # Right plot: Wavefront only
    wavefront_img_right = axs[1].imshow(
        frames[0].T,
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

    # Animation update function
    def update(i):
        wavefront_img_right.set_data(frames[i].T)
        time_in_ns = i * dt / 1e-9
        axs[1].set_title(f'Wavefront at t = {time_in_ns:.2f} ns')
        return [wavefront_img_right]

    print('Animating...')
    print('Number of frames:', len(frames))
    fps = 30
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True, repeat=False
    )

    plt.tight_layout()
    ani.save('kanda_test_programs/wavefront_animation.mp4', writer='ffmpeg', fps=fps)
    plt.show()

if __name__ == '__main__':
    main()
