#!/usr/bin/env python3
import os
import glob
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.util.numpy_support import vtk_to_numpy
from matplotlib.colors import ListedColormap, BoundaryNorm
import mpl_toolkits.axes_grid1 as axgrid1


def read_vti_image(filename):
    """Reads a VTI file using VTK and returns the vtkImageData object."""
    reader = vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def extract_slice(img, field_name, target_z=0.0, tol=1e-6):
    """
    Extracts (x,y) coords and z-component of field on nearest target_z slice.
    Returns coords2d and values.
    """
    def choose_slice_z(z_coords):
        uniq = np.unique(z_coords)
        return uniq[np.argmin(np.abs(uniq - target_z))]

    pd = img.GetPointData()
    array = pd.GetArray(field_name)
    if array is None:
        cd = img.GetCellData()
        array = cd.GetArray(field_name)
        if array is None:
            raise KeyError(f"Field '{field_name}' not found.")
        center = vtkCellCenters()
        center.SetInputData(img)
        center.Update()
        mesh = center.GetOutput()
    else:
        mesh = img

    pts = vtk_to_numpy(mesh.GetPoints().GetData()).reshape(-1,3)
    vals = vtk_to_numpy(array)
    slice_z = choose_slice_z(pts[:,2])
    mask = np.isclose(pts[:,2], slice_z, atol=tol)
    coords2d = pts[mask][:,:2]
    vals2d = vals[mask]
    if vals2d.ndim>1 and vals2d.shape[1]>=3:
        vals2d = vals2d[:,2]
    return coords2d, vals2d


def main():
    print("[INFO] Starting k_plot_snapshot...")
    geometry_path = input("Enter path to geometry.vti: ").strip()
    print(f"[INFO] Geometry path: {geometry_path}")
    if not os.path.isfile(geometry_path):
        raise FileNotFoundError(f"Cannot find file: {geometry_path}")
    parent_dir = os.path.dirname(geometry_path)
    print(f"[INFO] Parent directory: {parent_dir}")

    # Zoom option
    do_zoom = input("Generate zoomed plot? (y/n): ").strip().lower() == 'y'
    if do_zoom:
        x_min = float(input("Enter zoom x_min [m]: ").strip())
        x_max = float(input("Enter zoom x_max [m]: ").strip())
        y_min = float(input("Enter zoom y_min [m]: ").strip())
        y_max = float(input("Enter zoom y_max [m]: ").strip())
        print(f"[INFO] Zoom region set to x:[{x_min}, {x_max}], y:[{y_min}, {y_max}]")

    snap_dirs = [d for d in os.listdir(parent_dir) if d.endswith("_snaps")]
    if len(snap_dirs)==1:
        snap_dir = os.path.join(parent_dir, snap_dirs[0])
    else:
        snap_dir = input("Enter snapshot directory: ").strip()
    print(f"[INFO] Snapshot directory: {snap_dir}")
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"Snap dir not found: {snap_dir}")

    ez_field = "E-field"
    fps = 10
    output_dir = os.path.join(parent_dir, "snapshot")
    if do_zoom:
        output_dir = os.path.join(output_dir, f'snapshot_zoom_{x_min}_{x_max}_{y_min}_{y_max}')
    output_video_path = os.path.join(output_dir, "snapshot_animation.mp4")
    print(f"[INFO] Video output path: {output_video_path}")

    # Geometry slice
    print("[INFO] Reading geometry VTI and extracting slice...")
    geom_img = read_vti_image(geometry_path)
    pd = geom_img.GetPointData()
    if pd.GetNumberOfArrays()>0:
        geom_field = pd.GetArrayName(0)
    else:
        cd = geom_img.GetCellData()
        if cd.GetNumberOfArrays()>0:
            geom_field = cd.GetArrayName(0)
        else:
            raise RuntimeError("No data in geometry.vti")
    print(f"[INFO] Geometry field: {geom_field}")
    geom_coords, geom_vals = extract_slice(geom_img, geom_field)
    print(f"[INFO] Geometry slice extracted: {geom_coords.shape[0]} points")

    xs = np.unique(geom_coords[:,0])
    ys = np.unique(geom_coords[:,1])
    nx, ny = xs.size, ys.size
    print(f"[INFO] Grid dimensions: nx={nx}, ny={ny}")
    ix = np.searchsorted(xs, geom_coords[:,0])
    iy = np.searchsorted(ys, geom_coords[:,1])

    # Snapshot list
    snap_paths = sorted(
        glob.glob(os.path.join(snap_dir, "snapshot*.vti")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].replace("snapshot",""))
    )
    print(f"[INFO] Found {len(snap_paths)} snapshot files.")

    # Compute max_abs from first 20 frames
    print("[INFO] Computing Ez max_abs from first 20 frames...")
    max_abs = 0.0
    for idx, path in enumerate(snap_paths[:20], start=1):
        coords, vals = extract_slice(read_vti_image(path), ez_field)
        if vals.size:
            frame_max = abs(vals).max()
            max_abs = max(max_abs, frame_max)
        print(f"  [INFO] Frame {idx}: local max={frame_max:.3e}, current global max={max_abs:.3e}")
    if max_abs == 0.0:
        raise RuntimeError("No Ez data found in first 20 frames.")
    print(f"[INFO] Final max_abs for normalization: {max_abs:.3e}")

    vmin, vmax = -0.03, 0.03
    print(f"[INFO] Ez normalization range: [{vmin}, {vmax}]")

    # Prepare geometry grid and axes
    print("[INFO] Preparing geometry grid and setting up plot...")
    geom_grid = np.zeros((ny, nx))
    geom_grid[iy, ix] = geom_vals
    unique_ids = np.unique(geom_vals)
    # Setup figure
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("X [cm]", fontsize=20)
    ax.set_ylabel("Y [cm]", fontsize=20)
    ax.tick_params(labelsize=16)
    # Apply zoom if requested
    if do_zoom:
        ax.set_xlim(x_min*100, x_max*100) # Convert to cm
        ax.set_ylim(y_min*100, y_max*100) # Convert to cm
        print("[INFO] Axes limits set for zoom.")

    # Plot geometry
    extent = [xs.min()*100, xs.max()*100, ys.min()*100, ys.max()*100] # Convert to cm
    cmap_geom = ListedColormap(['gray'])
    norm_geom = BoundaryNorm([unique_ids.min()-0.5, unique_ids.max()+0.5], ncolors=1)
    ax.imshow(geom_grid, extent=extent, origin='lower', cmap=cmap_geom, norm=norm_geom, zorder=0)
    print("[INFO] Geometry plotted.")
    # Draw material boundaries
    levels = unique_ids[:-1] + 0.5
    ax.contour(
        np.linspace(xs.min()*100, xs.max()*100, nx),
        np.linspace(ys.min()*100, ys.max()*100, ny),
        geom_grid, levels=levels,
        colors='white', linewidths=1.0, zorder=1
    )
    print("[INFO] Material boundaries drawn.")

    # Prepare Ez overlay
    ez_grid = np.zeros((ny, nx))
    ez_im = ax.imshow(
        ez_grid, extent=extent, origin='lower',
        cmap='seismic', vmin=vmin, vmax=vmax,
        alpha=0.6, zorder=2
    )
    divider = axgrid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(ez_im, cax=cax, ticks=[vmin, 0.0, vmax])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("Normalized Ez", fontsize=20)
    print("[INFO] Ez overlay prepared.")

    # Frame directory
    frame_dir = os.path.join(output_dir, 'snapshot_frames')
    os.makedirs(frame_dir, exist_ok=True)
    print(f"[INFO] Frame directory: {frame_dir}")

    dt_ns = 0.2
    def update(i):
        coords, vals = extract_slice(read_vti_image(snap_paths[i]), ez_field)
        grid = np.zeros((ny, nx))
        ix_i = np.searchsorted(xs, coords[:,0])
        iy_i = np.searchsorted(ys, coords[:,1])
        grid[iy_i, ix_i] = vals / max_abs
        ez_im.set_data(grid)
        ax.set_title(f"Time = {(i+1)*dt_ns:.1f} ns", fontsize=20)
        frame_path = os.path.join(frame_dir, f"frame_{i+1:03d}.png")
        fig.savefig(frame_path, dpi=300)
        print(f"[INFO] Saved frame {i+1}/{len(snap_paths)} to {frame_path}")
        return [ez_im]

    print("[INFO] Starting animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(snap_paths), blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata={'artist':'gprMax'})
    ani.save(output_video_path, writer=writer)
    print(f"[INFO] Saved animation to {output_video_path}")

if __name__=='__main__':
    main()
