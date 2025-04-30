#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.util.numpy_support import vtk_to_numpy


def read_vti_image(filename):
    """Reads a VTI file using VTK and returns the vtkImageData object."""
    reader = vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def extract_slice(img, field_name, target_z=0.0, tol=1e-6):
    """
    Extracts (x,y) coords and the z-component of field vals on the slice nearest target_z.
    Returns coords2d (N×2) and values (N,).
    """
    def choose_slice_z(z_coords):
        uniq = np.unique(z_coords)
        return uniq[np.argmin(np.abs(uniq - target_z))]

    # Try point_data then cell_data
    pd = img.GetPointData()
    array = pd.GetArray(field_name)
    if array is None:
        cd = img.GetCellData()
        array = cd.GetArray(field_name)
        if array is None:
            raise KeyError(f"Field '{field_name}' not found in VTI data.")
        center = vtkCellCenters()
        center.SetInputData(img)
        center.Update()
        mesh = center.GetOutput()
    else:
        mesh = img

    pts = vtk_to_numpy(mesh.GetPoints().GetData()).reshape(-1, 3)
    vals = vtk_to_numpy(array)
    slice_z = choose_slice_z(pts[:, 2])
    mask = np.isclose(pts[:, 2], slice_z, atol=tol)
    coords2d = pts[mask][:, :2]
    vals2d = vals[mask]
    # If vector, take z-component
    if vals2d.ndim > 1 and vals2d.shape[1] >= 3:
        vals2d = vals2d[:, 2]
    return coords2d, vals2d


def main():
    geometry_path = input("Enter path to geometry.vti: ").strip()
    if not os.path.isfile(geometry_path):
        raise FileNotFoundError(f"Cannot find file: {geometry_path}")
    print(f"[INFO] Geometry: {geometry_path}")

    parent_dir = os.path.dirname(geometry_path)
    snap_dirs = [d for d in os.listdir(parent_dir) if d.endswith("_snaps")]
    if len(snap_dirs) != 1:
        print(f"[WARN] Snap folders: {snap_dirs}")
        snap_dir = input("Enter snapshot dir: ").strip()
    else:
        snap_dir = os.path.join(parent_dir, snap_dirs[0])
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"Cannot find snaps directory: {snap_dir}")
    print(f"[INFO] Snap dir: {snap_dir}")

    ez_field = "E-field"
    fps = 15
    output_video = os.path.join(parent_dir, "snapshot_animation.mp4")

    # Geometry slice
    print("[INFO] Reading geometry and extracting slice...")
    geom_img = read_vti_image(geometry_path)
    pd = geom_img.GetPointData()
    if pd.GetNumberOfArrays() > 0:
        geom_field = pd.GetArrayName(0)
    else:
        cd = geom_img.GetCellData()
        if cd.GetNumberOfArrays() > 0:
            geom_field = cd.GetArrayName(0)
        else:
            raise RuntimeError("No data in geometry.vti")
    geom_coords, geom_vals = extract_slice(geom_img, geom_field)
    print(f"[INFO] Geometry slice: {geom_coords.shape[0]} pts, field '{geom_field}'")

    xs = np.unique(geom_coords[:, 0])
    ys = np.unique(geom_coords[:, 1])
    nx, ny = xs.size, ys.size
    ix = np.searchsorted(xs, geom_coords[:, 0])
    iy = np.searchsorted(ys, geom_coords[:, 1])
    geom_mask = np.zeros((ny, nx))
    geom_mask[iy, ix] = (geom_vals > 0.5).astype(float)

    # Snapshot files
    snap_paths = sorted(glob.glob(os.path.join(snap_dir, "snapshot*.vti")),
                         key=lambda p: int(os.path.splitext(os.path.basename(p))[0].replace("snapshot", "")))
    print(f"[INFO] {len(snap_paths)} snapshots found.")

    # Global symmetric Ez range
    print("[INFO] Computing Ez range...")
    max_abs = 0.0
    for i, path in enumerate(snap_paths, 1):
        _, vals = extract_slice(read_vti_image(path), ez_field)
        if vals.size:
            max_abs = max(max_abs, np.abs(vals).max())
        if i % 10 == 0 or i == len(snap_paths):
            print(f"  processed {i}/{len(snap_paths)}, max_abs={max_abs:.3e}")
    if max_abs == 0.0:
        raise RuntimeError("No Ez data.")
    vmin, vmax = -max_abs, max_abs

            # Initialize plot
    print("[INFO] Initializing plot...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    # Create 2D meshgrid for pcolormesh
    X, Y = np.meshgrid(xs, ys)
    # Geometry background: material classification (3 media)
    geom_grid = np.zeros((ny, nx))
    # Populate geometry grid with material IDs
    geom_grid[iy, ix] = geom_vals  # geom_vals are IDs for each slice point
    from matplotlib.colors import ListedColormap, BoundaryNorm
    unique_ids = np.unique(geom_vals)
    cmap_geom = ListedColormap(['lightgray', 'saddlebrown', 'peru'][:len(unique_ids)])
    norm = BoundaryNorm(unique_ids - 0.5, len(unique_ids))
    geom_pc = ax.pcolormesh(
        X, Y, geom_grid,
        cmap=cmap_geom,
        norm=norm,
        alpha=1,
        shading='auto',
        zorder=0
    )
    # Snapshot overlay: empty grid via pcolormesh
    ez_grid = np.zeros((ny, nx))
    snap_pc = ax.pcolormesh(
        X, Y, ez_grid,
        cmap='viridis',
        vmin=vmin, vmax=vmax,
        alpha=0.4,
        shading='auto',
        zorder=1
    )
    cbar = fig.colorbar(snap_pc, ax=ax)
    cbar.set_label("Ez [V/m]")

    dt_ns = 0.5
    def update(i):
        coords, vals = extract_slice(read_vti_image(snap_paths[i]), ez_field)
        grid = np.zeros((ny, nx))
        ix_i = np.searchsorted(xs, coords[:, 0])
        iy_i = np.searchsorted(ys, coords[:, 1])
        grid[iy_i, ix_i] = vals
        # Update pcolormesh array
        snap_pc.set_array(grid.ravel())
        ax.set_title(f"Time = {(i+1)*dt_ns:.1f} ns")
        if (i+1) % 10 == 0 or (i+1) == len(snap_paths):
            print(f"[INFO] Frame {i+1}/{len(snap_paths)} rendered")
        return [snap_pc]

    print(f"[INFO] Saving animation to {output_video}...")
    ani = animation.FuncAnimation(fig, update, frames=len(snap_paths), blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata={'artist':'gprMax'}, bitrate=1800)
    ani.save(output_video, writer=writer)
    print(f"[INFO] Animation saved to {output_video}")

if __name__ == '__main__':
    main()
