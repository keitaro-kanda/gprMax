import matplotlib.pyplot as plt
import numpy as np


# Kirchhoff migration
def kirchhoff_migration(data, dx, dt, velocity, tmin, tmax, xmin, xmax, zmin, zmax, xstep, zstep):
    """Kirchhoff migration

    Args:
        data (array): Array of A-scans, i.e. B-scan data.
        dx (float): Spatial resolution of the model.
        dt (float): Temporal resolution of the model.
        velocity (float): Velocity of the medium.
        tmin (float): Minimum time of the migration.
        tmax (float): Maximum time of the migration.
        xmin (float): Minimum x of the migration.
        xmax (float): Maximum x of the migration.
        zmin (float): Minimum z of the migration.
        zmax (float): Maximum z of the migration.
        xstep (float): Step of x of the migration.
        zstep (float): Step of z of the migration.

    Returns:
        migrated_data (array): Array of migrated data.
    """

    # データのサイズを取得
    (nt, nx) = data.shape

    # マイグレーションの範囲を設定
    tmin_index = int(tmin/dt)
    tmax_index = int(tmax/dt)
    xmin_index = int(xmin/dx)
    xmax_index = int(xmax/dx)
    zmin_index = int(zmin/dx)
    zmax_index = int(zmax/dx)

    # マイグレーションのステップ数を計算
    xstep_index = int(xstep/dx)
    zstep_index = int(zstep/dx)

