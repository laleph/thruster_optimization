from dataclasses import dataclass
from typing import List, Tuple, Any, Optional # Added Optional

import magpylib as magpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Added pandas for DataFrame type hint
from magpylib.magnet import Cylinder, CylinderSegment
from matplotlib.axes import Axes # Corrected import for Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.streamplot import StreamplotSet # Import for StreamplotSet
from scipy.interpolate import RegularGridInterpolator

# TODO put params and interpolation arrays in a class/dict

# parameters
strength: int = 1000  # mT magnetization
nres: int = 400  # resolution in z and r

# field lines
offset: int = 10  # distance from the symmetry and wall
nlines: int = 19  # +1, so 20 in total since rr[1] is added by hand


# definitions


def find_separatrix(z: np.ndarray, Bz: np.ndarray) -> Tuple[int, float]:
    """find the integration end near the axis where the magnetic field changes topologically,
    i.e, where B_z near the axis chnages sign"""
    Bzline: np.ndarray = Bz[1, :]  # 1 means one point in r above symmetry axis

    end_index: np.ndarray = np.where(Bzline > 0)[0]
    if end_index.size == 0:  # catch empty array
        zmax: float = z[1, -1]
        print(zmax)
        print(z)
        return nres, zmax
    end_idx: int = end_index[0]
    zsep: float = z[1, end_idx]
    return end_idx, zsep  # index and z coordinate, where B_z is positive


def add_lengths_to_df(df: pd.DataFrame, norm: float = 1) -> pd.DataFrame:
    """add the various measures to the dataframe df"""
    df.length = df.length / norm
    df["l_mr"] = df.length * np.sqrt(df.mr)
    df["r0_l"] = df.r0 * df.length
    df["r0_l_mr"] = df.r0 * df.length * np.sqrt(df.mr)
    df["r0_l_mr_exp_r0"] = df.r0 * df.length * np.sqrt(df.mr) * np.exp(-df.r0)
    return df


def parallelism(zz: np.ndarray, rr: np.ndarray, Binterp_z: RegularGridInterpolator, Binterp_r: RegularGridInterpolator, R: float, L: float, p: float = 1) -> float:
    """calculate average theta = arctan2(br / bz) within (0,0) and (p*L,p*R) at the nres positions"""
    rp: np.ndarray = rr[np.where(rr < p * R)]
    zp: np.ndarray = zz[np.where(zz < p * L)]

    i: int = 0
    br_sum: float = 0.0
    bz_sum: float = 0.0
    for r_val in rp:
        for z_val in zp:
            # Ensure the result of interpolation is treated as a float
            # and take the real part in case of complex results from extrapolation
            br_sum += float(Binterp_r([r_val, z_val]).item().real)
            bz_sum += float(Binterp_z([r_val, z_val]).item().real)
            i += 1

    if i == 0:
        return 0.0 # Avoid division by zero if rp or zp is empty
    return float(np.arctan2(br_sum / i, bz_sum / i))  # global angle theta (for that region)


def active_volume(z: np.ndarray, r: np.ndarray, L: float, R: float) -> float:
    """active volume"""
    if z[-1] < 0.5 * L:
        # add last point to compare with the rectangle
        z_app: np.ndarray = np.append(z, 0.5 * L)
        r_app: np.ndarray = np.append(r, R)
    else:
        z_app = z
        r_app = r

    # Ensure trapz result is float
    active: float = float(np.trapz(np.power(r_app, 2), z_app))

    # relative active volume
    return active / (R**2 * 0.5 * L)


# basic plot


def field_plot(z: np.ndarray, rho: np.ndarray, Bz: np.ndarray, Br: np.ndarray, R: float, L: float, dR: float, ax: Optional[Axes] = None) -> None:
    """plot the magnetic field of the ring magnet"""
    Bamp: np.ndarray = np.linalg.norm(np.array([Bz, Br]), axis=0)

    # plot
    current_ax: Axes
    if ax is None:
        fig, current_ax = plt.subplots(1, 1, figsize=(12, 7))
    else:
        current_ax = ax

    # The return type of streamplot is StreamplotSet
    sp: StreamplotSet = current_ax.streamplot(
        z,
        rho,
        Bz,
        Br,
        density=1.1,
        # color=Bamp,
        # linewidth=np.sqrt(Bamp)*3,
        cmap="coolwarm",
        # broken_streamlines=False
    )

    # figure styling
    current_ax.set(
        # title="Magnetic field of thin cylinder",
        ylabel="r / mm",
        xlabel="z / mm",
        aspect=1,
    )

    rect: Rectangle = Rectangle((0, R), 0.5 * L, dR)
    pc: PatchCollection = PatchCollection([rect], facecolor="green", alpha=0.5, edgecolor="k")
    # Add collection to axes
    current_ax.add_collection(pc)


# field calculation
@dataclass
class Ring:
    R: float
    L: float
    dR: float
    r0: List[float]
    bri: RegularGridInterpolator
    bzi: RegularGridInterpolator
    pa: List[float]
    z: np.ndarray
    r: np.ndarray
    zsep: float
    ring: CylinderSegment

    def __init__(self, R: float = 20, L: float = 50, dR: float = 2, plt_on: bool = False, ax: Optional[Axes] = None) -> None:
        self.R = R
        self.L = L
        self.dR = dR
        # Assign types to the results of ring_calculation
        r0_val, interp_r_val, interp_z_val, pa_val, zz_val, rr_val, zsep_val, magnet_val = ring_calculation(R, L, dR, plt_on, ax)
        self.r0 = r0_val
        self.bri = interp_r_val
        self.bzi = interp_z_val
        self.pa = pa_val
        self.z = zz_val
        self.r = rr_val
        self.zsep = zsep_val
        self.ring = magnet_val


def ring_calculation(R: float = 20, L: float = 50, dR: float = 2, plt_on: bool = False, ax: Optional[Axes] = None) -> Tuple[List[float], RegularGridInterpolator, RegularGridInterpolator, List[float], np.ndarray, np.ndarray, float, CylinderSegment]:
    """calculate the field and field line length(s)
    return r0, interp_r, interp_z, pa, zz, rr, zsep, magnet"""

    # generate ring magnet (same as two cylinders, see notebook)
    magnet: CylinderSegment = magpy.magnet.CylinderSegment(
        magnetization=(0, 0, strength),
        dimension=(R, R + dR, L, 0, 360),
        position=(0, 0, 0),
    )

    # max. plot/integration region
    rmax: float = 1.1 * R
    zmax: float = 0.5 * L + dR + 0.5 * R  # should be bigger than the separatrix!!!

    # pre-compute and plot field of thin_cylinder (faster)

    # create grid
    tr: np.ndarray = np.linspace(0, rmax, nres)
    tz: np.ndarray = np.linspace(0, zmax, nres)
    grid: np.ndarray = np.array([[(rh, 0, zh) for zh in tz] for rh in tr])

    # compute and plot field of thin_cylinder
    B: np.ndarray = magpy.getB(magnet, grid)

    z_grid: np.ndarray = grid[:, :, 2]  # r,z from grid
    r_grid: np.ndarray = grid[:, :, 0]  # r,z from grid
    Bz_grid: np.ndarray = np.ascontiguousarray(B[:, :, 2])  # r,z from grid
    Br_grid: np.ndarray = np.ascontiguousarray(B[:, :, 0])  # r,z from grid

    end_index_val, zsep_val = find_separatrix(z_grid, Bz_grid)
    if end_index_val == nres:
        print("!!! domain too small !!!")
    # zmax = np.ceil(zsep+1)

    ## plot with switch on/off
    if plt_on:
        field_plot(tz, tr, Bz_grid, Br_grid, R, L, dR, ax)

    # interpolation
    # input arrays need to be C_CONTIGUOUS !!!
    zz_interp: np.ndarray = np.ascontiguousarray(z_grid[0, :])
    rr_interp: np.ndarray = np.ascontiguousarray(r_grid[:, 0])

    # define the r0 positions of the integrated field lines
    # rr[1] is added specifically for showing the separatrix
    r0_list: List[float] = [float(rr_interp[i]) for i in np.append(1, np.linspace(offset, nres / rmax * R - offset, nlines, dtype=int))]

    interp_r_val: RegularGridInterpolator = RegularGridInterpolator([rr_interp, zz_interp], Br_grid)
    interp_z_val: RegularGridInterpolator = RegularGridInterpolator([rr_interp, zz_interp], Bz_grid)

    # parallelism
    pa_list: List[float] = []
    for p_val in [0.1, 0.3, 0.5, 0.9]:
        pa_list.append(parallelism(zz_interp, rr_interp, interp_z_val, interp_r_val, R, L, p_val))
    # pa = 1

    # return for integration and plotting
    return r0_list, interp_r_val, interp_z_val, pa_list, zz_interp, rr_interp, zsep_val, magnet
