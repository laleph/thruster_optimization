from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

import magpylib as magpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from magpylib.magnet import CylinderSegment
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp


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
    end_index_val: int = end_index[0]
    zsep: float = z[1, end_index_val]
    return end_index_val, zsep  # index and z coordinate, where B_z is positive


def add_lengths_to_df(df: pd.DataFrame, norm: float = 1) -> pd.DataFrame:
    """add the various measures to the dataframe df"""
    df["length"] = df["length"] / norm
    df["l_mr"] = df["length"] * np.sqrt(df["mr"])
    df["r0_l"] = df["r0"] * df["length"]
    df["r0_l_mr"] = df["r0"] * df["length"] * np.sqrt(df["mr"])
    df["r0_l_mr_exp_r0"] = df["r0"] * df["length"] * np.sqrt(df["mr"]) * np.exp(-df["r0"])
    return df


def parallelism(
    zz: np.ndarray,
    rr: np.ndarray,
    Binterp_z: RegularGridInterpolator,
    Binterp_r: RegularGridInterpolator,
    R: float,
    L: float,
    p: float = 1,
) -> float:
    """calculate average theta = arctan2(br / bz) within (0,0) and (p*L,p*R) at the nres positions"""
    rp: np.ndarray = rr[np.where(rr < p * R)]
    zp: np.ndarray = zz[np.where(zz < p * L)]

    i: int = 0
    br_sum: float = 0
    bz_sum: float = 0
    for r_val in rp:
        for z_val in zp:
            br_val_interp = Binterp_r([r_val, z_val])
            bz_val_interp = Binterp_z([r_val, z_val])
            
            # Ensure br_val_interp and bz_val_interp are correctly handled whether scalar or array
            br_item = br_val_interp.item() if hasattr(br_val_interp, 'item') else br_val_interp
            bz_item = bz_val_interp.item() if hasattr(bz_val_interp, 'item') else bz_val_interp

            br_sum += float(br_item)
            bz_sum += float(bz_item)
            i += 1

    return float(np.arctan2(br_sum / i, bz_sum / i))  # global angle theta (for that region)


def active_volume(z: np.ndarray, r: np.ndarray, L: float, R: float) -> float:
    """active volume"""
    if z[-1] < 0.5 * L:
        # add last point to compare with the rectangle
        z = np.append(z, 0.5 * L)
        r = np.append(r, R)

    # Calculate and immediately cast to float
    active: float = float(np.trapz(np.power(r, 2), z))

    # relative active volume
    return active / (R**2 * 0.5 * L)


# basic plot


def field_plot(
    z: np.ndarray,
    rho: np.ndarray,
    Bz: np.ndarray,
    Br: np.ndarray,
    R: float,
    L: float,
    dR: float,
    ax: Optional[plt.Axes] = None,
) -> None:
    """plot the magnetic field of the ring magnet"""
    Bamp: np.ndarray = np.linalg.norm(np.array([Bz, Br]), axis=0)

    # plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    sp: mpl.streamplot.StreamplotSet = ax.streamplot(
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
    ax.set(
        # title="Magnetic field of thin cylinder",
        ylabel="r / mm",
        xlabel="z / mm",
        aspect=1,
    )

    rect: Rectangle = Rectangle((0, R), 0.5 * L, dR)
    pc: PatchCollection = PatchCollection([rect], facecolor="green", alpha=0.5, edgecolor="k")
    # Add collection to axes
    ax.add_collection(pc)


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
    ring: magpy.magnet.CylinderSegment

    def __init__(self, R: float = 20, L: float = 50, dR: float = 2, plt_on: bool = False, ax: Optional[plt.Axes] = None):
        self.R = R
        self.L = L
        self.dR = dR
        r0, interp_r, interp_z, pa, zz, rr, zsep, magnet = ring_calculation(R, L, dR, plt_on, ax)
        self.r0 = r0
        self.bri = interp_r
        self.bzi = interp_z
        self.pa = pa
        self.z = zz
        self.r = rr
        self.zsep = zsep
        self.ring = magnet


def ring_calculation(
    R: float = 20, L: float = 50, dR: float = 2, plt_on: bool = False, ax: Optional[plt.Axes] = None
) -> Tuple[
    List[float],
    RegularGridInterpolator,
    RegularGridInterpolator,
    List[float],
    np.ndarray,
    np.ndarray,
    float,
    magpy.magnet.CylinderSegment,
]:
    """calculate the field and field line length(s)
    return r0, interp_r, interp_z, pa, zz, rr, zsep, magnet"""

    # generate ring magnet (same as two cylinders, see notebook)
    magnet: magpy.magnet.CylinderSegment = magpy.magnet.CylinderSegment(
        magnetization=(0, 0, strength),
        dimension=(R, R + dR, L, 0, 360),
        position=(0, 0, 0),
    )

    # max. plot/integration region
    rmax: float = 1.1 * R
    zmax: float = float(0.5 * L + dR + 0.5 * R)  # should be bigger than the separatrix!!!

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

    end_index, zsep = find_separatrix(z_grid, Bz_grid)
    if end_index == nres:
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
    r0_lines: List[float] = [
        rr_interp[i] for i in np.append(1, np.linspace(offset, nres / rmax * R - offset, nlines, dtype=int))
    ]

    interp_r_func: RegularGridInterpolator = RegularGridInterpolator([rr_interp, zz_interp], Br_grid)
    interp_z_func: RegularGridInterpolator = RegularGridInterpolator([rr_interp, zz_interp], Bz_grid)

    # parallelism
    pa_vals: List[float] = []
    for p_val in [0.1, 0.3, 0.5, 0.9]:
        pa_vals.append(parallelism(zz_interp, rr_interp, interp_z_func, interp_r_func, R, L, p_val))
    # pa = 1

    # return for integration and plotting
    return r0_lines, interp_r_func, interp_z_func, pa_vals, zz_interp, rr_interp, zsep, magnet


def integration(
    R: float,
    L: float,
    dR: float,
    df: pd.DataFrame,
    rlines: List[float],
    interp_r: RegularGridInterpolator,
    interp_z: RegularGridInterpolator,
    parallel: List[float], # This was identified as List[float] in ring_calculation, but seems to be a single float value in usage
    zsep: float,
    plt_on: bool = False,
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    def field(t: float, y: List[float]) -> List[float]:
        """return the interpolated magnetic field (Bz,Br) at point (z,r)
        here used as the derivative f(y) = dy/dx in an ODE"""
        z_val, r_val = y
        bz_val: float = float(interp_z([r_val, z_val], method="linear")[0])
        br_val: float = float(interp_r([r_val, z_val], method="linear")[0])
        b_norm: float = float(np.linalg.norm([bz_val, br_val]))
        # Handle potential division by zero if b_norm is zero
        if b_norm == 0:
            return [0.0, 0.0]
        dz_val: float = bz_val / b_norm
        dr_val: float = br_val / b_norm
        return [
            -dz_val,
            -dr_val,
        ]  # opposed direction is necessary for integration in positive z direction

    def hit_wall(t: float, y: List[float]) -> float:
        """end criterion for field line integration/ODE"""

        eps: float = 1e-1  # stop a bit before the wall, else motion integration parallel z is possible before the wall,
        # see scaling of R
        return y[1] - R + eps

    # settings
    # Set attributes directly on the function object
    setattr(hit_wall, 'terminal', True)
    setattr(hit_wall, 'direction', 1)

    maxstep: float = 0.3  # max. integration step

    # integration for all field lines
    z0_val: float = 0.0
    for r0_val in rlines:  # > 0!

        # bfield at starting point (0) -> mirror ratio mr
        bz0_val: float = float(interp_z([r0_val, z0_val], method="linear")[0])
        br0_val: float = float(interp_r([r0_val, z0_val], method="linear")[0])
        b0_norm: float = float(np.linalg.norm([bz0_val, br0_val]))

        # integration of the field line(s)
        sol: Any = solve_ivp(
            field,
            [0, 1000],
            [z0_val, r0_val],
            events=hit_wall,
            method="RK45",
            max_step=maxstep,
            atol=1e-4,
            rtol=1e-7,
        )  # was DOP853
        # TODO some integrations with DOP853 seem to quit after some steps??

        # calculate the length of each field line
        s_vals: np.ndarray = np.sqrt(np.power(np.diff(sol.y[0]), 2) + np.power(np.diff(sol.y[1]), 2))
        length_val: float = np.sum(s_vals)  # not weighted length, that is done later
        # print(s.shape, length)

        # bfield at wall (w) -> mirror ratio mr
        rw_val: float = float(sol.y[1][-1])
        zw_val: float = float(sol.y[0][-1])
        bzw_val: float = float(interp_z([rw_val, zw_val], method="linear")[0])
        brw_val: float = float(interp_r([rw_val, zw_val], method="linear")[0])
        bw_norm: float = float(np.linalg.norm([bzw_val, brw_val]))
        
        # Handle potential division by zero if b0_norm is zero
        mr_value: float
        if b0_norm == 0:
            mr_value = 0.0
        else:
            mr_value = bw_norm / b0_norm  # mirror ratio

        # active volume va
        va_val: float = active_volume(sol.y[0].astype(float), sol.y[1].astype(float), L, R)

        if plt_on:
            if ax is None:
                plt.plot(sol.y[0], sol.y[1], "red")
            else:
                ax.plot(sol.y[0], sol.y[1], "red")

        # put results of each field line into df
        # Ensure `parallel` is treated as a scalar if it's a list from `pa_vals`
        # Taking the first element as a placeholder, this might need domain specific logic
        current_parallel_val = parallel[0] if isinstance(parallel, list) and parallel else parallel 

        df = pd.concat(
            [
                pd.DataFrame(
                    [[R, L, dR, current_parallel_val, zsep - 0.5 * L, va_val, r0_val, mr_value, length_val]],
                    columns=df.columns,
                ),
                df,
            ],
            ignore_index=True,
        )

    if plt_on and ax is None:
        plt.savefig("R" + str(R) + "L" + str(L) + ".png", dpi=300)

    return df
