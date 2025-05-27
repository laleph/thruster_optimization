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


# TODO put params and interpolation arrays in a class/dict - Partially addressed by SimParams

# parameters
@dataclass
class SimParams:
    """Holds global simulation parameters.

    Attributes:
        strength: Magnetization strength in mT.
        nres: Resolution for z and r grids.
        offset: Distance from symmetry axis and wall for field line starting points.
        nlines: Number of field lines to integrate (excluding the one on the separatrix).
    """
    strength: int = 1000  # mT magnetization
    nres: int = 400  # resolution in z and r
    offset: int = 10  # distance from the symmetry and wall
    nlines: int = 19  # +1, so 20 in total since rr[1] is added by hand

sim_params = SimParams()

# definitions


def find_separatrix(z: np.ndarray, Bz: np.ndarray) -> Tuple[int, float]:
    """Finds the separatrix based on the change in sign of Bz near the axis.

    The separatrix is a key topological feature in the magnetic field,
    indicating a region where field lines change their connection. This function
    identifies its location along the z-axis.

    Args:
        z: A 2D numpy array representing the z-coordinates of the grid.
           Expected shape (n_r_points, n_z_points).
        Bz: A 2D numpy array representing the z-component of the magnetic field
            on the grid. Expected shape (n_r_points, n_z_points).

    Returns:
        A tuple (index, zsep):
            index (int): The index along the z-axis (second dimension of input arrays)
                         where Bz near the axis first becomes positive. If no such
                         point is found, `sim_params.nres` is returned.
            zsep (float): The z-coordinate corresponding to the found index. If no
                          such point is found, the maximum z-value from the input
                          `z` array (at r_index=1) is returned.
    """
    Bzline: np.ndarray = Bz[1, :]  # 1 means one point in r above symmetry axis

    end_index: np.ndarray = np.where(Bzline > 0)[0]
    if end_index.size == 0:  # catch empty array
        zmax: float = z[1, -1]
        print(zmax)
        print(z)
        return sim_params.nres, zmax
    end_index_val: int = end_index[0]
    zsep: float = z[1, end_index_val]
    return end_index_val, zsep  # index and z coordinate, where B_z is positive


def add_lengths_to_df(df: pd.DataFrame, norm: float = 1) -> pd.DataFrame:
    """Adds several derived length-related measures to a DataFrame.

    The function calculates new columns based on existing 'length', 'mr' (mirror ratio),
    and 'r0' (initial radial position of field line) columns.

    Args:
        df: Pandas DataFrame to which new columns will be added.
            It must contain 'length', 'mr', and 'r0' columns.
        norm: A normalization factor to apply to the 'length' column before
              deriving other measures. Defaults to 1 (no normalization).

    Returns:
        The input DataFrame `df` with added columns:
            - 'length': Original 'length' divided by `norm`.
            - 'l_mr': `length * sqrt(mr)`.
            - 'r0_l': `r0 * length`.
            - 'r0_l_mr': `r0 * length * sqrt(mr)`.
            - 'r0_l_mr_exp_r0': `r0 * length * sqrt(mr) * exp(-r0)`.
    """
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
    """Calculates the average angle theta = arctan2(Br_avg, Bz_avg) in a specified region.

    This angle represents the average field line direction relative to the z-axis
    within a rectangular subgrid defined by (0,0) and (p*L, p*R).

    Args:
        zz: 1D numpy array of z-coordinates for the grid.
        rr: 1D numpy array of r-coordinates for the grid.
        Binterp_z: A `scipy.interpolate.RegularGridInterpolator` for Bz component.
        Binterp_r: A `scipy.interpolate.RegularGridInterpolator` for Br component.
        R: The characteristic radial dimension (e.g., magnet radius).
        L: The characteristic axial dimension (e.g., magnet length).
        p: A factor (0 to 1) determining the subgrid size relative to R and L.
           The calculation is performed within r < p*R and z < p*L. Defaults to 1.

    Returns:
        The average angle theta in radians. Returns 0.0 if no grid points
        fall within the specified subregion (to avoid division by zero).
    """
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
    
    if i == 0: # Avoid division by zero if rp or zp is empty
        return 0.0
    return float(np.arctan2(br_sum / i, bz_sum / i))  # global angle theta (for that region)


def active_volume(z: np.ndarray, r: np.ndarray, L: float, R: float) -> float:
    """Calculates the relative active volume of a field line.

    The active volume is defined by revolving the field line (r(z)) around the
    z-axis and is calculated using `np.trapz(r^2, z)`. This volume is then
    normalized by the cylindrical volume defined by R and 0.5*L.
    The input arrays `z` and `r` may be modified by appending a point
    (0.5*L, R) if the field line ends before z = 0.5*L.

    Args:
        z: 1D numpy array of z-coordinates along a field line.
        r: 1D numpy array of r-coordinates along a field line.
        L: Characteristic length of the system (e.g., magnet length).
        R: Characteristic radius of the system (e.g., magnet radius).

    Returns:
        The relative active volume, normalized by (pi * R^2 * 0.5*L).
        Note: pi is implicitly handled as it cancels out if comparing to pi*R^2*L/2.
              The function returns (integral(r^2 dz)) / (R^2 * L/2).
    """
    if z[-1] < 0.5 * L:
        # add last point to compare with the rectangle
        z = np.append(z, 0.5 * L)
        r = np.append(r, R)

    # Calculate and immediately cast to float
    active: float = float(np.trapz(np.power(r, 2), z))

    # relative active volume
    return active / (R**2 * 0.5 * L)


# basic plot


@dataclass
class RingCalculationOutput:
    """Holds the output from the `ring_calculation` function.

    This dataclass serves as a structured container for the various results
    computed during the magnetic field characterization of a ring magnet.

    Attributes:
        r0_lines: List of initial radial positions (r0) for field line integration.
        interp_r_func: Interpolator for the radial component of the B-field (Br).
        interp_z_func: Interpolator for the axial component of the B-field (Bz).
        pa_val: Calculated parallelism value (average field angle) for p=0.1.
        zz_interp: 1D array of z-coordinates used for interpolation.
        rr_interp: 1D array of r-coordinates used for interpolation.
        zsep: z-coordinate of the separatrix.
        magnet: The `magpylib.magnet.CylinderSegment` object representing the ring.
    """
    r0_lines: List[float]
    interp_r_func: RegularGridInterpolator
    interp_z_func: RegularGridInterpolator
    pa_val: float
    zz_interp: np.ndarray
    rr_interp: np.ndarray
    zsep: float
    magnet: magpy.magnet.CylinderSegment


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
    """Plots the magnetic field streamplot of a ring magnet.

    Displays the magnetic field lines (streamplot) and the magnet's cross-section.

    Args:
        z: 1D numpy array of z-coordinates for the grid.
        rho: 1D numpy array of radial (rho or r) coordinates for the grid.
        Bz: 2D numpy array of the z-component of the magnetic field on the grid.
        Br: 2D numpy array of the r-component of the magnetic field on the grid.
        R: Inner radius of the ring magnet.
        L: Length of the ring magnet.
        dR: Thickness (radial extent) of the ring magnet.
        ax: Optional `matplotlib.axes.Axes` object to plot on. If None, a new
            figure and axes are created.
    """
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
        cmap="coolwarm",
    )

    # figure styling
    ax.set(
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
    """Represents a ring magnet and its calculated magnetic field properties.

    This class encapsulates the geometric parameters of a ring magnet and the
    results obtained from the `ring_calculation` function, such as field
    interpolators, separatrix location, and the magnet object itself.

    Attributes:
        R: Inner radius of the ring magnet (mm).
        L: Length of the ring magnet (mm).
        dR: Thickness (radial extent) of the ring magnet (mm).
        r0: List of initial radial positions (r0) for field line integration.
        bri: Interpolator for the radial component of the B-field (Br).
        bzi: Interpolator for the axial component of the B-field (Bz).
        pa: Calculated parallelism value (average field angle).
        z: 1D array of z-coordinates used for interpolation.
        r: 1D array of r-coordinates used for interpolation.
        zsep: z-coordinate of the separatrix.
        ring: The `magpylib.magnet.CylinderSegment` object representing the ring.
    """
    R: float
    L: float
    dR: float
    r0: List[float]
    bri: RegularGridInterpolator
    bzi: RegularGridInterpolator
    pa: float
    z: np.ndarray
    r: np.ndarray
    zsep: float
    ring: magpy.magnet.CylinderSegment

    def __init__(self, R: float = 20, L: float = 50, dR: float = 2, plt_on: bool = False, ax: Optional[plt.Axes] = None):
        """Initializes a Ring object by calculating its magnetic field properties.

        Args:
            R: Inner radius of the ring magnet in mm. Defaults to 20.
            L: Length of the ring magnet in mm. Defaults to 50.
            dR: Thickness (radial extent) of the ring magnet in mm. Defaults to 2.
            plt_on: If True, generates a plot of the magnetic field during calculation.
                    Defaults to False.
            ax: Optional `matplotlib.axes.Axes` object to plot on if `plt_on` is True.
                If None, a new figure and axes are created for the plot.
        """
        self.R = R
        self.L = L
        self.dR = dR
        calc_output: RingCalculationOutput = ring_calculation(R, L, dR, plt_on, ax)
        self.r0 = calc_output.r0_lines
        self.bri = calc_output.interp_r_func
        self.bzi = calc_output.interp_z_func
        self.pa = calc_output.pa_val
        self.z = calc_output.zz_interp
        self.r = calc_output.rr_interp
        self.zsep = calc_output.zsep
        self.ring = calc_output.magnet


def ring_calculation(
    R: float = 20, L: float = 50, dR: float = 2, plt_on: bool = False, ax: Optional[plt.Axes] = None
) -> RingCalculationOutput:
    """Calculates magnetic field properties of a ring magnet.

    This function models a ring magnet, computes its magnetic field on a grid,
    finds the separatrix, creates field interpolators, determines field line
    starting positions, and calculates field parallelism. Optionally, it can
    plot the field.

    Args:
        R: Inner radius of the ring magnet in mm. Defaults to 20.
        L: Length of the ring magnet in mm. Defaults to 50.
        dR: Thickness (radial extent) of the ring magnet in mm. Defaults to 2.
        plt_on: If True, generates a plot of the magnetic field. Defaults to False.
        ax: Optional `matplotlib.axes.Axes` object to plot on if `plt_on` is True.
            If None, a new figure and axes are created for the plot.

    Returns:
        A `RingCalculationOutput` object containing the results:
            - r0_lines: Initial radial positions for field lines.
            - interp_r_func: Interpolator for Br.
            - interp_z_func: Interpolator for Bz.
            - pa_val: Parallelism value.
            - zz_interp: z-coordinates for interpolation.
            - rr_interp: r-coordinates for interpolation.
            - zsep: z-coordinate of the separatrix.
            - magnet: Magpylib magnet object.
    """

    # generate ring magnet (same as two cylinders, see notebook)
    magnet: magpy.magnet.CylinderSegment = magpy.magnet.CylinderSegment(
        magnetization=(0, 0, sim_params.strength),
        dimension=(R, R + dR, L, 0, 360),
        position=(0, 0, 0),
    )

    # max. plot/integration region
    rmax: float = 1.1 * R
    zmax: float = float(0.5 * L + dR + 0.5 * R)  # should be bigger than the separatrix!!!

    # pre-compute and plot field of thin_cylinder (faster)

    # create grid
    tr: np.ndarray = np.linspace(0, rmax, sim_params.nres)
    tz: np.ndarray = np.linspace(0, zmax, sim_params.nres)
    grid: np.ndarray = np.array([[(rh, 0, zh) for zh in tz] for rh in tr])

    # compute and plot field of thin_cylinder
    B: np.ndarray = magpy.getB(magnet, grid)

    z_grid: np.ndarray = grid[:, :, 2]  # r,z from grid
    r_grid: np.ndarray = grid[:, :, 0]  # r,z from grid
    Bz_grid: np.ndarray = np.ascontiguousarray(B[:, :, 2])  # r,z from grid
    Br_grid: np.ndarray = np.ascontiguousarray(B[:, :, 0])  # r,z from grid

    end_index, zsep = find_separatrix(z_grid, Bz_grid)
    if end_index == sim_params.nres:
        print("!!! domain too small !!!")

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
        rr_interp[i] for i in np.append(1, np.linspace(sim_params.offset, sim_params.nres / rmax * R - sim_params.offset, sim_params.nlines, dtype=int))
    ]

    interp_r_func: RegularGridInterpolator = RegularGridInterpolator([rr_interp, zz_interp], Br_grid)
    interp_z_func: RegularGridInterpolator = RegularGridInterpolator([rr_interp, zz_interp], Bz_grid)

    # parallelism
    # Calculate a single parallelism value with p=0.1
    pa_val: float = parallelism(zz_interp, rr_interp, interp_z_func, interp_r_func, R, L, p=0.1)

    # return for integration and plotting
    return RingCalculationOutput(
        r0_lines=r0_lines,
        interp_r_func=interp_r_func,
        interp_z_func=interp_z_func,
        pa_val=pa_val,
        zz_interp=zz_interp,
        rr_interp=rr_interp,
        zsep=zsep,
        magnet=magnet
    )


def integration(
    R: float,
    L: float,
    dR: float,
    df: pd.DataFrame,
    rlines: List[float],
    interp_r: RegularGridInterpolator,
    interp_z: RegularGridInterpolator,
    parallel: float,
    zsep: float,
    plt_on: bool = False,
    ax: Optional[plt.Axes] = None,
) -> pd.DataFrame:
    """Integrates magnetic field lines and calculates their properties.

    For a given set of starting radial positions (`rlines`), this function
    traces magnetic field lines until they hit the magnet wall (r=R).
    It uses `scipy.integrate.solve_ivp` with the "RK45" method.
    The "DOP853" method was previously considered but showed instability for
    some integration cases. Properties like field line length, mirror ratio,
    and active volume are calculated and stored in a DataFrame.

    Args:
        R: Inner radius of the magnet system (wall boundary).
        L: Length of the magnet system.
        dR: Thickness of the magnet (used for record-keeping in DataFrame).
        df: Pandas DataFrame to append results to.
        rlines: List of initial radial positions (r0) to start field line integration.
        interp_r: Interpolator for the radial component of the B-field (Br).
        interp_z: Interpolator for the axial component of the B-field (Bz).
        parallel: Parallelism value (average field angle), for record-keeping.
        zsep: z-coordinate of the separatrix, used to calculate `zsep_L`.
        plt_on: If True, plots the integrated field lines. Defaults to False.
        ax: Optional `matplotlib.axes.Axes` object to plot on if `plt_on` is True.

    Returns:
        A Pandas DataFrame with the results of the field line integrations.
        Each row corresponds to an integrated field line and includes columns:
        'R', 'L', 'dR', 'parallelism', 'zsep_L' (zsep - 0.5*L), 'va' (active volume),
        'r0' (initial radial position), 'mr' (mirror ratio), 'length' (field line length).
    """
    def field(t: float, y: List[float]) -> List[float]:
        """Defines the ODE system dy/dt = [-Bz/|B|, -Br/|B|] for field line tracing.
        
        Args:
            t: Time (or path length variable, not explicitly used in field calculation).
            y: Current position [z, r].

        Returns:
            The derivatives [dz/ds, dr/ds] proportional to [-Bz, -Br] normalized.
        """
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
        """Event function for `solve_ivp` to detect when a field line hits the wall.

        Args:
            t: Time (or path length variable).
            y: Current position [z, r].
        
        Returns:
            Value that is zero when the event occurs (r - R + eps = 0).
        """

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
        )
        # Note: The 'DOP853' solver was previously explored but found to be unstable
        # for some integration scenarios, leading to premature termination.
        # 'RK45' is currently used as a more robust default.
        # The TODO comment regarding DOP853 can be removed as this note is added.
        # calculate the length of each field line
        s_vals: np.ndarray = np.sqrt(np.power(np.diff(sol.y[0]), 2) + np.power(np.diff(sol.y[1]), 2))
        length_val: float = np.sum(s_vals)  # not weighted length, that is done later

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
        df = pd.concat(
            [
                pd.DataFrame(
                    [[R, L, dR, parallel, zsep - 0.5 * L, va_val, r0_val, mr_value, length_val]], # Use parallel directly
                    columns=df.columns,
                ),
                df,
            ],
            ignore_index=True,
        )

    if plt_on and ax is None:
        plt.savefig("R" + str(R) + "L" + str(L) + ".png", dpi=300)

    return df
