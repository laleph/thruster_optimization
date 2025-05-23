from dataclasses import dataclass

import magpylib as magpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from magpylib.magnet import Cylinder, CylinderSegment
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp

# TODO put params and interpolation arrays in a class/dict

# parameters
strength = 1000  # mT magnetization
nres = 400  # resolution in z and r

# field lines
offset = 10  # distance from the symmetry and wall
nlines = 19  # +1, so 20 in total since rr[1] is added by hand


# definitions


def find_separatrix(z, Bz):
    """find the integration end near the axis where the magnetic field changes topologically,
    i.e, where B_z near the axis chnages sign"""
    Bzline = Bz[1, :]  # 1 means one point in r above symmetry axis

    end_index = np.where(Bzline > 0)[0]
    if end_index.size == 0:  # catch empty array
        zmax = z[1, -1]
        print(zmax)
        print(z)
        return nres, zmax
    end_index = end_index[0]
    zsep = z[1, end_index]
    return end_index, zsep  # index and z coordinate, where B_z is positive


def add_lengths_to_df(df, norm=1):
    """add the various measures to the dataframe df"""
    df.length = df.length / norm
    df["l_mr"] = df.length * np.sqrt(df.mr)
    df["r0_l"] = df.r0 * df.length
    df["r0_l_mr"] = df.r0 * df.length * np.sqrt(df.mr)
    df["r0_l_mr_exp_r0"] = df.r0 * df.length * np.sqrt(df.mr) * np.exp(-df.r0)
    return df


def parallelism(zz, rr, Binterp_z, Binterp_r, R, L, p=1):
    """calculate average theta = arctan2(br / bz) within (0,0) and (p*L,p*R) at the nres positions"""
    rp = rr[np.where(rr < p * R)]
    zp = zz[np.where(zz < p * L)]

    i = 0
    br = 0
    bz = 0
    for r in rp:
        for z in zp:
            br += Binterp_r([r, z])
            bz += Binterp_z([r, z])
            i += 1

    return np.arctan2(br / i, bz / i)  # global angle theta (for that region)


def active_volume(z, r, L, R):
    """active volume"""
    if z[-1] < 0.5 * L:
        # add last point to compare with the rectangle
        z = np.append(z, 0.5 * L)
        r = np.append(r, R)

    active = np.trapz(np.power(r, 2), z)

    # relative active volume
    return active / (R**2 * 0.5 * L)


# basic plot


def field_plot(z, rho, Bz, Br, R, L, dR, ax=None):
    """plot the magnetic field of the ring magnet"""
    Bamp = np.linalg.norm(np.array([Bz, Br]), axis=0)

    # plot
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    sp = ax.streamplot(
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

    rect = Rectangle((0, R), 0.5 * L, dR)
    pc = PatchCollection([rect], facecolor="green", alpha=0.5, edgecolor="k")
    # Add collection to axes
    ax.add_collection(pc)


# field calculation
@dataclass
class Ring:
    def __init__(self, R=20, L=50, dR=2, plt_on=False, ax=None):
        self.R = R
        self.L = L
        self.dR = dR
        [r0, interp_r, interp_z, pa, zz, rr, zsep, magnet] = ring_calculation(R, L, dR, plt_on, ax)
        self.r0 = r0
        self.bri = interp_r
        self.bzi = interp_z
        self.pa = pa
        self.z = zz
        self.r = rr
        self.zsep = zsep
        self.ring = magnet


def ring_calculation(R=20, L=50, dR=2, plt_on=False, ax=None):
    """calculate the field and field line length(s)
    return r0, interp_r, interp_z, pa, zz, rr, zsep, magnet"""

    # generate ring magnet (same as two cylinders, see notebook)
    magnet = magpy.magnet.CylinderSegment(
        magnetization=(0, 0, strength),
        dimension=(R, R + dR, L, 0, 360),
        position=(0, 0, 0),
    )

    # max. plot/integration region
    rmax = 1.1 * R
    zmax = 0.5 * L + dR + 0.5 * R  # should be bigger than the separatrix!!!

    # pre-compute and plot field of thin_cylinder (faster)

    # create grid
    tr = np.linspace(0, rmax, nres)
    tz = np.linspace(0, zmax, nres)
    grid = np.array([[(rh, 0, zh) for zh in tz] for rh in tr])

    # compute and plot field of thin_cylinder
    B = magpy.getB(magnet, grid)

    z = grid[:, :, 2]  # r,z from grid
    r = grid[:, :, 0]  # r,z from grid
    Bz = np.ascontiguousarray(B[:, :, 2])  # r,z from grid
    Br = np.ascontiguousarray(B[:, :, 0])  # r,z from grid

    end_index, zsep = find_separatrix(z, Bz)
    if end_index == nres:
        print("!!! domain too small !!!")
    # zmax = np.ceil(zsep+1)

    ## plot with switch on/off
    if plt_on:
        field_plot(tz, tr, Bz, Br, R, L, dR, ax)

    # interpolation
    # input arrays need to be C_CONTIGUOUS !!!
    zz = np.ascontiguousarray(z[0, :])
    rr = np.ascontiguousarray(r[:, 0])

    # define the r0 positions of the integrated field lines
    # rr[1] is added specifically for showing the separatrix
    r0 = [rr[i] for i in np.append(1, np.linspace(offset, nres / rmax * R - offset, nlines, dtype=int))]

    interp_r = RegularGridInterpolator([rr, zz], Br)
    interp_z = RegularGridInterpolator([rr, zz], Bz)

    # parallelism
    pa = []
    for p in [0.1, 0.3, 0.5, 0.9]:
        pa.append(parallelism(zz, rr, interp_z, interp_r, R, L, p))
    # pa = 1

    # return for integration and plotting
    return r0, interp_r, interp_z, pa, zz, rr, zsep, magnet


def integration(
    R, L, dR, df, rlines, interp_r, interp_z, parallel, zsep, plt_on=False, ax=None
):
    def field(t, y):
        """return the interpolated magnetic field (Bz,Br) at point (z,r)
        here used as the derivative f(y) = dy/dx in an ODE"""
        z, r = y
        bz = interp_z([r, z], method="linear")[0]
        br = interp_r([r, z], method="linear")[0]
        b = np.linalg.norm([bz, br])
        dz = bz / b
        dr = br / b
        return [
            -dz,
            -dr,
        ]  # opposed direction is necessary for integration in positive z direction

    def hit_wall(t, y):
        """end criterion for field line integration/ODE"""

        eps = 1e-1  # stop a bit before the wall, else motion integration parallel z is possible before the wall,
        # see scaling of R
        return y[1] - R + eps

    # settings
    hit_wall.terminal = True
    hit_wall.direction = 1

    maxstep = 0.3  # max. integration step

    # integration for all field lines
    z0 = 0
    for r0 in rlines:  # > 0!

        # bfield at starting point (0) -> mirror ratio mr
        bz0 = interp_z([r0, z0], method="linear")[0]
        br0 = interp_r([r0, z0], method="linear")[0]
        b0 = np.linalg.norm([bz0, br0])

        # integration of the field line(s)
        sol = solve_ivp(
            field,
            [0, 1000],
            [z0, r0],
            events=hit_wall,
            method="RK45",
            max_step=maxstep,
            atol=1e-4,
            rtol=1e-7,
        )  # was DOP853
        # TODO some integrations with DOP853 seem to quit after some steps??

        # calculate the length of each field line
        s = np.sqrt(np.power(np.diff(sol.y[0]), 2) + np.power(np.diff(sol.y[1]), 2))
        length = np.sum(s)  # not weighted length, that is done later
        # print(s.shape, length)

        # bfield at wall (w) -> mirror ratio mr
        rw = sol.y[1][-1]
        zw = sol.y[0][-1]
        bzw = interp_z([rw, zw], method="linear")[0]
        brw = interp_r([rw, zw], method="linear")[0]
        bw = np.linalg.norm([bzw, brw])
        mr = bw / b0  # mirror ratio

        # active volume va
        va = active_volume(sol.y[0], sol.y[1], L, R)

        if plt_on:
            if ax is None:
                plt.plot(sol.y[0], sol.y[1], "red")
            else:
                ax.plot(sol.y[0], sol.y[1], "red")

        # put results of each field line into df
        df = pd.concat(
            [
                pd.DataFrame(
                    [[R, L, dR, parallel, zsep - 0.5 * L, va, r0, mr, length]],
                    columns=df.columns,
                ),
                df,
            ],
            ignore_index=True,
        )

    if plt_on and ax is None:
        plt.savefig("R" + str(R) + "L" + str(L) + ".png", dpi=300)

    return df
