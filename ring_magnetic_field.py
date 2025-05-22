#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from magpylib.magnet import Cylinder, CylinderSegment
from typing import List, Tuple, Dict, Any, Optional, Callable, cast 
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
# from scipy.integrate._ivp.ivp import OdeSolution
from matplotlib.axes import Axes 
from matplotlib.figure import Figure 

from defs import active_volume, ring_calculation, add_lengths_to_df, Ring

mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["font.size"] = 20
plt.rc("legend", fontsize=17)


# In[2]:


ngeom: int = 41

def integration(
    R: float,
    L: float,
    dR: float,
    df: pd.DataFrame,
    rlines: List[float],
    interp_r: RegularGridInterpolator,
    interp_z: RegularGridInterpolator,
    parallel: List[float],
    zsep: float,
    plt_on: bool = False,
    ax: Optional[Axes] = None, 
) -> pd.DataFrame:
    def field(t: float, y: Tuple[float, float]) -> List[float]:
        z_coord: float = y[0]
        r_coord: float = y[1]
        bz_val: float = float(interp_z([r_coord, z_coord], method="linear")[0])
        br_val: float = float(interp_r([r_coord, z_coord], method="linear")[0])
        b_norm: float = float(np.linalg.norm([bz_val, br_val])) # Cast to float
        if b_norm == 0:
            return [0.0, 0.0]
        dz_dt: float = bz_val / b_norm
        dr_dt: float = br_val / b_norm
        return [-dz_dt, -dr_dt]

    def hit_wall(t: float, y: Tuple[float, float]) -> float:
        eps: float = 1e-1
        return y[1] - R + eps

    hit_wall.terminal = True # type: ignore
    hit_wall.direction = 1 # type: ignore
    maxstep: float = 0.3
    z0_coord: float = 0.0
    
    current_df = df.copy()

    r0_val: float
    for r0_val in rlines:
        bz0_val: float = float(interp_z([r0_val, z0_coord], method="linear")[0])
        br0_val: float = float(interp_r([r0_val, z0_coord], method="linear")[0])
        b0_norm: float = float(np.linalg.norm([bz0_val, br0_val])) # Cast to float
        
        sol: Any = solve_ivp(
            field, [0.0, 1000.0], (z0_coord, r0_val),
            events=hit_wall, method="RK45", max_step=maxstep, atol=1e-4, rtol=1e-7
        )
        
        s_arr: np.ndarray = np.sqrt(np.power(np.diff(sol.y[0]), 2) + np.power(np.diff(sol.y[1]), 2))
        length_val: float = float(np.sum(s_arr)) # Cast to float
        
        rw_val: float = float(sol.y[1][-1])
        zw_val: float = float(sol.y[0][-1])
        bzw_val: float = float(interp_z([rw_val, zw_val], method="linear")[0])
        brw_val: float = float(interp_r([rw_val, zw_val], method="linear")[0])
        bw_norm: float = float(np.linalg.norm([bzw_val, brw_val])) # Cast to float
        mr_val: float = float(bw_norm / b0_norm) if b0_norm != 0 else 0.0 # Cast to float
        
        va_val: float = float(active_volume(sol.y[0], sol.y[1], L, R)) 
        
        if plt_on and ax is not None:
            ax.plot(sol.y[0], sol.y[1], "red")
            
        new_row_data: List[Any] = [[R, L, dR, parallel, zsep - 0.5 * L, va_val, r0_val, mr_val, length_val]]
        new_row_df = pd.DataFrame(new_row_data, columns=current_df.columns)
        current_df = pd.concat([new_row_df, current_df], ignore_index=True)
        
    if plt_on and ax is None:
        plt.savefig(f"R{R}L{L}.png", dpi=300)
        
    return current_df


# In[3]:


plt_on_scan: bool = False
df_columns_scan: List[str] = ["R", "L", "dR", "parallelism", "zsep_L", "va", "r0", "mr", "length"]
df_scan: pd.DataFrame = pd.DataFrame(columns=cast(Any, df_columns_scan)) 
R_scan_val: float = 20.0
dR_scan_val: float
L_scan_val: float

for dR_scan_val in [2.0]:
    for L_scan_val in np.linspace(10, 90, ngeom):
        rlines_res_scan: List[float]
        Brinterp_res_scan: RegularGridInterpolator
        Bzinterp_res_scan: RegularGridInterpolator
        parallel_res_scan: List[float]
        zsep_res_scan: float
        
        rlines_res_scan, Brinterp_res_scan, Bzinterp_res_scan, parallel_res_scan, _, _, zsep_res_scan, _ = ring_calculation(
            R_scan_val, L_scan_val, dR_scan_val, plt_on_scan
        )
        df_scan = integration(
            R_scan_val, L_scan_val, dR_scan_val, df_scan, rlines_res_scan, Brinterp_res_scan, Bzinterp_res_scan, parallel_res_scan, zsep_res_scan, plt_on_scan
        )


# In[4]:


def format_func(value: float, tick_number: int) -> str:
    N: int = int(np.round(2 * value / np.pi))
    if N == 0: return "0"
    elif N == 1: return r"$\pi/2$"
    elif N == 2: return r"$\pi$"
    elif N % 2 > 0: return rf"${N}\pi/2$"
    else: return rf"${N // 2}\pi$"

pb_fig4: float = df_scan.R.iloc[0] * np.sqrt(6) if not df_scan.empty and 'R' in df_scan.columns and not df_scan.R.empty else 0.0

fig_fig4: Figure 
ax_fig4: Axes 
fig_fig4, ax_fig4 = plt.subplots(figsize=(10, 7))

pp_fig4: List[float] = [0.1, 0.3, 0.5, 0.9]
pas_fig4: np.ndarray = np.array(df_scan.parallelism.tolist(), dtype=float) if not df_scan.empty and 'parallelism' in df_scan.columns else np.array([])

if pas_fig4.size > 0 and pas_fig4.ndim == 2 and pas_fig4.shape[1] > 0: 
   ax_fig4.plot(df_scan.L, np.sin(pas_fig4[:, 0]))

ax_fig4.axvline(x=pb_fig4, color="tab:olive")
ax_fig4.set_xlabel("L / mm")
ax_fig4.set_ylabel(r"$\sin (\alpha)$")
ax_fig4.ticklabel_format(axis="y", scilimits=(-3, 3))
fig_fig4.tight_layout()
plt.grid()
plt.savefig("parallelism.png", dpi=300)


# In[5]:


fig_fig5: Figure 
ax_fig5_main: Axes 
fig_fig5, ax_fig5_main = plt.subplots(figsize=(10, 7), dpi=300)
ax_fig5_twin: Axes = ax_fig5_main.twinx() 

if not df_scan.empty:
    lns1_fig5: Any = ax_fig5_main.plot(df_scan.L, df_scan.zsep_L / df_scan.L, label=r"$\widetilde{z}$")
    ax_fig5_main.set_xlabel("L / mm")
    ax_fig5_main.set_ylabel(r"$\widetilde{z}$ / mm")
    ax_fig5_main.grid()

    nn_fig5: int = 20
    line_indices_fig5: List[int] = [0 + nn_fig5 * rrr for rrr in range(ngeom)] 
    valid_indices_fig5: List[int] = [idx for idx in line_indices_fig5 if idx < len(df_scan)]
    
    if valid_indices_fig5: 
        lns2_fig5: Any = ax_fig5_twin.plot(df_scan.L.iloc[valid_indices_fig5], df_scan.va.iloc[valid_indices_fig5], "r", label=r"$\widetilde{V}$")
        ax_fig5_twin.set_ylabel(r"$\widetilde{V}$")
        if 'lns1_fig5' in locals(): 
            lns_combined_fig5: List[Any] = lns1_fig5 + lns2_fig5
            labs_fig5: List[str] = [l.get_label() for l in lns_combined_fig5]
            ax_fig5_main.legend(lns_combined_fig5, labs_fig5)

fig_fig5.tight_layout()
plt.savefig("zsep_va.png", dpi=300, bbox_inches="tight")


# In[6]:

df_processed_scan: Optional[pd.DataFrame] = None
dfsum_scan: Optional[pd.DataFrame] = None

if not df_scan.empty:
    df_scan.to_csv("scan_l2r_mr.csv")
    df_processed_scan = add_lengths_to_df(df_scan.copy())
    dfsum_scan = df_processed_scan.groupby(["R", "L"]).sum().reset_index()
    print(dfsum_scan) 
else:
    print("df_scan is empty, skipping processing and CSV save.")


# In[7]:
# Empty cell

# In[8]:
# Empty cell

# In[9]:


def plot_lengths(df_plot: pd.DataFrame, dfsum_plot: pd.DataFrame) -> None:
    if df_plot.empty or dfsum_plot.empty:
        print("DataFrame for plot_lengths is empty. Skipping plot.")
        return
        
    pb_plot: float = df_plot.R.iloc[0] * np.sqrt(6) if 'R' in df_plot.columns and not df_plot.R.empty else 0.0
    
    fig_lengths: Figure 
    ax_lengths: np.ndarray 
    fig_lengths, ax_lengths = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    nlines_lengths: int = 20
    rhos_lengths: List[float] = []
    if 'r0' in df_plot.columns and len(df_plot.r0) > 17:
        rhos_lengths = [float(df_plot.r0.iloc[2]), float(df_plot.r0.iloc[10]), float(df_plot.r0.iloc[17])] 
        
    label_avg_lengths: str = "average"

    ax_lengths[0].plot(dfsum_plot.L, dfsum_plot.length / nlines_lengths, label=label_avg_lengths)
    ax_lengths[0].axvline(x=pb_plot, color="tab:olive")
    ax_lengths[0].set_ylabel(r"$\ell$ / a.u.")
    
    ax_lengths[1].plot(dfsum_plot.L, dfsum_plot.mr / nlines_lengths, label=label_avg_lengths)
    ax_lengths[1].axvline(x=pb_plot, color="tab:olive")
    ax_lengths[1].set_ylabel(r"$R_m$")

    ax_lengths[2].plot(dfsum_plot.L, dfsum_plot["r0_l_mr_exp_r0"] / nlines_lengths, label=label_avg_lengths)
    ax_lengths[2].axvline(x=pb_plot, color="tab:olive")
    ax_lengths[2].set_ylabel(r"$\tau$ / a.u.")

    ls_lengths: List[str] = ["-", ":", "--"]
    i_len: int
    l_rho_len: float
    for i_len, l_rho_len in enumerate(rhos_lengths):
        dff_lengths: pd.DataFrame = df_plot.where(np.abs(df_plot.r0 - l_rho_len) < 0.2).dropna()
        if not dff_lengths.empty:
            ax_lengths[0].plot(dff_lengths.L, dff_lengths.length, label=f"{np.round(l_rho_len, decimals=1)} mm", linestyle=ls_lengths[i_len])
            ax_lengths[1].plot(dff_lengths.L, dff_lengths.mr, label=f"{np.round(l_rho_len, decimals=1)} mm", linestyle=ls_lengths[i_len])
            ax_lengths[2].plot(dff_lengths.L, dff_lengths["r0_l_mr_exp_r0"], label=f"{np.round(l_rho_len, decimals=1)} mm", linestyle=ls_lengths[i_len])
            
    ax_lengths[1].legend()
    ax_lengths[2].set_xlabel("L / mm")

if df_processed_scan is not None and dfsum_scan is not None: 
    plot_lengths(df_processed_scan, dfsum_scan)
    plt.savefig("lengths.png", dpi=300)
else:
    print("Skipping plot_lengths as input DataFrames are not defined.")


# In[10]:
# Empty cell

# In[11]:

ndf_scan: Optional[pd.DataFrame] = None
ndfsum_scan: Optional[pd.DataFrame] = None

if df_processed_scan is not None: 
    ndf_scan = df_processed_scan.copy()
    ndf_scan = add_lengths_to_df(ndf_scan, ndf_scan.L) 
    ndfsum_scan = ndf_scan.groupby(["R", "L"]).sum().reset_index()
    print(ndfsum_scan) 
    if ndf_scan is not None and ndfsum_scan is not None: # Check again for plot_lengths
        plot_lengths(ndf_scan, ndfsum_scan)
        plt.savefig("normed_lengths.png", dpi=300)
else:
    print("Skipping normed_lengths plot as df_processed_scan is not defined.")


# In[12]:
# Empty cell

# In[13]:
# Empty cell

# In[14]:


plt_on_fig3_val: bool = True
df_fig3_cols_val: List[str] = ["R", "L", "dR", "parallelism", "zsep_L", "va", "r0", "mr", "length"]
df_fig3_val: pd.DataFrame = pd.DataFrame(columns=cast(Any, df_fig3_cols_val)) 

fig_fig3_val: Figure = plt.figure(constrained_layout=True, figsize=(11, 8), dpi=300) 

gs_fig3: GridSpec = GridSpec(2, 6, figure=fig_fig3_val)
ax1_fig3: Axes = fig_fig3_val.add_subplot(gs_fig3[0, :2]) 
ax2_fig3: Axes = fig_fig3_val.add_subplot(gs_fig3[0, 2:]) 
ax3_fig3: Axes = fig_fig3_val.add_subplot(gs_fig3[-1, :]) 

axes_list_fig3: List[Axes] = [ax1_fig3, ax2_fig3, ax3_fig3] 
label_list_fig3: List[str] = ["(a)", "(b)", "(c)"]

R_fig3_val: float = 20.0
dR_fig3_val_loop: float
L_fig3_val_loop: float
i_fig3_val: int

for dR_fig3_val_loop in [2.0]:
    for i_fig3_val, L_fig3_val_loop in enumerate([10.0, 50.0, 90.0]):
        m_ring: Ring = Ring(R_fig3_val, L_fig3_val_loop, dR_fig3_val_loop, plt_on_fig3_val, axes_list_fig3[i_fig3_val])
        df_fig3_val = integration(
            m_ring.R, m_ring.L, m_ring.dR, df_fig3_val, m_ring.r0, m_ring.bri, m_ring.bzi, m_ring.pa, m_ring.zsep, plt_on_fig3_val, axes_list_fig3[i_fig3_val]
        )
        axes_list_fig3[i_fig3_val].plot(m_ring.zsep, 0, "k*", markersize=14)
        axes_list_fig3[i_fig3_val].set_title(label_list_fig3[i_fig3_val], fontfamily="serif", loc="left", fontsize="medium")

plt.savefig("RL_field.png", dpi=300, bbox_inches="tight")


# In[15]:
# Empty cell

# In[16]:


ngeom_r_scan_val: int = 18
plt_on_r_scan_val: bool = False
dfr_cols_val: List[str] = ["R", "L", "dR", "parallelism", "zsep_L", "va", "r0", "mr", "length"]
dfr_val: pd.DataFrame = pd.DataFrame(columns=cast(Any, dfr_cols_val)) 

L_r_scan_fixed_val: float = 50.0
dR_r_scan_loop: float
R_r_scan_loop: float

for dR_r_scan_loop in [2.0]:
    for R_r_scan_loop in np.linspace(6, 40, ngeom_r_scan_val):
        rlines_r_val: List[float]
        Brinterp_r_val: RegularGridInterpolator
        Bzinterp_r_val: RegularGridInterpolator
        parallel_r_val: List[float]
        zsep_r_val: float
        
        rlines_r_val, Brinterp_r_val, Bzinterp_r_val, parallel_r_val, _, _, zsep_r_val, _ = ring_calculation(
            R_r_scan_loop, L_r_scan_fixed_val, dR_r_scan_loop, plt_on_r_scan_val
        )
        dfr_val = integration(
            R_r_scan_loop, L_r_scan_fixed_val, dR_r_scan_loop, dfr_val, rlines_r_val, Brinterp_r_val, Bzinterp_r_val, parallel_r_val, zsep_r_val, plt_on_r_scan_val
        )


# In[17]:

dfr_processed_val: Optional[pd.DataFrame] = None 
dfrsum_val: Optional[pd.DataFrame] = None 

if 'dfr_val' in locals() and isinstance(dfr_val, pd.DataFrame) and not dfr_val.empty: 
    print(dfr_val) 
    dfr_processed_val = add_lengths_to_df(dfr_val.copy()) 
    dfrsum_val = dfr_processed_val.groupby(["R", "L"]).sum().reset_index() 
    print(dfrsum_val)
else:
    print("dfr_val is not defined or is empty.")


# In[18]:
# Empty cell, content merged above.

# In[19]:


def plot_lengths_R(df_plot_r_arg: pd.DataFrame, dfsum_plot_r_arg: pd.DataFrame) -> None:
    if df_plot_r_arg.empty or dfsum_plot_r_arg.empty:
        print("DataFrame for plot_lengths_R is empty. Skipping plot.")
        return

    pb_plot_r_val: float = df_plot_r_arg.L.iloc[0] / np.sqrt(6) if not df_plot_r_arg.L.empty else 0.0
    
    fig_plot_r_val: Figure 
    ax_plot_r_val: np.ndarray 
    fig_plot_r_val, ax_plot_r_val = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    nlines_plot_r_val: int = 20 
    rhos_indices_r: List[int] = [2, 10, 17] 
    rlabel_r: List[str] = ["0.85", "0.5", "0.1"] 
    label_avg_r_val: str = "average"

    ax_plot_r_val[0].plot(dfsum_plot_r_arg.R, dfsum_plot_r_arg.length / nlines_plot_r_val, label=label_avg_r_val)
    ax_plot_r_val[0].axvline(x=pb_plot_r_val, color="tab:olive")
    ax_plot_r_val[0].set_ylabel(r"$\ell$ / a.u.")
    
    ax_plot_r_val[1].plot(dfsum_plot_r_arg.R, dfsum_plot_r_arg.mr / nlines_plot_r_val, label=label_avg_r_val)
    ax_plot_r_val[1].axvline(x=pb_plot_r_val, color="tab:olive")
    ax_plot_r_val[1].set_ylabel("$R_m$")

    ax_plot_r_val[2].plot(dfsum_plot_r_arg.R, dfsum_plot_r_arg["r0_l_mr_exp_r0"] / nlines_plot_r_val, label=label_avg_r_val)
    ax_plot_r_val[2].axvline(x=pb_plot_r_val, color="tab:olive")
    ax_plot_r_val[2].set_ylabel(r"$\tau$ / a.u.")

    ls_r_val: List[str] = ["-", ":", "--"]
    i_r_val: int
    l_idx_r: int
    for i_r_val, l_idx_r in enumerate(rhos_indices_r):
        line_indices_r: List[int] = [l_idx_r + nlines_plot_r_val * rrr for rrr in range(ngeom_r_scan_val)] 
        valid_line_indices_r: List[int] = [idx for idx in line_indices_r if idx < len(df_plot_r_arg)]
        dff_r_val: pd.DataFrame = df_plot_r_arg.iloc[valid_line_indices_r]
        
        if not dff_r_val.empty:
           ax_plot_r_val[0].plot(dff_r_val.R, dff_r_val.length, label=f"$r_0/R = {rlabel_r[i_r_val]}$", linestyle=ls_r_val[i_r_val])
           ax_plot_r_val[1].plot(dff_r_val.R, dff_r_val.mr, label=f"$r_0/R = {rlabel_r[i_r_val]}$", linestyle=ls_r_val[i_r_val])
           ax_plot_r_val[2].plot(dff_r_val.R, dff_r_val["r0_l_mr_exp_r0"], label=rlabel_r[i_r_val], linestyle=ls_r_val[i_r_val])
            
    ax_plot_r_val[0].legend()
    ax_plot_r_val[2].set_xlabel("R / mm")

if dfr_processed_val is not None and dfrsum_val is not None: 
    plot_lengths_R(dfr_processed_val, dfrsum_val)
    plt.savefig("lengths_R_varied.png", dpi=300)
else:
    print("Skipping plot_lengths_R as input DataFrames are not defined.")


# In[20]:
# Empty cell

# In[21]:


ring0_example: CylinderSegment = CylinderSegment(magnetization=(0, 0, 1000), dimension=(2, 4, 1, 0, 360))
inner_cyl: Cylinder = Cylinder(magnetization=(0, 0, -1000), dimension=(4, 1))
outer_cyl: Cylinder = Cylinder(magnetization=(0, 0, 1000), dimension=(8, 1))
ring1_collection: Any = inner_cyl + outer_cyl 

b_field_ring0_ex: np.ndarray = ring0_example.getB((1, 2, 4))
b_field_ring1_ex: np.ndarray = ring1_collection.getB((1, 2, 4))

print("getB from Cylindersegment", b_field_ring0_ex)
print("getB from Cylinder cut-out", b_field_ring1_ex)


# In[22]:
# Empty cell
