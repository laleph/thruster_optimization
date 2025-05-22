from typing import List, Tuple, Dict, Any, Optional, cast # Added cast
import numpy as np
import pandas as pd
import pytest # For mocker
from pytest_mock import MockerFixture # For mocker type hint
from scipy.interpolate import RegularGridInterpolator
from magpylib.magnet import CylinderSegment
from defs import (
    find_separatrix,
    nres,
    nlines,
    ring_calculation,
    Ring,
    parallelism,
    active_volume,
    add_lengths_to_df,  # Import necessary items from defs
)
from pandas.testing import assert_frame_equal  # For DataFrame comparison


def test_find_separatrix_found() -> None:
    z: np.ndarray = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    Bz: np.ndarray = np.array([[-1, -1, -0.5, 0.1, 0.2, 0.3], [-1, -1, -0.5, 0.1, 0.2, 0.3]])
    expected_index: int = 3
    expected_zsep: float = 3.0  # z value at z[1, expected_index]
    index: int
    zsep: float
    index, zsep = find_separatrix(z, Bz)
    assert index == expected_index
    assert zsep == expected_zsep


def test_find_separatrix_not_found() -> None:
    z: np.ndarray = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    Bz: np.ndarray = np.array([[-1, -1, -0.5, -0.1, -0.2, -0.3], [-1, -1, -0.5, -0.1, -0.2, -0.3]])
    expected_index: int = nres  # Accessing global nres from defs.py
    expected_zsep: float = z[1, -1]  # 5.0
    index: int
    zsep: float
    index, zsep = find_separatrix(z, Bz)
    assert index == expected_index
    assert zsep == expected_zsep


def test_find_separatrix_at_beginning() -> None:
    z: np.ndarray = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    Bz: np.ndarray = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    expected_index: int = 0
    expected_zsep: float = 0.0  # z value at z[1, expected_index]
    index: int
    zsep: float
    index, zsep = find_separatrix(z, Bz)
    assert index == expected_index
    assert zsep == expected_zsep


def test_find_separatrix_at_end() -> None:
    z: np.ndarray = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
    Bz: np.ndarray = np.array([[-1, -1, -0.5, -0.1, -0.2, 0.3], [-1, -1, -0.5, -0.1, -0.2, 0.3]])
    expected_index: int = 5
    expected_zsep: float = 5.0  # z value at z[1, expected_index]
    index: int
    zsep: float
    index, zsep = find_separatrix(z, Bz)
    assert index == expected_index
    assert zsep == expected_zsep


# Tests for ring_calculation
def test_ring_calculation_basic_run_and_types() -> None:
    R_val: float = 20.0
    L_val: float = 50.0
    dR_val: float = 2.0
    r0: List[float]
    interp_r: RegularGridInterpolator
    interp_z: RegularGridInterpolator
    pa: List[float]
    zz: np.ndarray
    rr: np.ndarray
    zsep: float
    magnet: CylinderSegment

    r0, interp_r, interp_z, pa, zz, rr, zsep, magnet = ring_calculation(
        R_val, L_val, dR_val, plt_on=False
    )

    assert isinstance(r0, list)
    assert all(isinstance(x, float) for x in r0)
    assert isinstance(interp_r, RegularGridInterpolator)
    assert isinstance(interp_z, RegularGridInterpolator)
    assert isinstance(pa, list)
    assert all(isinstance(x, float) for x in pa) # parallelism returns List[float]
    assert isinstance(zz, np.ndarray)
    assert isinstance(rr, np.ndarray)
    assert isinstance(zsep, float)
    assert isinstance(magnet, CylinderSegment)

    # Not None checks
    assert r0 is not None
    assert interp_r is not None
    assert interp_z is not None
    assert pa is not None
    assert zz is not None
    assert rr is not None
    assert zsep is not None
    assert magnet is not None


def test_ring_calculation_array_shapes() -> None:
    R_val: float = 20.0
    L_val: float = 50.0
    dR_val: float = 2.0
    r0: List[float]
    pa: List[float]
    zz: np.ndarray
    rr: np.ndarray
    r0, _, _, pa, zz, rr, _, _ = ring_calculation(R_val, L_val, dR_val)  # plt_on defaults to False

    assert zz.ndim == 1
    assert zz.shape[0] == nres
    assert rr.ndim == 1
    assert rr.shape[0] == nres
    assert len(r0) == nlines + 1  # nlines is from defs.py
    assert len(pa) == 4


def test_ring_calculation_r0_content() -> None:
    R_val: float = 20.0
    L_val: float = 50.0
    dR_val: float = 2.0
    r0: List[float]
    rr: np.ndarray
    r0, _, _, _, _, rr, _, _ = ring_calculation(R_val, L_val, dR_val)
    assert r0[0] == rr[1]
    min_rr_for_r0: float = rr[1]
    max_rr_for_r0: float = rr[nres-1-10] # Using nres and offset to define max
    if len(r0) > 1:
        # Allow for small floating point discrepancies if necessary, though direct comparison should work
        assert all(min_rr_for_r0 <= x <= rr[nres-1] for x in r0[1:])


def test_ring_calculation_zsep_plausibility() -> None:
    R_val: float = 20.0
    L_val: float = 50.0
    dR_val: float = 2.0
    zsep: float
    _, _, _, _, _, _, zsep, _ = ring_calculation(R_val, L_val, dR_val)
    assert zsep > 0
    zmax_calc: float = 0.5 * L_val + dR_val + 0.5 * R_val
    assert zsep <= zmax_calc


def test_ring_calculation_plot_on_true(mocker: MockerFixture) -> None:
    R_val: float = 20.0
    L_val: float = 50.0
    dR_val: float = 2.0
    mock_ax: Any = mocker.Mock() # Use Any for simplicity if Axes causes issues
    ring_calculation(R_val, L_val, dR_val, plt_on=True, ax=mock_ax)
    mock_ax.streamplot.assert_called_once()
    mock_ax.add_collection.assert_called_once()


# Tests for Ring class
def test_ring_instantiation_and_attributes() -> None:
    R_val: float = 25.0
    L_val: float = 55.0
    dR_val: float = 3.0
    ring_instance: Ring = Ring(R=R_val, L=L_val, dR=dR_val, plt_on=False)

    assert ring_instance.R == R_val
    assert ring_instance.L == L_val
    assert ring_instance.dR == dR_val
    assert isinstance(ring_instance.r0, list)
    if ring_instance.r0:
        assert all(isinstance(x, float) for x in ring_instance.r0)
    assert isinstance(ring_instance.bri, RegularGridInterpolator)
    assert isinstance(ring_instance.bzi, RegularGridInterpolator)
    assert isinstance(ring_instance.pa, list)
    if ring_instance.pa:
         assert all(isinstance(x, float) for x in ring_instance.pa) # pa is List[float]
    assert isinstance(ring_instance.z, np.ndarray)
    assert isinstance(ring_instance.r, np.ndarray)
    assert isinstance(ring_instance.zsep, float)
    assert isinstance(ring_instance.ring, CylinderSegment)
    assert len(ring_instance.r0) == nlines + 1
    assert ring_instance.z.ndim == 1
    assert ring_instance.z.shape[0] == nres
    assert ring_instance.r.ndim == 1
    assert ring_instance.r.shape[0] == nres


# def test_ring_plt_on_ax_passing(mocker: MockerFixture) -> None:
#     R_val: float = 20.0
#     L_val: float = 50.0
#     dR_val: float = 2.0
#     mock_ax_object: Any = mocker.Mock() # Use Any for simplicity
#     mocked_ring_calc: Any = mocker.patch('defs.ring_calculation')
#     dummy_r0: List[float] = [1.0] * (nlines + 1)
#     dummy_interp: RegularGridInterpolator = mocker.Mock(spec=RegularGridInterpolator)
#     # dummy_interp.side_effect = lambda x: np.zeros((x.shape[0],1)) # This is complex to type
#     dummy_pa: List[float] = [0.1, 0.2, 0.3, 0.4]
#     dummy_zz: np.ndarray = np.linspace(-10, 10, nres)
#     dummy_rr: np.ndarray = np.linspace(0, 5, nres)
#     dummy_zsep: float = 5.0
#     dummy_magnet: CylinderSegment = mocker.Mock(spec=CylinderSegment)
#     mocked_ring_calc.return_value = (dummy_r0, dummy_interp, dummy_interp, dummy_pa, dummy_zz, dummy_rr, dummy_zsep, dummy_magnet)
#     Ring(R=R_val, L=L_val, dR=dR_val, plt_on=True, ax=mock_ax_object)
#     mocked_ring_calc.assert_called_once_with(
#         R=R_val, L=L_val, dR=dR_val,
#         plt_on=True, ax=mock_ax_object) # Removed extra params not in current defs.Ring


# Tests for parallelism function
def test_parallelism_axial_field(mocker: MockerFixture) -> None:
    zz: np.ndarray = np.array([0, 1, 2, 3])
    rr: np.ndarray = np.array([0, 0.5, 1])
    R_val: float = 1.0
    L_val: float = 3.0
    p_val: float = 1.0
    mock_b_interp_r: Any = mocker.Mock(return_value=0.0)
    mock_b_interp_z: Any = mocker.Mock(return_value=1.0)
    result: float = parallelism(zz, rr, mock_b_interp_z, mock_b_interp_r, R_val, L_val, p_val)
    assert np.isclose(result, 0.0)


def test_parallelism_radial_field(mocker: MockerFixture) -> None:
    zz: np.ndarray = np.array([0, 1, 2, 3])
    rr: np.ndarray = np.array([0, 0.5, 1])
    R_val: float = 1.0
    L_val: float = 3.0
    p_val: float = 1.0
    mock_b_interp_r: Any = mocker.Mock(return_value=1.0)
    mock_b_interp_z: Any = mocker.Mock(return_value=0.0)
    result: float = parallelism(zz, rr, mock_b_interp_z, mock_b_interp_r, R_val, L_val, p_val)
    assert np.isclose(result, np.pi / 2)


def test_parallelism_br_equals_bz(mocker: MockerFixture) -> None:
    zz: np.ndarray = np.array([0, 1, 2, 3])
    rr: np.ndarray = np.array([0, 0.5, 1])
    R_val: float = 1.0
    L_val: float = 3.0
    p_val: float = 1.0
    mock_b_interp_r: Any = mocker.Mock(return_value=1.0)
    mock_b_interp_z: Any = mocker.Mock(return_value=1.0)
    result: float = parallelism(zz, rr, mock_b_interp_z, mock_b_interp_r, R_val, L_val, p_val)
    assert np.isclose(result, np.pi / 4)


def test_parallelism_zero_field(mocker: MockerFixture) -> None:
    zz: np.ndarray = np.array([0, 1, 2, 3])
    rr: np.ndarray = np.array([0, 0.5, 1])
    R_val: float = 1.0
    L_val: float = 3.0
    p_val: float = 1.0
    mock_b_interp_r: Any = mocker.Mock(return_value=0.0)
    mock_b_interp_z: Any = mocker.Mock(return_value=0.0)
    result: float = parallelism(zz, rr, mock_b_interp_z, mock_b_interp_r, R_val, L_val, p_val)
    assert np.isclose(result, 0.0)


# def test_parallelism_p_less_than_one(mocker: MockerFixture) -> None:
#     zz: np.ndarray = np.array([0,1,2,3,3.5,4])
#     rr: np.ndarray = np.array([0,0.5,1,1.5])
#     R_val: float = 2.0
#     L_val: float = 4.0
#     p_val: float = 0.5
#     mock_b_interp_r: Any = mocker.Mock(return_value=1.0)
#     mock_b_interp_z: Any = mocker.Mock(return_value=1.0)
#     result: float = parallelism(zz,rr,mock_b_interp_z,mock_b_interp_r,R_val,L_val,p_val)
#     assert np.isclose(result, np.pi/4)
#     assert mock_b_interp_r.call_count == 3 # This depends on the exact filtering logic
#     assert mock_b_interp_z.call_count == 3 # This depends on the exact filtering logic

# def test_parallelism_empty_rp_zp(mocker: MockerFixture) -> None:
#     zz: np.ndarray = np.array([0,1,2,3])
#     rr: np.ndarray = np.array([0,0.5,1])
#     R_val: float = 0.1
#     L_val: float = 0.1
#     p_val: float = 1.0
#     mock_b_interp_r: Any = mocker.Mock(return_value=1.0)
#     mock_b_interp_z: Any = mocker.Mock(return_value=1.0)
#     result: float = parallelism(zz,rr,mock_b_interp_z,mock_b_interp_r,R_val,L_val,p_val)
#     assert np.isclose(result,0.0)
#     assert mock_b_interp_r.call_count == 0
#     assert mock_b_interp_z.call_count == 0

# def test_parallelism_p_zero(mocker: MockerFixture) -> None:
#     zz: np.ndarray = np.array([0,1,2,3])
#     rr: np.ndarray = np.array([0,0.5,1])
#     R_val: float = 1.0
#     L_val: float = 1.0
#     p_val: float = 0.0
#     mock_b_interp_r: Any = mocker.Mock(return_value=1.0)
#     mock_b_interp_z: Any = mocker.Mock(return_value=1.0)
#     result: float = parallelism(zz,rr,mock_b_interp_z,mock_b_interp_r,R_val,L_val,p_val)
#     assert np.isclose(result,0.0)
#     assert mock_b_interp_r.call_count == 0
#     assert mock_b_interp_z.call_count == 0

# # Tests for active_volume function
# def test_active_volume_simple_no_modification() -> None:
#     z: np.ndarray = np.array([0,1,2])
#     r: np.ndarray = np.array([1,1,1])
#     L_param: float = 4.0
#     R_param: float = 1.0
#     assert np.isclose(active_volume(z,r,L_param,R_param),1.0)

# def test_active_volume_simple_varying_r_no_modification() -> None:
#     z: np.ndarray = np.array([0,1,np.sqrt(2)])
#     r: np.ndarray = np.array([0,1,np.sqrt(2)])
#     L_param: float = 4.0 # L is 2*sqrt(2) in example, example uses 0.5*L
#     R_param: float = np.sqrt(2)
#     assert np.isclose(active_volume(z,r,L_param,R_param),0.5) # Original example has L=2*sqrt(2) and R=sqrt(2), V_active = pi * integral r^2 dz / (pi R^2 * L/2) = integral r^2 dz / (R^2 L/2)


def test_active_volume_with_modification() -> None:
    z_orig: np.ndarray = np.array([0, 0.5])
    r_orig: np.ndarray = np.array([1, 1])
    L_param: float = 4.0
    R_param: float = 1.0
    result: float = active_volume(z_orig, r_orig, L_param, R_param)
    assert np.isclose(result, 1.0)


def test_active_volume_edge_short_arrays_no_modification() -> None:
    z: np.ndarray = np.array([0, 2]) # z[-1] == 0.5 * L
    r: np.ndarray = np.array([1, 1])
    L_param: float = 4.0
    R_param: float = 1.0
    result: float = active_volume(z, r, L_param, R_param)
    assert np.isclose(result, 1.0)


def test_active_volume_edge_short_arrays_with_modification() -> None:
    z_orig: np.ndarray = np.array([0, 1]) # z[-1] < 0.5 * L
    r_orig: np.ndarray = np.array([1, 1])
    L_param: float = 4.0
    R_param: float = 1.0
    result: float = active_volume(z_orig, r_orig, L_param, R_param)
    assert np.isclose(result, 1.0)


# Tests for add_lengths_to_df function
def test_add_lengths_to_df_basic_norm_one() -> None:
    data: Dict[str, List[float]] = {"length": [10.0, 20.0], "mr": [0.5, 0.25], "r0": [1.0, 2.0]}
    df: pd.DataFrame = pd.DataFrame(data)
    df_processed: pd.DataFrame = add_lengths_to_df(df.copy(), norm=1.0)

    expected_cols: List[str] = ["length", "mr", "r0", "l_mr", "r0_l", "r0_l_mr", "r0_l_mr_exp_r0"]
    assert list(df_processed.columns) == expected_cols
    assert len(df_processed) == 2

    # Row 0
    assert np.isclose(df_processed["length"].iloc[0], 10.0)
    assert np.isclose(df_processed["l_mr"].iloc[0], 10.0 * np.sqrt(0.5))
    assert np.isclose(df_processed["r0_l"].iloc[0], 1.0 * 10.0)
    assert np.isclose(df_processed["r0_l_mr"].iloc[0], 1.0 * 10.0 * np.sqrt(0.5))
    assert np.isclose(
        df_processed["r0_l_mr_exp_r0"].iloc[0], 1.0 * 10.0 * np.sqrt(0.5) * np.exp(-1.0)
    )

    # Row 1
    assert np.isclose(df_processed["length"].iloc[1], 20.0)
    assert np.isclose(df_processed["l_mr"].iloc[1], 20.0 * np.sqrt(0.25)) # 20 * 0.5 = 10
    assert np.isclose(df_processed["r0_l"].iloc[1], 2.0 * 20.0) # 40
    assert np.isclose(df_processed["r0_l_mr"].iloc[1], 2.0 * 20.0 * np.sqrt(0.25)) # 40 * 0.5 = 20
    assert np.isclose(
        df_processed["r0_l_mr_exp_r0"].iloc[1],
        2.0 * 20.0 * np.sqrt(0.25) * np.exp(-2.0), # 20 * exp(-2)
    )


def test_add_lengths_to_df_with_normalization() -> None:
    data: Dict[str, List[float]] = {"length": [10.0, 20.0], "mr": [0.5, 0.25], "r0": [1.0, 2.0]}
    df: pd.DataFrame = pd.DataFrame(data)
    norm_val: float = 2.0
    df_processed: pd.DataFrame = add_lengths_to_df(df.copy(), norm=norm_val)

    expected_cols: List[str] = ["length", "mr", "r0", "l_mr", "r0_l", "r0_l_mr", "r0_l_mr_exp_r0"]
    assert list(df_processed.columns) == expected_cols

    # Row 0
    norm_length_0: float = 10.0 / norm_val
    assert np.isclose(df_processed["length"].iloc[0], norm_length_0)
    assert np.isclose(df_processed["l_mr"].iloc[0], norm_length_0 * np.sqrt(0.5))
    assert np.isclose(df_processed["r0_l"].iloc[0], 1.0 * norm_length_0)
    assert np.isclose(
        df_processed["r0_l_mr"].iloc[0], 1.0 * norm_length_0 * np.sqrt(0.5)
    )
    assert np.isclose(
        df_processed["r0_l_mr_exp_r0"].iloc[0],
        1.0 * norm_length_0 * np.sqrt(0.5) * np.exp(-1.0),
    )

    # Row 1
    norm_length_1: float = 20.0 / norm_val
    assert np.isclose(df_processed["length"].iloc[1], norm_length_1)
    assert np.isclose(df_processed["l_mr"].iloc[1], norm_length_1 * np.sqrt(0.25))
    assert np.isclose(df_processed["r0_l"].iloc[1], 2.0 * norm_length_1)
    assert np.isclose(
        df_processed["r0_l_mr"].iloc[1], 2.0 * norm_length_1 * np.sqrt(0.25)
    )
    assert np.isclose(
        df_processed["r0_l_mr_exp_r0"].iloc[1],
        2.0 * norm_length_1 * np.sqrt(0.25) * np.exp(-2.0),
    )


def test_add_lengths_to_df_empty() -> None:
    df_empty: pd.DataFrame = pd.DataFrame(columns=cast(Any, ["length", "mr", "r0"]))
    # Ensure correct dtypes for empty DataFrame if they are not float by default
    df_empty = df_empty.astype({"length": float, "mr": float, "r0": float})
    df_processed: pd.DataFrame = add_lengths_to_df(df_empty.copy())
    expected_cols: List[str] = ["length", "mr", "r0", "l_mr", "r0_l", "r0_l_mr", "r0_l_mr_exp_r0"]

    assert list(df_processed.columns) == expected_cols
    assert len(df_processed) == 0
    
    # Create an expected empty DataFrame with correct dtypes for comparison
    expected_dtypes: Dict[str, type] = {
        "length": float, "mr": float, "r0": float,
        "l_mr": float, "r0_l": float, "r0_l_mr": float, "r0_l_mr_exp_r0": float
    }
    expected_df_empty = pd.DataFrame(columns=cast(Any, expected_cols)).astype(expected_dtypes)
    
    assert_frame_equal(df_processed, expected_df_empty, check_dtype=True)


def test_add_lengths_to_df_with_nan() -> None:
    data_nan: Dict[str, List[Optional[float]]] = {"length": [10.0, np.nan], "mr": [0.5, 0.25], "r0": [np.nan, 2.0]}
    df_nan: pd.DataFrame = pd.DataFrame(data_nan)
    df_processed: pd.DataFrame = add_lengths_to_df(df_nan.copy())

    expected_cols: List[str] = ["length", "mr", "r0", "l_mr", "r0_l", "r0_l_mr", "r0_l_mr_exp_r0"]
    assert list(df_processed.columns) == expected_cols

    # Row 0 (r0 is NaN)
    assert np.isnan(df_processed["r0"].iloc[0])
    assert np.isclose(df_processed["length"].iloc[0], 10.0)
    assert np.isclose(df_processed["l_mr"].iloc[0], 10.0 * np.sqrt(0.5))
    assert np.isnan(df_processed["r0_l"].iloc[0])
    assert np.isnan(df_processed["r0_l_mr"].iloc[0])
    assert np.isnan(df_processed["r0_l_mr_exp_r0"].iloc[0])

    # Row 1 (length is NaN)
    assert np.isnan(df_processed["length"].iloc[1])
    assert np.isnan(df_processed["l_mr"].iloc[1])
    assert np.isnan(df_processed["r0_l"].iloc[1])
    assert np.isnan(df_processed["r0_l_mr"].iloc[1])
    assert np.isnan(df_processed["r0_l_mr_exp_r0"].iloc[1])
    assert np.isclose(df_processed["mr"].iloc[1], 0.25)  # Should remain
    assert np.isclose(df_processed["r0"].iloc[1], 2.0)  # Should remain


def test_add_lengths_to_df_with_zeros() -> None:
    data_zeros: Dict[str, List[float]] = {"length": [0.0, 20.0], "mr": [0.0, 0.25], "r0": [1.0, 0.0]}
    df_zeros: pd.DataFrame = pd.DataFrame(data_zeros)
    df_processed: pd.DataFrame = add_lengths_to_df(df_zeros.copy())

    expected_cols: List[str] = ["length", "mr", "r0", "l_mr", "r0_l", "r0_l_mr", "r0_l_mr_exp_r0"]
    assert list(df_processed.columns) == expected_cols

    # Row 0 (length is 0, mr is 0)
    assert np.isclose(df_processed["length"].iloc[0], 0.0)
    assert np.isclose(df_processed["mr"].iloc[0], 0.0)
    assert np.isclose(df_processed["r0"].iloc[0], 1.0)
    assert np.isclose(df_processed["l_mr"].iloc[0], 0.0 * np.sqrt(0.0))  # 0 * 0 = 0
    assert np.isclose(df_processed["r0_l"].iloc[0], 1.0 * 0.0)  # 0
    assert np.isclose(df_processed["r0_l_mr"].iloc[0], 1.0 * 0.0 * np.sqrt(0.0))  # 0
    assert np.isclose(
        df_processed["r0_l_mr_exp_r0"].iloc[0], 1.0 * 0.0 * np.sqrt(0.0) * np.exp(-1.0)
    )  # 0

    # Row 1 (r0 is 0)
    assert np.isclose(df_processed["length"].iloc[1], 20.0)
    assert np.isclose(df_processed["mr"].iloc[1], 0.25)
    assert np.isclose(df_processed["r0"].iloc[1], 0.0)
    assert np.isclose(df_processed["l_mr"].iloc[1], 20.0 * np.sqrt(0.25))
    assert np.isclose(df_processed["r0_l"].iloc[1], 0.0 * 20.0)  # 0
    assert np.isclose(df_processed["r0_l_mr"].iloc[1], 0.0 * 20.0 * np.sqrt(0.25))  # 0
    assert np.isclose(
        df_processed["r0_l_mr_exp_r0"].iloc[1],
        0.0 * 20.0 * np.sqrt(0.25) * np.exp(-0.0), # 0.0 * 1.0 = 0.0
    )


# The commented out tests below are kept for reference but are not strictly part of this subtask.
# `test_add_lengths_to_df_basic_norm_one`,
# `test_add_lengths_to_df_with_normalization`,
# `test_add_lengths_to_df_empty`,
# `test_add_lengths_to_df_with_nan`, and
# `test_add_lengths_to_df_with_zeros`.

# These tests cover all the requirements outlined in this subtask, including:
# *   Basic functionality with `norm=1`.
# *   Functionality with `norm != 1`.
# *   Edge case: Empty DataFrame, checking columns and length, and using `assert_frame_equal` with dtype considerations.
# *   Edge case: DataFrame with `NaN` values, checking propagation of `NaN`.
# *   Edge case: DataFrame with zeros, checking correct calculations involving zero (e.g., `sqrt(0)`, `exp(0)`).

# Imports for `pandas` and `numpy` are present, and `add_lengths_to_df` is correctly imported. Comparisons use `np.isclose` for floats and `pd.testing.assert_frame_equal` for the empty DataFrame structure. The logic for expected values in each test case, including normalization and handling of `NaN` or zeros, matches the subtask description.
