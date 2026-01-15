from __future__ import annotations

import numpy as np
import pandas as pd

import time
from typing import Callable

from utilities.data.lvm import lvm_to_df
import CoolProp.CoolProp as CP

type Formula = Callable[[pd.DataFrame], pd.Series]


def propagate_mc(
    values: pd.DataFrame,
    sigmas: pd.DataFrame,
    formulas: list[tuple[str, Formula]],
    n_draws: int = 1_000,
    seed: int | None = None,
    ddof: int = 0,
    batch_size: int = 1_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate per-cell 1-sigma uncertainties through derived formulas via Monte Carlo.

    Parameters
    ----------
    values : DataFrame
        Measured values. Rows are observations, columns are features.
    sigmas : DataFrame
        1-sigma uncertainties for each cell in ``values`` (same shape/index/cols).
    formulas : list of (name, function)
        Each function takes a sampled DataFrame and returns a Series aligned to
        ``values.index``. The name is used as the output column label.
    n_draws : int
        Number of Monte Carlo draws.
    seed : int | None
        Random seed for reproducibility.
    ddof : int
        Delta degrees of freedom for the output standard deviation (0 or 1).
    batch_size : int
        Draws per batch to keep memory usage bounded.

    Returns
    -------
    (means, sigmas) : (DataFrame, DataFrame)
        Two DataFrames with the same index as ``values`` and one column per formula.
    """

    if values.shape != sigmas.shape:
        raise ValueError(
            f"Shape mismatch: values={values.shape}, sigmas={sigmas.shape}"
        )

    if list(values.columns) != list(sigmas.columns):
        raise ValueError("Column mismatch between values and sigmas")

    if not values.index.equals(sigmas.index):
        raise ValueError("Index mismatch between values and sigmas")

    if n_draws <= 0:
        raise ValueError("n_draws must be > 0")

    if ddof not in (0, 1):
        raise ValueError("ddof must be 0 or 1")

    if ddof == 1 and n_draws < 2:
        raise ValueError("ddof=1 requires n_draws >= 2")

    if not formulas:
        raise ValueError("At least one formula is required")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    for name, fn in formulas:
        if not name:
            raise ValueError("Formula names must be non-empty")
        if not callable(fn):
            raise TypeError(f"Formula '{name}' is not callable")

    # Work in NumPy for sampling speed and to avoid per-draw DataFrame overhead.
    values_arr = values.to_numpy(dtype=float, copy=False)
    sigmas_arr = sigmas.to_numpy(dtype=float, copy=False)

    rng = np.random.default_rng(seed)
    n_rows = values_arr.shape[0]
    n_formulas = len(formulas)
    sum_values = np.zeros((n_formulas, n_rows), dtype=float)
    sumsq_values = np.zeros((n_formulas, n_rows), dtype=float)

    # Use user-provided names for stable output columns.
    names = [name for name, _ in formulas]
    for name, fn in formulas:
        # Dry run to catch index/shape issues before Monte Carlo work.
        out = fn(values)
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=values.index)
        if not out.index.equals(values.index):
            raise ValueError(f"Formula '{name}' output index mismatch")

    # Stream draws in batches to bound memory while keeping Python overhead reasonable.
    remaining = n_draws
    while remaining > 0:
        draw_count = min(batch_size, remaining)
        sample = rng.normal(
            loc=values_arr, scale=sigmas_arr, size=(draw_count, *values_arr.shape)
        )
        for draw in sample:
            sample_df = pd.DataFrame(draw, index=values.index, columns=values.columns)
            for idx, (_, fn) in enumerate(formulas):
                # Accumulate sum and sum of squares to avoid storing all draws.
                out = fn(sample_df)
                arr = np.asarray(out, dtype=float)
                sum_values[idx] += arr
                sumsq_values[idx] += arr * arr
        remaining -= draw_count

    means = sum_values / n_draws
    variances = sumsq_values / n_draws - means * means

    if ddof:
        variances *= n_draws / (n_draws - ddof)

    variances = np.maximum(variances, 0.0)
    sigmas_out = np.sqrt(variances)

    means_df = pd.DataFrame(means.T, index=values.index, columns=names)
    sigmas_df = pd.DataFrame(sigmas_out.T, index=values.index, columns=names)
    return means_df, sigmas_df


if __name__ == "__main__":
    def _test_basic_zero_sigma() -> tuple(pd.DataFrame,pd.DataFrame):
        # Minimal self-test for quick sanity checks.
        df_values = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [2.0, 2.0]},
            index=["x", "y"],
        )
        df_sigmas = pd.DataFrame(
            {"a": [0.0, 0.0], "b": [0.0, 0.0], "c": [0.0, 0.0]},
            index=["x", "y"],
        )

        formulas = [
            ("ratio", lambda df: (df["a"] + df["b"]) / df["c"]),
            ("sum", lambda df: df["a"] + df["b"]),
        ]

        means, sigmas = propagate_mc(
            df_values, df_sigmas, formulas, n_draws=10, seed=123
        )

        expected_ratio = (df_values["a"] + df_values["b"]) / df_values["c"]
        expected_sum = df_values["a"] + df_values["b"]

        assert means["ratio"].equals(expected_ratio)
        assert means["sum"].equals(expected_sum)
        assert (sigmas == 0.0).all().all()
        return(means, sigmas)

    def _test_shape_mismatch_raises() -> None:
        df_values = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [2.0, 2.0]},
            index=["x", "y"],
        )
        df_sigmas = pd.DataFrame(
            {"a": [0.0, 0.0], "b": [0.0, 0.0]},
            index=["x", "y"],
        )
        formulas = [("sum", lambda df: df["a"] + df["b"])]

        try:
            propagate_mc(df_values, df_sigmas, formulas)
        except ValueError as exc:
            assert "Shape mismatch" in str(exc)
        else:
            raise AssertionError("Expected shape mismatch error")

    def _bench_perf() -> None:
        # Performance bench (adjust sizes or limits for your machine if needed).
        perf_rows = 60
        perf_cols = 25
        perf_draws = 500
        perf_batch = 10
        perf_limit_seconds = 30.0

        perf_values = pd.DataFrame(
            np.ones((perf_rows, perf_cols), dtype=float),
            columns=[f"c{i}" for i in range(perf_cols)],
        )
        perf_sigmas = pd.DataFrame(
            0.01 * np.ones((perf_rows, perf_cols), dtype=float),
            columns=perf_values.columns,
            index=perf_values.index,
        )

        perf_formulas = [
            ("sum", lambda df: df.sum(axis=1)),
            ("ratio", lambda df: (df["c0"] + df["c1"]) / df["c2"]),
        ]

        start = time.perf_counter()
        mean_df,format_df = propagate_mc(
            perf_values,
            perf_sigmas,
            perf_formulas,
            n_draws=perf_draws,
            seed=42,
            batch_size=perf_batch,
        )
        elapsed = time.perf_counter() - start

        if elapsed > perf_limit_seconds:
            raise RuntimeError(
                f"Perf check failed: {elapsed:.2f}s > {perf_limit_seconds:.2f}s"
            )

        print(f"perf ok: {elapsed:.2f}s")

    def _perf_reels() -> pd.DataFrame :
        """Test with real data"""
        df = lvm_to_df(r'C:\Users\yberton\OneDrive - INSA Lyon\Expérimental\Acquisition\Etalonnage\E_in_imm\17_03_2025.lvm')
        udf = pd.DataFrame({"T" : [8.8E-2 for i in range(len(df))],
                            "P" : [6E-3 for i in range(len(df))]})
        df = pd.DataFrame({"T" : df['202 - E_in_imm [°C]'].values, "P" : df['118 - P_TS_in [bars]'].values})
        perf_formulas = [
            ("H" , lambda T,P : PropsSI('H','T',T,'P',P,'R245fa'))
        ]
        means, errors = propagate_mc(df, udf, perf_formulas)
    #means, sigmas = _test_basic_zero_sigma()
    #_test_shape_mismatch_raises()
    #_bench_perf()
    _perf_reels()
    print('Banana')