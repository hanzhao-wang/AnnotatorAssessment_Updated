from __future__ import annotations

import os
import random
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import click
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from matplotlib.ticker import MaxNLocator

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = REPO_ROOT / "paper_experiment_configs"

DEFAULT_DATASETS = ["PKU", "sky", "Helpsteer", "Ultra"]
DEFAULT_DELTA_VALUES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
DEFAULT_UTILITY_PROFILES = ["risk_averse_poly", "high_cost_exp"]
DEFAULT_N_LIST = np.arange(1, 202, 10).tolist()
DEFAULT_PKU_N_LIST = np.arange(1, 202, 10).tolist()
PUBLIC_HUB_DATASETS = {
    "PKU": "RLHFlow/PKU-SafeRLHF-30K-standard",
    "Helpsteer": "RLHFlow/Helpsteer-preference-standard",
    "Ultra": "RLHFlow/UltraFeedback-preference-standard",
    "sky": "BigCatc/Skywork-Reward-Preference-80K-v0.2-ordinal",
}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calibrate_data(train_dataset, eval_dataset, n_bins: int = 20):
    from calibration_module.calibrator import HistogramCalibrator

    preference_score = np.array(train_dataset["preference_score"])
    labels = np.array(train_dataset["label"])
    histogram = HistogramCalibrator(n_bins=n_bins)
    histogram.fit(preference_score, labels)
    histogram_probs = histogram.predict(np.array(eval_dataset["preference_score"]))
    return histogram_probs


@dataclass(frozen=True)
class UtilityProfile:
    name: str
    g_exp_terms: Tuple[Tuple[float, float], ...]
    E_func: Callable[[np.ndarray], np.ndarray]
    mu_func: Callable[[np.ndarray], np.ndarray]


def build_G_func(g_exp_terms: Sequence[Tuple[float, float]]) -> Callable[[np.ndarray], np.ndarray]:
    def _g_func(x):
        x_arr = np.asarray(x)
        out_dtype = np.complex128 if np.iscomplexobj(x_arr) else np.float64
        out = np.zeros_like(x_arr, dtype=out_dtype)
        for alpha, beta in g_exp_terms:
            out = out + alpha * (1.0 - np.exp(-beta * x_arr))
        return out

    return _g_func


def get_utility_profiles() -> Dict[str, UtilityProfile]:
    return {
        "baseline": UtilityProfile(
            name="baseline",
            g_exp_terms=((1.0, 1.0),),
            E_func=lambda eta: 0.02 * np.asarray(eta) + 0.18 * np.asarray(eta) ** 2,
            mu_func=lambda eta: 0.50 * (1.0 - np.exp(-1.2 * np.asarray(eta))),
        ),
        "risk_averse_poly": UtilityProfile(
            name="risk_averse_poly",
            g_exp_terms=((0.7, 0.7), (0.3, 2.0)),
            E_func=lambda eta: (
                0.24
                * (
                    0.03 * np.asarray(eta)
                    + 0.10 * np.asarray(eta) ** 2
                    + 0.12 * np.asarray(eta) ** 3
                )
            ),
            mu_func=lambda eta: 0.35 * np.log(1.0 + 1.8 * np.asarray(eta)),
        ),
        "high_cost_exp": UtilityProfile(
            name="high_cost_exp",
            g_exp_terms=((1.0, 1.0),),
            E_func=lambda eta: 0.22 * (0.03 * np.asarray(eta) + 0.10 * np.asarray(eta) ** 2 + 0.12 * np.asarray(eta) ** 3),
            mu_func=lambda eta: 0.20 * np.log(1.0 + 1.8 * np.asarray(eta)),
        ),
    }


def _complex_step_derivatives(
    func: Callable[[np.ndarray], np.ndarray], x_grid: np.ndarray, h: float = 1e-2
) -> Tuple[np.ndarray, np.ndarray]:
    x_real = np.asarray(x_grid, dtype=np.float64)
    f_real = np.real(np.asarray(func(x_real), dtype=np.complex128))
    f_complex = np.asarray(func(x_real.astype(np.complex128) + 1j * h), dtype=np.complex128)
    first_derivative = np.imag(f_complex) / h
    second_derivative = -2.0 * (np.real(f_complex) - f_real) / (h**2)
    return first_derivative, second_derivative


def validate_utility_profile_assumptions(profile: UtilityProfile) -> None:
    g_func = build_G_func(profile.g_exp_terms)
    payment_grid = np.linspace(-10.0, 20.0, 3001)
    eta_grid = np.linspace(0.0, 1.0, 2001)

    g_first, g_second = _complex_step_derivatives(g_func, payment_grid)
    e_first, e_second = _complex_step_derivatives(profile.E_func, eta_grid)
    mu_first, mu_second = _complex_step_derivatives(profile.mu_func, eta_grid)

    strict_tol = 1e-15
    weak_tol = 1e-10

    if float(np.min(g_first)) <= strict_tol:
        raise ValueError(
            f"Profile '{profile.name}' violates dG/dw > 0 on payment grid (min={np.min(g_first):.4e})."
        )
    if float(np.max(g_second)) > weak_tol:
        raise ValueError(
            f"Profile '{profile.name}' violates d2G/dw2 < 0 on payment grid (max={np.max(g_second):.4e})."
        )
    if float(np.min(e_first)) <= strict_tol:
        raise ValueError(
            f"Profile '{profile.name}' violates dE/deta > 0 on [0,1] (min={np.min(e_first):.4e})."
        )
    if float(np.min(e_second)) < -weak_tol:
        raise ValueError(
            f"Profile '{profile.name}' violates d2E/deta2 >= 0 on [0,1] (min={np.min(e_second):.4e})."
        )
    if float(np.min(mu_first)) < -weak_tol:
        raise ValueError(
            f"Profile '{profile.name}' violates dmu/deta >= 0 on [0,1] (min={np.min(mu_first):.4e})."
        )
    if float(np.max(mu_second)) > weak_tol:
        raise ValueError(
            f"Profile '{profile.name}' violates d2mu/deta2 <= 0 on [0,1] (max={np.max(mu_second):.4e})."
        )


def parse_float_list(value: str) -> List[float]:
    if not value.strip():
        return []
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> List[int]:
    if not value.strip():
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_string_list(value: str) -> List[str]:
    if not value.strip():
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def safe_relative_gap(fb_util: float, other_util: float) -> float:
    if np.isclose(fb_util, 0.0):
        return 0.0
    gap = (fb_util - other_util) / fb_util
    return float(np.clip(gap, 0.0, 1.0))


def resolve_n_list(dataset_name: str, n_list_override: Optional[List[int]]) -> List[int]:
    if n_list_override:
        return n_list_override
    if dataset_name == "PKU":
        return DEFAULT_PKU_N_LIST
    return DEFAULT_N_LIST


def _build_train_eval_from_public_hub(dataset_name: str, seed: int = 42):
    from datasets import load_dataset

    if dataset_name not in PUBLIC_HUB_DATASETS:
        raise ValueError(
            f"No public dataset mapping found for '{dataset_name}'. Available: {list(PUBLIC_HUB_DATASETS.keys())}"
        )

    hub_name = PUBLIC_HUB_DATASETS[dataset_name]
    ds = load_dataset(hub_name, split="train")
    ds = ds.shuffle(seed=seed)

    chosen_scores = np.asarray(ds["chosen_score"], dtype=np.float64)
    rejected_scores = np.asarray(ds["rejected_score"], dtype=np.float64)
    score_gap = chosen_scores - rejected_scores
    preference_scores = 1.0 / (1.0 + np.exp(-score_gap))

    split_idx = len(preference_scores) // 2
    eval_pref = preference_scores[:split_idx]
    train_pref = preference_scores[split_idx:]

    rng = np.random.default_rng(seed)
    train_flips = rng.binomial(1, 0.5, size=train_pref.shape[0]).astype(np.int64)
    eval_flips = rng.binomial(1, 0.5, size=eval_pref.shape[0]).astype(np.int64)

    train_dataset = {
        "label": (1 - train_flips).tolist(),
        "preference_score": np.abs(train_flips - train_pref).tolist(),
    }
    eval_dataset = {
        "label": (1 - eval_flips).tolist(),
        "preference_score": np.abs(eval_flips - eval_pref).tolist(),
    }
    return train_dataset, eval_dataset


@lru_cache(maxsize=None)
def load_preference_scores(dataset_name: str) -> np.ndarray:
    config_path = CONFIG_ROOT / f"llama-{dataset_name}.json"
    try:
        from utils.data import get_data

        train_dataset, eval_dataset = get_data(script_config_path=str(config_path))
    except Exception as exc:
        print(
            f"[Section4] Falling back to public scored dataset for '{dataset_name}' because config-driven data loading failed: {exc}"
        )
        train_dataset, eval_dataset = _build_train_eval_from_public_hub(dataset_name, seed=42)

    if dataset_name in ["PKU", "sky"]:
        histogram_probs = calibrate_data(train_dataset, eval_dataset, n_bins=30)
    else:
        histogram_probs = np.array(eval_dataset["preference_score"])
    return histogram_probs


def build_contract_space(contract_type: str):
    if contract_type == "linear":
        c0 = np.arange(0, 1, 1).tolist()
        c1 = np.arange(0, 10, 0.05).tolist()
        c2 = np.arange(-10, 10, 0.05).tolist()
    elif contract_type == "binary":
        c0 = np.arange(0, 1.02, 0.02).tolist()
        c1 = np.arange(0, 10, 0.05).tolist()
        c2 = np.arange(-10, 10, 0.05).tolist()
    else:
        raise ValueError(f"Unknown contract type: {contract_type}")
    return [c0, c1, c2]


class solver:
    def __init__(
        self,
        preference_scores,
        effort_space,
        contract_space,
        G_func,
        E_func,
        mu_func,
        n,
        U_0=0,
        delta=0,
        contract_type="linear",
        monitor_type="self",
        simulation_num=5000,
        g_exp_terms: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        self.original_preference_scores = np.array(preference_scores)
        self.preference_scores = self.original_preference_scores
        self.preference_scores[self.original_preference_scores < 0.5] = 1 - self.original_preference_scores[
            self.original_preference_scores < 0.5
        ]
        self.original_preference_scores = self.preference_scores
        if not np.all(self.original_preference_scores >= 0.5):
            raise ValueError("preference_scores must be >=0.5")

        self.effort_space = effort_space
        self.contract_space = contract_space
        self.G_func = G_func
        self.E_func = E_func
        self.mu_func = mu_func
        self.contract_type = contract_type
        self.monitor_type = monitor_type
        self.simulation_num = simulation_num
        self.delta = delta
        self.n = n
        self.U_0 = U_0
        self.g_exp_terms = tuple(g_exp_terms) if g_exp_terms is not None else None

    def Util_compute_exact(self):
        if self.g_exp_terms is None:
            raise ValueError("g_exp_terms must be provided for exact utility computation.")

        c0_values, c1_values, c2_values = self.contract_space
        E = len(self.effort_space)
        C0 = len(c0_values)
        C1 = len(c1_values)
        C2 = len(c2_values)

        c0_grid, c1_grid, c2_grid = np.meshgrid(c0_values, c1_values, c2_values, indexing="ij")
        c0_grid_4d = c0_grid[np.newaxis, ...]
        c1_grid_4d = c1_grid[np.newaxis, ...]
        c2_grid_4d = c2_grid[np.newaxis, ...]

        efforts = np.array(self.effort_space)
        if self.monitor_type == "self":
            mean_values = (1 + efforts * (1 - self.delta)) / 2
        elif self.monitor_type == "expert":
            mean_values = efforts * np.mean(self.original_preference_scores - 0.5) + 0.5
        else:
            raise ValueError(f"Unknown monitor type: {self.monitor_type}")

        if self.contract_type == "linear":
            mean_values_4d = mean_values[:, np.newaxis, np.newaxis, np.newaxis]
            payments = mean_values_4d * c1_grid_4d + c2_grid_4d
            G_payments = np.zeros((E, C0, C1, C2), dtype=np.float64)
            for alpha, beta in self.g_exp_terms:
                mgf_component = np.exp(-beta * c2_grid_4d) * (
                    1 - mean_values_4d + mean_values_4d * np.exp(-beta * c1_grid_4d / self.n)
                ) ** self.n
                G_payments = G_payments + alpha * (1.0 - mgf_component)
        elif self.contract_type == "binary":
            survival_2d = binom.sf(np.array(c0_values)[np.newaxis, :] * self.n, self.n, mean_values[:, np.newaxis])
            probs = survival_2d[:, :, np.newaxis, np.newaxis]
            probs = np.broadcast_to(probs, (E, C0, C1, C2))
            payments = probs * c1_grid_4d + c2_grid_4d
            G_payments = probs * self.G_func(c1_grid_4d + c2_grid_4d) + (1 - probs) * self.G_func(c2_grid_4d)
        else:
            raise ValueError(f"Unknown contract type: {self.contract_type}")

        cost_efforts = np.array([self.E_func(e) for e in efforts])
        cost_efforts_4d = cost_efforts[:, np.newaxis, np.newaxis, np.newaxis]

        agent_utility = G_payments - cost_efforts_4d

        mu_efforts = np.array([self.mu_func(e) for e in efforts])
        mu_efforts_4d = mu_efforts[:, np.newaxis, np.newaxis, np.newaxis]

        principal_utility = -payments + mu_efforts_4d

        self.agent_utility = agent_utility
        self.principal_utility = principal_utility
        return agent_utility, principal_utility

    def FB_solve(self):
        agent_utility, principal_utility = self.agent_utility, self.principal_utility
        above_reservation = agent_utility >= self.U_0
        feasible_mask = above_reservation
        if not np.any(feasible_mask):
            return (None, 0, 0, 0), self.mu_func(0), 0, self.U_0

        feasible_principal_util = principal_utility[feasible_mask]
        best_index_in_feasible = np.argmax(feasible_principal_util)
        feasible_indices = np.where(feasible_mask)
        best_e_idx = feasible_indices[0][best_index_in_feasible]
        best_c0_idx = feasible_indices[1][best_index_in_feasible]
        best_c1_idx = feasible_indices[2][best_index_in_feasible]
        best_c2_idx = feasible_indices[3][best_index_in_feasible]
        c0_values, c1_values, c2_values = self.contract_space
        best_c0 = c0_values[best_c0_idx]
        best_c1 = c1_values[best_c1_idx]
        best_c2 = c2_values[best_c2_idx]
        return (
            (best_c0, best_c1, best_c2),
            principal_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx],
            self.effort_space[best_e_idx],
            agent_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx],
        )

    def SB_solve(self):
        agent_utility, principal_utility = self.agent_utility, self.principal_utility
        max_util_over_e = agent_utility.max(axis=0)
        is_best_response = agent_utility + 0.01 >= max_util_over_e[np.newaxis, ...]
        above_reservation = agent_utility >= self.U_0
        feasible_mask = is_best_response & above_reservation

        if not np.any(feasible_mask):
            return (None, 0, 0, 0), self.mu_func(0), 0, self.U_0

        feasible_principal_util = principal_utility[feasible_mask]
        best_index_in_feasible = np.argmax(feasible_principal_util)
        feasible_indices = np.where(feasible_mask)
        best_e_idx = feasible_indices[0][best_index_in_feasible]
        best_c0_idx = feasible_indices[1][best_index_in_feasible]
        best_c1_idx = feasible_indices[2][best_index_in_feasible]
        best_c2_idx = feasible_indices[3][best_index_in_feasible]
        c0_values, c1_values, c2_values = self.contract_space
        best_c0 = c0_values[best_c0_idx]
        best_c1 = c1_values[best_c1_idx]
        best_c2 = c2_values[best_c2_idx]
        return (
            (best_c0, best_c1, best_c2),
            principal_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx],
            self.effort_space[best_e_idx],
            agent_utility[best_e_idx, best_c0_idx, best_c1_idx, best_c2_idx],
        )

    def SB_solve_tilde(self, e_star):
        agent_utility, principal_utility = self.agent_utility, self.principal_utility
        try:
            e_star_index = self.effort_space.index(e_star)
        except ValueError:
            return (0, 0, 0), self.mu_func(0), 0, self.U_0

        e_star_index = int(e_star_index)
        max_util_over_efforts = agent_utility.max(axis=0)
        part_max = np.max(agent_utility[e_star_index - 1 : max(e_star_index + 2, len(agent_utility))], axis=0)
        is_e_star_best = part_max >= max_util_over_efforts
        above_reservation = agent_utility[e_star_index] >= self.U_0
        feasible_mask = is_e_star_best & above_reservation

        if not np.any(feasible_mask):
            return (0, 0, 0), self.mu_func(0), 0, self.U_0

        principal_slice = principal_utility[e_star_index]
        feasible_princ_util = principal_slice[feasible_mask]
        best_idx_in_feasible = np.argmax(feasible_princ_util)
        feasible_indices = np.where(feasible_mask)
        best_c0_idx = feasible_indices[0][best_idx_in_feasible]
        best_c1_idx = feasible_indices[1][best_idx_in_feasible]
        best_c2_idx = feasible_indices[2][best_idx_in_feasible]
        c0_values, c1_values, c2_values = self.contract_space
        best_c0 = c0_values[best_c0_idx]
        best_c1 = c1_values[best_c1_idx]
        best_c2 = c2_values[best_c2_idx]

        return (
            (best_c0, best_c1, best_c2),
            principal_utility[e_star_index][best_c0_idx, best_c1_idx, best_c2_idx],
            e_star,
            agent_utility[e_star_index][best_c0_idx, best_c1_idx, best_c2_idx],
        )


def plot_func(monitor_type_list, contract_type_list, n_list, results, name, U_0, delta, plot_mode="gap", save_name=""):
    plt.figure(figsize=(10, 10))
    axis_label_fontsize = 38
    tick_fontsize = 30
    legend_fontsize = 24
    legend_title_fontsize = 24
    marker_monitor = {"self": "o", "expert": "s"}
    line_style = {"linear": "-", "binary": "--"}

    for monitor_type in monitor_type_list:
        for contract_type in contract_type_list:
            FB_utilities = []
            SB_utilities = []
            FT_effort = []
            SB_effort = []
            SB_agent_utilities = []
            SB_tilde_utilities = []
            SB_tilde_agent_utilities = []
            SB_tilde_effort = []

            for n in n_list:
                FB_utilities.append(results[(name, monitor_type, contract_type, n, "FB")]["best_principal_util"])
                SB_utilities.append(results[(name, monitor_type, contract_type, n, "SB")]["best_principal_util"])
                FT_effort.append(results[(name, monitor_type, contract_type, n, "FB")]["best_effort"])
                SB_effort.append(results[(name, monitor_type, contract_type, n, "SB")]["best_effort"])
                SB_agent_utilities.append(results[(name, monitor_type, contract_type, n, "SB")]["agent_util"])

                SB_tilde_utilities.append(results[(name, monitor_type, contract_type, n, "SB_tilde")]["best_principal_util"])
                SB_tilde_agent_utilities.append(
                    results[(name, monitor_type, contract_type, n, "SB_tilde")]["agent_util"]
                )
                SB_tilde_effort.append(results[(name, monitor_type, contract_type, n, "SB_tilde")]["best_effort"])

            if plot_mode == "gap":
                fb_arr = np.array(FB_utilities, dtype=np.float64)
                sb_arr = np.array(SB_utilities, dtype=np.float64)
                sb_tilde_arr = np.array(SB_tilde_utilities, dtype=np.float64)

                y_sb = np.divide(fb_arr - sb_arr, fb_arr, out=np.zeros_like(fb_arr), where=np.abs(fb_arr) > 1e-12)
                y_sb = np.clip(y_sb, 0, 1)

                y_sb_tilde = np.divide(
                    fb_arr - sb_tilde_arr, fb_arr, out=np.zeros_like(fb_arr), where=np.abs(fb_arr) > 1e-12
                )
                y_sb_tilde = np.clip(y_sb_tilde, 0, 1)

                plt.plot(
                    n_list,
                    y_sb,
                    marker=marker_monitor[monitor_type],
                    linestyle=line_style[contract_type],
                    markersize=10,
                    linewidth=4,
                    color="blue",
                )
                plt.plot(
                    n_list,
                    y_sb_tilde,
                    marker=marker_monitor[monitor_type],
                    linestyle=line_style[contract_type],
                    markersize=10,
                    linewidth=4,
                    color="red",
                )
            elif plot_mode == "effort":
                plt.plot(
                    n_list,
                    np.array(SB_effort),
                    marker=marker_monitor[monitor_type],
                    linestyle=line_style[contract_type],
                    markersize=10,
                    linewidth=4,
                    color="blue",
                )
                plt.plot(
                    n_list,
                    np.array(SB_tilde_effort),
                    marker=marker_monitor[monitor_type],
                    linestyle=line_style[contract_type],
                    markersize=10,
                    linewidth=4,
                    color="red",
                )
            elif plot_mode == "agent_util":
                plt.plot(
                    n_list,
                    np.array(SB_agent_utilities),
                    marker=marker_monitor[monitor_type],
                    linestyle=line_style[contract_type],
                    markersize=10,
                    linewidth=4,
                    color="blue",
                )
                plt.plot(
                    n_list,
                    np.array(SB_tilde_agent_utilities),
                    marker=marker_monitor[monitor_type],
                    linestyle=line_style[contract_type],
                    markersize=10,
                    linewidth=4,
                    color="red",
                )

    plt.xlabel("n", fontsize=axis_label_fontsize)
    if plot_mode == "gap":
        plt.ylabel("Utility Gap", fontsize=axis_label_fontsize)
        plt.yscale("linear")
        plt.ylim(0, 1.1)
    elif plot_mode == "effort":
        plt.axhline(y=np.max(FT_effort), color="black", linestyle="-.", linewidth=4, label="FB Effort")
        plt.ylabel("Agent Effort", fontsize=axis_label_fontsize)
        plt.ylim(-0.1, 1.1)
    elif plot_mode == "agent_util":
        plt.axhline(y=U_0, color="black", linestyle="-.", linewidth=4, label="$U_0$")
        plt.ylabel("Agent Utility", fontsize=axis_label_fontsize)
        plt.ylim(-0.1, 0.5)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    monitor_type_handle = {"self": "Self", "expert": "Expert"}
    monitor_legend_handles = []
    for monitor in monitor_type_list:
        line = mlines.Line2D(
            [],
            [],
            color="blue",
            marker=marker_monitor[monitor],
            linestyle="-",
            label=monitor_type_handle[monitor],
            markersize=16,
            linewidth=4,
        )
        monitor_legend_handles.append(line)

    legend_monitor = plt.legend(
        handles=monitor_legend_handles,
        title="Monitor",
        loc="upper left",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )

    contract_type_handle = {"linear": "Linear", "binary": "Binary"}
    contract_type_legend_handles = []
    for contract in contract_type_list:
        line = mlines.Line2D(
            [],
            [],
            color="blue",
            linestyle=line_style[contract],
            label=contract_type_handle[contract],
            markersize=10,
            linewidth=4,
        )
        contract_type_legend_handles.append(line)

    legend_contract_type = plt.legend(
        handles=contract_type_legend_handles,
        title="Contract",
        loc="upper right",
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )

    order_type_handle = {"SB": "$\\mathcal{C}_n$", "SB_tilde": r"$\tilde{\mathcal{C}}_n$"}
    colors = {"SB": "blue", "SB_tilde": "red"}
    order_type_legend_handles = []
    for order_type in ["SB", "SB_tilde"]:
        line = mlines.Line2D(
            [],
            [],
            color=colors[order_type],
            linestyle="-",
            label=order_type_handle[order_type],
            markersize=10,
            linewidth=4,
            markerfacecolor="white",
        )
        order_type_legend_handles.append(line)

    legend_order_type = plt.legend(handles=order_type_legend_handles, loc="center right", fontsize=legend_fontsize)
    plt.gca().add_artist(legend_order_type)
    plt.gca().add_artist(legend_monitor)
    plt.gca().add_artist(legend_contract_type)

    plt.grid()
    os.makedirs(f"./fig_contract/{plot_mode}", exist_ok=True)
    if save_name == "":
        plt.savefig(f"./fig_contract/{plot_mode}/{name}" + "delta_" + str(delta) + "U0_" + str(U_0) + ".eps", bbox_inches="tight")
    else:
        plt.savefig(f"./fig_contract/{plot_mode}/{name}" + save_name + ".eps", bbox_inches="tight")
    plt.close()


def _write_summary_csv(summary_rows: List[Dict[str, float]], csv_path: Path) -> None:
    fieldnames = list(summary_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _aggregate_summary_rows(summary_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, str, str, float], Dict[str, float]] = {}
    for row in summary_rows:
        key = (
            str(row["dataset"]),
            str(row["utility_profile"]),
            str(row["monitor_type"]),
            str(row["contract_type"]),
            float(row["delta"]),
        )
        if key not in grouped:
            grouped[key] = {"sum_gap_sb": 0.0, "sum_gap_sb_tilde": 0.0, "count": 0}
        grouped[key]["sum_gap_sb"] += float(row["gap_sb"])
        grouped[key]["sum_gap_sb_tilde"] += float(row["gap_sb_tilde"])
        grouped[key]["count"] += 1

    aggregated_rows: List[Dict[str, float]] = []
    for (dataset, utility_profile, monitor_type, contract_type, delta), stats in grouped.items():
        aggregated_rows.append(
            {
                "dataset": dataset,
                "utility_profile": utility_profile,
                "monitor_type": monitor_type,
                "contract_type": contract_type,
                "delta": float(delta),
                "gap_sb": stats["sum_gap_sb"] / stats["count"],
                "gap_sb_tilde": stats["sum_gap_sb_tilde"] / stats["count"],
            }
        )
    aggregated_rows.sort(
        key=lambda x: (x["dataset"], x["utility_profile"], x["monitor_type"], x["contract_type"], x["delta"])
    )
    return aggregated_rows


def plot_sensitivity_summary(summary_rows: List[Dict[str, float]], save_tag: str = "") -> None:
    output_dir = Path("./fig_contract/sensitivity")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_csv_path = output_dir / "figure3_delta_utility_summary.csv"
    _write_summary_csv(summary_rows, base_csv_path)
    if save_tag:
        _write_summary_csv(summary_rows, output_dir / f"figure3_delta_utility_summary_{save_tag}.csv")

    agg_rows = _aggregate_summary_rows(summary_rows)
    utility_profiles = list(dict.fromkeys(str(row["utility_profile"]) for row in summary_rows))
    axis_label_fontsize = 34
    tick_fontsize = 24
    legend_fontsize = 22
    legend_title_fontsize = 22
    target_dataset = "PKU"
    if target_dataset not in {str(row["dataset"]) for row in agg_rows}:
        target_dataset = str(agg_rows[0]["dataset"])
    monitor_type = "self"
    contract_type_list = ["linear", "binary"]
    marker_monitor = {"self": "o", "expert": "s"}
    line_style = {"linear": "-", "binary": "--"}
    contract_type_handle = {"linear": "Linear", "binary": "Binary"}

    def _render_single_profile(profile_name: str, eps_path: Path, png_path: Path) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 7))
        for contract_type in contract_type_list:
            subset = [
                row
                for row in agg_rows
                if row["utility_profile"] == profile_name
                and row["dataset"] == target_dataset
                and row["monitor_type"] == monitor_type
                and row["contract_type"] == contract_type
            ]
            if not subset:
                continue
            subset.sort(key=lambda x: x["delta"])
            x_vals = [row["delta"] for row in subset]
            y_sb = [row["gap_sb"] for row in subset]
            y_sb_tilde = [row["gap_sb_tilde"] for row in subset]
            style = line_style[contract_type]
            marker = marker_monitor[monitor_type]
            ax.plot(x_vals, y_sb, marker=marker, linestyle=style, markersize=8, linewidth=3, color="blue")
            ax.plot(x_vals, y_sb_tilde, marker=marker, linestyle=style, markersize=8, linewidth=3, color="red")

        contract_type_legend_handles = []
        for contract in contract_type_list:
            line = mlines.Line2D(
                [],
                [],
                color="black",
                marker=marker_monitor[monitor_type],
                linestyle=line_style[contract],
                label=contract_type_handle[contract],
                markersize=10,
                linewidth=3,
            )
            contract_type_legend_handles.append(line)

        order_type_legend_handles = [
            mlines.Line2D([], [], color="blue", linestyle="-", label=r"$\mathcal{C}_n$", markersize=10, linewidth=3),
            mlines.Line2D([], [], color="red", linestyle="-", label=r"$\tilde{\mathcal{C}}_n$", markersize=10, linewidth=3),
        ]
        legend_contract = ax.legend(
            handles=contract_type_legend_handles,
            title="Contract",
            loc="upper left",
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
        )
        legend_order_type = ax.legend(
            handles=order_type_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=legend_fontsize,
        )
        ax.add_artist(legend_contract)
        ax.add_artist(legend_order_type)
        ax.set_xlabel(r"$\delta$", fontsize=axis_label_fontsize)
        ax.set_ylabel("Utility Gap", fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(eps_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

    first_profile = utility_profiles[0]
    base_eps_path = output_dir / "figure3_delta_utility_summary.eps"
    base_png_path = output_dir / "figure3_delta_utility_summary.png"
    _render_single_profile(first_profile, base_eps_path, base_png_path)
    if save_tag:
        _render_single_profile(
            first_profile,
            output_dir / f"figure3_delta_utility_summary_{save_tag}.eps",
            output_dir / f"figure3_delta_utility_summary_{save_tag}.png",
        )
    for profile_name in utility_profiles:
        suffix = f"figure3_delta_utility_summary_{profile_name}"
        if save_tag:
            suffix = f"{suffix}_{save_tag}"
        _render_single_profile(
            profile_name,
            output_dir / f"{suffix}.eps",
            output_dir / f"{suffix}.png",
        )


def run_single_setting(
    monitor_type_list: Sequence[str],
    contract_type_list: Sequence[str],
    dataset_names: Sequence[str],
    n_list_override: Optional[List[int]],
    U_0: float,
    delta: float,
    effort_space: Sequence[float],
    utility_profile: UtilityProfile,
    save_name: str = "",
) -> List[Dict[str, float]]:
    summary_rows: List[Dict[str, float]] = []

    g_func = build_G_func(utility_profile.g_exp_terms)
    e_func = utility_profile.E_func
    mu_func = utility_profile.mu_func

    for name in dataset_names:
        n_list = resolve_n_list(name, n_list_override)
        histogram_probs = load_preference_scores(name)
        results = {}

        for monitor_type in monitor_type_list:
            for contract_type in contract_type_list:
                contract_space = build_contract_space(contract_type)
                for n in n_list:
                    print("===============================================")
                    print(
                        f"Data: {name}, Profile: {utility_profile.name}, Delta: {delta}, "
                        f"Monitor: {monitor_type}, Contract: {contract_type}, n: {n}"
                    )

                    solver_instance = solver(
                        preference_scores=histogram_probs,
                        effort_space=effort_space,
                        contract_space=contract_space,
                        G_func=g_func,
                        E_func=e_func,
                        mu_func=mu_func,
                        n=n,
                        U_0=U_0,
                        delta=delta,
                        contract_type=contract_type,
                        monitor_type=monitor_type,
                        g_exp_terms=utility_profile.g_exp_terms,
                    )
                    solver_instance.Util_compute_exact()

                    (best_c0, best_c1, best_c2), fb_principal, fb_effort, fb_agent = solver_instance.FB_solve()
                    results[(name, monitor_type, contract_type, n, "FB")] = {
                        "best_c0": best_c0,
                        "best_c1": best_c1,
                        "best_c2": best_c2,
                        "best_principal_util": fb_principal,
                        "best_effort": fb_effort,
                        "agent_util": fb_agent,
                    }

                    e_star = fb_effort
                    (
                        (best_c0, best_c1, best_c2),
                        sb_tilde_principal,
                        sb_tilde_effort,
                        sb_tilde_agent,
                    ) = solver_instance.SB_solve_tilde(e_star)
                    results[(name, monitor_type, contract_type, n, "SB_tilde")] = {
                        "best_c0": best_c0,
                        "best_c1": best_c1,
                        "best_c2": best_c2,
                        "best_principal_util": sb_tilde_principal,
                        "best_effort": sb_tilde_effort,
                        "agent_util": sb_tilde_agent,
                    }

                    (best_c0, best_c1, best_c2), sb_principal, sb_effort, sb_agent = solver_instance.SB_solve()
                    results[(name, monitor_type, contract_type, n, "SB")] = {
                        "best_c0": best_c0,
                        "best_c1": best_c1,
                        "best_c2": best_c2,
                        "best_principal_util": sb_principal,
                        "best_effort": sb_effort,
                        "agent_util": sb_agent,
                    }

                    summary_rows.append(
                        {
                            "dataset": name,
                            "utility_profile": utility_profile.name,
                            "delta": float(delta),
                            "monitor_type": monitor_type,
                            "contract_type": contract_type,
                            "n": int(n),
                            "fb_principal_util": float(fb_principal),
                            "sb_principal_util": float(sb_principal),
                            "sb_tilde_principal_util": float(sb_tilde_principal),
                            "gap_sb": safe_relative_gap(float(fb_principal), float(sb_principal)),
                            "gap_sb_tilde": safe_relative_gap(float(fb_principal), float(sb_tilde_principal)),
                            "fb_effort": float(fb_effort),
                            "sb_effort": float(sb_effort),
                            "sb_tilde_effort": float(sb_tilde_effort),
                            "fb_agent_util": float(fb_agent),
                            "sb_agent_util": float(sb_agent),
                            "sb_tilde_agent_util": float(sb_tilde_agent),
                        }
                    )

        plot_func(monitor_type_list, contract_type_list, n_list, results, name, U_0, delta, plot_mode="gap", save_name=save_name)
        plot_func(
            monitor_type_list,
            contract_type_list,
            n_list,
            results,
            name,
            U_0,
            delta,
            plot_mode="effort",
            save_name=save_name,
        )
        plot_func(
            monitor_type_list,
            contract_type_list,
            n_list,
            results,
            name,
            U_0,
            delta,
            plot_mode="agent_util",
            save_name=save_name,
        )

    return summary_rows


@click.command()
@click.option(
    "--delta-values",
    default="0.0,0.01,0.02,0.05,0.1,0.2",
    show_default=True,
    help="Comma-separated disagreement probabilities under full commitment.",
)
@click.option(
    "--utility-profiles",
    default="risk_averse_poly,high_cost_exp",
    show_default=True,
    help="Comma-separated utility profile names.",
)
@click.option(
    "--datasets",
    default="PKU,sky,Helpsteer,Ultra",
    show_default=True,
    help="Comma-separated dataset names.",
)
@click.option(
    "--n-list",
    default="",
    show_default=False,
    help="Comma-separated n values. If empty, use dataset-specific defaults.",
)
@click.option("--save-tag", default="", show_default=False, help="Optional suffix tag for output files.")
@click.option("--seed", default=32, show_default=True, type=int, help="Random seed.")
def main(delta_values: str, utility_profiles: str, datasets: str, n_list: str, save_tag: str, seed: int):
    set_random_seed(seed)
    plt.rcParams["font.size"] = 32

    selected_delta_values = parse_float_list(delta_values)
    if not selected_delta_values:
        selected_delta_values = DEFAULT_DELTA_VALUES

    selected_datasets = parse_string_list(datasets)
    if not selected_datasets:
        selected_datasets = DEFAULT_DATASETS

    selected_n_list = parse_int_list(n_list) if n_list.strip() else None

    profiles_registry = get_utility_profiles()
    selected_profiles = parse_string_list(utility_profiles)
    if not selected_profiles:
        selected_profiles = DEFAULT_UTILITY_PROFILES

    invalid_profiles = [p for p in selected_profiles if p not in profiles_registry]
    if invalid_profiles:
        raise click.BadParameter(
            f"Unknown utility profile(s): {invalid_profiles}. Available: {list(profiles_registry.keys())}"
        )

    for profile_name in selected_profiles:
        validate_utility_profile_assumptions(profiles_registry[profile_name])

    monitor_type_list = ["expert", "self"]
    contract_type_list = ["linear", "binary"]
    effort_space = np.arange(0, 1.01, 0.01).tolist()
    U_0 = 0.0

    all_rows: List[Dict[str, float]] = []
    for profile_name in selected_profiles:
        utility_profile = profiles_registry[profile_name]
        for delta in selected_delta_values:
            run_name = f"profile_{profile_name}_U0_{U_0}_delta_{delta}"
            if save_tag:
                run_name = f"{run_name}_{save_tag}"

            rows = run_single_setting(
                monitor_type_list=monitor_type_list,
                contract_type_list=contract_type_list,
                dataset_names=selected_datasets,
                n_list_override=selected_n_list,
                U_0=U_0,
                delta=delta,
                effort_space=effort_space,
                utility_profile=utility_profile,
                save_name=run_name,
            )
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No experiment rows were generated; check dataset/profile arguments.")

    plot_sensitivity_summary(all_rows, save_tag=save_tag)
    print("Saved summary CSV: fig_contract/sensitivity/figure3_delta_utility_summary.csv")
    print("Saved summary figure: fig_contract/sensitivity/figure3_delta_utility_summary.eps")


if __name__ == "__main__":
    main()
