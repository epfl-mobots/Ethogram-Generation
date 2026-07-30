#!/usr/bin/env python
"""Interactive UMAP neighbor inspector.

Left-click a point on a project's UMAP density map to see the averaged raw sensor traces of its
nearest neighbors on the LEFT panel; right-click a (different, possibly far away) point to show
its own averaged traces on the RIGHT panel -- so you can visually compare two distinct areas of
the embedding side by side. Neighbors are nearest in UMAP space, pooled across every dataset in
the selected project (matching how the density map/watershed itself was built).

The raw-sensor window pulled around each neighbor isn't a fixed guess: it's derived from that
neighbor's own wavelet amplitude spectrum (amplitude-weighted average period, summed across PCA
modes) -- i.e. the timescale the pipeline's own wavelet transform says characterizes that point.
The average of that window across the k neighbors is used for all of them, so they stay aligned
on one shared relative-time axis. This requires findEmbeddings(..., saveWaveletAmps=True) to have
been (re-)run for the datasets involved; neighbors from datasets without that field fall back to
--window-minutes.

Usage:
    python umap_neighbor_inspector.py [--project PATH] [--k 20] [--window-minutes 60] [--intermediate-dir PATH]

Run inside the motionmapper37 conda env. If --project is omitted, you'll be prompted to pick
one from the projects discovered under Results/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from umap_interp_utils import (
    average_neighbor_traces,
    derive_neighbor_windows,
    discover_projects,
    find_nearest_neighbors,
    load_project_index,
)
from umap_video_utils import render_umap_background


def pick_project(explicit_path):
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_dir():
            print(f"Project directory not found: {path}")
            sys.exit(1)
        return path

    projects = discover_projects("Results")
    if not projects:
        print("No projects found under Results/ (each needs UMAP/zVals_wShed_groups.mat + Projections/).")
        sys.exit(1)

    print("Available projects:")
    for i, p in enumerate(projects):
        print(f"  [{i}] {p}")
    while True:
        choice = input(f"Select a project [0-{len(projects) - 1}]: ").strip()
        try:
            idx = int(choice)
            if 0 <= idx < len(projects):
                return projects[idx]
        except ValueError:
            pass
        print("Invalid selection, try again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=str, default=None,
                         help="Project directory (contains UMAP/ and Projections/). Prompts interactively if omitted.")
    parser.add_argument("--k", type=int, default=20,
                         help="Initial number of nearest neighbors to average (default: 20). "
                              "Adjustable live from a slider in the app.")
    parser.add_argument("--k-max", type=int, default=200,
                         help="Upper bound of the in-app k slider (default: 200).")
    parser.add_argument("--window-minutes", type=int, default=60,
                         help="Fallback half-width, in minutes, used only for neighbors whose dataset "
                              "doesn't have wavelet amplitudes saved yet. Otherwise the window is derived "
                              "per-neighbor from its own wavelet amplitude-weighted average period, and "
                              "the average of those across the k neighbors is used for all of them "
                              "(default fallback: 60).")
    parser.add_argument("--intermediate-dir", type=str, default="Results/Intermediate_Results",
                         help="Directory holding per-dataset raw sensor pickles (default: Results/Intermediate_Results).")
    args = parser.parse_args()

    project_dir = pick_project(args.project)
    print(f"Loading project index for {project_dir} ...")
    index = load_project_index(project_dir)
    print(f"Loaded {index.pooled_xy.shape[0]} pooled points across {len(index.dataset_status)} datasets.")
    for source_id, status in index.dataset_status.items():
        if status != "ok":
            print(f"  WARNING: {source_id}: {status}")

    # Layout: [left traces column] [density map] [right traces column], with a k-slider in a
    # thin row below the map. Left-click updates the left column, right-click updates the right
    # one, so two distinct (possibly far apart) areas of the embedding can be compared side by
    # side without one click overwriting the other.
    fig = plt.figure(figsize=(19, 7.5))
    outer_gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.4, 1], height_ratios=[20, 1], wspace=0.35, hspace=0.35)
    traces_slot_left = outer_gs[0, 0]
    ax_map = fig.add_subplot(outer_gs[0, 1])
    traces_slot_right = outer_gs[0, 2]
    ax_slider = fig.add_subplot(outer_gs[1, 1])

    current_k = args.k
    last_click = {"left": None, "right": None}

    render_umap_background(ax_map, index.density, extent=index.extent, wbounds=index.wbounds)
    ax_map.scatter(index.pooled_xy[:, 0], index.pooled_xy[:, 1], s=1, c="white", alpha=0.08, linewidths=0)
    ax_map.set_title(f"{project_dir.name} ({index.pooled_xy.shape[0]} points, {len(index.dataset_status)} datasets)\n"
                      f"left-click -> left panel, right-click -> right panel")
    ax_map.set_xlabel("UMAP 1")
    ax_map.set_ylabel("UMAP 2")

    k_slider = Slider(ax_slider, "k (neighbors)", valmin=1, valmax=max(args.k_max, args.k), valinit=args.k, valstep=1)

    highlight_left = ax_map.scatter([], [], s=40, facecolors="none", edgecolors="red", linewidths=1.4,
                                     zorder=5, label="left-click")
    highlight_right = ax_map.scatter([], [], s=40, facecolors="none", edgecolors="dodgerblue", linewidths=1.4,
                                      zorder=5, label="right-click")
    ax_map.legend(loc="upper right", fontsize=8)

    sides = {}

    def init_side(name, slot, color, placeholder_text):
        axes = []
        ax = fig.add_subplot(slot)
        ax.set_title(placeholder_text, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
        sides[name] = {"slot": slot, "axes": axes, "color": color}

    init_side("left", traces_slot_left, "tab:red", "Left-click a point to show its neighbor traces here")
    init_side("right", traces_slot_right, "tab:blue", "Right-click a (different) point to show its traces here")

    def rebuild_trace_axes(name, n):
        state = sides[name]
        for ax in state["axes"]:
            ax.remove()
        state["axes"] = []
        if n == 0:
            ax = fig.add_subplot(state["slot"])
            ax.set_title("No raw sensor data available for these neighbors", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            state["axes"].append(ax)
            return state["axes"]
        inner_gs = state["slot"].subgridspec(n, 1, hspace=0.6)
        for i in range(n):
            state["axes"].append(fig.add_subplot(inner_gs[i]))
        return state["axes"]

    def update_side(name, click_xy, highlight, label):
        neighbors = find_nearest_neighbors(index, click_xy, k=current_k)
        highlight.set_offsets(neighbors[["umap_x", "umap_y"]].to_numpy())

        window_minutes, window_detail = derive_neighbor_windows(
            neighbors, project_dir, default_window_minutes=args.window_minutes
        )
        averaged, report = average_neighbor_traces(
            neighbors, window_minutes=window_minutes, intermediate_dir=args.intermediate_dir
        )

        window_hours = window_minutes / 60.0
        n_included = len(report["included"])
        n_wavelet = int((window_detail["source"] == "wavelet").sum()) if not window_detail.empty else 0
        print(f"\n[{label}] Clicked ({click_xy[0]:.2f}, {click_xy[1]:.2f}): {n_included}/{len(neighbors)} "
              f"neighbors used. Window = {window_hours:.2f} h ({n_wavelet}/{len(neighbors)} neighbors had a "
              f"wavelet-derived window; averaged).")
        if report["skipped"]:
            print("  Skipped:")
            for source_id, reason in report["skipped"]:
                print(f"    {source_id}: {reason}")

        axes = rebuild_trace_axes(name, 0 if averaged.empty else len(averaged.columns))
        if not averaged.empty:
            relative_hours = averaged.index.total_seconds() / 3600.0
            for ax, col in zip(axes, averaged.columns):
                ax.plot(relative_hours, averaged[col], color=sides[name]["color"])
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_ylabel(col, fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.3)
            axes[-1].set_xlabel("Hours relative to each neighbor's own timestamp", fontsize=8)
            axes[0].set_title(f"[{label}] {n_included} neighbor(s), window ±{window_hours:.2f} h", fontsize=9)

    def on_click(event):
        if event.inaxes is not ax_map:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # a pan/zoom tool is active; don't treat this as a data click

        click_xy = (event.xdata, event.ydata)
        if event.button == 1:
            last_click["left"] = click_xy
            update_side("left", click_xy, highlight_left, "left-click")
        elif event.button == 3:
            last_click["right"] = click_xy
            update_side("right", click_xy, highlight_right, "right-click")
        else:
            return

        # The trace panels are (re)created here, after the one-time layout pass at startup, so
        # their y-axis labels are never accounted for by it -- without redoing the layout now,
        # those labels can render past the edge of their column and overlap ax_map.
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_k_changed(val):
        nonlocal current_k
        current_k = int(val)
        # Re-run whichever side(s) already have a point selected, so the change is immediately
        # visible instead of only taking effect on the next click.
        if last_click["left"] is not None:
            update_side("left", last_click["left"], highlight_left, "left-click")
        if last_click["right"] is not None:
            update_side("right", last_click["right"], highlight_right, "right-click")
        fig.tight_layout()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    k_slider.on_changed(on_k_changed)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
