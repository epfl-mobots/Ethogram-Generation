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
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from umap_interp_utils import (
    average_neighbor_traces,
    derive_neighbor_windows,
    discover_projects,
    find_nearest_neighbors,
    load_project_index,
)
from umap_video_utils import render_umap_background


def build_plot_groups(columns, co2_mode):
    """Group an averaged-traces DataFrame's columns into an ordered list of (label, member_cols)
    plot groups. Every non-CO2 column gets its own single-member group, as before. CO2 columns
    (co2_UL/UR/LL/LR for WholeHive datasets, co2_L/co2_R otherwise) are collapsed into one
    "co2" group -- rendered as the across-sensor mean +/- std -- when co2_mode is "averaged" and
    there's more than one of them; otherwise each stays its own group, same as any other column.
    """
    cols = sorted(columns)
    co2_cols = [c for c in cols if c.lower().startswith("co2")]
    other_cols = [c for c in cols if c not in co2_cols]

    groups = [(c, [c]) for c in other_cols]
    if co2_mode == "averaged" and len(co2_cols) > 1:
        groups.append(("co2", co2_cols))
    else:
        groups.extend((c, [c]) for c in co2_cols)

    # Sort by each group's lowest member column name so the co2 group lands where its members
    # would have, keeping the overall panel order close to plain alphabetical.
    groups.sort(key=lambda item: min(item[1]))
    return groups


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
    parser.add_argument(
        "--figures-dir", type=str,
        default="/Users/cyrilmonette/Desktop/EPFL 2018-2026/PhD - Mobots/Publishing/Publications/Metabolism/figures",
        help="Directory the 'Save image' button writes into (default: %(default)s).",
    )
    args = parser.parse_args()

    project_dir = pick_project(args.project)
    figures_dir = Path(args.figures_dir)
    print(f"Loading project index for {project_dir} ...")
    index = load_project_index(project_dir)
    print(f"Loaded {index.pooled_xy.shape[0]} pooled points across {len(index.dataset_status)} datasets.")
    for source_id, status in index.dataset_status.items():
        if status != "ok":
            print(f"  WARNING: {source_id}: {status}")

    # Layout: [left traces column] [density map] [right traces column] in row 0, with every
    # control (CO2 mode, k-slider, sync checkbox + save button) in row 1 below. Every one of
    # these -- including the widgets -- is a genuine gridspec cell (fig.add_subplot), not a
    # fixed figure-fraction axes (fig.add_axes). Mixing the two is what caused the earlier bugs:
    # plt.tight_layout() explicitly does not know how to account for fixed-position axes (it
    # warns "Axes that are not compatible with tight_layout"), so it would happily let row 0's
    # trace panels grow down into where a fixed-position widget visually sat, since gridspec
    # cells are the only thing tight_layout guarantees won't overlap each other.
    fig = plt.figure(figsize=(19, 7.5))
    outer_gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.4, 1], height_ratios=[6, 1], wspace=0.35, hspace=0.55)
    traces_slot_left = outer_gs[0, 0]
    ax_map = fig.add_subplot(outer_gs[0, 1])
    traces_slot_right = outer_gs[0, 2]

    ax_co2_mode = fig.add_subplot(outer_gs[1, 0])
    ax_co2_mode.set_title("CO2 display", fontsize=8)
    ax_slider = fig.add_subplot(outer_gs[1, 1])
    right_controls_gs = outer_gs[1, 2].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
    ax_sync_check = fig.add_subplot(right_controls_gs[0, 0])
    ax_save_button = fig.add_subplot(right_controls_gs[0, 1])

    current_k = args.k
    save_message = {"artist": None, "timer": None, "shown_at": None}
    co2_mode = "averaged"
    sync_y = False
    last_click = {"left": None, "right": None}

    render_umap_background(ax_map, index.density, extent=index.extent, wbounds=index.wbounds)
    ax_map.scatter(index.pooled_xy[:, 0], index.pooled_xy[:, 1], s=1, c="white", alpha=0.08, linewidths=0)
    ax_map.set_title(f"{project_dir.name} ({index.pooled_xy.shape[0]} points, {len(index.dataset_status)} datasets)")
    ax_map.set_xlabel("UMAP 1")
    ax_map.set_ylabel("UMAP 2")

    k_slider = Slider(ax_slider, "k (neighbors)", valmin=1, valmax=max(args.k_max, args.k), valinit=args.k, valstep=1)

    highlight_left = ax_map.scatter([], [], s=40, facecolors="none", edgecolors="red", linewidths=1.4, zorder=5)
    highlight_right = ax_map.scatter([], [], s=40, facecolors="none", edgecolors="dodgerblue", linewidths=1.4, zorder=5)

    sides = {}

    def init_side(name, slot, color, placeholder_text):
        axes = []
        ax = fig.add_subplot(slot)
        ax.set_title(placeholder_text, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
        sides[name] = {"slot": slot, "axes": axes, "color": color}

    init_side("left", traces_slot_left, "tab:red", "Click a point on the map to show its neighbor traces here")
    init_side("right", traces_slot_right, "tab:blue", "A second selection appears here for comparison")

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

        plot_groups = [] if averaged.empty else build_plot_groups(averaged.columns, co2_mode)
        axes = rebuild_trace_axes(name, len(plot_groups))
        label_axes = {}
        if not averaged.empty:
            relative_hours = averaged.index.total_seconds() / 3600.0
            for ax, (group_label, member_cols) in zip(axes, plot_groups):
                if len(member_cols) > 1:
                    # CO2 averaged across sensors (e.g. the 4 WholeHive corners): one mean line
                    # plus a +/-std band showing spread across sensors at each relative time.
                    values = averaged[member_cols]
                    mean_vals = values.mean(axis=1)
                    std_vals = values.std(axis=1)
                    ax.plot(relative_hours, mean_vals, color=sides[name]["color"])
                    ax.fill_between(relative_hours, mean_vals - std_vals, mean_vals + std_vals,
                                     color=sides[name]["color"], alpha=0.25, linewidth=0)
                    ax.set_ylabel(group_label, fontsize=9)
                else:
                    ax.plot(relative_hours, averaged[member_cols[0]], color=sides[name]["color"])
                    ax.set_ylabel(member_cols[0], fontsize=9)
                ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
                ax.tick_params(labelsize=8)
                ax.grid(alpha=0.3)
                label_axes[group_label] = ax
            axes[-1].set_xlabel("Hours relative to each neighbor's own timestamp", fontsize=8)
            axes[0].set_title(f"{n_included} neighbor(s), window ±{window_hours:.2f} h", fontsize=9)
        sides[name]["label_axes"] = label_axes

    def apply_y_sync():
        # Only "activity" and any co2 group/column are linked -- rel_humid/Tamb keep their own
        # independent scale, since the ask was specifically to make activity and co2 directly
        # comparable between the left and right panels, not every trace.
        if not sync_y:
            return
        left_axes = sides["left"].get("label_axes", {})
        right_axes = sides["right"].get("label_axes", {})
        for group_label in set(left_axes) & set(right_axes):
            if group_label != "activity" and not group_label.lower().startswith("co2"):
                continue
            ax_l, ax_r = left_axes[group_label], right_axes[group_label]
            ymin = min(ax_l.get_ylim()[0], ax_r.get_ylim()[0])
            ymax = max(ax_l.get_ylim()[1], ax_r.get_ylim()[1])
            ax_l.set_ylim(ymin, ymax)
            ax_r.set_ylim(ymin, ymax)

    def refresh_both_sides():
        # Re-run whichever side(s) already have a point selected, so a control change (k,
        # CO2 display mode, y-sync) is immediately visible instead of only taking effect on the
        # next click.
        if last_click["left"] is not None:
            update_side("left", last_click["left"], highlight_left, "left-click")
        if last_click["right"] is not None:
            update_side("right", last_click["right"], highlight_right, "right-click")
        apply_y_sync()
        fig.tight_layout()
        fig.canvas.draw_idle()

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

        apply_y_sync()
        # The trace panels are (re)created here, after the one-time layout pass at startup, so
        # their y-axis labels are never accounted for by it -- without redoing the layout now,
        # those labels can render past the edge of their column and overlap ax_map.
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_k_changed(val):
        nonlocal current_k
        current_k = int(val)
        refresh_both_sides()

    def on_co2_mode_changed(selected_label):
        nonlocal co2_mode
        co2_mode = "averaged" if selected_label.startswith("avg") else "individual"
        refresh_both_sides()

    def on_sync_changed(_selected_label):
        nonlocal sync_y
        sync_y = not sync_y
        refresh_both_sides()

    def row0_bbox_inches():
        # Crop the saved figure to row 0 (map + trace columns) only, excluding row 1's control
        # strip below it. The boundary is fixed by the gridspec's own height_ratios/hspace, not
        # by content, so this reads it straight from the gridspec rather than needing a renderer
        # or backend-specific tight-bbox machinery. Cropping exactly at row 0's cell-bottom cuts
        # into its own x-tick labels/xlabel (they render into the hspace gap below the cell, not
        # strictly inside it); cropping at row 1's cell-top catches the "CO2 display" title
        # spilling above *its* cell for the same reason. The midpoint of that gap clears both.
        bottoms, tops, _lefts, _rights = outer_gs.get_grid_positions(fig)
        crop_y = (bottoms[0] + tops[1]) / 2
        width_in, height_in = fig.get_size_inches()
        return Bbox.from_extents(0, crop_y * height_in, width_in, height_in)

    def on_save_clicked(event):
        # No file dialog: the installed matplotlib's native macOS save panel
        # (NavigationToolbar2Mac -> _macosx.choose_save_file) takes only a title and a bare
        # filename, no directory, and Cocoa's fallback when none is set is whatever it last
        # remembers -- not something this app can point at --figures-dir. Saving straight there
        # with an auto-generated name sidesteps that entirely (and avoids tkinter, which is what
        # previously made the button unresponsive and crashed the app on close).
        # Clear any still-visible confirmation from a rapid previous click *before* saving, so a
        # leftover banner from the last save never ends up baked into this one's PNG.
        clear_save_message()

        figures_dir.mkdir(parents=True, exist_ok=True)
        path = figures_dir / f"{project_dir.name}_k{current_k}_{datetime.now():%Y%m%d_%H%M%S}.png"
        fig.savefig(path, dpi=150, bbox_inches=row0_bbox_inches())
        print(f"\nSaved current view to {path}")
        show_save_message(f"Saved to {path.name}")

    def clear_save_message():
        if save_message["timer"] is not None:
            save_message["timer"].stop()
            save_message["timer"] = None
        if save_message["artist"] is not None:
            save_message["artist"].remove()
            save_message["artist"] = None
            save_message["shown_at"] = None
            fig.canvas.draw_idle()

    def show_save_message(text):
        save_message["artist"] = fig.text(
            0.5, 0.97, text, ha="center", va="top", fontsize=11, color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="tab:green", alpha=0.85, edgecolor="none"),
            zorder=100,
        )
        save_message["shown_at"] = time.monotonic()
        fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=3000)
        timer.single_shot = True
        timer.add_callback(clear_save_message)
        timer.start()
        save_message["timer"] = timer

    def clear_save_message_if_expired(_event):
        # Backup for the timer above: matplotlib's native macOS timer can be unreliable once
        # embedded in a widget-heavy app (observed: it can silently never fire here, even though
        # it fires fine in isolation), so any subsequent mouse movement also clears an expired
        # message. Mouse-move events fire constantly during normal use, so in practice this still
        # clears within a fraction of a second of the 3s mark even if the timer never fires.
        if save_message["shown_at"] is not None and time.monotonic() - save_message["shown_at"] >= 3.0:
            clear_save_message()

    save_button = Button(ax_save_button, "Save image")
    save_button.on_clicked(on_save_clicked)

    # Default (index 0) matches co2_mode = "averaged" above.
    co2_mode_radio = RadioButtons(ax_co2_mode, ["avg ± std", "per sensor"], active=0)
    for radio_label in co2_mode_radio.labels:
        radio_label.set_fontsize(8)
    co2_mode_radio.on_clicked(on_co2_mode_changed)

    sync_check = CheckButtons(ax_sync_check, ["Sync Y (L/R)"], [sync_y])
    for check_label in sync_check.labels:
        check_label.set_fontsize(8)
    sync_check.on_clicked(on_sync_changed)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", clear_save_message_if_expired)
    k_slider.on_changed(on_k_changed)

    # matplotlib widgets are only weakly referenced by the event system -- without an explicit
    # strong reference that outlives this function's own locals, they can silently stop firing
    # once GC runs (e.g. triggered by the axes churn in rebuild_trace_axes()). plt.show() blocks
    # here for real interactive backends, which incidentally keeps main()'s locals alive for the
    # whole session, but that's not something to rely on -- attaching to the figure is the robust
    # fix matplotlib's own docs recommend.
    fig._umap_inspector_widgets = (k_slider, save_button, co2_mode_radio, sync_check)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
