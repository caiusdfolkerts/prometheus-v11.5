# prometheus/sim/plotting.py
"""
Generates diagnostic plots from simulation data and saves them to a PDF file.

Provides helper functions for common plot types (time series, histograms, etc.)
and a main function `generate_plots_pdf` to create the multi-page report based
on collected graph data and configuration settings.
"""

import matplotlib
matplotlib.use('Agg') # use non-interactive backend before importing pyplot to prevent display issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import datetime
import traceback
from typing import Dict, Any, List, Optional

# --- plotting constants for consistent styling ---
TITLE_FONTSIZE = 10
LABEL_FONTSIZE = 8
TICK_FONTSIZE = 7
LEGEND_FONTSIZE = 7
MARKER_SIZE = 3
LINE_WIDTH = 1.5
GRID_ALPHA = 0.6
SCATTER_SIZE = 1 # default size for scatter plot points
SCATTER_ALPHA = 0.5 # default alpha for scatter plot points


def _get_plot_setting(graph_settings: Dict, key: str, default: bool = False) -> bool:
    """safely retrieves a boolean plot setting from the graph configuration dictionary."""
    return graph_settings.get(key, default)

def _get_valid_data(data_dict: Dict, keys: List[str]) -> Optional[Dict[str, np.ndarray]]:
    """
    validates that required data keys exist, are non-empty lists, converts them to
    numpy arrays of equal length, and returns them in a dictionary.
    handles potential NaN values gracefully by not erroring on them directly.
    """
    if not data_dict: return None
    valid_data = {}
    min_len = float('inf')
    all_keys_present = True

    for k in keys:
        data_list = data_dict.get(k)
        if not isinstance(data_list, list):
            all_keys_present = False; break
        valid_data[k] = np.asarray(data_list) # convert to numpy array
        # allow some specific keys to be empty if not collected during simulation
        if len(valid_data[k]) == 0 and k != 'final_snapshot' and k not in ['min_separation', 'max_separation', 'bh_num_nodes']:
            all_keys_present = False; break # if essential time series data is empty, can't plot
        if len(valid_data[k]) > 0 : min_len = min(min_len, len(valid_data[k]))

    if not all_keys_present: return None
    if min_len == float('inf') and any(len(arr) == 0 for arr in valid_data.values()): # all lists were empty
        return None

    # ensure all arrays have the same length by trimming to the shortest non-empty array
    if len(keys) > 1 and min_len != float('inf'):
        for k_trim in keys:
            if len(valid_data[k_trim]) > min_len:
                valid_data[k_trim] = valid_data[k_trim][:min_len]
            elif len(valid_data[k_trim]) < min_len and len(valid_data[k_trim]) > 0:
                 return None # mismatched non-empty length after initial check, probablj an issue
    return valid_data


def _plot_time_series(ax: plt.Axes, time_data: np.ndarray, series_data: Dict[str, np.ndarray], title: str, ylabel: str, yscale: str = 'linear'):
    """
    helper to plot one or more time series on a given axes object.
    
    args:
        ax: matplotlib axes to plot on.
        time_data: numpy array of time values.
        series_data: dictionary mapping series labels to numpy arrays of data values.
        title: plot title.
        ylabel: y-axis label.
        yscale: y-axis scale ('linear' or 'log').
    """
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Simulation Time", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    has_valid_series = False
    for label, data_arr in series_data.items():
        # filter out nans for plotting to avoid gaps/errors if some steps didn't produce data
        valid_indices = ~np.isnan(data_arr)
        if np.any(valid_indices):
            ax.plot(time_data[valid_indices], data_arr[valid_indices], label=label, lw=LINE_WIDTH)
            has_valid_series = True
    if not has_valid_series:
        ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', transform=ax.transAxes)
        ax.grid(False)
        return
    if len(series_data) > 1: ax.legend(fontsize=LEGEND_FONTSIZE) # show legend for mulltiple series
    ax.grid(True, linestyle=':', alpha=GRID_ALPHA)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_yscale(yscale)
    if yscale == 'log':
        # collect all valid positive data points to determine sensible log scale limits
        all_series_data_flat = np.concatenate([d[~np.isnan(d) & (d > 0)] for d in series_data.values()])
        if all_series_data_flat.size > 0:
             min_positive = np.min(all_series_data_flat)
             max_val = np.max(np.concatenate([d[~np.isnan(d)] for d in series_data.values()]))
             if min_positive > 0 and max_val > min_positive :
                  ax.set_ylim(bottom=min_positive * 0.5, top=max_val * 1.5 if max_val > 0 else min_positive * 10)
             elif min_positive > 0 : # all values are the same positive number
                  ax.set_ylim(bottom=min_positive * 0.5, top=min_positive * 1.5)


def _plot_histogram(ax: plt.Axes, data: Optional[np.ndarray], bins: int, title: str, xlabel: str, xscale: str = 'linear'):
    """
    helper to plot a histogram on a given axes object, handling invalid data.
    
    args:
        ax: matplotlib axes to plot on.
        data: numpy array of data for the histogram.
        bins: number of bins for the histogram.
        title: plot title.
        xlabel: x-axis label.
        xscale: x-axis scale ('linear' or 'log').
    """
    if data is None or data.size == 0:
         ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
         ax.set_title(title, fontsize=TITLE_FONTSIZE); ax.grid(False); return

    finite_data = data[np.isfinite(data)] # filter out nan/inf values before binning
    if finite_data.size == 0:
        ax.text(0.5, 0.5, "No Finite Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=TITLE_FONTSIZE); ax.grid(False); return

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Number of Particles", fontsize=LABEL_FONTSIZE)

    if xscale == 'log':
        positive_data = finite_data[finite_data > 1e-20] # use a small positive floor for log scale
        if positive_data.size > 0:
            data_min, data_max = np.min(positive_data), np.max(positive_data)
            if data_max > data_min * (1 + 1e-6): # ensure range is valid for logspace bins
                 log_bins = np.logspace(np.log10(data_min), np.log10(data_max), bins + 1)
                 ax.hist(positive_data, bins=log_bins, log=False) # use log bins, linear counts
                 ax.set_xscale('log')
            else: # handle case where all positive data is the same value or very close
                 ax.hist(positive_data, bins=bins, log=False)
                 ax.set_xscale('linear') # fallback to linear if range is effectively zero for log scale
                 if positive_data.size > 0: ax.set_xlim(positive_data[0]*0.9, positive_data[0]*1.1)
        else:
             ax.text(0.5, 0.5, "No Positive Data for Log Scale", ha='center', va='center', transform=ax.transAxes)
    else: # linear scale
        ax.hist(finite_data, bins=bins, log=False)
        ax.set_xscale('linear')

    ax.grid(True, linestyle=':', alpha=GRID_ALPHA)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)

def _plot_radial_profile(ax: plt.Axes, radii: Optional[np.ndarray], values: Optional[np.ndarray], bins: int, title: str, ylabel: str, yscale: str = 'linear'):
    """
    helper to plot a binned radial profile on a given axes object.
    
    args:
        ax: matplotlib axes to plot on.
        radii: numpy array of radial distances.
        values: numpy array of corresponding physical quantity values.
        bins: number of radial bins.
        title: plot title.
        ylabel: y-axis label (for the binned quantity).
        yscale: y-axis scale ('linear' or 'log').
    """
    if radii is None or values is None or radii.size == 0 or values.size == 0 or radii.shape != values.shape:
         ax.text(0.5, 0.5, "No/Mismatched Data", ha='center', va='center', transform=ax.transAxes)
         ax.set_title(title, fontsize=TITLE_FONTSIZE); ax.grid(False); return

    mask = np.isfinite(radii) & np.isfinite(values) # ensure both radius and value are finite
    radii_f = radii[mask]; values_f = values[mask]
    if radii_f.size == 0:
        ax.text(0.5, 0.5, "No Finite Data Pairs", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=TITLE_FONTSIZE); ax.grid(False); return

    try:
        min_r_data, max_r_data = np.min(radii_f), np.max(radii_f)
        if max_r_data <= min_r_data: # handle case with single radius or no radial range
            if radii_f.size > 0 : # if there are points, plot them directly
                 ax.plot(radii_f, values_f, 'o', markersize=MARKER_SIZE)
            else:
                 ax.text(0.5, 0.5, "No Radial Range", ha='center', va='center', transform=ax.transAxes)
        else:
            # create linear bins for radii
            radius_bins = np.linspace(min_r_data, max_r_data, bins + 1)
            bin_centers = 0.5 * (radius_bins[:-1] + radius_bins[1:])
            # calculate sum and count in each bin to find the mean
            sum_in_bin, _ = np.histogram(radii_f, bins=radius_bins, weights=values_f)
            count_in_bin, _ = np.histogram(radii_f, bins=radius_bins)
            valid_bins = count_in_bin > 0
            bin_means = np.full_like(sum_in_bin, np.nan, dtype=np.float64) # initialize with nans
            bin_means[valid_bins] = sum_in_bin[valid_bins] / count_in_bin[valid_bins] # compute mean only for non-empty bins
            if np.any(valid_bins):
                ax.plot(bin_centers[valid_bins], bin_means[valid_bins], 'o-', markersize=MARKER_SIZE, lw=LINE_WIDTH)
            else: ax.text(0.5, 0.5, "No Points in Bins", ha='center', va='center', transform=ax.transAxes)

    except Exception as e:
        print(f"Error plotting radial profile '{title}': {e}"); traceback.print_exc()
        ax.text(0.5, 0.5, "Plotting Error", ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Radius from CoM", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.grid(True, linestyle=':', alpha=GRID_ALPHA)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_yscale(yscale)
    if yscale == 'log':
        # attempt to set reasonable log scale limits for y-axis
        plotted_means = bin_means[valid_bins & ~np.isnan(bin_means) & (bin_means > 0)] if 'bin_means' in locals() and 'valid_bins' in locals() else np.array([])
        if plotted_means.size > 0:
             min_plot = np.min(plotted_means)
             max_plot = np.max(bin_means[valid_bins & ~np.isnan(bin_means)]) if 'bin_means' in locals() and 'valid_bins' in locals() and np.any(valid_bins & ~np.isnan(bin_means)) else min_plot *10
             ax.set_ylim(bottom=min_plot * 0.5, top=max_plot * 1.5 if max_plot > min_plot else min_plot * 10)

def _plot_scatter(ax: plt.Axes, x_data: Optional[np.ndarray], y_data: Optional[np.ndarray], title: str, xlabel: str, ylabel: str, xscale: str = 'linear', yscale: str = 'linear', sample_frac: float = 1.0):
    """
    helper to plot a scatter diagram on a given axes object, with optional subsampling.
    
    args:
        ax: matplotlib axes to plot on.
        x_data: numpy array of x-coordinates.
        y_data: numpy array of y-coordinates.
        title: plot title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        xscale: x-axis scale ('linear' or 'log').
        yscale: y-axis scale ('linear' or 'log').
        sample_frac: fraction of points to sample for plotting (0.0 to 1.0).
    """
    if x_data is None or y_data is None or x_data.size == 0 or y_data.size == 0:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=TITLE_FONTSIZE); ax.grid(False); return

    min_len = min(len(x_data), len(y_data)) # ensure x and y data have the same length
    x = x_data[:min_len]; y = y_data[:min_len]
    mask = np.isfinite(x) & np.isfinite(y) # filter oukt pairs with nan/inf values
    x = x[mask]; y = y[mask]
    if x.size == 0:
        ax.text(0.5, 0.5, "No Finite Data Pairs", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=TITLE_FONTSIZE); ax.grid(False); return

    n_points = x.size
    # subsample if requested and if there are enough points to warrant it
    if sample_frac < 1.0 and n_points > 500:
        n_sample = max(100, int(n_points * sample_frac)) # ensure a minimum number of samples
        if n_sample < n_points:
            indices = np.random.choice(n_points, n_sample, replace=False) # random sampling without replacement
            x = x[indices]; y = y[indices]

    ax.scatter(x, y, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, edgecolors='none', rasterized=True) # rasterized for large N to reduce pdf file size
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE); ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.grid(True, linestyle=':', alpha=GRID_ALPHA)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    # auto-adjust log limits to fit positive data
    if xscale == 'log':
        x_pos = x[x > 0]; ax.set_xlim(np.min(x_pos)*0.8, np.max(x_pos)*1.2) if x_pos.size > 0 else None
    if yscale == 'log':
        y_pos = y[y > 0]; ax.set_ylim(np.min(y_pos)*0.8, np.max(y_pos)*1.2) if y_pos.size > 0 else None


def generate_plots_pdf(graph_data: Dict[str, Any], graph_settings: Dict, output_dir: str) -> str:
    """
    Generates a multi-page PDF containing diagnostic simulation plots.
    """
    if not graph_data: print("Plotting Error: No graph data provided."); return ""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(output_dir, f"prometheus_graphs_{timestamp}.pdf")
    print(f"Generating plots PDF: {os.path.normpath(pdf_filename)}")
    plot_pages = [
        {
            "title": "Energy Diagnostics", "rows": 1, "cols": 2, "figsize": (10, 4), 
            "subplots": [
                {"setting": "plot_energy_components", "func": _plot_time_series, "keys": ['time', 'total_ke', 'total_pe', 'total_u'], "args": ["Energy Components", "Energy"], "kwargs": {"series_data": {'KE': 'total_ke', 'PE': 'total_pe', 'U': 'total_u'}}},
                {"setting": "plot_energy_rates", "func": _plot_time_series, "keys": ['time', 'total_cooling_rate', 'total_fusion_rate', 'total_sph_work_rate'], "args": ["Energy Change Rates", "Rate"], "kwargs": {"series_data": {'Cooling': 'total_cooling_rate', 'Fusion': 'total_fusion_rate', 'SPH Work': 'total_sph_work_rate'}}},
            ]
        },
        {
            "title": "Momentum & System Properties", "rows": 1, "cols": 2, "figsize": (10, 4),
            "subplots": [
                {"setting": "plot_angular_momentum", "func": _plot_time_series, "keys": ['time', 'total_lx', 'total_ly', 'total_lz'], "args": ["Angular Momentum (CoM)", "Ang. Momentum"], "kwargs": {"series_data": {'Lx': 'total_lx', 'Ly': 'total_ly', 'Lz': 'total_lz'}}},
                {"setting": "plot_com_velocity", "func": _plot_time_series, "keys": ['time', 'com_vx', 'com_vy', 'com_vz'], "args": ["Center of Mass Velocity", "Velocity"], "kwargs": {"series_data": {'Vx': 'com_vx', 'Vy': 'com_vy', 'Vz': 'com_vz'}}},
            ]
        },
        # Page 3: Global Properties 
        {
            "title": "Global Physics Properties", "rows": 1, "cols": 3, "figsize": (12, 4),
            "subplots": [
                {"setting": "plot_max_rho", "func": _plot_time_series, "keys": ['time', 'max_rho'], "args": ["Maximum Density", "Density"], "kwargs": {"yscale": "log"}},
                {"setting": "plot_max_temp", "func": _plot_time_series, "keys": ['time', 'max_temp'], "args": ["Maximum Temperature", "Temperature (K)"], "kwargs": {"yscale": "log"}},
                {"setting": "plot_avg_temp", "func": _plot_time_series, "keys": ['time', 'avg_temp'], "args": ["Average Temperature", "Temperature (K)"], "kwargs": {"yscale": "log"}},
            ]
        },
        # Page 4: Final State Histograms
        {
            "title": "Final State Distributions", "rows": 1, "cols": 3, "figsize": (12, 4),
            "subplots": [
                {"setting": "plot_hist_speed", "func": _plot_histogram, "keys": ['final_snapshot', 'speeds'], "args": [graph_settings.get('histogram_bins', 50), "Speed Distribution", "Speed"]},
                {"setting": "plot_hist_temp", "func": _plot_histogram, "keys": ['final_snapshot', 'temperatures'], "args": [graph_settings.get('histogram_bins', 50), "Temperature Distribution", "Temperature (K)"], "kwargs": {"xscale": "log"}},
                {"setting": "plot_hist_density", "func": _plot_histogram, "keys": ['final_snapshot', 'densities'], "args": [graph_settings.get('histogram_bins', 50), "Density Distribution", "Density"], "kwargs": {"xscale": "log"}},
            ]
        },
        # page 5: final state profiles  Phase Diagram 
        {
            "title": "Final State Structure & Phase Diagram", "rows": 1, "cols": 3, "figsize": (12, 4),
            "subplots": [
                {"setting": "plot_profile_density", "func": _plot_radial_profile, "keys": ['final_snapshot', 'radii', 'densities'], "args": [graph_settings.get('radial_profile_bins', 30), "Density Profile", "Density"], "kwargs": {"yscale": "log"}},
                {"setting": "plot_profile_temp", "func": _plot_radial_profile, "keys": ['final_snapshot', 'radii', 'temperatures'], "args": [graph_settings.get('radial_profile_bins', 30), "Temperature Profile", "Temperature (K)"], "kwargs": {"yscale": "log"}},
                {"setting": "plot_phase_T_rho", "func": _plot_scatter, "keys": ['final_snapshot', 'densities', 'temperatures'], "args": ["Phase Diagram (T-rho)", "Density", "Temperature (K)"], "kwargs": {"xscale": "log", "yscale": "log", "sample_frac": graph_settings.get('phase_plot_sample_frac', 0.2)}},
            ]
        },
    ]

    try:
        with PdfPages(pdf_filename) as pdf:
            # loop through defined pages
            for page_num, page_layout in enumerate(plot_pages):
                # check if any plot on this page is enabled
                is_page_enabled = any(_get_plot_setting(graph_settings, plot_def["setting"]) for plot_def in page_layout["subplots"])
                if not is_page_enabled: continue # skip page if all subplots disabled

                # xreate figure for the page
                fig, axes = plt.subplots(page_layout["rows"], page_layout["cols"], figsize=page_layout["figsize"])
                fig.suptitle(page_layout["title"], fontsize=14)
                if isinstance(axes, plt.Axes): axes = np.array([axes]) # ensure axes is always array-like
                axes = axes.flatten() # flatten for easy iteration

                # adjust index for subplots if page layout changed
                subplot_idx = 0 
                for plot_def in page_layout["subplots"]:
                    if subplot_idx >= len(axes): # safety break if more plot_defs than actual subplots
                        print(f"Warning: Not enough subplots for page '{page_layout['title']}'. Skipping remaining plot defs.")
                        break
                    
                    ax = axes[subplot_idx]
                    plot_func = plot_def["func"]
                    plot_args = plot_def.get("args", [])
                    plot_kwargs = plot_def.get("kwargs", {})
                    keys_needed = plot_def.get("keys", [])
                    setting_key = plot_def["setting"]

                    # default placeholder if plot disabled
                    ax.set_title(plot_args[1] if len(plot_args) > 1 and isinstance(plot_args[1], str) else "Plot", fontsize=TITLE_FONTSIZE) # Guess title
                    ax.text(0.5, 0.5, "Disabled", ha='center', va='center', transform=ax.transAxes)

                    if _get_plot_setting(graph_settings, setting_key):
                        data_source = graph_data
                        plot_data_args = []
                        is_snapshot_plot = 'final_snapshot' in keys_needed
                        if is_snapshot_plot:
                            data_source = graph_data.get('final_snapshot', {})
                            plot_data_args = [data_source.get(k) for k in keys_needed if k != 'final_snapshot']
                        else: 
                            valid_ts_data = _get_valid_data(data_source, keys_needed)
                            if valid_ts_data:
                                time_key = 'time' 
                                time_data = valid_ts_data.get(time_key)
                                if time_data is not None:
                                     series_data_arg = {}
                                     if "series_data" in plot_kwargs:
                                         name_map = plot_kwargs.pop("series_data") 
                                         for series_name, data_key in name_map.items():
                                             if data_key in valid_ts_data:
                                                 series_data_arg[series_name] = valid_ts_data[data_key]
                                     else: 
                                          data_key = keys_needed[1]
                                          series_data_arg = {plot_args[1]: valid_ts_data[data_key]} 

                                     plot_data_args = [time_data, series_data_arg]
                                else: plot_data_args = None 
                            else: plot_data_args = None 

                        if plot_data_args is not None:
                            ax.cla() 
                            try:
                                plot_func(ax, *plot_data_args, *plot_args, **plot_kwargs)
                            except Exception as e_plot:
                                print(f"ERROR plotting '{setting_key}': {e_plot}"); traceback.print_exc()
                                ax.text(0.5, 0.5, "Plotting Error", ha='center', va='center', transform=ax.transAxes)
                        else: 
                            ax.cla() 
                            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(plot_args[1] if len(plot_args) > 1 and isinstance(plot_args[1], str) else "Plot", fontsize=TITLE_FONTSIZE)
                    subplot_idx += 1 # increment subplot index

                # hide any unused subplots if the number of definitions is less than rows*cols
                for k in range(subplot_idx, len(axes)):
                    axes[k].axis('off')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
                pdf.savefig(fig)
                plt.close(fig) 

            print(f"Successfully generated PDF: {pdf_filename}")
            return pdf_filename

    except Exception as e:
        print(f"ERROR during PDF generation process: {e}"); traceback.print_exc()
        if os.path.exists(pdf_filename):
            try: os.remove(pdf_filename)
            except OSError: pass
        return ""