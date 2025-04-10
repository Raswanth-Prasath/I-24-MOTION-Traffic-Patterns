import requests
import ijson
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import datetime
import os
import io
import shutil
import sys # For exiting gracefully
try:
    import zoneinfo # Requires Python 3.9+
except ImportError:
    print("Error: 'zoneinfo' module not found.")
    print("This script requires Python 3.9 or later for timezone support.")
    print("Alternatively, install 'tzdata' on some systems: pip install tzdata")
    sys.exit(1)


# --- Configuration ---
DATA_URL = "https://huggingface.co/datasets/Raswanth/I24_MOTION/resolve/main/637b023440527bf2daa5932f__post1.json"
LOCAL_JSON_FILENAME = "637b023440527bf2daa5932f__post1.json" # Name to save the file locally
OUTPUT_PLOT_FILENAME = "trajectory_plot_time_filtered_lines_tz.png" # Updated filename
REPORTING_INTERVAL = 1000 # Print progress every N trajectories processed within the window
SPEED_VMIN = 0
SPEED_VMAX = 80
LINE_WIDTH = 0.5

# --- Time Filtering Configuration ---
# Define the time window in HH:MM:SS format (using the date of the data)
START_TIME_STR = "06:00:00" # Example: Start time
END_TIME_STR   = "10:00:00" # Example: End time
# Define the timezone for the I-24 MOTION data (Nashville, TN)
# This handles Daylight Saving Time (CST/CDT) automatically.
DATA_TIMEZONE = "America/Chicago"

# --- Function to download file ---
def download_file(url, filename):
    """Downloads a file from a URL to a local filename, streaming the content."""
    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Skipping download.")
        return True
    else:
        print(f"Downloading '{url}' to '{filename}'...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                # Use a larger chunk size for potentially faster downloads
                with open(filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192*2): # 16MB chunks
                        f.write(chunk)
            print("Download complete.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            if os.path.exists(filename): os.remove(filename)
            return False
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            if os.path.exists(filename): os.remove(filename)
            return False

# --- Function to get date context and calculate time boundaries (TIMEZONE-AWARE) ---
def get_time_boundaries(filename, start_str, end_str, data_tz_name):
    """
    Reads the first record to get the date IN THE SPECIFIED TIMEZONE,
    then calculates start/end Unix timestamps based on timezone-aware datetimes.
    Requires Python 3.9+ and the zoneinfo module.
    """
    print(f"Determining date context from first record in '{filename}' using timezone '{data_tz_name}'...")
    first_timestamp = None
    try:
        # Get the timezone object
        data_tz = zoneinfo.ZoneInfo(data_tz_name)
    except zoneinfo.ZoneInfoNotFoundError:
        print(f"Error: Timezone '{data_tz_name}' not found. Check spelling or system timezone data.")
        return None, None
    except Exception as e:
        print(f"Error initializing timezone '{data_tz_name}': {e}")
        return None, None

    try:
        with open(filename, 'rb') as f:
            # Increase buffer size for ijson if reading is slow or errors occur
            parser = ijson.items(f, 'item', buf_size=64*1024, use_float=True)
            for record in parser:
                if "timestamp" in record and isinstance(record["timestamp"], list) and len(record["timestamp"]) > 0:
                    if isinstance(record["timestamp"][0], (float, int)):
                        first_timestamp = record["timestamp"][0]
                        print(f"Found first Unix timestamp (UTC): {first_timestamp}")
                        break
            if first_timestamp is None:
                 print("Error: Could not find a valid timestamp in the first few records.")
                 return None, None
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found for getting date context.")
        return None, None
    except ijson.JSONError as e:
        print(f"Error parsing JSON while getting date context: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error getting date context: {e}")
        return None, None

    try:
        # Convert the FIRST Unix timestamp (UTC) to a timezone-aware datetime
        # This correctly determines the date in the specified timezone, handling DST
        reference_datetime_aware = datetime.datetime.fromtimestamp(first_timestamp, tz=data_tz)
        reference_date = reference_datetime_aware.date()
        print(f"Reference date determined as: {reference_date.isoformat()} in timezone {data_tz_name}")

        # Parse the naive time strings
        start_time_obj = datetime.time.fromisoformat(start_str)
        end_time_obj = datetime.time.fromisoformat(end_str)

        # Combine the determined date with the naive times
        start_dt_naive = datetime.datetime.combine(reference_date, start_time_obj)
        end_dt_naive = datetime.datetime.combine(reference_date, end_time_obj)

        # Make the combined datetimes timezone-aware using the specific data timezone
        # This is crucial for correct conversion to Unix timestamp
        start_dt_aware = start_dt_naive.replace(tzinfo=data_tz)
        end_dt_aware = end_dt_naive.replace(tzinfo=data_tz)

        # Get the Unix timestamp (seconds since epoch, UTC) for the start/end times
        start_unix_ts = start_dt_aware.timestamp()
        end_unix_ts = end_dt_aware.timestamp()

        # Print informative messages including the timezone
        print(f"Processing data from {start_dt_aware.strftime('%Y-%m-%d %H:%M:%S %Z%z')} (Unix: {start_unix_ts:.2f})")
        print(f"Processing data until {end_dt_aware.strftime('%Y-%m-%d %H:%M:%S %Z%z')} (Unix: {end_unix_ts:.2f})")

        if start_unix_ts >= end_unix_ts:
            print("Error: Start time is after or equal to end time.")
            return None, None

        return start_unix_ts, end_unix_ts

    except ValueError as e:
        print(f"Error parsing time string ('{start_str}' or '{end_str}'). Use HH:MM:SS format. Details: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error calculating time boundaries: {e}")
        return None, None

# --- Main Script Logic ---
def main():
    print(f"Using Python version: {sys.version}")
    if sys.version_info < (3, 9):
      print("Warning: This script requires Python 3.9+ for reliable timezone support via 'zoneinfo'.")
      print("Results may be incorrect if run on older Python versions or without 'tzdata'.")

    # Download the JSON file if needed
    if not download_file(DATA_URL, LOCAL_JSON_FILENAME):
        print("Failed to obtain data file. Exiting.")
        sys.exit(1)

    # Get time boundaries based on the data's date and SPECIFIED TIMEZONE
    start_unix_timestamp, end_unix_timestamp = get_time_boundaries(
        LOCAL_JSON_FILENAME, START_TIME_STR, END_TIME_STR, DATA_TIMEZONE
    )
    if start_unix_timestamp is None or end_unix_timestamp is None:
        print("Could not determine time boundaries. Exiting.")
        sys.exit(1)

    # Set up the plot
    print("Setting up plot...")
    plt.rc('font', family='serif', size=14)
    fig, ax = plt.subplots(figsize=(67, 14), dpi=300)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)

    # Apply black background theme
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    # Update title with time window and timezone context
    ax.set_title(f"Westbound Trajectories ({START_TIME_STR} - {END_TIME_STR})", color='white')

    # Define the color range
    colors = [(1, 0, 0), (1, 1, 0), (0, 0.8, 0), (0, 0.3, 0)] # Red-Yellow-Green-DarkGreen
    custom_cmap = LinearSegmentedColormap.from_list('CustomGreenToRed', colors, N=256)
    norm = Normalize(vmin=SPEED_VMIN, vmax=SPEED_VMAX)

    print(f"Processing trajectories from '{LOCAL_JSON_FILENAME}' within the time window...")
    t_start = time.time()
    i = 0 # Counter for PROCESSED trajectories within window
    records_processed = 0 # Counter for total records checked

    # --- Data Accumulation for LineCollection ---
    all_segments = []
    all_segment_speeds = []
    # Initialize bounds with the filter times to ensure plot covers the window
    min_time, max_time = start_unix_timestamp, end_unix_timestamp # X bounds set by filter
    min_pos, max_pos = np.inf, -np.inf # Position bounds still need data
    found_data_in_window = False
    # -------------------------------------------

    # Process trajectories using ijson
    try:
        # Use 'rb' mode for ijson with byte streams
        with open(LOCAL_JSON_FILENAME, 'rb') as f:
             # Increase buffer size for potentially better performance on large files
            parser = ijson.items(f, 'item', buf_size=64*1024, use_float=True)
            for record in parser:
                records_processed += 1
                if records_processed % (REPORTING_INTERVAL * 20) == 0: # Report overall progress less often
                     print(f"  ...scanned {records_processed} records...")

                # Basic check for required fields and type
                if not ("timestamp" in record and "x_position" in record and \
                        isinstance(record["timestamp"], list) and len(record["timestamp"]) > 1):
                    continue

                # Check direction *after* timestamp check
                if record.get("direction") != -1: # Westbound direction
                    continue

                # --- Time Filtering Logic ---
                # Check if the first timestamp is valid before accessing
                if not isinstance(record["timestamp"][0], (float, int)):
                    continue # Skip if first timestamp isn't usable
                record_start_timestamp = record["timestamp"][0] # Timestamps are Unix (UTC)

                # Compare record's UTC timestamp directly with calculated boundary UTC timestamps
                if record_start_timestamp < start_unix_timestamp:
                    continue # Skip trajectories starting before the window

                if record_start_timestamp > end_unix_timestamp:
                    # Optimization: If trajectories are roughly ordered by start time,
                    # we can stop scanning once we pass the end time boundary.
                    # This assumes the primary ordering in the JSON is time-based.
                    print(f"Trajectory start time ({record_start_timestamp:.2f}) exceeds end time ({end_unix_timestamp:.2f}). Stopping scan.")
                    break
                # --- End Time Filtering ---

                # If we reach here, the trajectory starts within the window. Process it.
                # Ensure the rest of the data is valid
                if len(record["timestamp"]) == len(record["x_position"]):
                    try:
                        # Convert to numpy arrays
                        # *** Assume x_position is in FEET ***
                        position_ft = np.array(record["x_position"], dtype=float)
                        timestamp_s = np.array(record["timestamp"], dtype=float)

                        # --- Calculations ---
                        # *** Convert FEET to MILES for the Y-axis ***
                        position_miles = position_ft / 5280.0

                        time_diff = np.diff(timestamp_s)
                        # *** Calculate position difference in FEET ***
                        pos_diff_ft = np.diff(position_ft)

                        # Calculate speed in feet per second (fps), then convert to mph
                        speed_fps = np.zeros_like(timestamp_s)
                        valid_diff = time_diff > 1e-6 # Avoid division by zero or tiny dt
                        calculated_speeds_fps = np.zeros_like(pos_diff_ft)
                        calculated_speeds_fps[valid_diff] = pos_diff_ft[valid_diff] / time_diff[valid_diff]

                        # Handle the first point's speed (backward fill)
                        speed_fps[1:] = calculated_speeds_fps
                        if len(speed_fps) > 1: speed_fps[0] = speed_fps[1]

                        # Convert fps to mph (1 fps = 3600/5280 mph ≈ 0.681818 mph)
                        speed_mph = np.abs(speed_fps * (3600.0 / 5280.0)) # Use the precise fraction

                        # --- Prepare data for LineCollection ---
                        # *** Use Unix timestamps (UTC) for X-axis, position_MILES for Y-axis ***
                        points = np.array([timestamp_s, position_miles]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)

                        # Average speed for the segment color
                        # Use speed_mph calculated for each point
                        segment_speeds = (speed_mph[:-1] + speed_mph[1:]) / 2.0

                        # --- Accumulate data ---
                        all_segments.extend(segments.tolist())
                        all_segment_speeds.extend(segment_speeds.tolist())
                        found_data_in_window = True

                        # *** Update actual data bounds using position_MILES ***
                        min_pos = min(min_pos, position_miles.min())
                        max_pos = max(max_pos, position_miles.max())
                        # min_time/max_time remain fixed by the filter window start/end

                        i += 1 # Increment processed trajectory count within window

                        if i % REPORTING_INTERVAL == 0:
                             print(f"Processed {i} valid trajectories within window...")

                    except (ValueError, TypeError, IndexError) as calc_e:
                        print(f"Warning: Error calculating data for record {records_processed}: {calc_e}. Skipping.")
                        continue
                    except Exception as calc_e:
                         print(f"Warning: Unexpected calculation error for record {records_processed}: {calc_e}. Skipping.")
                         continue
                # else: # Handle length mismatch if necessary
                #    print(f"Warning: Timestamp/Position length mismatch for record {records_processed}. Skipping.")

    except ijson.JSONError as e:
        print(f"Error parsing JSON file during main processing: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Local JSON file '{LOCAL_JSON_FILENAME}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during main processing loop: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    t_elapsed = time.time() - t_start
    print(f"Finished processing. Found and processed {i} trajectories within window from {records_processed} records scanned. Elapsed: {t_elapsed:.2f} seconds")

    # --- Create and Add LineCollection (after loop) ---
    if not found_data_in_window or not all_segments:
        print("No valid trajectories found within the specified time window. Cannot create plot.")
        # Still create a blank plot with the correct time range for context
        ax.set_xlim(start_unix_timestamp, end_unix_timestamp)
        # Try to set reasonable Y limits even if blank
        if np.isinf(min_pos) or np.isinf(max_pos):
            ax.set_ylim(0, 1) # Placeholder Y limits if no data at all
        else:
             pos_buffer = (max_pos - min_pos) * 0.05 if (max_pos - min_pos) > 0 else 0.1
             ax.set_ylim(min_pos - pos_buffer, max_pos + pos_buffer)

        ax.set_xlabel("Time (UTC Timestamp)")
        ax.set_ylabel("Mile marker")
        ax.invert_yaxis()
        ax.grid(True, color='white', alpha=0.2, linestyle='-', linewidth=0.5)
        # Format X ticks even for blank plot
        try:
            data_tz = zoneinfo.ZoneInfo(DATA_TIMEZONE) # Need TZ obj again for formatting
            locator = mticker.MaxNLocator(nbins=10, prune='both')
            ax.xaxis.set_major_locator(locator)
            ticks_loc = ax.get_xticks()
            ticks_loc = ticks_loc[(ticks_loc >= start_unix_timestamp) & (ticks_loc <= end_unix_timestamp)]

            if len(ticks_loc) > 0:
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                # Convert Unix ticks back to aware datetimes in the target timezone for labels
                x_datetime_aware = [datetime.datetime.fromtimestamp(ts, tz=data_tz) for ts in ticks_loc]
                labels = [d.strftime('%H:%M:%S') for d in x_datetime_aware] # Show local time H:M:S
                ax.set_xticklabels(labels, rotation=45, ha='right', color='white')
                ax.set_xlabel(f"Time") # Label axis with local timezone
            else:
                ax.tick_params(axis='x', labelcolor='white')
                ax.set_xlabel("Time") # Fallback label

        except Exception as tick_e:
             print(f"Warning: Could not format X-axis time ticks for blank plot: {tick_e}")
             ax.tick_params(axis='x', labelcolor='white')
             ax.set_xlabel("Time")

        print(f"Saving blank plot frame to '{OUTPUT_PLOT_FILENAME}'...")
        fig.savefig(OUTPUT_PLOT_FILENAME, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())

    else:
        print(f"Creating LineCollection for {len(all_segments)} segments...")
        lc = LineCollection(all_segments, cmap=custom_cmap, norm=norm, linewidths=LINE_WIDTH, zorder=2)
        lc.set_array(np.array(all_segment_speeds))
        ax.add_collection(lc)

        print("Finalizing plot...")

        # --- Set Axis Limits ---
        # Use the filter start/end Unix timestamps for X axis
        ax.set_xlim(start_unix_timestamp, end_unix_timestamp)
        # Use determined min/max pos for Y axis with a buffer
        if np.isfinite(min_pos) and np.isfinite(max_pos):
             pos_buffer = (max_pos - min_pos) * 0.05 # Buffer
             ax.set_ylim(min_pos - pos_buffer, max_pos + pos_buffer)
        else: # Fallback if something went wrong with pos calculation
             ax.set_ylim(0,1)
             print("Warning: Could not determine valid position range despite having data.")

        # --- Add Colorbar ---
        cbar = fig.colorbar(lc, cax=cax)
        cbar.set_label('Speed (mph)', rotation=270, labelpad=20, color='white')
        cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cbar.outline.set_edgecolor('white')
        cbar.outline.set_linewidth(1)

        # --- Axes Formatting ---
        ax.set_ylabel("Mile marker")

        # Format x-axis time ticks to show local time based on the DATA_TIMEZONE
        try:
            data_tz = zoneinfo.ZoneInfo(DATA_TIMEZONE) # Need TZ obj again for formatting
            locator = mticker.MaxNLocator(nbins=10, prune='both') # Suggest ~10 ticks
            ax.xaxis.set_major_locator(locator)
            ticks_loc = ax.get_xticks() # Get suggested tick locations (Unix timestamps)
            # Ensure ticks are within the actual plot range
            ticks_loc = ticks_loc[(ticks_loc >= start_unix_timestamp) & (ticks_loc <= end_unix_timestamp)]

            if len(ticks_loc) > 0:
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc)) # Use these specific locations
                # Convert Unix ticks back to aware datetimes in the target timezone for labels
                x_datetime_aware = [datetime.datetime.fromtimestamp(ts, tz=data_tz) for ts in ticks_loc]
                # Format labels to show local time H:M:S
                labels = [d.strftime('%H:%M:%S') for d in x_datetime_aware]
                ax.set_xticklabels(labels, rotation=45, ha='right', color='white')
                ax.set_xlabel(f"Time") # Label axis with local timezone
            else:
                 print("Warning: Could not determine valid X-axis ticks within the time window.")
                 ax.tick_params(axis='x', labelcolor='white')
                 ax.set_xlabel("Time") # Fallback label

        except Exception as tick_e:
             print(f"Warning: Could not format X-axis time ticks: {tick_e}")
             ax.tick_params(axis='x', labelcolor='white')
             ax.set_xlabel("Time") # Fallback label

        # Invert y-axis
        ax.invert_yaxis()

        # Add grid (behind lines)
        ax.grid(True, color='white', alpha=0.2, linestyle='-', linewidth=0.5, zorder=1)

        # --- Save Plot ---
        print(f"Saving plot to '{OUTPUT_PLOT_FILENAME}'...")
        fig.savefig(OUTPUT_PLOT_FILENAME, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print("Plot saved successfully.")

    plt.close(fig) # Close the figure to free memory

# Run the main function
if __name__ == "__main__":
    main()