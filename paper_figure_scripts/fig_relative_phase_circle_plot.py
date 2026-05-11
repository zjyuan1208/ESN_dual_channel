import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example peak timings
peak_timings1 = np.array([1, 4, 8, 12, 16, 20])
peak_timings2 = np.array([2, 5, 9, 13, 17, 21])
peak_timings3 = np.array([3, 7, 11, 15, 19, 23])


def calculate_basic_interval(peak_timings3, name_second):
    intervals = []
    for t in peak_timings3[2:]:
        closest_peak_idx = np.argmin(np.abs(peak_timings3 - t))
        if closest_peak_idx >= 2:
            if name_second == 2:
                interval = (peak_timings3[closest_peak_idx] - peak_timings3[closest_peak_idx - 2]) / 4
            elif name_second == 3:
                interval = (peak_timings3[closest_peak_idx] - peak_timings3[closest_peak_idx - 2]) / 6
            intervals.append(interval)
    return intervals


def update_peak_timings(peak_timings, name_first):
    updated_timings = peak_timings.copy()
    if name_first == 2:
        for i in range(3, len(peak_timings), 3):
            updated_timings[i] -= peak_timings[i - 3]
            updated_timings[i - 1] -= peak_timings[i - 3]
            updated_timings[i - 2] -= peak_timings[i - 3]
    elif name_first == 3:
        for i in range(2, len(peak_timings), 2):
            updated_timings[i] -= peak_timings[i - 2]
            updated_timings[i - 1] -= peak_timings[i - 2]
    return updated_timings


def plot_circle_seaborn(peak_timings1, peak_timings2, basic_interval):
    intervals1 = np.diff(peak_timings1)
    intervals2 = np.diff(peak_timings2)

    for i in range(len(intervals1)-1):
        basic_interval_value = basic_interval[i]
        phases1 = (intervals1 / (6 * basic_interval_value)) * 2 * np.pi
        phases2 = (intervals2 / (6 * basic_interval_value)) * 2 * np.pi

    data = {
        'Phases': np.concatenate([phases1, phases2]),
        'Intervals': np.concatenate([intervals1, intervals2]),
        'Source': ['Peak Timings 1'] * len(phases1) + ['Peak Timings 2'] * len(phases2)
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection='polar')
    sns.scatterplot(data=df, x='Phases', y='Intervals', hue='Source', ax=ax)
    ax.set_title('Circle Plot of Inter-beat Intervals')
    plt.show()


name_second = 2  # Example value
name_first = 2  # Example value

basic_interval = calculate_basic_interval(peak_timings3, name_second)
peak_timings1_updated = update_peak_timings(peak_timings1, name_first)
peak_timings2_updated = update_peak_timings(peak_timings2, name_first)

plot_circle_seaborn(peak_timings1_updated, peak_timings2_updated, basic_interval)


# def calculate_relative_phase_angles(taps, beats):
#     phase_angles = []
#
#     for tap in taps:
#         # Find the closest preceding beat
#         B_n_index = np.searchsorted(beats, tap) - 1
#         B_n = beats[B_n_index]
#
#         # Find the following beat
#         B_n1 = beats[B_n_index + 1]
#
#         # Calculate the relative phase angle
#         phi = 360 * (tap - B_n) / (B_n1 - B_n)
#         phase_angles.append(phi)
#
#     return np.array(phase_angles)
#
#
# def calculate_resultant_vector_length(relative_phase_angles):
#     N = len(relative_phase_angles)
#     complex_sum = np.sum(np.exp(1j * np.deg2rad(relative_phase_angles)))
#     R = np.abs(complex_sum / N)
#     return R
#
#
# def main(taps, metronome):
#     # Step 1: Calculate the relative phase angles
#     relative_phase_angles = calculate_relative_phase_angles(taps, metronome)
#
#     # Step 2: Calculate the Resultant Vector Length (R)
#     R = calculate_resultant_vector_length(relative_phase_angles)
#
#     return relative_phase_angles, R
#
#
# # Example usage
# taps = np.array([0.15, 0.35, 0.55, 0.75, 0.95])  # Participant's taps
# metronome = np.array([0.0, 0.5, 1.0])  # Metronome beats
#
# relative_phase_angles, R = main(taps, metronome)
# print(f"Relative Phase Angles: {relative_phase_angles}")
# print(f"Resultant Vector Length (R): {R}")
# exit()
#
# def plot_two_peaks_on_circle(peak_timings1, peak_timings2, basic_interval):
#     # Apply Seaborn styling
#     sns.set(style="whitegrid")
#
#     # Calculate normalization factor
#     normalization_factor = 6 * basic_interval
#
#     # Normalize peak timings for both series
#     normalized_timings1 = [t / normalization_factor for t in peak_timings1]
#     normalized_timings2 = [t / normalization_factor for t in peak_timings2]
#
#     # Convert normalized timings to radians for both series
#     angles1 = [2 * np.pi * t for t in normalized_timings1]
#     angles2 = [2 * np.pi * t for t in normalized_timings2]
#
#     # Calculate distances from the center for both series
#     distances1 = [peak_timings1[i] - peak_timings1[i - 1] for i in range(1, len(peak_timings1))]
#     distances2 = [peak_timings2[i] - peak_timings2[i - 1] for i in range(1, len(peak_timings2))]
#
#     # Skip the first element for both series
#     angles1 = angles1[1:]
#     angles2 = angles2[1:]
#
#     # Setup the polar plot
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#
#     # Add a padding factor to the radius to ensure outer points are visible
#     max_distance = max(max(distances1), max(distances2))
#     padding_factor = 1.1
#     ax.set_ylim(0, max_distance * padding_factor)
#
#     # Plot each peak as a point on the circle for both series
#     ax.plot(angles1, distances1, 'bo', label='Series 1')  # 'bo' indicates blue color and circle marker
#     ax.plot(angles2, distances2, 'ro', label='Series 2')  # 'ro' indicates red color and circle marker
#
#     # Plot the circle with radius related to 6 * basic_interval
#     circle = plt.Circle((0, 0), max_distance, transform=ax.transData._b, color='white', alpha=0.3)
#     ax.add_artist(circle)
#
#     # Add a legend
#     ax.legend(loc='upper right')
#
#     # Customize the angle labels to show radians
#     ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
#     ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
#
#     # Remove the grid for angular distances
#     ax.xaxis.set_tick_params(grid_linewidth=0)
#
#     # Customize the radial ticks for both series
#     ax.set_yticks([max_distance / 2, max_distance])  # Two divisions for the first series
#     ax.set_yticklabels([])  # Remove the radial labels
#     ax.yaxis.set_tick_params(colors='blue')  # Color for the first series
#
#     # Manually add the radial grid lines for the second series
#     for r in [max_distance / 3, 2 * max_distance / 3, max_distance]:
#         ax.plot([0, 2 * np.pi], [r, r], color='red', linestyle='--', linewidth=0.5)
#
#     # Show the plot
#     plt.show()
#
# # Example usage
# peak_timings1 = [0, 1, 2.5, 4, 7]  # Example peak timings for series 1
# peak_timings2 = [0, 1.5, 3, 5, 8]  # Example peak timings for series 2
# basic_interval = 1  # Example basic interval
#
# plot_two_peaks_on_circle(peak_timings1, peak_timings2, basic_interval)
#
# def plot_three_peaks_on_circle(peak_timings1, peak_timings2, peak_timings3, basic_interval):
#     # Apply Seaborn styling
#     sns.set(style="whitegrid")
#
#     # Calculate normalization factor
#     normalization_factor = 6 * basic_interval
#
#     # Normalize peak timings for all three series
#     normalized_timings1 = [t / normalization_factor for t in peak_timings1]
#     normalized_timings2 = [t / normalization_factor for t in peak_timings2]
#     normalized_timings3 = [t / normalization_factor for t in peak_timings3]
#
#     # Convert normalized timings to radians for all three series
#     angles1 = [2 * np.pi * t for t in normalized_timings1]
#     angles2 = [2 * np.pi * t for t in normalized_timings2]
#     angles3 = [2 * np.pi * t for t in normalized_timings3]
#
#     # Calculate distances from the center for all three series
#     distances1 = [peak_timings1[i] - peak_timings1[i - 1] for i in range(1, len(peak_timings1))]
#     distances2 = [peak_timings2[i] - peak_timings2[i - 1] for i in range(1, len(peak_timings2))]
#     distances3 = [peak_timings3[i] - peak_timings3[i - 1] for i in range(1, len(peak_timings3))]
#
#     # Skip the first element for both angles and distances
#     angles1 = angles1[1:]
#     angles2 = angles2[1:]
#     angles3 = angles3[1:]
#
#     # Setup the polar plot
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#
#     # Add a padding factor to the radius to ensure outer points are visible
#     max_distance = max(max(distances1), max(distances2), max(distances3))
#     padding_factor = 1.1
#     ax.set_ylim(0, max_distance * padding_factor)
#
#     # Plot each peak as a point on the circle for all three series
#     ax.plot(angles1, distances1, 'bo', label='Series 1')  # 'bo' indicates blue color and circle marker
#     ax.plot(angles2, distances2, 'ro', label='Series 2')  # 'ro' indicates red color and circle marker
#     ax.plot(angles3, distances3, 'go', label='Series 3')  # 'go' indicates green color and circle marker
#
#     # Plot the circle with radius related to 6 * basic_interval
#     circle = plt.Circle((0, 0), max_distance, transform=ax.transData._b, color='white', alpha=0.3)
#     ax.add_artist(circle)
#
#     # Add a legend
#     ax.legend(loc='upper right')
#
#     # Customize the angle labels to show radians
#     ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
#     ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
#
#     # Remove the grid for angular distances
#     ax.xaxis.set_tick_params(grid_linewidth=0)
#
#     # Customize the radial ticks for all three series
#     ax.set_yticks([max_distance / 2, max_distance])  # Two divisions for the first series
#     ax.set_yticklabels([])  # Remove the radial labels
#     ax.yaxis.set_tick_params(colors='blue')  # Color for the first series
#
#     # Manually add the radial grid lines for the second series
#     for r in [max_distance / 3, 2 * max_distance / 3, max_distance]:
#         ax.plot([0, 2 * np.pi], [r, r], color='red', linestyle='--', linewidth=0.5)
#
#     # Manually add the radial grid lines for the third series
#     for r in [max_distance / 4, max_distance / 2, 3 * max_distance / 4, max_distance]:
#         ax.plot([0, 2 * np.pi], [r, r], color='green', linestyle=':', linewidth=0.5)
#
#     # Show the plot
#     plt.show()
#
# # Example usage
# peak_timings1 = [0, 1, 2.5, 4, 7]  # Example peak timings for series 1
# peak_timings2 = [0, 1.5, 3, 5, 8]  # Example peak timings for series 2
# peak_timings3 = [0, 2, 4, 6, 9]    # Example peak timings for series 3
# basic_interval = 1  # Example basic interval
#
# plot_three_peaks_on_circle(peak_timings1, peak_timings2, peak_timings3, basic_interval)
#
#
# # import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def plot_two_peaks_on_circle(peak_timings1, peak_timings2, basic_interval):
#     # Apply Seaborn styling
#     sns.set(style="whitegrid")
#
#     # Calculate normalization factor
#     normalization_factor = 6 * basic_interval
#
#     # Normalize peak timings for both series
#     normalized_timings1 = [t / normalization_factor for t in peak_timings1]
#     normalized_timings2 = [t / normalization_factor for t in peak_timings2]
#
#     # Convert normalized timings to radians for both series
#     angles1 = [2 * np.pi * t for t in normalized_timings1]
#     angles2 = [2 * np.pi * t for t in normalized_timings2]
#
#     # Calculate distances from the center for both series
#     distances1 = [peak_timings1[i] - peak_timings1[i - 1] for i in range(1, len(peak_timings1))]
#     distances2 = [peak_timings2[i] - peak_timings2[i - 1] for i in range(1, len(peak_timings2))]
#
#     # Skip the first element for both series
#     angles1 = angles1[1:]
#     angles2 = angles2[1:]
#
#     # Setup the polar plot
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#
#     # Add a padding factor to the radius to ensure outer points are visible
#     max_distance = max(max(distances1), max(distances2))
#     padding_factor = 1.1
#     ax.set_ylim(0, max_distance * padding_factor)
#
#     # Plot each peak as a point on the circle for both series
#     ax.plot(angles1, distances1, 'bo', label='Series 1')  # 'bo' indicates blue color and circle marker
#     ax.plot(angles2, distances2, 'ro', label='Series 2')  # 'ro' indicates red color and circle marker
#
#     # Plot the circle with radius related to 6 * basic_interval
#     circle = plt.Circle((0, 0), max_distance, transform=ax.transData._b, color='white', alpha=0.3)
#     ax.add_artist(circle)
#
#     # Add a legend
#     ax.legend(loc='upper right')
#
#     # Customize the angle labels to show radians
#     ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
#     ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
#
#     # Remove the grid for angular distances
#     ax.xaxis.set_tick_params(grid_linewidth=0)
#
#     # Customize the radial ticks for both series
#     ax.set_yticks([max_distance / 2, max_distance])  # Two divisions for the first series
#     ax.yaxis.set_tick_params(colors='blue')  # Color for the first series
#
#     # Manually add the radial grid lines and labels for the second series
#     for r in [max_distance / 3, 2 * max_distance / 3, max_distance]:
#         ax.plot([0, 2 * np.pi], [r, r], color='red', linestyle='--', linewidth=0.5)
#         ax.text(np.pi / 2, r, f'{r:.2f}', color='red', fontsize=10, ha='center', va='bottom')
#
#     # Show the plot
#     plt.show()
#
# # Example usage
# peak_timings1 = [0, 1, 2.5, 4, 7]  # Example peak timings for series 1
# peak_timings2 = [0, 1.5, 3, 5, 8]  # Example peak timings for series 2
# basic_interval = 1  # Example basic interval
#
# plot_two_peaks_on_circle(peak_timings1, peak_timings2, basic_interval)