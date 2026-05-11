import numpy as np
import torch
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import wasserstein_distance


save_path = '/home/zhyuan/Desktop/ESN/results_human_behavior/paper_figures_2'
save_state = False

### human computer interact: after learning in the first 30s, the output weight is not going to change anymore
### fb * 2 for 02_03

def extract_and_calculate_time_interval(file_path):
    # Extract numbers from the file name using regular expressions
    match = re.search(r'TAP_(\d+)_(\d+)_([A-Za-z\d]+)?_PERF\.txt', file_path)

    if match:
        # Extract the two numbers from the match
        first_number = int(match.group(1))
        second_number = int(match.group(2))

        # Define time intervals based on the second number
        if second_number == 3:
            time_interval = 533
        elif second_number == 2:
            time_interval = 800
        else:
            # Handle other cases if needed
            time_interval = None

        return first_number, second_number, time_interval
    else:
        # Handle the case when the file name doesn't match the expected pattern
        return None


def process_human_data(file_path, downsample_factor=6):
    data = np.loadtxt(file_path, delimiter='\t', dtype=str)

    # Separate data for Var2 values 1 and 2
    data_1 = data[data[:, 1] == '1']
    data_2 = data[data[:, 1] == '2']

    # y_values = np.diff(data_1[1:, 0].astype(float))

    data_1 = data_1[:, 0].astype(float) * 1000 / downsample_factor  # Convert seconds to milliseconds
    data_2 = data_2[:, 0].astype(float) * 1000 / downsample_factor  # Convert seconds to milliseconds

    if max(data_1) > max(data_2):
        total_length = int(max(data_1)) + 100
    else:
        total_length = int(max(data_2)) + 100


    '''Generate the input data'''
    # Create a tensor with zeros
    input_channel_1 = np.zeros(total_length)

    # Set amplitudes at given time stamps to 80
    input_channel_1[np.round(data_1).astype(int)] = 80

    # Fixed parameters
    # first_number: person 2 or a pc
    # second_number: person 1
    first_number, second_number, time_interval = extract_and_calculate_time_interval(file_path)
    time_interval = time_interval / downsample_factor

    # Determine start time from data2
    start_time = data_2[0]

    # Generate binary time series in the first channel
    input_channel_0 = np.zeros((total_length, 1))
    current_time = start_time
    while current_time < 30000 / downsample_factor:
        index = int((current_time))
        input_channel_0[index] = 80
        current_time += time_interval

    # Create input_data
    input_channel_0 = torch.tensor(input_channel_0).float()
    input_channel_1 = torch.tensor(input_channel_1).float()
    input_data = torch.cat([input_channel_0, input_channel_1.unsqueeze(-1)], dim=-1)

    # print(input_data.size())
    # plt.figure(figsize=(10, 6))
    # plt.plot(input_channel_0[:5000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(input_channel_1[:5000], 'b', label=f'{0}_feat1_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    '''Generate the target stick data'''

    # Generate binary time series in the first channel of target_stick
    cut_length = int(200 / downsample_factor)

    target_stick_channel_0 = np.zeros((total_length, 1))
    current_time = start_time - cut_length
    while current_time < total_length:
        index = int((current_time))
        target_stick_channel_0[index, 0] = 80
        current_time += time_interval

    # Generate binary time series in the second channel of target_stick
    cut_tensor = input_channel_1[cut_length:]  # Cut the first 200 values
    padding_tensor = torch.zeros((cut_length))  # Pad zeros after cutting
    target_stick_channel_1 = torch.cat((cut_tensor, padding_tensor), dim=0)
    target_stick_channel_0 = torch.tensor(target_stick_channel_0).float()
    target_stick_data = torch.cat([target_stick_channel_0, target_stick_channel_1.unsqueeze(1)], dim=-1)
    # print(target_stick_data.size())

    # plt.figure(figsize=(10, 6))
    # plt.plot(target_stick_data[:5000, 0], 'r', label=f'{0}_feat0_target')
    # plt.plot(input_channel_0[:5000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(target_stick_data[:5000, 1], 'r', label=f'{0}_feat1_target')
    # plt.plot(input_channel_1[:5000], 'b', label=f'{0}_feat1_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    '''Generate the target affected data'''
    target_affected_channel_0 = target_stick_channel_0
    target_affected_channel_0[int(30000 / downsample_factor):] = target_stick_channel_1[int(30000 / downsample_factor):].unsqueeze(1)
    target_affected_data = target_stick_data.clone()
    target_affected_data[:, 0] = target_affected_channel_0.squeeze(1)
    # print(target_affected_data.size())

    # plt.figure(figsize=(10, 6))
    # plt.plot(target_affected_data[:5000, 0], 'r', label=f'{0}_feat0_target')
    # plt.plot(input_channel_0[:5000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(target_affected_data[:5000, 1], 'r', label=f'{0}_feat1_target')
    # plt.plot(input_channel_1[:5000], 'b', label=f'{0}_feat1_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    '''Generate the final output target for the first channel'''
    # Generate binary time series in the first channel
    # Create a tensor with zeros
    target_channel_2 = np.zeros(total_length)

    # Set amplitudes at given time stamps to 80
    target_channel_2[np.round(data_2).astype(int)] = 80

    final_target = torch.tensor(target_channel_2).float()
    cut_tensor = final_target[cut_length:]  # Cut the first 200 values
    padding_tensor = torch.zeros((cut_length))  # Pad zeros after cutting
    final_target = torch.cat((cut_tensor, padding_tensor), dim=0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(final_target[:10000], 'r', label=f'{0}_feat0_target')
    # plt.plot(input_channel_0[:10000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    return input_data, target_stick_data, target_affected_data, final_target

def process_human_data_2(file_path, downsample_factor=6):
    data = np.loadtxt(file_path, delimiter='\t', dtype=str)

    # Separate data for Var2 values 1 and 2
    data_1 = data[data[:, 1] == '1']
    data_2 = data[data[:, 1] == '2']

    # y_values = np.diff(data_1[1:, 0].astype(float))

    data_1 = data_1[:, 0].astype(float) * 1000 / downsample_factor  # Convert seconds to milliseconds
    data_2 = data_2[:, 0].astype(float) * 1000 / downsample_factor  # Convert seconds to milliseconds

    if max(data_1) > max(data_2):
        total_length = int(max(data_1)) + 100
    else:
        total_length = int(max(data_2)) + 100


    '''Generate the input data'''
    # Create a tensor with zeros
    input_channel_1 = np.zeros(total_length)

    # Set amplitudes at given time stamps to 80
    input_channel_1[np.round(data_1).astype(int)] = 80

    # Fixed parameters
    # first_number: person 2 or a pc
    # second_number: person 1
    first_number, second_number, time_interval = extract_and_calculate_time_interval(file_path)
    time_interval = time_interval / downsample_factor

    # Determine start time from data2
    start_time = data_2[0]

    # Generate binary time series in the first channel
    input_channel_0 = np.zeros((total_length, 1))
    current_time = start_time
    while current_time < 30000 / downsample_factor:
        index = int((current_time))
        input_channel_0[index] = 80
        current_time += time_interval

    # Create input_data
    input_channel_0 = torch.tensor(input_channel_0).float()
    input_channel_1 = torch.tensor(input_channel_1).float()
    input_data = torch.cat([input_channel_0, input_channel_1.unsqueeze(-1)], dim=-1)

    # print(input_data.size())
    # plt.figure(figsize=(10, 6))
    # plt.plot(input_channel_0[:5000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(input_channel_1[:5000], 'b', label=f'{0}_feat1_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    '''Generate the target stick data'''

    # Generate binary time series in the first channel of target_stick
    cut_length = int(200 / downsample_factor)

    target_stick_channel_0 = np.zeros((total_length, 1))
    current_time = start_time - cut_length
    while current_time < total_length:
        index = int((current_time))
        target_stick_channel_0[index, 0] = 80
        current_time += time_interval

    # Generate binary time series in the second channel of target_stick
    cut_tensor = input_channel_1[cut_length:]  # Cut the first 200 values
    padding_tensor = torch.zeros((cut_length))  # Pad zeros after cutting
    target_stick_channel_1 = torch.cat((cut_tensor, padding_tensor), dim=0)
    target_stick_channel_0 = torch.tensor(target_stick_channel_0).float()
    target_stick_data = torch.cat([target_stick_channel_0, target_stick_channel_1.unsqueeze(1)], dim=-1)
    # print(target_stick_data.size())

    # plt.figure(figsize=(10, 6))
    # plt.plot(target_stick_data[:5000, 0], 'r', label=f'{0}_feat0_target')
    # plt.plot(input_channel_0[:5000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(target_stick_data[:5000, 1], 'r', label=f'{0}_feat1_target')
    # plt.plot(input_channel_1[:5000], 'b', label=f'{0}_feat1_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    '''Generate the target affected data'''
    target_affected_channel_0 = target_stick_channel_0
    target_affected_channel_0[int(30000 / downsample_factor):] = target_stick_channel_1[int(30000 / downsample_factor):].unsqueeze(1)
    target_affected_data = target_stick_data.clone()
    target_affected_data[:, 0] = target_affected_channel_0.squeeze(1)
    # print(target_affected_data.size())

    # plt.figure(figsize=(10, 6))
    # plt.plot(target_affected_data[:5000, 0], 'r', label=f'{0}_feat0_target')
    # plt.plot(input_channel_0[:5000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(target_affected_data[:5000, 1], 'r', label=f'{0}_feat1_target')
    # plt.plot(input_channel_1[:5000], 'b', label=f'{0}_feat1_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    '''Generate the final output target for the first channel'''
    # Generate binary time series in the first channel
    # Create a tensor with zeros
    target_channel_2 = np.zeros(total_length)

    # Set amplitudes at given time stamps to 80
    target_channel_2[np.round(data_2).astype(int)] = 80

    final_target = torch.tensor(target_channel_2).float()
    cut_tensor = final_target[cut_length:]  # Cut the first 200 values
    padding_tensor = torch.zeros((cut_length))  # Pad zeros after cutting
    final_target = torch.cat((cut_tensor, padding_tensor), dim=0)

    target_channel_1 = np.zeros(total_length)

    # Set amplitudes at given time stamps to 80
    target_channel_1[np.round(data_1).astype(int)] = 80

    final_target_2 = torch.tensor(target_channel_1).float()
    cut_tensor = final_target_2[cut_length:]  # Cut the first 200 values
    padding_tensor = torch.zeros((cut_length))  # Pad zeros after cutting
    final_target_2 = torch.cat((cut_tensor, padding_tensor), dim=0)

    # plt.figure(figsize=(10, 6))
    # plt.plot(final_target[:10000], 'r', label=f'{0}_feat0_target')
    # plt.plot(input_channel_0[:10000], 'b', label=f'{0}_feat0_input')
    # plt.legend()
    # plt.title(f'{file_path[-33:-4]}')
    # plt.show()
    # exit()

    return input_data, target_stick_data, target_affected_data, final_target, final_target_2

def filter_peaks(peaks, min_distance=20):
    filtered_peaks = []
    current_peak = None

    for peak in peaks:
        if current_peak is None or peak - current_peak > min_distance:
            filtered_peaks.append(peak)
            current_peak = peak

    return np.array(filtered_peaks)


def kl_divergence(p, q):
    return np.sum(p * np.log(p / q))


def jensen_shannon_divergence(p, q):
    # Calculate the average distribution
    m = 0.5 * (p + q)

    # Calculate KL divergence from P to M and from Q to M
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)

    # Calculate Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd

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



def plot_two_peaks_on_circle(ax, peak_timings1, peak_timings2, basic_interval, label1, label2):
    # Apply Seaborn styling
    sns.set(style="whitegrid")

    # Calculate normalization factor
    normalization_factor = 6 * basic_interval

    peak_timings1 = peak_timings1 - peak_timings1[0]
    peak_timings2 = peak_timings2 - peak_timings2[0]

    # Normalize peak timings for both series
    normalized_timings1 = [t % normalization_factor / normalization_factor for t in peak_timings1]
    normalized_timings2 = [t % normalization_factor / normalization_factor for t in peak_timings2]

    # Convert normalized timings to radians for both series
    angles1 = [2 * np.pi * t for t in normalized_timings1]
    angles2 = [2 * np.pi * t for t in normalized_timings2]

    # Calculate distances from the center for both series
    distances1 = [peak_timings1[i] - peak_timings1[i - 1] for i in range(1, len(peak_timings1))]
    distances2 = [peak_timings2[i] - peak_timings2[i - 1] for i in range(1, len(peak_timings2))]

    # Skip the first element for both series
    angles1 = angles1[1:]
    angles2 = angles2[1:]


    # # Define circles with radii 533 and 800
    # circles = [533/6, 800/6]
    # for radius in circles:
    #     circle = plt.Circle((0, 0), radius, transform=ax.transData._b, color='white', alpha=0.3)
    #     ax.add_artist(circle)


    # Plot each peak as a point on the circle for both series
    lines1, = ax.plot(angles2, distances2, 'o', label=label2, markerfacecolor='None', markeredgecolor=reds[0], alpha=0.5)
    lines2, = ax.plot(angles1, distances1, 'o', label=label1, markerfacecolor='None', markeredgecolor=blues[0], alpha=0.8)


    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 0]
    ax.legend([lines2, lines1], [labels[idx] for idx in order], loc='upper right')


    # Customize the angle labels to show radians
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])

    # # Remove the grid for angular distances
    # ax.xaxis.set_tick_params(grid_linewidth=0)

    # # Customize the radial ticks for both series
    # ax.set_yticks(circles)  # Two divisions for the first series
    # ax.set_yticklabels([])  # Remove the radial labels
    # ax.yaxis.set_tick_params(colors='blue')  # Color for the first series

    # circles = [533/6, 800/6]
    # # for r in [max_distance / 3, 2 * max_distance / 3, max_distance]:
    # for r in circles:
    #     ax.plot([0, 2 * np.pi], [r, r], color='black', linestyle='--', linewidth=0.5)
    #     ax.text(np.pi / 2, r, f'{r*6:.0f}', color='black', fontsize=10, ha='center', va='bottom')


def plot_three_peaks_on_circle(peak_timings1, peak_timings2, peak_timings3, basic_interval):
    # Apply Seaborn styling
    sns.set(style="whitegrid")

    # Calculate normalization factor
    normalization_factor = 6 * basic_interval

    # Normalize peak timings for all three series
    normalized_timings1 = [t % normalization_factor / normalization_factor for t in peak_timings1]
    normalized_timings2 = [t % normalization_factor / normalization_factor for t in peak_timings2]
    normalized_timings3 = [t % normalization_factor / normalization_factor for t in peak_timings3]

    # Convert normalized timings to radians for all three series
    angles1 = [2 * np.pi * t for t in normalized_timings1]
    angles2 = [2 * np.pi * t for t in normalized_timings2]
    angles3 = [2 * np.pi * t for t in normalized_timings3]

    # Calculate distances from the center for all three series
    distances1 = [peak_timings1[i] - peak_timings1[i - 1] for i in range(1, len(peak_timings1))]
    distances2 = [peak_timings2[i] - peak_timings2[i - 1] for i in range(1, len(peak_timings2))]
    distances3 = [peak_timings3[i] - peak_timings3[i - 1] for i in range(1, len(peak_timings3))]

    # Skip the first element for both angles and distances
    angles1 = angles1[1:]
    angles2 = angles2[1:]
    angles3 = angles3[1:]

    # Setup the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Add a padding factor to the radius to ensure outer points are visible
    max_distance = max(max(distances1), max(distances2), max(distances3))
    padding_factor = 1.1
    ax.set_ylim(0, max_distance * padding_factor)

    # Plot each peak as a point on the circle for all three series
    ax.plot(angles1, distances1, 'bo', label='Series 1')  # 'bo' indicates blue color and circle marker
    ax.plot(angles2, distances2, 'ro', label='Series 2')  # 'ro' indicates red color and circle marker
    ax.plot(angles3, distances3, 'go', label='Series 3')  # 'go' indicates green color and circle marker

    # Plot the circle with radius related to 6 * basic_interval
    circle = plt.Circle((0, 0), max_distance, transform=ax.transData._b, color='white', alpha=0.3)
    ax.add_artist(circle)

    # Add a legend
    ax.legend(loc='upper right')

    # Customize the angle labels to show radians
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])

    # Remove the grid for angular distances
    ax.xaxis.set_tick_params(grid_linewidth=0)

    # Customize the radial ticks for all three series
    ax.set_yticks([max_distance / 2, max_distance])  # Two divisions for the first series
    ax.set_yticklabels([])  # Remove the radial labels
    ax.yaxis.set_tick_params(colors='blue')  # Color for the first series

    # Manually add the radial grid lines for the second series
    for r in [max_distance / 3, 2 * max_distance / 3, max_distance]:
        ax.plot([0, 2 * np.pi], [r, r], color='red', linestyle='--', linewidth=0.5)

    # Manually add the radial grid lines for the third series
    for r in [max_distance / 4, max_distance / 2, 3 * max_distance / 4, max_distance]:
        ax.plot([0, 2 * np.pi], [r, r], color='green', linestyle=':', linewidth=0.5)

    # Show the plot
    plt.show()

# Define colors
blues = ['#115699', '#0E6DB3', '#5CAAD7', '#95C6DE']
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']


samples = {
    '2_2': {
        'file_paths': [
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_2_2_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_2_2_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_2_2_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_2_2_PC_PERF.txt',
        ],
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/human_computer_interact_model_performance_2024',
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/wdist_lr_update_both/human_human_interact_model_performance_2024',
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_computer/human_computer_interact_model_performance_2024',
    },
    '3_2': {
        'file_paths': [
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_3_2_PC_PERF.txt',
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_3_2_PC_PERF.txt',
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_3_2_PC_PERF.txt',
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_3_2_PC_PERF.txt',
        ],
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/human_computer_interact_model_performance_2024',
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/wdist_lr_update_both/human_human_interact_model_performance_2024',
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_computer/human_computer_interact_model_performance_2024',
    },
    '2_3': {
        'file_paths': [
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_2_3_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_2_3_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_2_3_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_2_3_PC_PERF.txt',
        ],
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/human_computer_interact_model_performance_2024',
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/wdist_lr_update_both/human_human_interact_model_performance_2024',
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_computer/human_computer_interact_model_performance_2024',
    },
    '3_3': {
        'file_paths': [
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_3_3_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_3_3_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_3_3_PC_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_3_3_PC_PERF.txt',
        ],
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/human_computer_interact_model_performance_2024',
        # 'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/human_computer/wdist_lr_update_both/human_human_interact_model_performance_2024',
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_computer/human_computer_interact_model_performance_2024',
    }
}

# for sample_key, sample in samples.items():
#     plt.figure(figsize=(10, 6))
#
#     human_peak_lst = [find_peaks(process_human_data(file_path)[-1])[0].tolist() for file_path in
#                          sample['file_paths']]
#     model_peak_lst = []
#
#     time_interval_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
#                          sample['file_paths']]
#     model_performance_lst = []
#     dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']
#     for i, date in enumerate(dates):
#         file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'
#         with open(file_path_json, 'r') as json_file:
#             loaded_data = json.load(json_file)
#             p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=10)[0]
#             p = filter_peaks(p, min_distance=40)
#             model_performance = np.diff(p).tolist()
#             model_peak_lst.append(p)
#
#             model_performance_lst.append(model_performance)
#
#     # for i in range(len(human_peak_lst)):
#     #     distance = wasserstein_distance(human_peak_lst[i], model_peak_lst[i])
#     #
#     #     print(f"The Wasserstein distance for {dates[i]}_TAP_{sample_key} is: {distance}")
#
#     first_value = int(sample_key.split('_')[1])
#     second_value = int(sample_key.split('_')[0])
#     if first_value == 2:
#         dashed_line_x = 800 / 6
#     elif first_value == 3:
#         dashed_line_x = 533 / 6
#
#
#     for i, data in enumerate(model_performance_lst):
#         sns.kdeplot(data, label=f'{first_value}_{second_value}_model_{i+1}', color=blues[i])
#
#     for i, data in enumerate(time_interval_lst):
#         sns.kdeplot(data, label=f'{first_value}_{second_value}_participant_{i+1}', color=reds[i])
#
#
#     # Add vertical dashed line
#     plt.axvline(x=dashed_line_x, color='gray', linestyle='--')
#
#     # Add labels and legend
#     plt.xlabel('Inter-Beat Interval (time steps)')
#     plt.ylabel('Probability Density')
#     plt.legend()
#     if save_state:
#         plt.savefig(f'{save_path}/human_computer_distribution_compare_{first_value}_{second_value}.png', dpi=300)
#
#     # Show the plot
#     plt.show()

def closest_prediction(peak_times, reference_time, closest_distance=401/6, low=20/6, high=400/6):
    """Find the peak closest to reference_time within the 20-400 ms window."""
    closest_peak = None
    # closest_distance = 401/6  # Initialize above the maximum allowed distance.
    for peak_time in peak_times:
        distance = reference_time - peak_time
        # if low <= distance <= high and np.abs(distance) < closest_distance:
        if low <= distance <= high:
            closest_peak = peak_time
            # closest_distance = distance
    return closest_peak


def generate_timing_list_until_end(start_time, interval, end_time):
    """Generate a timing list up to a specific end time.

    Args:
        start_time (int or float): Start time.
        interval (int or float): Time interval.
        end_time (int or float): End time.

    Returns:
        list: List containing the generated timings.
    """
    # Compute the number of elements without exceeding the end time.
    num_elements = int((end_time - start_time) / interval)
    # Generate and return the timing list.
    return [start_time + i * interval for i in range(num_elements + 1) if start_time + i * interval <= end_time]


def calculate_f1_score(TP, FP, TN, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def classify_beats(model_time_lst, reference_time_lst, window_start=-200 / 6, window_end=250 / 6):
    TP, FP, TN, FN = 0, 0, 0, 0

    # Generate time windows for each reference beat
    time_windows = [(beat + window_start, beat + window_end, beat) for beat in reference_time_lst]

    # Sort lists for efficient iteration
    model_time_lst.sort()
    time_windows.sort(key=lambda x: x[0])  # Sort by window start time

    # Track model beats that have been matched to a reference beat
    matched_model_beats = set()

    # Check for TP: Model beats closest to the center of the reference time windows
    for start, end, center in time_windows:
        closest_beat = None
        closest_distance = float('inf')
        for model_beat in model_time_lst:
            if start <= model_beat <= end:
                distance = abs(model_beat - center)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_beat = model_beat
        if closest_beat is not None:
            TP += 1
            matched_model_beats.add(closest_beat)

    # All model beats within a window are considered matched, others are TN (outside any window)
    TN = len([beat for beat in model_time_lst if beat not in matched_model_beats])

    # FN: Reference windows without any model beats marked as closest
    FN = len(reference_time_lst) - TP

    # FP: Incorrectly identified beats; in this context, it's the total model beats - matched model beats
    # FP = len(model_time_lst) - len(matched_model_beats)
    FP = len(reference_time_lst) - len(matched_model_beats)

    return TP, TN, FP, FN

def calculate_relative_phase_angles(taps, beats):
    phase_angles = []

    for tap in taps:
        # Find the closest beat
        B_n_index = np.argmin(np.abs(beats - tap))
        B_n = beats[B_n_index]

        # Find the following beat
        if B_n_index + 1 < len(beats):
            B_n1 = beats[B_n_index + 1]
        else:
            # If B_n is the last beat, wrap around to the first beat
            B_n1 = beats[0]

        # Calculate the relative phase angle
        if B_n1 != B_n:
            phi = 360 * (tap - B_n) / (B_n1 - B_n)
        else:
            # If B_n and B_n1 are the same, avoid division by zero
            phi = 0
        phase_angles.append(phi)

    return np.array(phase_angles)



def calculate_resultant_vector_length(relative_phase_angles):
    N = len(relative_phase_angles)
    complex_sum = np.sum(np.exp(1j * np.deg2rad(relative_phase_angles)))
    R = np.abs(complex_sum / N)
    return R


def R_score(taps, metronome):
    # Step 1: Calculate the relative phase angles
    relative_phase_angles = calculate_relative_phase_angles(taps, metronome)

    # Step 2: Calculate the Resultant Vector Length (R)
    R = calculate_resultant_vector_length(relative_phase_angles)

    return relative_phase_angles, R


# # # Example data
# model_time_lst = [1, 3, 5, 7, 9, 11, 13, 15]  # Example model detected beats
# reference_time_lst = [2, 4, 6, 8, 12]  # Example reference beats
#
# TP, TN, FP, FN = classify_beats(model_time_lst, reference_time_lst)
# print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
#
# exit()

f1_score_seperate = False
FFT_plot = False
matching_rate = False
f1_score = False
interval_distribution = False
circle_plot = True
R_score_state = False

if R_score_state:
    fig, axs = plt.subplots(4, 4, subplot_kw={'projection': 'polar'}, figsize=(20, 20))
    for i, (sample_key, sample) in enumerate(samples.items()):
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']
        for j, date in enumerate(dates):
            ax = axs[i, j]
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'

            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0])[0]
                p = filter_peaks(p, min_distance=45)

                first_value, second_value = map(int, sample_key.split('_'))
                reference_time_interval = 800 / 6 if second_value == 2 else 533 / 6

                real_time_lst = find_peaks(process_human_data(sample['file_paths'][j])[-1])[0]
                model_time_lst = p[4:]
                basic_interval = 800 / 6 / 3  # Example basic interval

                relative_phase_angles, R = R_score(model_time_lst, real_time_lst[3:])
                # print(relative_phase_angles)
                print(date, sample_key)
                print(R)
                relative_phase_angles, R = R_score(model_time_lst, real_time_lst[3:])
                print(R)

if circle_plot:
    # for sample_key, sample in samples.items():
    #     dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']
    #
    #     for i, date in enumerate(dates):
    #         file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'
    #
    #
    #         with open(file_path_json, 'r') as json_file:
    #             loaded_data = json.load(json_file)
    #             p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=0)[0]
    #             p = filter_peaks(p, min_distance=45)
    #
    #
    #             first_value, second_value = map(int, sample_key.split('_'))
    #             reference_time_interval = 800/6 if second_value == 2 else 533/6
    #
    #             real_time_lst = find_peaks(process_human_data(sample['file_paths'][i])[-1])[0]
    #             real_time_lst = real_time_lst
    #             model_time_lst = p[4:]
    #             basic_interval = 800/6/3  # Example basic interval
    #             # plot_two_peaks_on_circle(model_time_lst, real_time_lst, basic_interval)
    #             # Create figure with 4x4 subplots
    #             fig, axs = plt.subplots(4, 4, subplot_kw={'projection': 'polar'}, figsize=(20, 20))
    fig, axs = plt.subplots(4, 4, subplot_kw={'projection': 'polar'}, figsize=(20, 20))
    for i, (sample_key, sample) in enumerate(samples.items()):
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']
        for j, date in enumerate(dates):
            ax = axs[i, j]
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'

            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=10)[0]
                p = filter_peaks(p, min_distance=45)

                first_value, second_value = map(int, sample_key.split('_'))
                reference_time_interval = 800 / 6 if second_value == 2 else 533 / 6

                real_time_lst = find_peaks(process_human_data(sample['file_paths'][j])[-1])[0]
                model_time_lst = p[4:]
                basic_interval = 800 / 3  # Example basic interval

                if first_value == 2 and second_value == 2:
                    name_first = 3
                    name_second = 3
                elif first_value == 2 and second_value == 3:
                    name_first = 2
                    name_second = 3
                elif first_value == 3 and second_value == 2:
                    name_first = 3
                    name_second = 2
                else:
                    name_first = 2
                    name_second = 2

                model_time_lst = (np.array(model_time_lst) * 6)
                real_time_lst = (np.array(real_time_lst) * 6)

                plot_two_peaks_on_circle(ax, model_time_lst, real_time_lst, basic_interval,
                                         label1=f'{name_first}:{name_second} model',
                                         label2=f'{name_first}:{name_second} participant')

    plt.tight_layout()

    # Save the figure if required
    if save_state:
        plt.savefig(f'{save_path}/human_computer_dots_compare.png', dpi=300)
    plt.show()


''' F1 score corresponding to the reference reserpectively'''
if f1_score_seperate:
    for sample_key, sample in samples.items():
        time_interval_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
                             sample['file_paths']]
        # time_interval_2_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
        #                      sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'


            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=0)[0]
                p = filter_peaks(p, min_distance=45)


                first_value, second_value = map(int, sample_key.split('_'))
                reference_time_interval = 800/6 if second_value == 2 else 533/6

                real_time_lst = find_peaks(process_human_data(sample['file_paths'][i])[-1])[0]
                # real_time_lst = [x + 200 / 6 for x in real_time_lst]
                reference_time_lst = generate_timing_list_until_end(start_time=real_time_lst[0],
                                                                    interval=reference_time_interval,
                                                                    end_time=real_time_lst[-1])
                real_time_lst = real_time_lst
                reference_time_lst = reference_time_lst[10:]
                model_time_lst = p

                ### human results
                # TP, TN, FP, FN = classify_beats(real_time_lst, reference_time_lst, window_start=-150 / 6, window_end=200 / 6)
                # print('group', i+1)
                # print(sample_key)
                # # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
                #
                # print('precision', calculate_f1_score(TP, FP, TN, FN)[0])
                # print('recall', calculate_f1_score(TP, FP, TN, FN)[1])
                # print('f1-score', calculate_f1_score(TP, FP, TN, FN)[2])


                ### model results
                TP, TN, FP, FN = classify_beats(model_time_lst, reference_time_lst, window_start=-100 / 6, window_end=250 / 6)
                print('group', i+1)
                print(sample_key)
                # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

                print('precision', calculate_f1_score(TP, FP, TN, FN)[0])
                print('recall', calculate_f1_score(TP, FP, TN, FN)[1])
                print('f1-score', calculate_f1_score(TP, FP, TN, FN)[2])




''' FFT on human and model'''
if FFT_plot:
    plt.figure(figsize=(20, 20))  # Adjusted for 4x4 grid of subfigures
    # fig, axs = plt.subplots(4, 4, figsize=(20, 20))


    sample_keys_sorted = sorted(samples.keys(), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[0])))
    subplot_index = 1  # Start with the first subplot

    for sample_key in sample_keys_sorted:
        sample = samples[sample_key]
        time_interval_lst = [process_human_data_2(file_path)[-2] for file_path in sample['file_paths']]
        time_interval_2_lst = [process_human_data_2(file_path)[-1] for file_path in sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'

            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                model_signal = torch.FloatTensor(loaded_data['predicted'])[0, :, 0]
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0])[0]
                p = filter_peaks(p, min_distance=45)
                total_length = len(model_signal) + 100

                '''Generate the input data'''
                # Create a tensor with zeros
                model_signal = np.zeros(total_length)

                # Set amplitudes at given time stamps to 80
                model_signal[np.round(p).astype(int)] = 80
                # model_performance = (np.diff(p)*6).tolist()
                model_performance_lst.append(model_signal)

        first_value, second_value = map(int, sample_key.split('_'))

        # # Assuming the calculation for dashed_line_x and dashed_line_x_2 is correct and needed
        # dashed_line_x = 800 if first_value == 2 else 533
        # dashed_line_x_2 = 800 if second_value == 2 else 533

        # Frequency threshold for noise reduction (in Hz)
        threshold_freq = 10 # Example threshold; adjust based on your data

        for j, (model_data, participant_data, participant_data_2) in enumerate(
                zip(model_performance_lst, time_interval_lst, time_interval_2_lst)):
            # Calculate FFT for each dataset
            fft_model = np.fft.fft(model_data)
            fft_participant_1 = np.fft.fft(participant_data)
            fft_participant_2 = np.fft.fft(participant_data_2)

            # Calculate frequencies for each dataset
            freqs_model = np.fft.fftfreq(len(model_data), d=1. / 10)
            freqs_participant_1 = np.fft.fftfreq(len(participant_data), d=1. / 10)
            freqs_participant_2 = np.fft.fftfreq(len(participant_data_2), d=1. / 10)

            fft_model[np.abs(freqs_model) > threshold_freq] = 0
            fft_participant_1[np.abs(freqs_participant_1) > threshold_freq] = 0
            fft_participant_2[np.abs(freqs_participant_2) > threshold_freq] = 0


            pos_indices_model = range(1, len(freqs_model) // 2)
            pos_indices_participant_1 = range(1, len(freqs_participant_1) // 2)
            pos_indices_participant_2 = range(1, len(freqs_participant_2) // 2)

            # # plt.subplot(4, 4, subplot_index)
            #
            # # plt.plot(freqs_model[pos_indices_model], np.abs(fft_model)[pos_indices_model], label=f'model_{j + 1}',
            # #          color=blues[0])
            # plt.plot(freqs_participant_1[pos_indices_participant_1], np.abs(fft_participant_1)[pos_indices_participant_1],
            #          label=f'group_{j + 1}_participant1', color=reds[0])
            # plt.plot(freqs_participant_2[pos_indices_participant_2], np.abs(fft_participant_2)[pos_indices_participant_2],
            #          label=f'group_{j + 1}_participant2', color=reds[3])
            #
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Magnitude')
            # # plt.xlim(0, 0.24)
            # plt.title(f'{first_value}_{second_value}')
            # plt.legend()
            #
            # subplot_index += 1  # Move to the next subplot position


            ax2 = plt.subplot(4, 4, subplot_index)

            # Create a second y-axis for Features 2 and 3
            # ax2 = ax1.twinx()
            color_feature2 = reds[0]  # Ensure 'reds' is defined
            ax2.set_ylabel('Participants Magnitude', color=color_feature2)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.plot(freqs_participant_1[pos_indices_participant_1]*10, np.abs(fft_participant_1)[pos_indices_participant_1],
                     label=f'group {j + 1} participant 1', color=color_feature2, alpha=0.6)
            # ax2.plot(freqs_participant_2[pos_indices_participant_2]*10, np.abs(fft_participant_2)[pos_indices_participant_2],
            #          label=f'group_{j + 1}_participant2', color=reds[3], alpha=0.5)  # Use a different color for Feature 3
            ax2.tick_params(axis='y', labelcolor=color_feature2)

            # Plot for Feature 1 (model) on left y-axis
            ax1 = ax2.twinx()
            color = blues[0]  # Ensure 'blues' is defined
            # ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Model Magnitude', color=color)
            ax1.plot(freqs_model[pos_indices_model]*10, np.abs(fft_model)[pos_indices_model], label=f'model {j + 1}', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            if first_value == 2 and second_value == 2:
                name_first = 3
                name_second = 3
            elif first_value == 2 and second_value == 3:
                name_first = 2
                name_second = 3
            elif first_value == 3 and second_value == 2:
                name_first = 3
                name_second = 2
            else:
                name_first = 2
                name_second = 2
            plt.title(f'{name_first}:{name_second}')

            plt.xlim(0, 6)


            # Combining legends from both axes (if necessary)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')

            subplot_index += 1

    # Adjust layout and show/save the figure
    plt.tight_layout()

    if save_state:
        plt.savefig(f'{save_path}/human_computer_frequency_compare.png', dpi=300)


    plt.show()

''' Calculate the prediction ratio according to the reference '''
if matching_rate:
    for sample_key, sample in samples.items():
        time_interval_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
                             sample['file_paths']]
        # time_interval_2_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
        #                      sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'


            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=0)[0]
                p = filter_peaks(p, min_distance=45)

                # # a_filtered, b_filtered = filter_arrays(p, find_peaks(process_human_data(sample['file_paths'][i])[-2])[0], tolerance_low=20, tolerance_high=30)
                # # distance = wasserstein_distance(a_filtered, b_filtered)
                # #
                # # print(date)
                # # print(sample_key)
                # # print(len(a_filtered)/len(find_peaks(process_human_data(sample['file_paths'][i])[-2])[0]))
                # # print(distance)
                #
                # # p = filter_peaks(p, min_distance=interval)
                # model_performance = np.diff(p).tolist()
                # # model_performance = np.diff(find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=10)[0]).tolist()
                # model_performance_lst.append(model_performance)



                # # Initialize time lists.
                # real_time_lst = [100, 500, 900]  # Observed peak times.
                # model_time_lst = [80, 480, 850]  # Model-predicted peak times.
                # reference_time_lst = [500, 1000]  # Reference peak times.

                first_value, second_value = map(int, sample_key.split('_'))
                reference_time_interval = 800/6 if second_value == 2 else 533/6

                real_time_lst = find_peaks(process_human_data(sample['file_paths'][i])[-1])[0]
                # real_time_lst = [x + 200 / 6 for x in real_time_lst]
                reference_time_lst = generate_timing_list_until_end(start_time=real_time_lst[0],
                                                                    interval=reference_time_interval,
                                                                    end_time=real_time_lst[-1])
                real_time_lst = real_time_lst
                reference_time_lst = reference_time_lst[10:]
                # print(len(real_time_lst))
                # print(len(p))
                # print(len(reference_time_lst))
                # exit()
                model_time_lst = p

                # Initialize counters.
                TP = FP = FN = TN = 0
                real_p = model_p = 0


                # For each reference time point, check whether there is a corresponding nearest prediction.
                for ref_time in reference_time_lst:
                    # real_pred_time = closest_prediction(real_time_lst, ref_time, closest_distance=451/6, low=-150/6, high=250/6)
                    # model_pred_time = closest_prediction(model_time_lst, ref_time, closest_distance=451/6, low=-100/6, high=350/6)
                    real_pred_time = closest_prediction(real_time_lst, ref_time, closest_distance=451/6, low=-150/6, high=200/6)
                    model_pred_time = closest_prediction(model_time_lst, ref_time, closest_distance=451/6, low=-100/6, high=250/6)

                    real_pred = real_pred_time is not None
                    model_pred = model_pred_time is not None

                    if real_pred:
                        real_p += 1
                    if model_pred:
                        model_p += 1

                print(f'group_{i+1}')
                print(f'{sample_key}')
                # print(f"p_human: {real_p/len(reference_time_lst)}")
                print(f"p_model: {model_p/len(reference_time_lst)}")
                # print(reference_time_lst)
                # print(p)
                # print(f"{model_p/len(reference_time_lst)}")
                # print(np.diff(reference_time_lst))
                # print(np.diff(p))
                # print(model_p/len(reference_time_lst))


                # print('precision', calculate_f1_score(TP, FP, TN, FN)[0])
                # print('recall', calculate_f1_score(TP, FP, TN, FN)[1])
                # print('f1-score', calculate_f1_score(TP, FP, TN, FN)[2])
                # Total number of observations
                # total_observations = TP + FP + TN + FN
                #
                # # Observed agreement (p_o)
                # observed_agreement = (TP + TN) / total_observations
                #
                # # Expected agreement (p_e)
                # positive_observed_rate = (TP + FP) / total_observations
                # negative_observed_rate = (TN + FN) / total_observations
                # positive_expected_rate = (TP + FN) / total_observations
                # negative_expected_rate = (TN + FP) / total_observations
                #
                # expected_agreement = (positive_observed_rate * positive_expected_rate) + (
                #             negative_observed_rate * negative_expected_rate)
                #
                # # Cohen's Kappa
                # kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
                # print(kappa)



''' Calculate the F1 score '''
if f1_score:
    for sample_key, sample in samples.items():
        time_interval_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
                             sample['file_paths']]
        # time_interval_2_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
        #                      sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'


            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=0)[0]
                p = filter_peaks(p, min_distance=45)

                # # a_filtered, b_filtered = filter_arrays(p, find_peaks(process_human_data(sample['file_paths'][i])[-2])[0], tolerance_low=20, tolerance_high=30)
                # # distance = wasserstein_distance(a_filtered, b_filtered)
                # #
                # # print(date)
                # # print(sample_key)
                # # print(len(a_filtered)/len(find_peaks(process_human_data(sample['file_paths'][i])[-2])[0]))
                # # print(distance)
                #
                # # p = filter_peaks(p, min_distance=interval)
                # model_performance = np.diff(p).tolist()
                # # model_performance = np.diff(find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=10)[0]).tolist()
                # model_performance_lst.append(model_performance)



                # # Initialize time lists.
                # real_time_lst = [100, 500, 900]  # Observed peak times.
                # model_time_lst = [80, 480, 850]  # Model-predicted peak times.
                # reference_time_lst = [500, 1000]  # Reference peak times.

                first_value, second_value = map(int, sample_key.split('_'))
                reference_time_interval = 800/6 if second_value == 2 else 533/6

                real_time_lst = find_peaks(process_human_data(sample['file_paths'][i])[-1])[0]
                # real_time_lst = [x + 200 / 6 for x in real_time_lst]
                reference_time_lst = generate_timing_list_until_end(start_time=real_time_lst[0],
                                                                    interval=reference_time_interval,
                                                                    end_time=real_time_lst[-1])
                real_time_lst = real_time_lst[10:]
                reference_time_lst = reference_time_lst[10:]
                # print(len(real_time_lst))
                # print(len(p))
                # print(len(reference_time_lst))
                # exit()
                model_time_lst = p[10:]

                # Initialize counters.
                TP = FP = FN = TN = 0

                # For each reference time point, check whether there is a corresponding nearest prediction.
                for ref_time in reference_time_lst:
                    real_pred_time = closest_prediction(real_time_lst, ref_time, closest_distance=451/6, low=-150/6, high=200/6)
                    model_pred_time = closest_prediction(model_time_lst, ref_time, closest_distance=451/6, low=-100/6, high=250/6)


                    real_pred = real_pred_time is not None
                    model_pred = model_pred_time is not None

                    if real_pred and model_pred:
                        TP += 1
                    elif real_pred and not model_pred:
                        FP += 1
                    elif not real_pred and model_pred:
                        FN += 1
                    elif not real_pred and not model_pred:
                        TN += 1

                print('group', i+1)
                print(sample_key)
                # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

                print('precision', calculate_f1_score(TP, FP, TN, FN)[0])
                print('recall', calculate_f1_score(TP, FP, TN, FN)[1])
                print('f1-score', calculate_f1_score(TP, FP, TN, FN)[2])
                # Total number of observations
                # total_observations = TP + FP + TN + FN
                #
                # # Observed agreement (p_o)
                # observed_agreement = (TP + TN) / total_observations
                #
                # # Expected agreement (p_e)
                # positive_observed_rate = (TP + FP) / total_observations
                # negative_observed_rate = (TN + FN) / total_observations
                # positive_expected_rate = (TP + FN) / total_observations
                # negative_expected_rate = (TN + FP) / total_observations
                #
                # expected_agreement = (positive_observed_rate * positive_expected_rate) + (
                #             negative_observed_rate * negative_expected_rate)
                #
                # # Cohen's Kappa
                # kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
                # print(kappa)


''' Plot the interval distribution '''
if interval_distribution:
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    # sample_keys_sorted = sorted(samples.keys(), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[0])))
    for plot_row, (sample_key, sample) in enumerate(samples.items()):
        human_peak_lst = [find_peaks(process_human_data(file_path)[-1])[0].tolist() for file_path in sample['file_paths']]
        model_peak_lst = []

        time_interval_lst = [(np.diff(find_peaks(process_human_data(file_path)[-1])[0])*6).tolist() for file_path in
                             sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for date in dates:
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_PC_PERF.json'
            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0], height=10)[0]
                p = filter_peaks(p, min_distance=45)
                model_performance = (np.diff(p)*6).tolist()
                model_peak_lst.append(p)
                model_performance_lst.append(model_performance)

        # Assuming first_value and second_value are determined as per your original logic
        first_value = int(sample_key.split('_')[0])
        second_value = int(sample_key.split('_')[1])
        if first_value == 2:
            dashed_line_x = 800
        elif first_value == 3:
            dashed_line_x = 533
        if second_value == 2:
            dashed_line_x_2 = 800
        elif second_value == 3:
            dashed_line_x_2 = 533

        for plot_col in range(4):
            ax = axs[plot_row, plot_col]
            # sns.kdeplot(model_performance_lst[plot_col], label=f'{first_value}_{second_value}_model_{plot_col + 1}',
            #             color=blues[0], ax=ax)
            # sns.kdeplot(time_interval_lst[plot_col], label=f'{first_value}_{second_value}_participant_{plot_col + 1}',
            #             color=reds[0], ax=ax)
            plt.title(f'')
            if first_value == 2 and second_value == 2:
                name_first = 3
                name_second = 3
            elif first_value == 2 and second_value == 3:
                name_first = 2
                name_second = 3
            elif first_value == 3 and second_value == 2:
                name_first = 3
                name_second = 2
            else:
                name_first = 2
                name_second = 2

            sns.kdeplot(model_performance_lst[plot_col], label=f'{name_first}:{name_second} model {plot_col + 1}',
                        color=blues[0], ax=ax)
            sns.kdeplot(time_interval_lst[plot_col], label=f'{name_first}:{name_second} participant {plot_col + 1}',
                        color=reds[0], ax=ax)

            # Add vertical dashed line if applicable
            # if plot_col in [2, 3]:  # Adjust this condition based on your dashed_line_x logic
            ax.axvline(x=dashed_line_x, color=reds[2], linestyle='dashed')
            ax.axvline(x=dashed_line_x_2, color=reds[0], linestyle='dashdot')
            # ax.axvline(x=dashed_line_x, color=reds[0], linestyle='--')
            # plt.axvline(x=dashed_line_x, color=reds[0], linestyle='dashed')
            # plt.axvline(x=dashed_line_x_2, color=reds[2], linestyle='dashdot')
            ax.set_xlim(40*6, 200*6)


            # Set labels and legend for each subplot
            ax.set_xlabel('Inter-Beat Interval (ms)')
            ax.set_ylabel('Probability Density')
            ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure if required
    if save_state:
        plt.savefig(f'{save_path}/human_computer_distribution_compare.png', dpi=300)

    # Show the plot
    plt.show()

