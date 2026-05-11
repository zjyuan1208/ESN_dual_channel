import numpy as np
import torch
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import wasserstein_distance
from scipy.fft import fft
from scipy.interpolate import interp1d


save_path = '/home/zhyuan/Desktop/ESN/results_human_behavior/paper_figures_2_v1'
save_state = True

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

# Define colors
blues = ['#115699', '#0E6DB3', '#5CAAD7', '#95C6DE']
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']


samples = {
    '2_2': {
        'file_paths': [
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_2_2_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_2_2_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_2_2_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_2_2_P2_PERF.txt',
        ],
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_human/human_human_interact_model_performance_2024',
    },
    '3_2': {
        'file_paths': [
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_3_2_P2_PERF.txt',
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_3_2_P2_PERF.txt',
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_3_2_P2_PERF.txt',
            '/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_3_2_P2_PERF.txt',
        ],
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_human/human_human_interact_model_performance_2024',
    },
    '2_3': {
        'file_paths': [
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_2_3_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_2_3_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_2_3_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_2_3_P2_PERF.txt',
        ],
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_human/human_human_interact_model_performance_2024',
    },
    '3_3': {
        'file_paths': [
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_AM/TAP_3_3_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_3_3_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_03_AM/TAP_3_3_P2_PERF.txt',
            f'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_02_10_AM/TAP_3_3_P2_PERF.txt',
        ],
        'model_json_path_prefix': '/home/zhyuan/Desktop/ESN/results_human_behavior/sin_replace/human_human/human_human_interact_model_performance_2024',
    }
}


def filter_arrays(a, b, tolerance_low=20, tolerance_high=30):
    a_filtered = []
    b_filtered = []
    for value in a:
        # Find the closest value in array b within the range [value - tolerance_low, value + tolerance_high]
        close_values = b[(b >= value - tolerance_low) & (b <= value + tolerance_high)]
        if len(close_values) > 0:
            closest_value = close_values[np.argmin(np.abs(close_values - value))]
            a_filtered.append(value)
            b_filtered.append(closest_value)
    return np.array(a_filtered), np.array(b_filtered)


def closest_prediction(peak_times, reference_time, closest_distance=401/6, low=20/6, high=400/6):
    """Find the peak closest to reference_time within the 20-400 ms window."""
    closest_peak = None
    # closest_distance = 401/6  # Initialize above the maximum allowed distance.
    for peak_time in peak_times:
        distance = reference_time - peak_time
        if low <= distance <= high and np.abs(distance) < closest_distance:
            closest_peak = peak_time
            closest_distance = distance
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
    FP = len(model_time_lst) - len(matched_model_beats)
    # FP = len(reference_time_lst) - len(matched_model_beats)

    return TP, TN, FP, FN


def filter_dots(dist_lst, angle_lst):
    filterd_dist_lst = []
    filtered_angle_lst = []
    for i in range(len(dist_lst)):
        # if dist_lst[i] < 1600/6:
        if dist_lst[i] < 1600*1.5:
            filterd_dist_lst.append(dist_lst[i])
            filtered_angle_lst.append(angle_lst[i])

    return filterd_dist_lst, filtered_angle_lst

def calculate_basic_interval(peak_timings3, peak_timings_other, name_second):
    intervals = []
    for t in peak_timings_other:
        closest_peak_idx = np.argmin(np.abs(peak_timings3 - t))
        if closest_peak_idx >= 2:
            if name_second == 2:
                interval = (peak_timings3[closest_peak_idx] - peak_timings3[closest_peak_idx - 2]) / 4
            elif name_second == 3:
                interval = (peak_timings3[closest_peak_idx] - peak_timings3[closest_peak_idx - 2]) / 6
            intervals.append(interval)
    return intervals


def update_peak_timings(peak_timings, name_first):
    updated_timings = peak_timings.copy() - peak_timings[0]
    if name_first == 2:
        for i in range(3, len(peak_timings), 3):
            updated_timings[i] = peak_timings[i] - peak_timings[i - 2]
            updated_timings[i - 1] = peak_timings[i-1] - peak_timings[i - 2]
            updated_timings[i - 2] = 0
            # updated_timings[i - 2] = peak_timings[i-2] - peak_timings[i - 3]
    elif name_first == 3:
        for i in range(2, len(peak_timings), 2):
            updated_timings[i] = peak_timings[i] - peak_timings[i - 1]
            updated_timings[i - 1] = 0
            # updated_timings[i - 1] = peak_timings[i-1] - peak_timings[i - 2]
    return updated_timings


def plot_three_peaks_on_circle(ax, peak_timings1, peak_timings2, peak_timings3, label1, label2, label3, name_first, name_second):
    basic_interval_model = calculate_basic_interval(peak_timings3, peak_timings1, name_second)
    basic_interval_P1 = calculate_basic_interval(peak_timings3, peak_timings2, name_second)
    basic_interval_P2 = calculate_basic_interval(peak_timings3, peak_timings3, name_second)

    peak_timings1_updated = update_peak_timings(peak_timings1, name_first)[1:]
    peak_timings2_updated = update_peak_timings(peak_timings2, name_first)[1:]
    peak_timings3_updated = update_peak_timings(peak_timings3, name_second)[1:]


    # intervals1 = np.diff(peak_timings1_updated)
    # intervals2 = np.diff(peak_timings2_updated)
    # intervals3 = np.diff(peak_timings3_updated)
    intervals1 = np.diff(peak_timings1)
    intervals2 = np.diff(peak_timings2)
    intervals3 = np.diff(peak_timings3)

    # plot_circle_seaborn(peak_timings1_updated, peak_timings2_updated, basic_interval)

    for i in range(len(basic_interval_model)):
        basic_interval_value = basic_interval_model[i]
        phases1 = (peak_timings1_updated / (6 * basic_interval_value)) * 2 * np.pi
    for i in range(len(basic_interval_P1)):
        basic_interval_value = basic_interval_P1[i]
        phases2 = (peak_timings2_updated / (6 * basic_interval_value)) * 2 * np.pi
    for i in range(len(basic_interval_P2)):
        basic_interval_value = basic_interval_P2[i]
        phases3 = (peak_timings3_updated / (6 * basic_interval_value)) * 2 * np.pi

    intervals1, phases1 = filter_dots(intervals1, phases1[:])
    intervals2, phases2 = filter_dots(intervals2, phases2[:])
    intervals3, phases3 = filter_dots(intervals3, phases3[:])

    # Plot each peak as a point on the circle for both series
    lines1, = ax.plot(phases3, intervals3, 'o', label=label3, markerfacecolor='None', markeredgecolor='#01844F', alpha=1)
    lines2, = ax.plot(phases2, intervals2, 'o', label=label2, markerfacecolor='None', markeredgecolor=reds[0], alpha=0.5)
    lines3, = ax.plot(phases1, intervals1, 'o', label=label1, markerfacecolor='None', markeredgecolor=blues[0], alpha=0.8)
    # lines2, = ax.plot(phases2, intervals2, 'o', label=label2, markerfacecolor='None', markeredgecolor=reds[0], alpha=0)
    # lines3, = ax.plot(phases1, intervals1, 'o', label=label1, markerfacecolor='None', markeredgecolor=blues[0], alpha=0)


    # Handles and labels
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0]  # [Line 3, Line 2, Line 1]

    # Create legend with the specified order
    ax.legend([lines3, lines2, lines1], [labels[idx] for idx in order], loc='upper right')

    # Customize the angle labels to show radians
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])


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

def part_R_score(taps, metronome):
    l = int(taps[-1] / 5000)
    r_scores = []

    for i in range(l):
        start = i * 5000
        end = (i + 1) * 5000
        taps_segment = [tap for tap in taps if start <= tap < end]
        metronome_segment = [m for m in metronome if start <= m < end]

        if taps_segment and metronome_segment:  # Ensure there are taps and metronome beats in the segment
            _, r = R_score(taps_segment, metronome_segment)
            r_scores.append(r)

    return r_scores



f1_score_seperate = False
FFT_plot = False
matching_rate = False
f1_score = True
interval_distribution = False
circle_plot = False
inter_beat_interval_plot = False

# relative_phase_angles, R = R_score(model_time_lst, participant_1_time_lst[3:])
# # print(relative_phase_angles)
# print(date, sample_key)
# print(R)
# relative_phase_angles, R = R_score(model_time_lst, participant_2_time_lst[3:])
# print(R)

if inter_beat_interval_plot:
    dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    for i, (sample_key, sample) in enumerate(samples.items()):
        for j, date in enumerate(dates):
            ax = axs[i, j]
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_P2_PERF.json'

            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0])[0]
                p = filter_peaks(p, min_distance=45)

                first_value, second_value = map(int, sample_key.split('_'))
                reference_time_interval = 800 / 6 if second_value == 2 else 533 / 6

                participant_1_time_lst = find_peaks(process_human_data(sample['file_paths'][j])[-2])[0]
                participant_2_time_lst = find_peaks(process_human_data(sample['file_paths'][j])[-1])[0]
                model_time_lst = p[4:]

                # relative_phase_angles, R = R_score(model_time_lst, participant_1_time_lst[3:])
                print(date, sample_key)
                # print(R)
                # relative_phase_angles, R = R_score(model_time_lst, participant_2_time_lst[3:])
                # print(R)
                R = part_R_score(model_time_lst, participant_1_time_lst[3:])
                print('with P1', R)
                R = part_R_score(model_time_lst, participant_2_time_lst[3:])
                print('with P2', R)

                if first_value == 2 and second_value == 2:
                    name_first = 3
                    name_second = 3
                    target_ibi1 = 800 / 6
                    target_ibi2 = 800 / 6
                elif first_value == 2 and second_value == 3:
                    name_first = 2
                    name_second = 3
                    target_ibi1 = 533 / 6
                    target_ibi2 = 800 / 6
                elif first_value == 3 and second_value == 2:
                    name_first = 3
                    name_second = 2
                    target_ibi1 = 800 / 6
                    target_ibi2 = 533 / 6
                else:
                    name_first = 2
                    name_second = 2
                    target_ibi1 = 533 / 6
                    target_ibi2 = 533 / 6



                # Plot the IBIs
                ax.plot(model_time_lst[1:], np.diff(model_time_lst), label='model', marker='o')
                ax.plot(participant_1_time_lst[1:], np.diff(participant_1_time_lst), label='Participant 1', marker='x')
                ax.plot(participant_2_time_lst[1:], np.diff(participant_2_time_lst), label='Participant 2', marker='s')

                # Adding horizontal lines for target IBIs
                ax.axhline(y=target_ibi1, color='r', linestyle='--', label=f'Target IBI {int(target_ibi1)}')
                ax.axhline(y=target_ibi2, color='g', linestyle='--', label=f'Target IBI {int(target_ibi2)}')

                # Adding labels and title
                ax.set_xlabel('Time')
                ax.set_ylabel('Inter-Beat Interval (IBI)')
                ax.set_title(f'Inter-Beat Interval for {sample_key} on {date}')
                ax.set_ylim(0, 400)
                ax.legend()

    plt.tight_layout()
    plt.show()



if circle_plot:
    fig, axs = plt.subplots(4, 4, subplot_kw={'projection': 'polar'}, figsize=(20, 20))
    for i, (sample_key, sample) in enumerate(samples.items()):
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']
        for j, date in enumerate(dates):
            ax = axs[i, j]
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_P2_PERF.json'

            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0])[0]
                p = filter_peaks(p, min_distance=45)

                first_value, second_value = map(int, sample_key.split('_'))
                reference_time_interval = 800 / 6 if second_value == 2 else 533 / 6

                participant_1_time_lst = find_peaks(process_human_data(sample['file_paths'][j])[-2])[0]
                participant_2_time_lst = find_peaks(process_human_data(sample['file_paths'][j])[-1])[0]
                model_time_lst = p[0:]
                # basic_interval = 800 / 6 / 3  # Example basic interval

                # relative_phase_angles, R = R_score(model_time_lst, participant_1_time_lst[3:])
                # print(relative_phase_angles)
                # print(date, sample_key)
                # print(R)
                # relative_phase_angles, R = R_score(model_time_lst, participant_2_time_lst[3:])
                # print(R)



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
                participant_1_time_lst = (np.array(participant_1_time_lst) * 6)
                participant_2_time_lst = (np.array(participant_2_time_lst) * 6)


                plot_three_peaks_on_circle(ax, model_time_lst, participant_1_time_lst, participant_2_time_lst,
                                         label1=f'{name_first}:{name_second} model',
                                         label2=f'{name_first}:{name_second} participant 1',
                                         label3=f'{name_first}:{name_second} participant 2', name_first=name_first, name_second=name_second)

    plt.tight_layout()

    # Save the figure if required
    if save_state:
        plt.savefig(f'{save_path}/human_human_dots_compare.pdf', dpi=300)
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
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_P2_PERF.json'


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

                # ## human results
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
        time_interval_lst = [process_human_data(file_path)[-2] for file_path in sample['file_paths']]
        time_interval_2_lst = [process_human_data(file_path)[-1] for file_path in sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_P2_PERF.json'

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

            # Apply noise reduction by zeroing out components above the threshold frequency
            fft_model[np.abs(freqs_model) > threshold_freq] = 0
            fft_participant_1[np.abs(freqs_participant_1) > threshold_freq] = 0
            fft_participant_2[np.abs(freqs_participant_2) > threshold_freq] = 0

            # plt.subplot(4, 4, subplot_index)

            # Plot the positive frequencies, starting from index 1 to exclude the DC component
            pos_indices_model = range(1, len(freqs_model) // 2)
            pos_indices_participant_1 = range(1, len(freqs_participant_1) // 2)
            pos_indices_participant_2 = range(1, len(freqs_participant_2) // 2)

            # plt.plot(freqs_model[pos_indices_model], np.abs(fft_model)[pos_indices_model], label=f'model_{j + 1}',
            #          color=blues[0])
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
                     label=f'group {j + 1} participant 1', color=color_feature2, alpha=0.8)
            ax2.plot(freqs_participant_2[pos_indices_participant_2]*10, np.abs(fft_participant_2)[pos_indices_participant_2],
                     label=f'group {j + 1} participant 2', color=reds[3], alpha=0.5)  # Use a different color for Feature 3
            ax2.tick_params(axis='y', labelcolor=color_feature2)

            # Plot for Feature 1 (model) on left y-axis
            ax1 = ax2.twinx()
            color = blues[0]  # Ensure 'blues' is defined
            # ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Model Magnitude', color=color)
            ax1.plot(freqs_model[pos_indices_model]*10, np.abs(fft_model)[pos_indices_model], label=f'model {j + 1}', color=color, alpha=0.9)
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
        plt.savefig(f'{save_path}/human_human_frequency_compare.pdf', dpi=300)


    plt.show()


''' Calculate the prediction ratio according to the reference '''
if matching_rate:
    for sample_key, sample in samples.items():
        time_interval_lst = [np.diff(find_peaks(process_human_data(file_path)[-2])[0]).tolist() for file_path in
                             sample['file_paths']]
        time_interval_2_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
                             sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_P2_PERF.json'


            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0])[0]
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

                real_time_lst = find_peaks(process_human_data(sample['file_paths'][i])[-2])[0]
                # real_time_lst = [x + 200 / 6 for x in real_time_lst]
                reference_time_lst = generate_timing_list_until_end(start_time=real_time_lst[0],
                                                                    interval=reference_time_interval,
                                                                    end_time=real_time_lst[-1])
                # print(i)
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
                    # real_pred_time = closest_prediction(real_time_lst, ref_time, closest_distance=301/6, low=-100/6, high=200/6)
                    # model_pred_time = closest_prediction(model_time_lst, ref_time, closest_distance=401/6, low=-50/6, high=400/6)
                    real_pred_time = closest_prediction(real_time_lst, ref_time, closest_distance=451 / 6, low=-150 / 6, high=200 / 6)
                    model_pred_time = closest_prediction(model_time_lst, ref_time, closest_distance=451 / 6, low=-100 / 6, high=250 / 6)

                    real_pred = real_pred_time is not None
                    model_pred = model_pred_time is not None

                    if real_pred:
                        real_p += 1
                    if model_pred:
                        model_p += 1

                print(f'group_{i+1}')
                print(f'{sample_key}')
                print(f"p_human: {real_p/len(reference_time_lst)}, p_model: {model_p/len(reference_time_lst)}")
                # print(model_p/len(reference_time_lst))


                # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
                # print(calculate_f1_score(TP, FP, TN, FN))
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
        time_interval_lst = [np.diff(find_peaks(process_human_data(file_path)[-2])[0]).tolist() for file_path in
                             sample['file_paths']]
        time_interval_2_lst = [np.diff(find_peaks(process_human_data(file_path)[-1])[0]).tolist() for file_path in
                             sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_P2_PERF.json'


            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0])[0]
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

                real_time_lst = find_peaks(process_human_data(sample['file_paths'][i])[-2])[0]
                real_time_lst = [x + 200 / 6 for x in real_time_lst]
                reference_time_lst = generate_timing_list_until_end(start_time=real_time_lst[0],
                                                                    interval=reference_time_interval,
                                                                    end_time=real_time_lst[-1])
                # print(len(real_time_lst))
                # print(len(p))
                # print(len(reference_time_lst))
                # exit()
                model_time_lst = p

                # Initialize counters.
                TP = FP = FN = TN = 0

                # For each reference time point, check whether there is a corresponding nearest prediction.
                for ref_time in reference_time_lst:
                    # real_pred_time = closest_prediction(real_time_lst, ref_time)
                    # model_pred_time = closest_prediction(model_time_lst, ref_time)
                    # real_pred_time = closest_prediction(peak_times=real_time_lst, reference_time=ref_time, closest_distance=351/6, low=-150/6, high=201/6)
                    # model_pred_time = closest_prediction(peak_times=model_time_lst, reference_time=ref_time, closest_distance=351/6, low=-50/6, high=300/6)
                    real_pred_time = closest_prediction(real_time_lst, ref_time, closest_distance=451 / 6, low=-150 / 6, high=200 / 6)
                    model_pred_time = closest_prediction(model_time_lst, ref_time, closest_distance=451 / 6, low=-100 / 6, high=250 / 6)

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

                print('group', i + 1)
                print(sample_key)
                # print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

                print('precision', calculate_f1_score(TP, FP, TN, FN)[0])
                print('recall', calculate_f1_score(TP, FP, TN, FN)[1])
                print('f1-score', calculate_f1_score(TP, FP, TN, FN)[2])


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

if interval_distribution:
    plt.figure(figsize=(20, 20))  # Adjusted for 4x4 grid of subfigures
    # fig, axs = plt.subplots(4, 4, figsize=(20, 20))


    sample_keys_sorted = sorted(samples.keys(), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[0])))
    subplot_index = 1  # Start with the first subplot

    for sample_key in sample_keys_sorted:
        sample = samples[sample_key]
        time_interval_lst = [(np.diff(find_peaks(process_human_data(file_path)[-2])[0])*6).tolist() for file_path in
                             sample['file_paths']]
        time_interval_2_lst = [(np.diff(find_peaks(process_human_data(file_path)[-1])[0])*6).tolist() for file_path in
                               sample['file_paths']]
        model_performance_lst = []
        dates = ['01_30_AM', '01_30_PM', '02_03_AM', '02_10_AM']

        for i, date in enumerate(dates):
            file_path_json = f'{sample["model_json_path_prefix"]}_{date}_TAP_{sample_key}_P2_PERF.json'

            with open(file_path_json, 'r') as json_file:
                loaded_data = json.load(json_file)
                p = find_peaks(np.array(loaded_data['predicted'])[0, :, 0])[0]
                p = filter_peaks(p, min_distance=45)
                model_performance = (np.diff(p)*6).tolist()
                model_performance_lst.append(model_performance)

        first_value, second_value = map(int, sample_key.split('_'))

        # Assuming the calculation for dashed_line_x and dashed_line_x_2 is correct and needed
        dashed_line_x = 800 if first_value == 2 else 533
        dashed_line_x_2 = 800 if second_value == 2 else 533

        # Plotting the subfigures
        for j, (model_data, participant_data, participant_data_2) in enumerate(zip(model_performance_lst, time_interval_lst, time_interval_2_lst)):
            plt.subplot(4, 4, subplot_index)
            sns.kdeplot(model_data, label=f'model {j+1}', color=blues[0])
            sns.kdeplot(participant_data, label=f'group {j+1} participant 1', color=reds[0])
            sns.kdeplot(participant_data_2, label=f'group {j+1} participant 2', color=reds[3])

            plt.axvline(x=dashed_line_x, color=reds[3], linestyle='dashed')
            plt.axvline(x=dashed_line_x_2, color=reds[0], linestyle='dashdot')

            plt.xlim(0*6, 500*6)
            plt.xlabel('Inter-Beat Interval (ms)')
            plt.ylabel('Probability Density')
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
            plt.legend()

            subplot_index += 1  # Move to the next subplot position

    # Adjust layout and show/save the figure
    plt.tight_layout()

    if save_state:
        plt.savefig(f'{save_path}/human_human_distribution_compare.pdf', dpi=300)


    plt.show()


