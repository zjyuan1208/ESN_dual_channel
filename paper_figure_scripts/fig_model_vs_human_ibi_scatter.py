import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks
import json
import re

save_path = '/home/zhyuan/Desktop/ESN/results_human_behavior/paper_figures_2_v1'
save_state = True


blues = ['#115699', '#0E6DB3', '#5CAAD7', '#95C6DE']
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']

'''Plot for all the human data'''
#
# def extract_and_calculate_time_interval(file_path):
#     # Extract numbers from the file name using regular expressions
#     match = re.search(r'TAP_(\d+)_(\d+)_([A-Za-z\d]+)?_PERF\.txt', file_path)
#
#     if match:
#         # Extract the two numbers from the match
#         first_number = int(match.group(1))
#         second_number = int(match.group(2))
#
#         # Define time intervals based on the second number
#         if second_number == 3:
#             time_interval_first_channel = 533
#         elif second_number == 2:
#             time_interval_first_channel = 800
#         else:
#             # Handle other cases if needed
#             time_interval_first_channel = None
#
#         # Define time intervals based on the first number
#         if second_number == 3:
#             time_interval_second_channel = 533
#         elif second_number == 2:
#             time_interval_second_channel = 800
#         else:
#             # Handle other cases if needed
#             time_interval_second_channel = None
#
#         return first_number, second_number, time_interval_first_channel, time_interval_second_channel
#     else:
#         # Handle the case when the file name doesn't match the expected pattern
#         return None
#
#
# file_path = r'/home/zhyuan/Desktop/ESN/data/human_exp/PERF_2024_01_30_PM/TAP_3_3_PC_PERF.txt'
# second_channel_tap, first_channel_tap, time_interval_first_channel, time_interval_second_channel \
#     = extract_and_calculate_time_interval(file_path)
#
#
# data = np.loadtxt(file_path, delimiter='\t', dtype=str)
#
# # Separate data for Var2 values 1 and 2
# data_1 = data[data[:, 1] == '1']
# data_2 = data[data[:, 1] == '2']
#
# # y_values = np.diff(data_1[1:, 0].astype(float))
#
# data_1 = data_1[:, 0].astype(float) * 1000  # Convert seconds to milliseconds
# data_2 = data_2[:, 0].astype(float) * 1000  # Convert seconds to milliseconds
#
#
# total_length = int(max(data_1)) + 10
#
# x_values = data_1[2:].astype(float)
# y_values = np.diff(data_1[1:].astype(float))
# x_2_values = data_2[2:].astype(float)
# y_2_values = np.diff(data_2[1:].astype(float))
# plt.figure(figsize=(8, 6))
#
# time_interval = y_values[0] * 2 / 3
# time_interval_equal = y_values[0]
#
#
# plt.plot(x_values, y_values, marker='o', color='b', linestyle='-', label='person 2 or a pc')
# plt.plot(x_2_values, y_2_values, marker='x', color='r', linestyle='-', label='person 1')
# plt.axhline(y=time_interval, color='g', linestyle='--', label=f'2:3 Target interval at {np.round(time_interval, 3)}')
# plt.axhline(y=time_interval_equal, color='y', linestyle='--', label=f'1:1 Target interval at {np.round(time_interval_equal, 3)}')
# # plt.axhline(y=time_interval_equal, color='c', linestyle='--', label=f'1:1 Target interval at {np.round(time_interval_equal, 3)}')
#
# # Adding labels and title
# plt.xlabel('Time')
# plt.ylabel('Interval')
# plt.legend()
#
# # plt.title(f'example_{i}_feature1 Line Plot of peak interval')
# # if save_fig:
# #     plt.savefig(f'{save_path}/fig_{leaky_rate[0]}_{i}_feature1_peak_interval.png')
# plt.show()


'''Plot for the one dim increasing intervals'''
# Load data from the JSON file

# file_path = r'/home/zhyuan/Desktop/ESN/results_human_behavior/1dim/1dim_wo_update_interval_400to3000.json'
# file_path = '/home/zhyuan/Desktop/ESN/results_human_behavior/1dim_wo_update_interval_400to300_all.json'
file_path = '/home/zhyuan/Desktop/ESN/results_human_behavior/1dim_wo_update_k_interval_400to3000_tcyb_v2.json'

with open(file_path, 'r') as json_file:
    loaded_data = json.load(json_file)


# Access the lists from the loaded data
real_interval_lst = loaded_data['real_interval_lst']
model_interval_lst = loaded_data['model_interval_lst']
model_interval_lst_mean = loaded_data['model_interval_lst_mean']
model_interval_lst_std = loaded_data['model_interval_lst_std']
real_interval_lst = [x * 6 for x in real_interval_lst]
model_interval_lst = [[element * 6 for element in sublist[2:]] for sublist in model_interval_lst]
model_interval_lst_mean = [x * 6 for x in model_interval_lst_mean]
model_interval_lst_std = [x * 6 for x in model_interval_lst_std]

# Plotting the scatter plot with error bars
# plt.errorbar(real_interval_lst, model_interval_lst_mean, yerr=model_interval_lst_std, fmt='o', color=blues[1])
#              # label='Before update')
# Function to calculate the occurrence times
def calculate_occurrences(model_interval_lst):
    occurrences = []
    for sublist in model_interval_lst:
        counts = np.bincount(sublist)
        occurrences.append(counts)
    return occurrences

# Calculate occurrence times
occurrences = calculate_occurrences(model_interval_lst)

# Prepare the figure
plt.figure(figsize=(12, 8))

# Plot bubble chart
for i, x in enumerate(real_interval_lst):
    y = range(len(occurrences[i]))  # Possible y-values (model intervals)
    sizes = occurrences[i] * 5  # Scale bubble sizes for better visibility
    plt.scatter([x]*len(y), y, s=sizes, alpha=0.5, color=blues[1])
    # plt.scatter([x]*len(y), y, s=sizes, alpha=0.5, label=f"Interval {x}")



# plt.errorbar(real_interval_lst_update, model_interval_lst_mean_update, yerr=model_interval_lst_std_update, fmt='o', color='blue',
#              label='After update')

# Adding labels and title
plt.xlabel('Real Inter-Beat Intervals (ms)', fontsize=20)
plt.ylabel('Model Inter-Beat Intervals (ms)', fontsize=20)
# plt.title('Scatter Plot of Real vs. Model Intervals with Variance')

# Displaying legend

# Adding diagonal line where x equals y
plt.plot([min(real_interval_lst), max(real_interval_lst)], [min(real_interval_lst), max(real_interval_lst)],
         color=reds[0], linestyle='--', label='target interval')

# Adding diagonal line where y = x - 150
x_values = np.array(real_interval_lst)
plt.plot(x_values, x_values / 2, color=reds[0], linestyle='--', label='one half target interval')
plt.plot(x_values, x_values / 3, color=reds[1], linestyle='--', label='one third target interval')
plt.plot(x_values, x_values / 4, color=reds[2], linestyle='--', label='one quarter target interval')
plt.plot(x_values, x_values / 6, color=reds[3], linestyle='--', label='one sixth target interval')


# Set y-axis limit to only show the part above y=0
plt.ylim(bottom=0)
plt.legend()
if save_state:
    plt.savefig(f'{save_path}/scatter_plot_1dim_wo_update_interval_400to3000.pdf', dpi=300)

# Show the plot
plt.show()


# file_path_update = r'/home/zhyuan/Desktop/ESN/results_human_behavior/1dim/1dim_update_c_k_interval_400to3000.json'
# file_path_update = r'/home/zhyuan/Desktop/ESN/results_human_behavior/1dim_update_interval_400to300_all.json'
file_path_update = r'/home/zhyuan/Desktop/ESN/results_human_behavior/1dim_with_update_k_interval_400to3000_tcyb_v2.json'

with open(file_path_update, 'r') as json_file:
    loaded_data = json.load(json_file)

# Access the lists from the loaded data
# real_interval_lst_update = loaded_data['real_interval_lst']
# model_interval_lst_mean_update = loaded_data['model_interval_lst_mean']
# model_interval_lst_std_update = loaded_data['model_interval_lst_std']
#
# real_interval_lst_update = [x * 6 for x in real_interval_lst_update]
# model_interval_lst_mean_update = [x * 6 for x in model_interval_lst_mean_update]
# model_interval_lst_std_update = [x * 6 for x in model_interval_lst_std_update]

# Access the lists from the loaded data
real_interval_lst_update = loaded_data['real_interval_lst']
model_interval_lst_update = loaded_data['model_interval_lst']
model_interval_lst_mean_update = loaded_data['model_interval_lst_mean']
model_interval_lst_std_update = loaded_data['model_interval_lst_std']

real_interval_lst_update = [x * 6 for x in real_interval_lst_update]
model_interval_lst_update = [[element * 6 for element in sublist[2:]] for sublist in model_interval_lst_update]
model_interval_lst_mean_update = [x * 6 for x in model_interval_lst_mean_update]
model_interval_lst_std_update = [x * 6 for x in model_interval_lst_std_update]

# Plotting the scatter plot with error bars
# plt.errorbar(real_interval_lst, model_interval_lst_mean, yerr=model_interval_lst_std, fmt='o', color=blues[1])
#              # label='Before update')


# Calculate occurrence times
occurrences = calculate_occurrences(model_interval_lst_update)

# Prepare the figure
plt.figure(figsize=(12, 8))

# Plot bubble chart
for i, x in enumerate(real_interval_lst):
    y = range(len(occurrences[i]))  # Possible y-values (model intervals)
    sizes = occurrences[i] * 5  # Scale bubble sizes for better visibility
    plt.scatter([x]*len(y), y, s=sizes, alpha=0.5, color=blues[1])
    # plt.scatter([x]*len(y), y, s=sizes, alpha=0.5, label=f"Interval {x}")


# Plotting the scatter plot with error bars
# plt.errorbar(real_interval_lst_update, model_interval_lst_mean_update, yerr=model_interval_lst_std_update, fmt='o', color=blues[1])
#              # label='After adpatation of c and k')

# plt.errorbar(real_interval_lst_update, model_interval_lst_mean_update, yerr=model_interval_lst_std_update, fmt='o', color='blue',
#              label='After update')

# Adding labels and title
plt.xlabel('Real Inter-Beat Intervals (ms)', fontsize=20)
plt.ylabel('Model Inter-Beat Intervals (ms)', fontsize=20)
# plt.title('Scatter Plot of Real vs. Model Intervals with Variance')



# Adding diagonal line where x equals y
plt.plot([min(real_interval_lst_update), max(real_interval_lst_update)], [min(real_interval_lst_update), max(real_interval_lst_update)],
         color=reds[0], linestyle='--', label='target interval')
         # color='red', linestyle='--', label='Diagonal Line (x=y)')


# Adding diagonal line where y = x - 150
x_values = np.array(real_interval_lst_update)
plt.plot(x_values, x_values / 2, color=reds[0], linestyle='--', label='one half target interval')
plt.plot(x_values, x_values / 3, color=reds[1], linestyle='--', label='one third target interval')
plt.plot(x_values, x_values / 4, color=reds[2], linestyle='--', label='one quarter target interval')
plt.plot(x_values, x_values / 6, color=reds[3], linestyle='--', label='one sixth target interval')


# Set y-axis limit to only show the part above y=0
plt.ylim(bottom=0)
# Displaying legend
plt.legend()
if save_state:
    plt.savefig(f'{save_path}/scatter_plot_1dim_update_c_k_interval_400to3000.pdf', dpi=300)

# Show the plot
plt.show()
#
#
# exit()



