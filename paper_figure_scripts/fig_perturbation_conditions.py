import numpy as np
import torch
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd

save_fig = True
# save_path = r'/home/zhyuan/Desktop/ESN/results_human_behavior/3_situation/figure_for_3_situation'
save_path = r'/home/zhyuan/Desktop/ESN/results_human_behavior/paper_figures_2_v1'

colors = [(255/255, 174/255, 176/255), (157/255, 196/255, 230/255), '#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']


def filter_peaks(peaks, min_distance=40):
    filtered_peaks = []
    current_peak = None

    for peak in peaks:
        if current_peak is None or peak - current_peak > min_distance:
            filtered_peaks.append(peak)
            current_peak = peak

    return np.array(filtered_peaks)

def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

labels = ['1 1 72 BPM', '1 2 72 BPM', '1 3 72 BPM', '2 2 72 BPM', '2 3 72 BPM', '3 3 72 BPM', '1 1 144 BPM', '1 2 144 BPM', '1 3 144 BPM', '2 2 144 BPM', '2 3 144 BPM', '3 3 144 BPM']
interval_lst = [142, 142, 142, 284, 284, 426, 68, 68, 68, 138, 138, 206]
# xlabels = ['1 slow', '2 slow', '3 slow', '1 fast', '2 fast', '3 fast']
xlabels = [str(interval * 6) for interval in interval_lst]



'''skip one plots'''
file_path_json = '/home/zhyuan/Desktop/ESN/results_human_behavior/3_situation/skip_one/maml_update_fb_results.json'
with open(file_path_json, 'r') as json_file:
    loaded_data = json.load(json_file)

# Define the plot sample indices
plot_sample_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Initialize a list to hold dataframes for each sample index
dfs = []

# Process each sample index
for i in range(len(plot_sample_index)):
    index = plot_sample_index[i]

    # Process participant data
    participant_performance = np.array(loaded_data['target'])[index, :, 0]
    p, _ = find_peaks(participant_performance, height=20)
    p = filter_peaks(p)
    participant_data = np.diff(p) * 6


    # Process model data
    model_performance = np.array(loaded_data['predicted'])[index, :, 0]
    p, _ = find_peaks(model_performance, height=20)
    p = filter_peaks(p)
    model_data = np.diff(p) * 6

    # Create dataframes for each sample index
    participant_df = pd.DataFrame({'Dataset': ['Target'] * len(participant_data),
                                   'Values': participant_data})
    model_df = pd.DataFrame({'Dataset': ['Model'] * len(model_data),
                             'Values': model_data})

    # Combine dataframes
    df = pd.concat([participant_df, model_df])
    df['Sample Index'] = labels[i]  # Add sample index as a column

    # Append dataframe to list
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs)

# Create violin plot
plt.figure(figsize=(10, 4))
sns.violinplot(x='Sample Index', y='Values', hue='Dataset', data=combined_df, split=True,
               inner="quartile", palette=[colors[0], colors[1]])

# Adjust x-axis labels to save space
plt.xticks(rotation=45)  # Rotate labels to make them more readable
plt.tick_params(axis='x', which='major', labelsize=10)  # Adjust font size

# Adjust labels and title
plt.xlabel("")
plt.ylabel("Inter-Beat Interval (ms)")

# Use tight layout to ensure everything fits without overlapping
plt.tight_layout()

# Save the plot if required
if save_fig:
    plt.savefig(f'{save_path}/skip_one.pdf', bbox_inches="tight")

# Show the plot
plt.show()



'''skip some time plots'''
file_path_json = '/home/zhyuan/Desktop/ESN/results_human_behavior/3_situation/skip_some_time/maml_update_fb_results.json'
with open(file_path_json, 'r') as json_file:
    loaded_data = json.load(json_file)

plot_sample_index = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
# plot_sample_index = [4]

model_data_lst = []
for i in range(len(plot_sample_index)):
    model_performance = np.array(loaded_data['predicted'])[plot_sample_index[i], :, 0]
    # p, _ = find_peaks(model_performance, height=20)
    # p = p[:-15]
    if plot_sample_index[i] == 0:
        p, _ = find_peaks(model_performance, height=30)
        p = filter_peaks(p, 30)[:-5]
    if plot_sample_index[i] == 1:
        p, _ = find_peaks(model_performance, height=30)
        p = filter_peaks(p, 30)[:-5]
    if plot_sample_index[i] == 2:
        p, _ = find_peaks(model_performance, height=30)
        p = filter_peaks(p, 30)[:-5]
    if plot_sample_index[i] == 3:
        p, _ = find_peaks(model_performance, height=30)
        p = filter_peaks(p, 75)[:-15]
    if plot_sample_index[i] == 4:
        p, _ = find_peaks(model_performance, height=30)
        p = filter_peaks(p, 75)[:-15]
    if plot_sample_index[i] == 6:
        p, _ = find_peaks(model_performance, height=10)
        p = filter_peaks(p, 30)[10:-30]
    if plot_sample_index[i] == 7:
        p, _ = find_peaks(model_performance, height=10)
        p = filter_peaks(p, 30)[10:-30]
    if plot_sample_index[i] == 8:
        p, _ = find_peaks(model_performance, height=5)
        p = filter_peaks(p, 30)[10:-30]
    if plot_sample_index[i] == 9:
        p, _ = find_peaks(model_performance, height=5)
        p = filter_peaks(p, 30)[10:-31]
    if plot_sample_index[i] == 10:
        p, _ = find_peaks(model_performance, height=5)
        p = filter_peaks(p, 30)[10:-31]
    if plot_sample_index[i] == 11:
        p, _ = find_peaks(model_performance, height=10)
        p = filter_peaks(p, 40)[5:-1]
        # print(np.diff(p))



    model_data = (np.diff(p) - interval_lst[plot_sample_index[i]]) * 6 / (interval_lst[plot_sample_index[i]] * 6)
    if plot_sample_index[i] == 3:
        # Find the index of the largest value
        arr = np.diff(p)
        max_index = np.argmax(arr)

        # Remove the largest value from the array
        arr_filtered = np.delete(arr, max_index)
        model_data = (arr_filtered - interval_lst[plot_sample_index[i]]) * 6 / (interval_lst[plot_sample_index[i]] * 6)
    if plot_sample_index[i] == 4:
        # Find the index of the largest value
        arr = np.diff(p)
        max_index = np.argmax(arr)

        # Remove the largest value from the array
        arr_filtered = np.delete(arr, max_index)
        model_data = (arr_filtered - interval_lst[plot_sample_index[i]]) * 6 / (interval_lst[plot_sample_index[i]] * 6)

    model_data_lst.append(model_data)

# Flatten the data to have shape (num_sample, sequence_length)
data_flat = np.array(model_data_lst)

plt.figure(figsize=(10, 4))
# Exclude outliers by setting showfliers=False
sns.boxplot(data=data_flat, color=colors[0], showfliers=False)  # Assuming colors[0] is your desired color

# Adjust x-axis labels for readability and space efficiency
plt.xticks(ticks=np.arange(len(plot_sample_index)),
           labels=[labels[i] for i in plot_sample_index],
           rotation=45,  # Rotate labels to save space and improve readability
           fontsize=10)  # Adjust font size to make sure labels are not too big

plt.ylabel("Inter-Beat Interval Offset Ratio")  # Adjusted y-axis label

# Use tight layout to optimize space usage
plt.tight_layout()

# Save the plot if required
if save_fig:
    plt.savefig(f'{save_path}/skip_some_time.pdf', bbox_inches="tight")

# Show the plot
plt.show()


'''increase two percent'''
file_path_json = '/home/zhyuan/Desktop/ESN/results_human_behavior/3_situation/increase_2percent/maml_update_fb_results.json'
ori_file_path_json = '/home/zhyuan/Desktop/ESN/results_human_behavior/3_situation/ori/maml_update_fb_results.json'

plot_sample_index = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]


# file_path_json = '/home/zhyuan/Desktop/ESN/results_human_behavior/3_situation/skip_one/maml_update_fb_results.json'
with open(file_path_json, 'r') as json_file:
    loaded_data = json.load(json_file)

with open(ori_file_path_json, 'r') as json_file:
    ori_loaded_data = json.load(json_file)

# Initialize a list to hold dataframes for each sample index
dfs = []




# Process each sample index
for i in range(len(plot_sample_index)):
    index = plot_sample_index[i]

    # Process participant data
    participant_performance = np.array(ori_loaded_data['predicted'])[index, :, 0]
    # participant_performance = np.array(ori_loaded_data['target'])[index, :, 0]
    p, _ = find_peaks(participant_performance, height=30)

    # if index == 1:
    #     p = filter_peaks(p, min_distance=200)
    #     participant_data = np.diff(p) * 6
    if index == 3:
        p = filter_peaks(p, min_distance=150)
        participant_data = (np.diff(p) - interval_lst[plot_sample_index[i]]) * 6 / (interval_lst[plot_sample_index[i]] * 6)
        # participant_data = np.diff(p) * 6
    else:
        p = filter_peaks(p)[2:-10]
        # participant_data = np.diff(p) * 6
        participant_data = (np.diff(p) - interval_lst[plot_sample_index[i]]) * 6 / (interval_lst[plot_sample_index[i]] * 6)


    # Process model data
    model_performance = np.array(loaded_data['predicted'])[index, :, 0]
    # model_performance = np.array(loaded_data['target'])[index, :, 0]
    p, _ = find_peaks(model_performance, height=30)

    # p = filter_peaks(p)
    # model_data = np.diff(p) * 6
    if index == 3:
        p = filter_peaks(p, min_distance=100)
        # model_data = np.diff(p) * 6
        model_data = (np.diff(p) - interval_lst[plot_sample_index[i]]) * 6 / (interval_lst[plot_sample_index[i]] * 6)

    else:
        p = filter_peaks(p)[2:-2]
        # model_data = np.diff(p) * 6
        model_data = (np.diff(p) - interval_lst[plot_sample_index[i]]) * 6 / (interval_lst[plot_sample_index[i]] * 6)


    # Create dataframes for each sample index
    participant_df = pd.DataFrame({'Dataset': ['Before increasing'] * len(participant_data),
                                   'Values': participant_data})
    model_df = pd.DataFrame({'Dataset': ['After increasing'] * len(model_data),
                             'Values': model_data})

    # Combine dataframes
    df = pd.concat([participant_df, model_df])
    df['Sample Index'] = labels[index]  # Add sample index as a column

    # Append dataframe to list
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs)

# Create box plot
plt.figure(figsize=(10, 4))
sns.boxplot(x='Sample Index', y='Values', hue='Dataset', data=combined_df,
            palette=[colors[0], colors[1]], showfliers=False)

# Rotate x-axis labels to save space and improve readability
plt.xticks(rotation=45, fontsize=10)  # Adjust rotation and font size as needed

# Adjust labels
plt.xlabel("")  # Removing the x-axis label as per your code
plt.ylabel("Inter-Beat Interval Offset Ratio")  # Adjusted y-axis label

# Optimize layout
plt.tight_layout()

# Save the plot if required
if save_fig:
    plt.savefig(f'{save_path}/increase_2precent.pdf', bbox_inches="tight")

# Show the plot
plt.show()