import numpy as np
# import sounddevice as sd
import scipy.io

import torch

import matplotlib.pyplot as plt
import copy

def list_sorter(input_list, sorted_indices):
    input_list_copy = copy.deepcopy(input_list)
    for l in range(len(input_list)):
        input_list_copy[l] = input_list[sorted_indices[l]]
    return input_list_copy


save_path = '/home/zhyuan/Desktop/ESN/figure_wcci_2024'
save_path = '/home/zhyuan/Desktop/ESN/results_human_behavior/paper_figures_2_v1'
save_state = False

'''ablation study'''
# ori = torch.load('/home/zhyuan/Desktop/ESN/ori.pt')
# only_damping = torch.load('/home/zhyuan/Desktop/ESN/only_damping.pt')
# combination = torch.load('/home/zhyuan/Desktop/ESN/damping_adaptc.pt')
# target = torch.load('/home/zhyuan/Desktop/ESN/target.pt')
# only_adapt_c = torch.load('/home/zhyuan/Desktop/ESN/only_adapt_c.pt')
# input_dim = 2
# output_dim = 2
# plot_length = 4000
# # print(only_damping.shape)
#
# # print(loaded_tensor)
# plt.plot(target.detach().cpu().reshape(12, -1, output_dim)[3, :plot_length, 1].data, 'r',
#          label=f'target')
# plt.plot(ori.detach().cpu().reshape(12, -1, output_dim)[3, :plot_length, 1].data, 'b',
#          label=f'wo_update')
# plt.plot(only_damping.detach().cpu().reshape(12, -1, output_dim)[3, :plot_length, 1].data, 'y', label=f'only add k')
# # plt.plot(only_adapt_c.detach().cpu().reshape(12, -1, input_dim)[3, :plot_length, 1].data, 'g', label=f'only_adapt_c')
# plt.plot(combination.detach().cpu().reshape(12, -1, input_dim)[3, :plot_length, 1].data, 'g', label=f'add k & adapt c')
# plt.legend()
# # plt.title(f'example_{i}_feature0_mse')
# plt.show()
#
# exit()

"""The box plot of the comparasion of the time interval of the MG and ours"""

# target_interval_lst = [
#     [86, 85, 85, 86, 85, 85, 86, 85, 85, 85, 86, 85, 85, 86, 85, 85, 86],
#     [171, 171, 171, 170, 171, 171, 170, 171],
#     # [257, 256, 256, 256, 256],
#     [42, 41, 41, 41, 41, 42, 41, 41, 41, 41, 42, 41, 41, 41, 41, 42, 41, 41, 41, 41, 42, 41, 41, 41, 41, 42, 41, 41, 41, 41, 42, 41, 41, 41, 41, 42],
#     [83, 82, 83, 82, 82, 83, 82, 83, 82, 82, 83, 82, 83, 82, 82, 83, 82, 83],
#     [124, 124, 123, 124, 123, 124, 124, 123, 124, 123, 124, 124]
# ]
pred_peak_lst = [
[868, 1011, 1154, 1295, 1438, 1579, 1722, 1864, 2006, 2148, 2290, 2432, 2574, 2717, 2859, 3001, 3143, 3285, 3427, 3569, 3712, 3854, 3996, 4138, 4280],
[1006, 1291, 1575, 1861, 2144, 2430, 2713, 2999, 3282, 3568, 3852, 4137],
# [1401, 1828, 2255, 2682, 3109, 3536, 3962, 4389],
[981, 1049, 1118, 1187, 1256, 1325, 1393, 1462, 1530, 1599, 1668, 1736, 1804, 1873, 1942, 2011, 2079, 2148, 2217, 2285, 2354, 2422, 2491, 2560, 2628, 2697, 2766, 2834, 2903, 2972, 3040, 3109, 3178],
[434, 571, 709, 847, 984, 1122, 1259, 1397, 1534, 1671, 1808, 1945, 2082, 2219, 2357, 2494, 2631, 2768, 2906, 3043, 3180, 3318, 3455, 3592, 3730, 3867, 4004, 4142, 4279],
[774, 978, 1185, 1389, 1598, 1802, 2009, 2213, 2421, 2625, 2833, 3037, 3245, 3449, 3657, 3861, 4069, 4273]
]

# target_interval_6ms = [85, 171, 256, 41, 82, 124]
target_interval_6ms = [142, 284, 68, 138, 206]
target_interval_10ms = [85, 171, 41, 82, 124]
# target_interval_10ms = [(x * 6/10) for x in target_interval_6ms]

interval_lst = target_interval_6ms
# xlabels = ['1 slow', '2 slow', '3 slow', '1 fast', '2 fast', '3 fast']
xlabels = [str(interval * 6) for interval in interval_lst]

original_indices = list(range(len(interval_lst)))

# Sort interval_lst and get sorted indices
sorted_indices = sorted(range(len(interval_lst)), key=lambda i: interval_lst[i])
xlabels = [xlabels[i] for i in sorted_indices]


# ours_interval_lst = [[140, 144, 142, 143, 142, 143, 141, 143, 142, 142, 142, 142, 143, 142, 142, 142],
#                      [290, 285, 285, 286, 285, 284],
#                      [70, 70, 68, 70, 69, 68, 70, 68, 69, 69, 68, 68, 69, 70, 68, 68
#                          , 69, 69, 69, 68, 69, 69, 68, 69, 69, 68, 69, 69, 68, 69],
#                      [138, 138, 139, 137, 147, 127, 137, 137, 137, 137, 137, 138, 137, 138, 137, 137, 137],
#                      [206, 205, 206, 206, 206, 205, 207, 206, 206, 206, 206]
#                      ]

ours_interval_lst = []
for l in range(len(pred_peak_lst)):
    ours_interval_lst.append(np.diff(pred_peak_lst[l]))

mg_interval_lst = [
    [100, 92, 99, 94, 101, 95, 100, 91, 105, 95, 100, 92, 105],
    [170, 160, 159, 163, 171, 151, 152, 173],
    [72, 69, 67, 69, 66, 71, 71, 72, 69, 70, 68, 70, 70, 69, 64, 72, 65, 68],
    [107, 101, 98, 96, 93, 91, 97, 97, 96, 97, 101, 96, 92],
    [135, 133, 125, 125, 119, 124, 122, 128, 128, 116]
]

def error(input_list, target_list, downsample_factor):
    error_lst = []
    for i in range(len(input_list)):
        error_ = []
        for j in range(len(input_list[i])):
            error_.append(np.abs(target_list[i] - input_list[i][j])*downsample_factor)
        error_lst.append(error_)
    return error_lst



def mean_and_var(input_lst, downsample_factor=None):
    mean_lst = []
    var_lst = []
    if downsample_factor is None:
        for interval in input_lst:
            mean = np.mean(interval)
            var = np.var(interval)
            # print(f"Mean: {round(mean)}, Variance: {var}")
            mean_lst.append(round(mean))
            var_lst.append(var)
    else:
        for interval in input_lst:
            mean = np.mean(np.array(interval*downsample_factor))
            var = np.var(np.array(interval*downsample_factor))
            mean_lst.append(round(mean))
            var_lst.append(var)
    return mean_lst, var_lst

ours_error_lst = error(ours_interval_lst, target_interval_6ms, 6)
mg_error_lst = error(mg_interval_lst, target_interval_10ms, 10)

# ours_mean_lst, ours_var_lst = mean_and_var(ours_error_lst)
# mg_mean_lst, mg_var_lst = mean_and_var(mg_error_lst)
ours_mean_lst, ours_var_lst = mean_and_var(ours_interval_lst, downsample_factor=6)
mg_mean_lst, mg_var_lst = mean_and_var(mg_interval_lst, downsample_factor=10)


# data = [ours_error_lst, mg_error_lst]
data = [list(zip(ours_mean_lst, ours_var_lst)), list(zip(mg_mean_lst, mg_var_lst))]
# data = [ours_interval_lst*6, mg_interval_lst*10]

# Sample labels
# xlabels = ['1 slow', '2 slow', '3 slow', '1 fast', '2 fast']
# xlabels = xlabels[:-1]

samples = list(range(len(xlabels)))

# Custom RGB colors
# colors = ['#86473F', '#E16B8C', '#90B44B']
# labels = ['Random', 'Wo_adaptation', 'Adaptation']
# colors = ['#0F2540', '#33A6B8']
# colors = ['#E4A8A8', '#DDCCD0']
# colors = ['#E5D2CF', '#92B6CA']
# colors = ['#80221E', '#AD7C59', '#B85C48', '#CABCAB']
# colors = ['#80221E', '#AD7C59', '#B85C48', '#CABCAB']
colors = [(255/255, 174/255, 176/255), (157/255, 196/255, 230/255), '#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']

labels = ['Ours', 'MG']

ours_interval_lst_copy = copy.deepcopy(ours_error_lst)
for l in range(len(ours_error_lst)):
    # for i in range(len(sorted_indices)):
    ours_interval_lst_copy[l] = ours_error_lst[sorted_indices[l]]
# print(ours_interval_lst_copy)

mg_interval_lst_copy = copy.deepcopy(mg_error_lst)
for l in range(len(mg_error_lst)):
    # for i in range(len(sorted_indices)):
    mg_interval_lst_copy[l] = mg_error_lst[sorted_indices[l]]


# Index for each sample
ind = np.arange(len(xlabels))

# Bar width
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))



# Plot bars for ours
ours_bars = ax.bar(ind - width/2, [np.mean(data) for data in ours_interval_lst_copy], width, label='Ours', yerr=[np.std(data) for data in ours_interval_lst_copy], capsize=2, color=colors[0])

# Plot bars for MG
mg_bars = ax.bar(ind + width/2, [np.mean(data) for data in mg_interval_lst_copy], width, label='MG', yerr=[np.std(data) for data in mg_interval_lst_copy], capsize=5, color=colors[1])

# Add error bars
for bar, data in zip(ours_bars + mg_bars, ours_interval_lst_copy + mg_interval_lst_copy):
    x = bar.get_x() + bar.get_width() / 2  # x-coordinate of the bar
    y = bar.get_height()  # Bar height
    ax.errorbar(x, y, yerr=np.std(data), fmt='none', ecolor='black', capsize=2)

# Set x-axis labels
ax.set_xticks(ind)
ax.set_xticklabels(xlabels)

# Set axis labels
ax.set_xlabel('Inter-Beat Interval (ms)')
ax.set_ylabel('Predicted Inter-Beat Interval Error (ms)')

ax.legend()

plt.xticks(rotation=45)  # Rotate labels for readability

plt.tight_layout()
if save_state:
    plt.savefig(f'{save_path}/compare_ours_mg.pdf', dpi=300)
plt.show()


# correct way to plot the box plot
'''data = [ours_interval_lst, mg_interval_lst]
# xlabels = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5', 'Group 6']  # X-axis labels for groups

fig, ax = plt.subplots()

positions = np.arange(len(xlabels)) + 1

boxplots = []  # To store boxplot artists for legend

for i, d in enumerate(data):
    pos = [p + i * 0.2 for p in positions]  # Positioning the boxplots
    bp = ax.boxplot(d, positions=pos, widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i]))
    boxplots.append(bp)  # Store boxplot artists for legend

# Adding labels and title
ax.set_xlabel('Inter-Beat Interval (ms)')
ax.set_ylabel('Predicted Inter-Beat Interval (ms)')
# ax.set_title('Comparison of Interval Data')

# Customizing x-axis ticks and labels
ax.set_xticks(positions + 0.2)
ax.set_xticklabels(xlabels)

# Adding legend
ax.legend([bp["boxes"][0] for bp in boxplots], ['Ours', 'MG'])

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed

plt.tight_layout()
plt.show()'''


# fig, ax = plt.subplots()
#
# positions = np.arange(len(xlabels)) + 1
#
# boxplots = []  # To store boxplot artists for legend
#
# for i, d in enumerate(data):
#     pos = [p + i * 0.2 for p in positions]  # Positioning the boxplots
#     bp = ax.boxplot(d, positions=pos, widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i]))
#     boxplots.append(bp)  # Store boxplot artists for legend
#
# # Adding labels and title
# ax.set_xlabel('Inter-Beat Interval (ms)')
# ax.set_ylabel('Predicted Inter-Beat Interval (ms)')
#
# # Customizing x-axis ticks and labels
# ax.set_xticks(positions + 0.2)
# ax.set_xticklabels(xlabels)
#
# # Adding legend
# ax.legend([bp["boxes"][0] for bp in boxplots], ['Ours', 'MG'])
#
# # # Create the figure with a single subplot
# # fig, ax = plt.subplots(figsize=(10, 5))
# #
# # # Apply logarithmic scale to y-axis
# # # ax.set_yscale('log')
# #
# # # Create legend handles with custom colors
# # legend_handles = []
# # for i, color in enumerate(colors):
# #     box = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10)
# #     # legend_handles.append((box, f'Table {i+1}'))
# #     legend_handles.append((box, labels[i]))
# #
# # # Plot box plots with custom RGB colors
# # data_sort = copy.deepcopy(data)
# # for l in range(len(data)):
# #     for i in range(len(sorted_indices)):
# #         data_sort[l][i] = data[l][sorted_indices[i]]
# #
# # for i, d in enumerate(data_sort):
# #     positions = [s + i*0.2 for s in samples]
# #     ax.boxplot(d, positions, widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i]))
# #     ax.set_xticks([s + 0.2 for s in samples])
# #     ax.set_xticklabels(xlabels)
# #     ax.set_xlabel('Inter-Beat Interval (ms)')
# #     ax.set_ylabel('Predicted Inter-Beat Interval (ms)')
#
#
# # Add legend with custom handles
# # ax.legend(*zip(*legend_handles))
#
# # Adjust layout
# plt.tight_layout()
# if save_state:
#     plt.savefig(f'{save_path}/compare_ours_mg.png', dpi=300)
#
# # Show the plot
# plt.show()


# exit()



"""The box plot of the comparasion of the wo and w adaptation"""
# data 0 1 2 6 7 8
target_peak_lst = [
[867, 1009, 1151, 1293, 1435, 1578, 1720, 1862, 2004, 2146, 2288, 2430, 2573, 2715, 2857, 2999, 3141, 3283, 3426, 3568, 3710, 3852, 3994, 4136, 4278],
[1010, 1294, 1578, 1863, 2147, 2432, 2716, 3001, 3285, 3570, 3854, 4139],
[1436, 1863, 2290, 2717, 3144, 3570, 3997],
[980, 1049, 1117, 1186, 1255, 1323, 1392, 1460, 1529, 1598, 1666, 1735, 1804, 1872, 1941, 2010, 2078, 2147, 2216, 2284, 2353, 2422, 2490, 2559, 2628, 2696, 2765, 2834, 2902, 2971, 3040, 3108, 3177],
[431, 568, 705, 843, 980, 1117, 1255, 1392, 1529, 1667, 1804, 1941, 2079, 2216, 2353, 2490, 2628, 2765, 2902, 3040, 3177, 3314, 3452, 3589, 3726, 3864, 4001, 4138, 4276],
[774, 980, 1186, 1392, 1598, 1804, 2010, 2216, 2422, 2628, 2834, 3040, 3246, 3452, 3658, 3864, 4070, 4276]
]

wo_ad_peak_lst = [
[868, 1011, 1154, 1295, 1438, 1579, 1722, 1864, 2006, 2148, 2290, 2432, 2574, 2717, 2859, 3001, 3143, 3285, 3427, 3569, 3712, 3854, 3996, 4138, 4280],
[1006, 1291, 1575, 1861, 2144, 2430, 2713, 2999, 3282, 3568, 3852, 4137],
[1401, 1828, 2255, 2682, 3109, 3536, 3962],
[981, 1049, 1118, 1187, 1256, 1325, 1393, 1462, 1530, 1599, 1668, 1736, 1804, 1873, 1942, 2011, 2079, 2148, 2217, 2285, 2354, 2422, 2491, 2560, 2628, 2697, 2766, 2834, 2903, 2972, 3040, 3109, 3178],
[434, 571, 709, 847, 984, 1122, 1259, 1397, 1534, 1671, 1808, 1945, 2082, 2219, 2357, 2494, 2631, 2768, 2906, 3043, 3180, 3318, 3455, 3592, 3730, 3867, 4004, 4142, 4279],
[774, 978, 1185, 1389, 1598, 1802, 2009, 2213, 2421, 2625, 2833, 3037, 3245, 3449, 3657, 3861, 4069, 4273]
]

# with_ad_peak_lst = [
# [867, 1010, 1154, 1293, 1436, 1578, 1721, 1862, 2005, 2146, 2289, 2430, 2572, 2713, 2846, 3001, 3143, 3286, 3428, 3571, 3713, 3855, 3998, 4140, 4283],
# [1007, 1291, 1576, 1861, 2146, 2430, 2715, 3000, 3284, 3568, 3853, 4138], #
# [1436, 1862, 2288, 2715, 3141, 3569, 3994], #
# [981, 1048, 1118, 1187, 1256, 1325, 1394, 1463, 1530, 1599, 1668, 1736, 1803, 1873, 1942, 2011, 2079, 2148, 2217, 2285, 2354, 2422, 2491, 2560, 2628, 2697, 2766, 2834, 2903, 2972, 3040, 3109, 3178], #
# [434, 574, 709, 844, 982, 1119, 1257, 1394, 1531, 1669, 1807, 1944, 2082, 2219, 2356, 2493, 2629, 2765, 2901, 3038, 3175, 3312, 3450, 3587, 3725, 3862, 3999, 4137, 4274], #
# [772, 977, 1184, 1389, 1596, 1802, 2009, 2214, 2421, 2626, 2834, 3039, 3247, 3451, 3659, 3863, 4071, 4275] #
# ]
with_ad_peak_lst = [
[867, 1010, 1154, 1293, 1436, 1578, 1721, 1862, 2005, 2146, 2289, 2430, 2572, 2713, 2855, 2995, 3138, 3278, 3422, 3562, 3706, 3847, 3990, 4132, 4276],
[1008, 1296, 1581, 1870, 2146, 2430, 2715, 3000, 3284, 3568, 3853, 4138], #
[1436, 1862, 2288, 2715, 3141, 3569, 3994], #
[981, 1048, 1118, 1187, 1256, 1325, 1394, 1463, 1530, 1599, 1668, 1736, 1803, 1873, 1942, 2011, 2079, 2148, 2217, 2285, 2354, 2422, 2491, 2560, 2628, 2697, 2766, 2834, 2903, 2972, 3040, 3109, 3178], #
[434, 574, 709, 844, 982, 1119, 1257, 1394, 1531, 1669, 1807, 1944, 2082, 2219, 2356, 2493, 2629, 2765, 2901, 3038, 3175, 3312, 3450, 3587, 3725, 3862, 3999, 4137, 4274], #
[772, 977, 1184, 1389, 1596, 1802, 2009, 2214, 2421, 2626, 2834, 3039, 3247, 3451, 3659, 3863, 4071, 4275] #
]

interval_lst = [142, 284, 426, 68, 138, 206]
# xlabels = ['1 slow', '2 slow', '3 slow', '1 fast', '2 fast', '3 fast']
xlabels = [str(interval * 6) for interval in interval_lst]

original_indices = list(range(len(interval_lst)))

# Sort interval_lst and get sorted indices
sorted_indices = sorted(range(len(interval_lst)), key=lambda i: interval_lst[i])
xlabels = [xlabels[i] for i in sorted_indices]
# Data from the tables
difference_wo_lst = []
sampels = []
data = []

mean_wo_lst = []
mean_with_lst = []
var_wo_lst = []
var_with_lst = []

error_wo = []
error_with = []


for i in range(len(target_peak_lst)):
    difference_wo_lst = []
    difference_with_lst = []
    difference_lst_sub = []
    # print(len(pred_peak_lst[i]))
    # print(len(add_10_lst[i]))
    for j in range(len(target_peak_lst[i])):
        difference_wo_lst.append((-wo_ad_peak_lst[i][j]+target_peak_lst[i][j])/interval_lst[i])
        difference_with_lst.append((-with_ad_peak_lst[i][j]+target_peak_lst[i][j])/interval_lst[i])
    # if i != 0:
    # difference_add_10_lst = normalize_list(difference_add_10_lst)
    # difference_add_5_lst = normalize_list(difference_add_5_lst)
    # difference_minus_5_lst = normalize_list(difference_minus_5_lst)
    # difference_minus_10_lst = normalize_list(difference_minus_10_lst)
    error_wo.append(difference_wo_lst)
    error_with.append(difference_with_lst)

    mean_wo_ele, var_wo_ele = np.mean(difference_wo_lst), np.var(difference_wo_lst)
    mean_with_ele, var_with_ele = np.mean(difference_with_lst), np.var(difference_with_lst)
    mean_wo_lst.append(mean_wo_ele)
    mean_with_lst.append(mean_with_ele)
    var_wo_lst.append(var_wo_ele)
    var_with_lst.append(var_with_ele)


# Sample labels
# samples = list(range(12))
# xlabels = ['1 slow', '2 slow', '3 slow', '1 fast', '2 fast', '3 fast']
# Sort interval_lst and get sorted indices
# sorted_indices = sorted(range(len(interval_lst)), key=lambda i: interval_lst[i])
# xlabels = [xlabels[i] for i in sorted_indices]
mean_wo_lst = [mean_wo_lst[i] for i in sorted_indices]
var_wo_lst = [var_wo_lst[i] for i in sorted_indices]
mean_with_lst = [mean_with_lst[i] for i in sorted_indices]
var_with_lst = [var_with_lst[i] for i in sorted_indices]


# data = [list(zip(mean_wo_lst, var_wo_lst)), list(zip(mean_with_lst, var_with_lst))]
#
# samples = list(range(len(target_peak_lst)))
#
# # Custom RGB colors
# # colors = ['#86473F', '#E16B8C', '#90B44B']
# # labels = ['Random', 'Wo_adaptation', 'Adaptation']
# # colors = ['#0F2540', '#33A6B8']
# colors = ['#92B6CA', '#E5D2CF']
# labels = ['Wo_Adaptation', 'Adaptation']
#
# # Create the figure with a single subplot
# fig, ax = plt.subplots(figsize=(10, 5))
#
# # Apply logarithmic scale to y-axis
# # ax.set_yscale('log')
#
# # Create legend handles with custom colors
# legend_handles = []
# for i, color in enumerate(colors):
#     box = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10)
#     # legend_handles.append((box, f'Table {i+1}'))
#     legend_handles.append((box, labels[i]))
#
# # Plot box plots with custom RGB colors
# for i, d in enumerate(data):
#     positions = [s + i*0.2 for s in samples]
#     ax.boxplot(d, positions=positions, widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i]))
#     ax.set_xticks([s + 0.2 for s in samples])
#     ax.set_xticklabels(xlabels)
#     ax.set_xlabel('Inter-Beat Interval (ms)')
#     ax.set_ylabel('Time Offset Ratio')
#
# # Add legend with custom handles
# ax.legend(*zip(*legend_handles))
#
# # Adjust layout
# plt.tight_layout()
# if save_state:
#     plt.savefig(f'{save_path}/compare_wo_and_with_adaptation.png', dpi=300)

# # Show the plot
# plt.show()

# colors = ['#92B6CA', '#E5D2CF']
# colors = ['#80221E', '#AD7C59', '#B85C48', '#CABCAB']
labels = ['Wo_Adaptation', 'Adaptation']

# data = [error_wo, error_with]


error_wo_copy = copy.deepcopy(error_wo)
for l in range(len(error_wo)):
    # for i in range(len(sorted_indices)):
    error_wo_copy[l] = error_wo[sorted_indices[l]]

error_with_copy = copy.deepcopy(error_with)
for l in range(len(error_with)):
    # for i in range(len(sorted_indices)):
    error_with_copy[l] = error_with[sorted_indices[l]]



# Index for each sample
ind = np.arange(len(xlabels))

# Bar width
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))


# Plot bars for the first condition
ours_bars = ax.bar(ind - width/2, [np.mean(data) for data in error_wo_copy], width, label=labels[0], yerr=[np.std(data) for data in error_wo_copy], capsize=2, color=colors[0])

# Plot bars for the second condition
mg_bars = ax.bar(ind + width/2, [np.mean(data) for data in error_with_copy], width, label=labels[1], yerr=[np.std(data) for data in error_with_copy], capsize=2, color=colors[1])

# Add error bars
for bar, data in zip(ours_bars + mg_bars, error_wo_copy + error_with_copy):
    x = bar.get_x() + bar.get_width() / 2  # x-coordinate of the bar
    y = bar.get_height()  # Bar height
    ax.errorbar(x, y, yerr=np.std(data), fmt='none', ecolor='black', capsize=2)

# Set x-axis labels
ax.set_xticks(ind)
ax.set_xticklabels(xlabels)

# Set axis labels
ax.set_xlabel('Inter-Beat Interval (ms)')
ax.set_ylabel('Time Offset Ratio')

ax.legend()

plt.xticks(rotation=45)  # Rotate labels for readability

plt.tight_layout()
if save_state:
    plt.savefig(f'{save_path}/compare_wo_and_with_adaptation.pdf', dpi=300)
plt.show()

'''fig, ax = plt.subplots()

positions = np.arange(len(xlabels)) + 1

boxplots = []  # To store boxplot artists for legend

for i, d in enumerate(data):
    pos = [p + i * 0.2 for p in positions]  # Positioning the boxplots
    bp = ax.boxplot(d, positions=pos, widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i]))
    boxplots.append(bp)  # Store boxplot artists for legend

# Adding labels and title
ax.set_xlabel('Inter-Beat Interval (ms)')
ax.set_ylabel('Time Offset Ratio')
# ax.set_title('Comparison of Interval Data')

# Customizing x-axis ticks and labels
ax.set_xticks(positions + 0.2)
ax.set_xticklabels(xlabels)

# Adding legend
ax.legend([bp["boxes"][0] for bp in boxplots], labels)

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed

plt.tight_layout()
plt.show()'''





""""Analysis c (singal channel)"""
# data 0 1 2 6 7 8
# target_peak_lst = [
# [156, 298, 440, 583, 725, 867, 1009, 1151, 1293, 1435, 1578, 1720, 1862, 2004, 2146, 2288, 2430, 2573, 2715, 2857, 2999, 3141, 3283, 3426, 3568, 3710, 3852, 3994, 4136, 4278],
# [156, 441, 725, 1010, 1294, 1578, 1863, 2147, 2432, 2716, 3001, 3285, 3570, 3854, 4139],
# [156, 583, 1010, 1436, 1863, 2290, 2717, 3144, 3570, 3997],
# [156, 225, 293, 362, 431, 499, 568, 637, 705, 774, 843, 911, 980, 1049,
#               1117, 1186, 1255, 1323, 1392, 1460, 1529, 1598, 1666, 1735, 1804, 1872,
#               1941, 2010, 2078, 2147, 2216, 2284, 2353, 2422, 2490, 2559, 2628, 2696,
#               2765, 2834, 2902, 2971, 3040, 3108, 3177, 3246, 3314, 3383, 3452, 3520,
#               3589, 3658, 3726, 3795, 3864, 3932, 4001, 4069, 4138, 4207, 4275],
# [156, 293, 431, 568, 705, 843, 980, 1117, 1255, 1392, 1529, 1667,
#               1804, 1941, 2079, 2216, 2353, 2490, 2628, 2765, 2902, 3040,
#               3177, 3314, 3452, 3589, 3726, 3864, 4001, 4138, 4276],
# [156, 362, 568, 774, 980, 1186, 1392, 1598, 1804, 2010, 2216,
#               2422, 2628, 2834, 3040, 3246, 3452, 3658, 3864, 4070, 4276]
# ]

# 0
pred_peak_lst = [
[868, 1011, 1154, 1295, 1438, 1579, 1722, 1864, 2006, 2148, 2290, 2432, 2574, 2717, 2859, 3001, 3143, 3285, 3427, 3569, 3712, 3854, 3996, 4138, 4280],
[1006, 1291, 1575, 1861, 2144, 2430, 2713, 2999, 3282, 3568, 3852, 4137],
[1401, 1828, 2255, 2682, 3109, 3536, 3962, 4389],
[981, 1049, 1118, 1187, 1256, 1325, 1393, 1462, 1530, 1599, 1668, 1736, 1804, 1873, 1942, 2011, 2079, 2148, 2217, 2285, 2354, 2422, 2491, 2560, 2628, 2697, 2766, 2834, 2903, 2972, 3040, 3109, 3178],
[434, 571, 709, 847, 984, 1122, 1259, 1397, 1534, 1671, 1808, 1945, 2082, 2219, 2357, 2494, 2631, 2768, 2906, 3043, 3180, 3318, 3455, 3592, 3730, 3867, 4004, 4142, 4279],
[774, 978, 1185, 1389, 1598, 1802, 2009, 2213, 2421, 2625, 2833, 3037, 3245, 3449, 3657, 3861, 4069, 4273]
]


# 0.1
add_10_lst = [
[862, 1005, 1145, 1289, 1430, 1574, 1715, 1859, 1999, 2143, 2284, 2427, 2568, 2711, 2853, 2996, 3137, 3280, 3421, 3564, 3705, 3849, 3990, 4133, 4274],
[844, 1129, 1414, 1699, 1984, 2269, 2552, 2837, 3121, 3406, 3690, 3974],
[1125, 1553, 1979, 2407, 2833, 3261, 3687, 4114],
[988, 1056, 1125, 1194, 1263, 1335, 1400, 1474, 1538, 1606, 1675, 1744, 1812, 1881, 1950, 2018, 2087, 2156, 2224, 2293, 2362, 2430, 2499, 2568, 2636, 2705, 2773, 2842, 2911, 2980, 3048, 3117, 3185],
[426, 567, 703, 844, 980, 1117, 1253, 1393, 1527, 1666, 1802, 1941, 2077, 2216, 2352, 2490, 2626, 2765, 2901, 3040, 3175, 3314, 3450, 3589, 3725, 3864, 3999, 4138, 4274],
[770, 980, 1187, 1393, 1599, 1805, 2012, 2217, 2424, 2629, 2836, 3040, 3248, 3452, 3659, 3864, 4071, 4276]
]

# 0.05
add_5_lst = [
[868, 1009, 1151, 1292, 1436, 1576, 1719, 1860, 2004, 2145, 2288, 2429, 2573, 2713, 2857, 2998, 3141, 3282, 3425, 3566, 3710, 3850, 3994, 4135, 4278],
[1005, 1290, 1574, 1859, 2144, 2428, 2713, 2997, 3281, 3566, 3850, 4135],
[1381, 1808, 2236, 2662, 3089, 3516, 3943, 4367],
[994, 1062, 1132, 1201, 1270, 1337, 1406, 1475, 1543, 1612, 1680, 1749, 1818, 1886, 1955, 2024, 2092, 2161, 2230, 2298, 2367, 2436, 2504, 2573, 2642, 2710, 2779, 2847, 2916, 2985, 3054, 3122, 3191],
[429, 569, 707, 845, 982, 1121, 1257, 1394, 1532, 1669, 1807, 1944, 2082, 2218, 2356, 2493, 2631, 2768, 2905, 3042, 3180, 3317, 3454, 3592, 3729, 3866, 4004, 4141, 4278],
[796, 987, 1191, 1397, 1600, 1808, 2011, 2220, 2423, 2631, 2835, 3043, 3246, 3455, 3658, 3867, 4070, 4279]
]

# -0.05
minus_5_lst = [
[868, 1011, 1154, 1296, 1439, 1582, 1724, 1866, 2009, 2150, 2293, 2434, 2576, 2718, 2861, 3003, 3145, 3287, 3430, 3572, 3714, 3856, 3998, 4140, 4282],
[1035, 1301, 1613, 1864, 2144, 2432, 2712, 3001, 3282, 3570, 3850, 4139],
[1401, 1828, 2255, 2682, 3109, 3536, 3962, 4389],
[985, 1054, 1123, 1192, 1259, 1329, 1396, 1466, 1534, 1603, 1671, 1741, 1808, 1876, 1945, 2014, 2083, 2151, 2220, 2288, 2357, 2426, 2494, 2563, 2632, 2701, 2769, 2838, 2906, 2975, 3044, 3112, 3181],
[435, 572, 707, 845, 981, 1120, 1256, 1395, 1531, 1669, 1805, 1944, 2080, 2218, 2355, 2492, 2629, 2767, 2904, 3041, 3178,
 3316, 3453, 3590, 3728, 3865, 4002, 4140, 4277],
[771, 978, 1180, 1387, 1592, 1799, 2005, 2211, 2416, 2623, 2829, 3035, 3241, 3447, 3653, 3859, 4065, 4271]
]


# -0.1
minus_10_lst = [
[871, 1012, 1155, 1296, 1439, 1580, 1722, 1864, 2007, 2149, 2291, 2433, 2575, 2717, 2859, 3001, 3144, 3286, 3428, 3570, 3712, 3854, 3996, 4138, 4281],
[1007, 1291, 1576, 1860, 2145, 2429, 2714, 2998, 3283, 3568, 3852, 4136],
[1435, 1861, 2288, 2715, 3141, 3569, 3995, 4390],
[985, 1054, 1123, 1192, 1260, 1329, 1397, 1466, 1534, 1603, 1671, 1741, 1808, 1878, 1946, 2015, 2083, 2152, 2221, 2289, 2358, 2427, 2495, 2564, 2633, 2701, 2770, 2839, 2907, 2976, 3045, 3113, 3182],
[439, 574, 710, 844, 983, 1120, 1257, 1394, 1531, 1669, 1806, 1943, 2081, 2218, 2355, 2493, 2630, 2767, 2905, 3042, 3179, 3316, 3454, 3591, 3728, 3866, 4003, 4140, 4278],
[772, 977, 1184, 1389, 1596, 1801, 2008, 2213, 2420, 2625, 2832, 3037, 3244, 3449, 3656, 3861, 4068, 4273]
]



def normalize_list(input_list):
    min_val = min(input_list)
    max_val = max(input_list)
    if max_val - min_val == 0:
        normalized_list = input_list
    else:
        normalized_list = [2 * (x - min_val) / (max_val - min_val) - 1 for x in input_list]
    # normalized_value = 2 * (x - min_val) / (max_val - min_val) - 1
    return normalized_list


difference_lst = []
sampels = []
data = []
diff_add_10_mean_lst = []
diff_add_5_mean_lst = []
diff_minus_5_mean_lst = []
diff_minus_10_mean_lst = []
diff_add_10_var_lst = []
diff_add_5_var_lst = []
diff_minus_5_var_lst = []
diff_minus_10_var_lst = []

mean_add10_lst = []
mean_add5_lst = []
mean_minus5_lst = []
mean_minus10_lst = []
var_add10_lst = []
var_add5_lst = []
var_minus5_lst = []
var_minus10_lst = []

interval_lst = [142, 284, 426, 68, 138, 206]

error_add_10 = []
error_add_5 = []
error_minus_5 = []
error_minus_10 = []


for i in range(len(pred_peak_lst)):
    difference_add_10_lst = []
    difference_add_5_lst = []
    difference_minus_5_lst = []
    difference_minus_10_lst = []
    difference_lst_sub = []
    # print(len(pred_peak_lst[i]))
    # print(len(add_10_lst[i]))
    for j in range(len(pred_peak_lst[i])):
        difference_add_10_lst.append((-pred_peak_lst[i][j]+add_10_lst[i][j])/interval_lst[i])
        difference_add_5_lst.append((-pred_peak_lst[i][j]+add_5_lst[i][j])/interval_lst[i])
        difference_minus_5_lst.append((-pred_peak_lst[i][j]+minus_5_lst[i][j])/interval_lst[i])
        difference_minus_10_lst.append((-pred_peak_lst[i][j]+minus_10_lst[i][j])/interval_lst[i])
    # if i != 0:
    # difference_add_10_lst = normalize_list(difference_add_10_lst)
    # difference_add_5_lst = normalize_list(difference_add_5_lst)
    # difference_minus_5_lst = normalize_list(difference_minus_5_lst)
    # difference_minus_10_lst = normalize_list(difference_minus_10_lst)
    error_add_10.append(difference_add_10_lst)
    error_add_5.append(difference_add_5_lst)
    error_minus_5.append(difference_minus_5_lst)
    error_minus_10.append(difference_minus_10_lst)

    mean_add10_ele, var_add10_ele = np.mean(difference_add_10_lst), np.var(difference_add_10_lst)
    mean_add5_ele, var_add5_ele = np.mean(difference_add_5_lst), np.var(difference_add_5_lst)
    mean_minus5_ele, var_minus5_ele = np.mean(difference_minus_5_lst), np.var(difference_minus_5_lst)
    mean_minus10_ele, var_minus10_ele = np.mean(difference_minus_10_lst), np.var(difference_minus_10_lst)
    mean_add10_lst.append(mean_add10_ele)
    mean_add5_lst.append(mean_add5_ele)
    mean_minus5_lst.append(mean_minus5_ele)
    mean_minus10_lst.append(mean_minus10_ele)
    var_add10_lst.append(var_add10_ele)
    var_add5_lst.append(var_add5_ele)
    var_minus5_lst.append(var_minus5_ele)
    var_minus10_lst.append(var_minus10_ele)


interval_lst = [142, 284, 426, 68, 138, 206]
# xlabels = ['1 slow', '2 slow', '3 slow', '1 fast', '2 fast', '3 fast']
xlabels = [str(interval * 6) for interval in interval_lst]

original_indices = list(range(len(interval_lst)))

# Sort interval_lst and get sorted indices
sorted_indices = sorted(range(len(interval_lst)), key=lambda i: interval_lst[i])
xlabels = [xlabels[i] for i in sorted_indices]
mean_add10_lst = [mean_add10_lst[i] for i in sorted_indices]
var_add10_lst = [var_add10_lst[i] for i in sorted_indices]
mean_add5_lst = [mean_add5_lst[i] for i in sorted_indices]
var_add5_lst = [var_add5_lst[i] for i in sorted_indices]
mean_minus5_lst = [mean_minus5_lst[i] for i in sorted_indices]
var_minus5_lst = [var_minus5_lst[i] for i in sorted_indices]
mean_minus10_lst = [mean_minus10_lst[i] for i in sorted_indices]
var_minus10_lst = [var_minus10_lst[i] for i in sorted_indices]


data = [list(zip(mean_add10_lst, var_add10_lst)), list(zip(mean_add5_lst, var_add5_lst)), list(zip(mean_minus5_lst, var_minus5_lst)), list(zip(mean_minus10_lst, var_minus10_lst))]

samples = list(range(len(pred_peak_lst)))


# Custom RGB colors
# colors = ['#0F2540', '#33A6B8', '#91AD70', '#4D5139']
# colors = ['#516A76', '#B3BCBB', '#D8B6AB', '#E4DBD3']
# colors = ['#546686', '#92B6CA', '#CBDDEE', '#E5D2CF']
# colors = ['#80221E', '#AD7C59', '#B85C48', '#CABCAB']
labels = ['+0.1', '+0.05', '-0.05', '-0.1']

# data = [error_add_10, error_add_5, error_minus_5, error_minus_10]


# error_wo_copy = copy.deepcopy(error_wo)
# for l in range(len(error_wo)):
#     # for i in range(len(sorted_indices)):
#     error_wo_copy[l] = error_wo[sorted_indices[l]]
#
# error_with_copy = copy.deepcopy(error_with)
# for l in range(len(error_with)):
#     # for i in range(len(sorted_indices)):
#     error_with_copy[l] = error_with[sorted_indices[l]]

error_add_10_copy = list_sorter(error_add_10, sorted_indices)
error_add_5_copy = list_sorter(error_add_5, sorted_indices)
error_minus_5_copy = list_sorter(error_minus_5, sorted_indices)
error_minus_10_copy = list_sorter(error_minus_10, sorted_indices)



# Index for each sample
ind = np.arange(len(xlabels))

# Bar width
width = 0.2  # Use a narrower width to fit four bar groups

fig, ax = plt.subplots(figsize=(10, 5))

# Plot bars for add_10
add_10_bars = ax.bar(ind - width * 3 / 2, [np.mean(data) for data in error_add_10_copy], width, label=labels[0], yerr=[np.std(data) for data in error_add_10_copy], capsize=2, color=colors[2])

# Plot bars for add_5
add_5_bars = ax.bar(ind - width / 2, [np.mean(data) for data in error_add_5_copy], width, label=labels[1], yerr=[np.std(data) for data in error_add_5_copy], capsize=2, color=colors[3])

# Plot bars for minus_5
minus_5_bars = ax.bar(ind + width / 2, [np.mean(data) for data in error_minus_5_copy], width, label=labels[2], yerr=[np.std(data) for data in error_minus_5_copy], capsize=2, color=colors[4])

# Plot bars for minus_10
minus_10_bars = ax.bar(ind + width * 3 / 2, [np.mean(data) for data in error_minus_10_copy], width, label=labels[3], yerr=[np.std(data) for data in error_minus_10_copy], capsize=2, color=colors[5])

# Add error bars
for bars, error_data in zip([add_10_bars, add_5_bars, minus_5_bars, minus_10_bars], [error_add_10_copy, error_add_5_copy, error_minus_5_copy, error_minus_10_copy]):
    for bar, data in zip(bars, error_data):
        x = bar.get_x() + bar.get_width() / 2  # x-coordinate of the bar
        y = bar.get_height()  # Bar height
        ax.errorbar(x, y, yerr=np.std(data), fmt='none', ecolor='black', capsize=2)

# Set x-axis labels
ax.set_xticks(ind)
ax.set_xticklabels(xlabels)

# Set axis labels
ax.set_xlabel('Inter-Beat Interval (ms)')
ax.set_ylabel('Time Offset Ratio')

ax.legend()

plt.xticks(rotation=45)  # Rotate labels for readability

plt.tight_layout()

plt.tight_layout()
if save_state:
    plt.savefig(f'{save_path}/adapt_c.pdf', dpi=300)
plt.show()


'''# Create the figure with a single subplot
fig, ax = plt.subplots(figsize=(10, 5))

# Create legend handles with custom colors
legend_handles = []
for i, color in enumerate(colors):
    box = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10)
    # legend_handles.append((box, f'Table {i+1}'))
    legend_handles.append((box, labels[i]))

# Plot box plots with custom RGB colors
for i, d in enumerate(data):
    positions = [s + i*0.2 for s in samples]
    # truncated_data = np.clip(d, -0.2, 0.2)
    ax.boxplot(d, positions=positions, widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i]), showfliers=False)
    ax.set_xticks([s + 0.2 for s in samples])
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Inter-Beat Interval (ms)')
    ax.set_ylabel('Time Offset Ratio')

# Add legend with custom handles
ax.legend(*zip(*legend_handles))

# Adjust layout
plt.tight_layout()
# plt.ylim(-0.45, 0.25)
if save_state:
    plt.savefig(f'{save_path}/adapt_c.png', dpi=300)

# Show the plot
plt.show()'''


# # Plotting histograms for each list in data
# for i, lst in enumerate(data, start=1):
#     plt.hist(lst, bins=20, alpha=0.7, label=f'List {i}')
#
# plt.legend()
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.title('Histogram of Data Lists')
# plt.show()



