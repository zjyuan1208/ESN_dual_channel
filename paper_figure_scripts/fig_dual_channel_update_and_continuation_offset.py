import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


save_path = '/home/zhyuan/Desktop/ESN/figure_arxiv_2024'
save_path = '/home/zhyuan/Desktop/ESN/results_human_behavior/paper_figures_2_v1'
save_state = False

blues = ['#115699', '#0E6DB3', '#5CAAD7', '#95C6DE']
reds = ['#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']
colors = [(255/255, 174/255, 176/255), (157/255, 196/255, 230/255), '#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']
# colors = [reds[2], blues[2], '#8E0D29', '#BB1E38', '#D35B4D', '#F6BCA9']


# Provided data
data = {
    'sample 0': {
        'target_peaks_feat_0': [441, 583, 726, 868, 1010, 1152, 1294, 1436, 1578, 1721, 1863, 2005, 2147, 2289, 2431, 2573],
        'pred_peaks_feat_0':   [439, 582, 724, 869, 1011, 1153, 1295, 1437, 1579, 1721, 1863, 1999, 2148, 2290, 2432, 2574, 2716, 2861, 3005, 3147, 3291, 3434, 3577, 3719, 3862, 4004, 4146, 4288, 4431, 4569, 4704, 4838],
        'target_peaks_feat_1': [441, 583, 726, 868, 1010, 1152, 1294, 1436, 1578, 1721, 1863, 2005, 2147, 2289, 2431, 2573, 2716, 2858, 3000, 3142, 3284, 3426, 3568, 3711, 3853, 3995, 4137, 4279],
        'pred_peaks_feat_1':   [440, 582, 724, 869, 1011, 1153, 1295, 1437, 1580, 1721, 1863, 1999, 2148, 2290, 2432, 2574, 2716, 2862, 3004, 3147, 3291, 3434, 3577, 3720, 3862, 4004, 4146, 4288, 4431, 4570, 4706, 4840]
    },
    'sample 1': {
        'target_peaks_feat_0': [299, 441, 583, 726, 868, 1010, 1152, 1294, 1436, 1578, 1721, 1863, 2005, 2147, 2289, 2431, 2573],
        'pred_peaks_feat_0':   [299, 432, 585, 723, 871, 1008, 1156, 1294, 1440, 1579, 1724, 1864, 2008, 2149, 2293, 2433, 2577, 2719, 2870, 3015, 3164, 3307, 3455, 3614, 3752, 3911, 4061, 4214, 4365, 4515, 4668, 4823],
        'target_peaks_feat_1': [726, 1010, 1295, 1579, 1864, 2148, 2433, 2717, 3002, 3286, 3570, 3855, 4139],
        'pred_peaks_feat_1':   [724, 1008, 1294, 1582, 1866, 2150, 2436, 2720, 3024, 3324, 3615, 3906, 4065, 4209, 4367, 4521, 4672, 4826]
    },
    'sample 2': {
        'target_peaks_feat_0': [299, 441, 583, 726, 868, 1010, 1152, 1294, 1436, 1578, 1721, 1863, 2005, 2147, 2289, 2431, 2573],
        'pred_peaks_feat_0':   [300, 443, 582, 728, 871, 1011, 1155, 1297, 1438, 1581, 1723, 1864, 2007, 2150, 2291, 2434, 2576, 2717, 2865, 3013, 3158, 3305, 3455, 3601, 3745, 3892, 4050, 4195, 4342],
        'target_peaks_feat_1': [1010, 1437, 1864, 2291, 2718, 3144, 3571, 3998], # the prediction has an extra peak between two peaks
        'pred_peaks_feat_1':   [1018, 1444, 1869, 2296, 2723, 3172, 3616, 3898, 4051, 4201, 4344, 4494, 4646, 4797, 4949]
        # 'pred_peaks_feat_1':   [1018, 1300, 1444, 1727, 1869, 2153, 2296, 2581, 2723, 3019, 3172, 3461, 3616, 3755, 3898, 4051, 4201, 4344, 4494, 4646, 4797, 4949]
    },
    'sample 3': {
        'target_peaks_feat_0': [1010, 1295, 1579, 1864, 2148, 2433],
        'pred_peaks_feat_0':   [993, 1289, 1574, 1859, 2147, 2432, 2719, 3009, 3298, 3587, 3874, 4162, 4448],
        'target_peaks_feat_1': [1295, 1579, 1864, 2148, 2433, 2717, 3002, 3286, 3570, 3855, 4139],
        'pred_peaks_feat_1':   [1292, 1577, 1861, 2147, 2431, 2716, 3006, 3297, 3593, 3882, 4168, 4443, 4617, 4740, 4909]
    },
    # 'sample 4': {
    #     'target_peaks_feat_0': [441, 726, 1010, 1295, 1579, 1864, 2148, 2433],
    #     'pred_peaks_feat_0':   [441, 707, 1000, 1296, 1563, 1854, 2150, 2425, 2710, 2997, 3276, 3549, 3842, 4141, 4398, 4447, 4695, 4747],
    #     'target_peaks_feat_1': [584, 1010, 1437, 1864, 2291, 2718, 3144, 3571, 3998], # fail (multiple 3 is too much)
    #     'pred_peaks_feat_1':   [585, 872, 2577, 3422, 4017, 4320, 4622, 4922]
    # },
    # 'sample 5': {
    #     'target_peaks_feat_0': [584, 1010, 1437, 1864, 2291],
    #     'pred_peaks_feat_0':   [441, 867, 1293, 1720, 2147, 2574, 2864, 2997, 3034, 3072, 3283, 3332, 3601, 3899, 4222, 4543, 4860],
    #     'target_peaks_feat_1': [157, 584, 1010, 1437, 1864, 2291, 2718, 3144, 3571, 3998], # fail (multiple 3 is too much)
    #     'pred_peaks_feat_1':   [2995, 3433, 4047, 4726]
    # },
    'sample 6':{
    'target_peaks_feat_0': [637, 706, 775, 843, 912, 980, 1049, 1118, 1186, 1255, 1324, 1392, 1461, 1530, 1598, 1667, 1736, 1804, 1873, 1942, 2010, 2079, 2148, 2216, 2285, 2354, 2422, 2491, 2560, 2628],
    'pred_peaks_feat_0':   [639, 708, 777, 846, 914, 982, 1050, 1118, 1187, 1256, 1324, 1393, 1462, 1531, 1599, 1668, 1737, 1806, 1874, 1943, 2012, 2080, 2149, 2217, 2286, 2355, 2423, 2492, 2561, 2629, 2697, 2770, 2839, 2909, 2980, 3050, 3120, 3188, 3257, 3326, 3395, 3463, 3532, 3601, 3670, 3738, 3807, 3875, 3944, 4013, 4081, 4150, 4219, 4287, 4356, 4428, 4497, 4566, 4634, 4704, 4773, 4841, 4911],
    'target_peaks_feat_1': [637, 706, 775, 843, 912, 980, 1049, 1118, 1186, 1255, 1324, 1392, 1461, 1530, 1598, 1667, 1736, 1804, 1873, 1942, 2010, 2079, 2148, 2216, 2285, 2354, 2422, 2491, 2560, 2628, 2697, 2766, 2834, 2903, 2972, 3040, 3109, 3178, 3246, 3315, 3384, 3452, 3521, 3589, 3658, 3727, 3795, 3864, 3933, 4001, 4070, 4139, 4207, 4276],
    'pred_peaks_feat_1':   [639, 708, 776, 846, 914, 982, 1050, 1118, 1187, 1256, 1324, 1393, 1463, 1531, 1600, 1668, 1737, 1806, 1875, 1943, 2012, 2081, 2149, 2218, 2286, 2355, 2424, 2492, 2561, 2629, 2697, 2771, 2839, 2910, 2981, 3051, 3120, 3189, 3258, 3327, 3396, 3464, 3533, 3602, 3671, 3739, 3808, 3877, 3945, 4014, 4083, 4151, 4220, 4288, 4357, 4430, 4499, 4567, 4636, 4705, 4774, 4842, 4912]
    }
}

# Initialize lists
target_peaks_feat_0_list = []
pred_peaks_feat_0_list = []
target_peaks_feat_1_list = []
pred_peaks_feat_1_list = []

# Organize data into lists
for sample_key, sample_data in data.items():
    target_peaks_feat_0_list.append(sample_data['target_peaks_feat_0'])
    pred_peaks_feat_0_list.append(sample_data['pred_peaks_feat_0'])
    target_peaks_feat_1_list.append(sample_data['target_peaks_feat_1'])
    pred_peaks_feat_1_list.append(sample_data['pred_peaks_feat_1'])

# Filled data for sample 7
data['sample 7'] = {
    'target_peaks_feat_0': [225, 294, 363, 431, 500, 569, 637, 706, 775, 843, 912, 980, 1049, 1118, 1186, 1255, 1324, 1392, 1461, 1530, 1598, 1667, 1736, 1804, 1873, 1942, 2010, 2079, 2148, 2216, 2285, 2354, 2422, 2491, 2560, 2628],
    'pred_peaks_feat_0':   [224, 296, 363, 435, 502, 571, 638, 709, 775, 845, 912, 982, 1049, 1119, 1187, 1256, 1323, 1394, 1461, 1531, 1598, 1667, 1735, 1805, 1872, 1942, 2010, 2079, 2147, 2216, 2284, 2354, 2421, 2492, 2559, 2629, 2696, 2769, 2837, 2907, 2977, 3048, 3116, 3187, 3254, 3325, 3392, 3462, 3530, 3600, 3667, 3737, 3805, 3874, 3942, 4012, 4079, 4149, 4216, 4286, 4354, 4424, 4492, 4561, 4628, 4696, 4764, 4831, 4899],
    'target_peaks_feat_1': [569, 706, 844, 981, 1118, 1255, 1393, 1530, 1667, 1805, 1942, 2079, 2217, 2354, 2491, 2629, 2766, 2903, 3041, 3178, 3315, 3452, 3590, 3727, 3864, 4002, 4139, 4276],
    'pred_peaks_feat_1':   [570, 707, 843, 981, 1118, 1255, 1392, 1529, 1666, 1803, 1940, 2077, 2215, 2352, 2490, 2627, 2766, 2905, 3046, 3186, 3323, 3460, 3598, 3736, 3873, 4010, 4147, 4285, 4422, 4566, 4633, 4701, 4769, 4836, 4902]
}

# Update the lists
target_peaks_feat_0_list.append(data['sample 7']['target_peaks_feat_0'])
pred_peaks_feat_0_list.append(data['sample 7']['pred_peaks_feat_0'])
target_peaks_feat_1_list.append(data['sample 7']['target_peaks_feat_1'])
pred_peaks_feat_1_list.append(data['sample 7']['pred_peaks_feat_1'])

# Filled data for sample 8
data['sample 8'] = {
    'target_peaks_feat_0': [225, 294, 363, 431, 500, 569, 637, 706, 775, 843, 912, 980, 1049, 1118, 1186, 1255, 1324, 1392, 1461, 1530, 1598, 1667, 1736, 1804, 1873, 1942, 2010, 2079, 2148, 2216, 2285, 2354, 2422, 2491, 2560, 2628],
    'pred_peaks_feat_0':   [223, 295, 361, 433, 502, 567, 638, 707, 773, 843, 912, 978, 1049, 1118, 1185, 1255, 1324, 1391, 1461, 1531, 1597, 1667, 1737, 1803, 1873, 1943, 2010, 2079, 2149, 2216, 2285, 2355, 2422, 2491, 2561, 2628, 2695, 2768, 2838, 2908, 2980, 3051, 3121, 3191, 3261, 3331, 3402, 3470, 3540, 3610, 3678, 3748, 3819, 3886, 3959, 4031, 4099, 4173, 4244, 4319, 4395, 4468, 4539, 4611, 4683, 4754, 4827, 4898],
    'target_peaks_feat_1': [569, 775, 981, 1187, 1393, 1599, 1805, 2011, 2217, 2423, 2629, 2835, 3041, 3247, 3452, 3658, 3864, 4070, 4276],
    'pred_peaks_feat_1':   [575, 776, 981, 1187, 1393, 1597, 1803, 2009, 2215, 2421, 2627, 2838, 3052, 3262, 3409, 3618, 3827, 4039, 4250, 4467, 4621, 4684, 4761, 4834]
}

# Update the lists
target_peaks_feat_0_list.append(data['sample 8']['target_peaks_feat_0'])
pred_peaks_feat_0_list.append(data['sample 8']['pred_peaks_feat_0'])
target_peaks_feat_1_list.append(data['sample 8']['target_peaks_feat_1'])
pred_peaks_feat_1_list.append(data['sample 8']['pred_peaks_feat_1'])


# Filled data for sample 9
data['sample 9'] = {
    'target_peaks_feat_0': [569, 706, 844, 981, 1118, 1255, 1393, 1530, 1667, 1805, 1942, 2079, 2217, 2354, 2491, 2629],
    'pred_peaks_feat_0':   [568, 705, 843, 981, 1118, 1255, 1393, 1530, 1667, 1805, 1942, 2079, 2217, 2354, 2492, 2630, 2768, 2909, 3049, 3185, 3323, 3461, 3599, 3736, 3873, 4011, 4148, 4285, 4423, 4561, 4695, 4827],
    'target_peaks_feat_1': [294, 432, 569, 706, 844, 981, 1118, 1255, 1393, 1530, 1667, 1805, 1942, 2079, 2217, 2354, 2491, 2629, 2766, 2903, 3041, 3178, 3315, 3452, 3590, 3727, 3864, 4002, 4139, 4276],
    'pred_peaks_feat_1':   [287, 426, 567, 705, 844, 982, 1119, 1256, 1393, 1530, 1668, 1805, 1943, 2080, 2218, 2354, 2492, 2630, 2768, 2911, 3050, 3188, 3327, 3464, 3602, 3739, 3876, 4014, 4151, 4288, 4428, 4565, 4699, 4831, 4955]
}

# Update the lists
target_peaks_feat_0_list.append(data['sample 9']['target_peaks_feat_0'])
pred_peaks_feat_0_list.append(data['sample 9']['pred_peaks_feat_0'])
target_peaks_feat_1_list.append(data['sample 9']['target_peaks_feat_1'])
pred_peaks_feat_1_list.append(data['sample 9']['pred_peaks_feat_1'])

# Filled data for sample 10
data['sample 10'] = {
    'target_peaks_feat_0': [294, 432, 569, 706, 844, 981, 1118, 1255, 1393, 1530, 1667, 1805, 1942, 2079, 2217, 2354, 2491, 2629],
    'pred_peaks_feat_0':   [294, 430, 569, 711, 841, 981, 1122, 1253, 1392, 1534, 1666, 1806, 1948, 2079, 2219, 2339, 2491, 2631, 2774, 2908, 3051, 3191, 3322, 3465, 3606, 3736, 3878, 4020, 4149, 4290, 4432, 4571, 4716, 4862],
    'target_peaks_feat_1': [569, 775, 981, 1187, 1393, 1599, 1805, 2011, 2217, 2423, 2629, 2835, 3041, 3247, 3452, 3658, 3864, 4070, 4276],
    'pred_peaks_feat_1':   [578, 767, 989, 1180, 1398, 1593, 1809, 2006, 2220, 2418, 2632, 2844, 3055, 3260, 3473, 3675, 3825, 4088, 4237, 4505, 4650, 4799, 4867]
}

# Update the lists
target_peaks_feat_0_list.append(data['sample 10']['target_peaks_feat_0'])
pred_peaks_feat_0_list.append(data['sample 10']['pred_peaks_feat_0'])
target_peaks_feat_1_list.append(data['sample 10']['target_peaks_feat_1'])
pred_peaks_feat_1_list.append(data['sample 10']['pred_peaks_feat_1'])

# Filled data for sample 11
data['sample 11'] = {
    'target_peaks_feat_0': [363, 569, 775, 981, 1187, 1393, 1599, 1805, 2011, 2217, 2423, 2629],
    'pred_peaks_feat_0':   [371, 574, 777, 983, 1189, 1396, 1602, 1806, 2014, 2219, 2424, 2631, 2837, 3045, 3253, 3461, 3668, 3875, 4082, 4290, 4501],
    'target_peaks_feat_1': [569, 775, 981, 1187, 1393, 1599, 1805, 2011, 2217, 2423, 2629, 2835, 3041, 3247, 3452, 3658, 3864, 4070, 4276],
    'pred_peaks_feat_1':   [556, 772, 973, 1188, 1395, 1602, 1808, 2014, 2220, 2426, 2632, 2838, 3045, 3257, 3466, 3672, 3879, 4086, 4294, 4504, 4581, 4652, 4778]
}

# Update the lists
target_peaks_feat_0_list.append(data['sample 11']['target_peaks_feat_0'])
pred_peaks_feat_0_list.append(data['sample 11']['pred_peaks_feat_0'])
target_peaks_feat_1_list.append(data['sample 11']['target_peaks_feat_1'])
pred_peaks_feat_1_list.append(data['sample 11']['pred_peaks_feat_1'])

# Check the results
# print("Target Peaks feat_0:", target_peaks_feat_0_list)
# print("Pred Peaks feat_0:", pred_peaks_feat_0_list)
# print("Target Peaks feat_1:", target_peaks_feat_1_list)
# print("Pred Peaks feat_1:", pred_peaks_feat_1_list)

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




'''For prediction'''
#### this part is during the reference there

def process_pure(input_lst, target_lst):
    pure_lst = []
    for i in range(len(pred_peaks_feat_0_list)):
        pure_lst.append(input_lst[i][:len(target_lst[i])])

    return pure_lst

pred_peaks_feat_0_pure_list = process_pure(pred_peaks_feat_0_list, target_peaks_feat_0_list)
pred_peaks_feat_1_pure_list = process_pure(pred_peaks_feat_1_list, target_peaks_feat_1_list)



# Calculate the mean errors and standard deviations for each sample and feature
mean_errors_feat_0 = [np.mean(np.abs(np.array(pred) - np.array(target))/ np.array(target) * 6) for pred, target in zip(pred_peaks_feat_0_pure_list, target_peaks_feat_0_list)]
std_dev_feat_0 = [np.std(np.abs(np.array(pred) - np.array(target))/ np.array(target) * 6) for pred, target in zip(pred_peaks_feat_0_pure_list, target_peaks_feat_0_list)]

mean_errors_feat_1 = [np.mean(np.abs(np.array(pred) - np.array(target))/ np.array(target) * 6) for pred, target in zip(pred_peaks_feat_1_pure_list, target_peaks_feat_1_list)]
std_dev_feat_1 = [np.std(np.abs(np.array(pred) - np.array(target))/ np.array(target) * 6) for pred, target in zip(pred_peaks_feat_1_pure_list, target_peaks_feat_1_list)]


# colors = ['#80221E', '#AD7C59', '#B85C48', '#CABCAB']

# labels = ['1 1 72 BPM', '1 2 72 BPM', '1 3 72 BPM', '2 2 72 BPM', '2 3 72 BPM', '3 3 72 BPM', '1 1 144 BPM', '1 2 144 BPM', '1 3 144 BPM', '2 2 144 BPM', '2 3 144 BPM', '3 3 144 BPM']
labels = ['1 1 72 BPM', '1 2 72 BPM', '1 3 72 BPM', '2 2 72 BPM', '1 1 144 BPM', '1 2 144 BPM', '1 3 144 BPM', '2 2 144 BPM', '2 3 144 BPM', '3 3 144 BPM']
# # Plotting
# fig, ax = plt.subplots(figsize=(12, 6))
#
# bar_width = 0.35
# index = np.arange(len(target_peaks_feat_0_list))
#
# # Plot errors for feature 0 with error bars
# bar1 = ax.bar(index, mean_errors_feat_0, bar_width, label='Channel 0', yerr=std_dev_feat_0, capsize=3, color=colors[0])
# # box1 = ax.boxplot(mean_errors_feat_0, widths=bar_width, patch_artist=True, showmeans=True, meanline=True, showfliers=False)
# # # Customize boxplot colors
# # for box in box1['boxes']:
# #     box.set(facecolor=colors[0])
# #
# # # Add error bars for feature 0
# # ax.errorbar(index + bar_width / 2, mean_errors_feat_0, yerr=std_dev_feat_0, fmt='none', ecolor='black', capsize=3)
#
# # Plot errors for feature 1 with error bars
# bar2 = ax.bar(index + bar_width, mean_errors_feat_1, bar_width, label='Channel 1', yerr=std_dev_feat_1, capsize=3, color=colors[1])
#
# # ax.set_xlabel('Sample')
# ax.set_ylabel('Time Offset Ratio')
# # Set y-axis limits
# ax.set_ylim(-0.05, 0.15)
#
# # ax.set_title('Mean Error Comparison between Features for Each Sample with Error Bars')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels([f'{labels[i]}' for i in range(len(target_peaks_feat_0_list))])
# ax.legend()

# Calculate errors directly without aggregating to means and standard deviations
errors_feat_0 = [np.abs(np.array(pred) - np.array(target)) / np.array(target) * 6 for pred, target in zip(pred_peaks_feat_0_pure_list, target_peaks_feat_0_list)]
errors_feat_1 = [np.abs(np.array(pred) - np.array(target)) / np.array(target) * 6 for pred, target in zip(pred_peaks_feat_1_pure_list, target_peaks_feat_1_list)]

# Prepare DataFrame for Seaborn
data = {'Error': [], 'Label': [], 'Feature': []}
for label, errors in zip(labels, errors_feat_0):
    data['Error'].extend(errors)
    data['Label'].extend([label] * len(errors))
    data['Feature'].extend(['Channel 1'] * len(errors))
for label, errors in zip(labels, errors_feat_1):
    data['Error'].extend(errors)
    data['Label'].extend([label] * len(errors))
    data['Feature'].extend(['Channel 2'] * len(errors))

# df = pd.DataFrame(data)
# custom_colors = {"Channel 1": colors[0], "Channel 2": colors[1]}
# # Plotting with Seaborn
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Label', y='Error', hue='Feature', data=df, palette=custom_colors)
#
# plt.xticks(rotation=45)
# plt.xlabel('')
# plt.ylabel('Time Offset Ratio')
df = pd.DataFrame(data)
df['Label'] = pd.Categorical(df['Label'], categories=labels, ordered=True)
custom_colors = {"Channel 1": colors[0], "Channel 2": colors[1]}

# Function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for each combination of Label and Feature
filtered_df = df.groupby(['Label', 'Feature']).apply(lambda x: remove_outliers(x, 'Error')).reset_index(drop=True)

# Plotting with Seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(x='Label', y='Error', hue='Feature', data=filtered_df, palette=custom_colors)

plt.xticks(rotation=45)
plt.xlabel('')
plt.ylabel('Time Offset Ratio')
# Optional: Set y-axis limits if needed, e.g., plt.ylim(-0.05, 0.15)

plt.legend(title='Channel')
plt.tight_layout()
if save_state:
    plt.savefig(f'{save_path}/dual_channel_update_output.pdf', dpi=300)

plt.show()


'''For continuation'''
#### this part is during the reference not there

def process_continuation(input_lst, target_lst):
    continuation_lst = []
    for i in range(len(pred_peaks_feat_0_list)):
        continuation_lst.append(input_lst[i][len(target_lst[i]):])

    return continuation_lst

def get_continuation_interval(input_lst):
    time_interval_lst = []
    for i in range(len(input_lst)):
        time_interval_lst.append(np.diff(input_lst[i]))
    return time_interval_lst


def get_interval(input_lst):
    time_interval_lst = []
    for i in range(len(input_lst)):
        time_interval_lst.append(np.mean(np.diff(input_lst[i])))
    return time_interval_lst

def get_interval_error(input_lst, target_lst):
    error_lst = []
    for i in range(len(input_lst)):
        error = []
        for j in range(len(input_lst[i]) - 3):
            error.append(np.abs(input_lst[i][j]-target_lst[i]))
        error_lst.append(error/target_lst[i])
        print(error)
        print('**************')
    return error_lst


pred_peaks_feat_0_cont_list = process_continuation(pred_peaks_feat_0_list, target_peaks_feat_0_list)
pred_peaks_feat_1_cont_list = process_continuation(pred_peaks_feat_1_list, target_peaks_feat_1_list)

pred_peaks_feat_0_cont_interval_lst = get_continuation_interval(pred_peaks_feat_0_cont_list)
pred_peaks_feat_1_cont_interval_lst = get_continuation_interval(pred_peaks_feat_1_cont_list)
# print(pred_peaks_feat_0_cont_interval_lst)
# exit()

target_feat_0_interval = get_interval(target_peaks_feat_0_list)
# print(target_feat_0_interval)
target_feat_1_interval = get_interval(target_peaks_feat_1_list)

error_feature_0 = get_interval_error(pred_peaks_feat_0_cont_interval_lst, target_feat_0_interval)
# error_feature_1 = get_interval_error(pred_peaks_feat_1_cont_interval_lst, target_feat_1_interval)



# # Calculate the mean errors and standard deviations for each sample and feature
# mean_errors_feat_0 = [np.mean(np.abs(np.array(error)) * 6) for error in error_feature_0]
# std_dev_feat_0 = [np.std(np.abs(np.array(error)) * 6) for error in error_feature_0]
#
# mean_errors_feat_1 = [np.mean(np.abs(np.array(error)) * 6) for error in error_feature_1]
# std_dev_feat_1 = [np.std(np.abs(np.array(error)) * 6) for error in error_feature_1]
# labels = ['1 1 72 BPM', '1 2 72 BPM', '1 3 72 BPM', '2 2 72 BPM', '1 1 144 BPM', '1 2 144 BPM', '1 3 144 BPM', '2 2 144 BPM', '2 3 144 BPM', '3 3 144 BPM']
# # Plotting
# fig, ax = plt.subplots(figsize=(12, 6))
#
# bar_width = 0.35
# index = np.arange(len(target_peaks_feat_0_list))
#
# # Plot errors for feature 0 with error bars
# bar1 = ax.bar(index, mean_errors_feat_0, bar_width, yerr=std_dev_feat_0, capsize=3, color=colors[0])
#
# # Plot errors for feature 1 with error bars
# # bar2 = ax.bar(index + bar_width, mean_errors_feat_1, bar_width, label='Feature 1', yerr=std_dev_feat_1, capsize=3, color=colors[1])
#
# # ax.set_xlabel('Sample')
# ax.set_ylabel('Continuation Inter-Beat Interval Error (ms)')
# # ax.set_title('Mean Error Comparison between Features for Each Sample with Error Bars')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels([f'{labels[i]}' for i in range(len(target_peaks_feat_0_list))])
# ax.legend()
# if save_state:
#     plt.savefig(f'{save_path}/continuation_error_two_channel.png', dpi=300)
# plt.show()



# Convert errors to a suitable format for box plotting. Each feature's errors should be a list of arrays/lists.
errors_feat_0_box = [np.abs(np.array(error)) for error in error_feature_0]
# errors_feat_1_box = [np.abs(np.array(error)) for error in error_feature_1]

labels = ['1 1 72 BPM', '1 2 72 BPM', '1 3 72 BPM', '2 2 72 BPM', '1 1 144 BPM', '1 2 144 BPM', '1 3 144 BPM', '2 2 144 BPM', '2 3 144 BPM', '3 3 144 BPM']

data = []
for i, errors in enumerate(errors_feat_0_box):
    for error in errors:
        data.append({'Error': error, 'Label': labels[i]})

# df = pd.DataFrame(data)
#
# # Plotting with Seaborn
# plt.figure(figsize=(12, 6))
# # sns.boxplot(x='Label', y='Error', data=df, palette='colorblind')  # 'colorblind' is an example palette
# sns.boxplot(x='Label', y='Error', data=df, color=colors[0])  # 'colorblind' is an example palette
#
# plt.xticks(rotation=45)  # Rotate labels to prevent overlap
# plt.xlabel('')
# plt.ylabel('Continuation Time Offset Ratio')
df = pd.DataFrame(data)
df['Label'] = pd.Categorical(df['Label'], categories=labels, ordered=True)

# Remove outliers for each 'Label'
filtered_df = df.groupby('Label').apply(lambda x: remove_outliers(x, 'Error')).reset_index(drop=True)

# Plotting with Seaborn
plt.figure(figsize=(12, 8))
sns.boxplot(x='Label', y='Error', data=filtered_df, color=colors[0])

plt.xticks(rotation=45)  # Rotate labels to prevent overlap
plt.xlabel('')
plt.ylabel('Continuation Time Offset Ratio')

if save_state:
    plt.savefig(f'{save_path}/feature_0_continuous_error.pdf', dpi=300)

plt.show()


'''Two channel contionuous comparision'''
# # Convert your errors and labels into a DataFrame format that Seaborn can use easily
# data = []
# for i, (errors_0, errors_1) in enumerate(zip(errors_feat_0_box, errors_feat_1_box)):
#     for error in errors_0:
#         data.append({'Error': error, 'Label': labels[i], 'Feature': 'Feature 0'})
#     for error in errors_1:
#         data.append({'Error': error, 'Label': labels[i], 'Feature': 'Feature 1'})
#
# df = pd.DataFrame(data)
# custom_colors = {"Feature 0": colors[0], "Feature 1": colors[1]}
#
#
# # Plotting with Seaborn
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Label', y='Error', hue='Feature', data=df, palette=custom_colors)  # 'colorblind' is an example palette
#
# plt.xticks(rotation=45)  # Rotate labels to prevent overlap
# plt.ylabel('Continuation Inter-Beat Interval Error (ms)')
# plt.xlabel('Condition')  # Optionally set the x-axis label
#
# plt.legend(title='Feature')
#
# if save_state:
#     plt.savefig(f'{save_path}/feature_comparison_boxplot_seaborn.png', dpi=300)
#
# plt.show()

