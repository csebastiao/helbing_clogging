# -*- coding: utf-8 -*-
"""
Analysis of results from helbing_model.py
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def get_cumulative_exit(h, x_hole, plot = 'on'):
    """
    Get the timestep where fish exit, and show the cumulative exit of fish
    through time and the histogram of the delay between two consecutive exit.

    Parameters
    ----------
    h : pandas.DataFrame
        Dataframe with the column 'Time', 'FishID', 'X' (and usually 'Y'
        and 'R').
    x_hole : float
        x coordinate of the hole from where the fish exit.

    Returns
    -------
    exit_timestep : numpy.array
        Sorted array of the timestep where the fish are going out.
    delay : list
        List of delay between 2 consecutive exit.

    """
    exit_timestep = []
    for i in range(max(h['FishID']) + 1):
        ind = h[h['FishID'] == i] #We filter for one fish
        #We add the first timestep where the fish is out, so when the x of
        #the fish is higher than the hole
        if len(list(ind[(ind['X'] - x_hole) > 0.]['Time'])) > 1 :
            exit_timestep.append(     
                list(ind[(ind['X'] - x_hole) > 0.]['Time'])[1])
    exit_timestep = np.asarray(sorted(exit_timestep)) #We sort the timestep
    ce = []
    count = 0
    c_et = np.copy(exit_timestep)
    tt = np.array(h['Time'])[::max(h['FishID'])+1]
    for time in tt:
        if len(c_et) > 0:
            if time == c_et[0]:
                n = len(c_et[c_et == time]) #If 2 fish got out on the same time
                count += n #Count goes up each time a fish exit
                for k in range(n):
                    c_et = np.delete(c_et, 0)
        ce.append(count)  #We take the count for every timestep
    delay = []
    for i in range(len(exit_timestep) - 1): #Delay between 2 consecutive exit
        delay.append(exit_timestep[i + 1] - exit_timestep[i])
    if plot == 'on' :
        plt.figure()
        plt.plot(tt, ce)
        plt.title('Cumulative exit')
        plt.xlabel('Time')
        plt.ylabel('Number of fish out')
        plt.figure()
        plt.hist(delay, bins = 20) #Make histogram of it
        plt.title('Histogram of delay between two consecutive exit')
    return ce, tt, exit_timestep, delay


all_12_delay = []
all_12_exit_timestep = []
all_12_count = []
all_12_tt = []
total_12 = []
for i in range(5):
    title = 'history_49_ds12_ka12_{}.csv'.format(i)
    
    h = pd.read_csv(title)
    
    count, tt, exit_timestep, delay = get_cumulative_exit(h, 5, plot = 'off')
    all_12_delay = np.concatenate((all_12_delay, delay))
    all_12_tt = np.concatenate((all_12_tt, tt))
    all_12_exit_timestep = np.concatenate((all_12_exit_timestep, exit_timestep))
    all_12_count = np.concatenate((all_12_count, count))
    total_12.append(exit_timestep[-1])
    

all_24_delay = []
all_24_exit_timestep = []
all_24_count = []
all_24_tt = []
total_24 = []
for i in range(15):
    title = 'history_49_ds12_ka24_{}.csv'.format(i)
    
    h = pd.read_csv(title)
    
    count, tt, exit_timestep, delay = get_cumulative_exit(h, 5, plot = 'off')
    all_24_delay = np.concatenate((all_24_delay, delay))
    all_24_tt = np.concatenate((all_24_tt, tt))
    all_24_exit_timestep = np.concatenate((all_24_exit_timestep, exit_timestep))
    all_24_count = np.concatenate((all_24_count, count))
    total_24.append(exit_timestep[-1])


all_48_delay = []
all_48_exit_timestep = []
all_48_count = []
all_48_tt = []
total_48 = []
for i in range(10):
    title = 'history_49_ds12_ka48_{}.csv'.format(i)
    
    h = pd.read_csv(title)
    
    count, tt, exit_timestep, delay = get_cumulative_exit(h, 5, plot = 'off')
    all_48_delay = np.concatenate((all_48_delay, delay))
    all_48_tt = np.concatenate((all_48_tt, tt))
    all_48_exit_timestep = np.concatenate((all_48_exit_timestep, exit_timestep))
    all_48_count = np.concatenate((all_48_count, count))
    total_48.append(exit_timestep[-1])

plt.figure()
plt.rcParams.update({'font.size': 22})
w = np.ones_like(all_24_delay)/float(len(all_24_delay))
plt.hist(all_24_delay, bins = 20, weights = w)
plt.xlabel('Time between two consecutives exit')


# plt.figure()
# plt.hist2d(all_exit_timestep, all_count, bins = 20)


# fixed_12 = np.ones((len(all_12_exit_timestep), 1))
# fixed_24 = np.ones((len(all_24_exit_timestep), 1)) * 2
# fixed_48 = np.ones((len(all_48_exit_timestep), 1)) * 3
# plt.figure()
# plt.scatter(fixed_12, all_12_exit_timestep, marker = '.')
# plt.scatter(fixed_24, all_24_exit_timestep, marker = '.')
# plt.scatter(fixed_48, all_48_exit_timestep, marker = '.')


# alt_fixed_12 = ['$1.2 * 10^5$'] * len(all_12_exit_timestep)
# alt_fixed_24 = ['$2.4 * 10^5$'] * len(all_24_exit_timestep)
# alt_fixed_48 = ['$4.8 * 10^5$'] * len(all_48_exit_timestep)
# df = pd.DataFrame({'Exit time (s)' : np.concatenate(
#     (all_12_exit_timestep, all_24_exit_timestep, all_48_exit_timestep)),
#                             '$\kappa$' : np.concatenate(
#     (alt_fixed_12, alt_fixed_24, alt_fixed_48))})
# plt.figure()
# sns.boxplot(x = '$\kappa$', y = 'Exit time (s)', data = df)

val = [np.mean(total_12), np.mean(total_24), np.mean(total_48)]
err = [np.std(total_12), np.std(total_24), np.std(total_48)]
kappa = [12, 24, 48]
plt.figure()
plt.errorbar(kappa, val, yerr = err, linestyle = '', marker = 'o')
plt.ylabel('Total exit time (s)')
plt.xlabel('Friction constant value $\kappa / 10^4$ ($kg.m^{-1}.s^{-1}$)')


