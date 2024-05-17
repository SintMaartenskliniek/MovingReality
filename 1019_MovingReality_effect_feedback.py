"""
Main script for:
    1019 Moving(g) Reality, feedback study
    
2024, Carmen Ensink, Sint Maartenskliniek,
c.ensink@maartenskliniek.nl

"""

# Import dependencies
# To run this script you need to pull the following repositories:
#    To import and analyze OMCS data: https://github.com/SintMaartenskliniek/OMCS_GaitAnalysis
#    To import and analyze IMU data: https://github.com/SintMaartenskliniek/IMU_GaitAnalysis
import sys
sys.path.insert(0, '/IMU_GaitAnalysis')
sys.path.insert(0, '/OMCS_GaitAnalysis')

import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import numpy as np
from scipy import signal
from scipy import stats
import os
import pingouin
import pandas as pd
from MovingReality_functions import person_characteristics, group_trialorder, corresponding_filenames, define_filepaths, import_vicondata, foot_kinematics_vicon, bland_altman_plot, define_filepaths_healthy
from gaitcharacteristics import propulsion

# Define if debugplots should be plotted or not
# debugplots = True
debugplots = False

# Define location of data
datafolder_stroke = os.path.abspath('data')
datafolder_healthy = '/IMU_GaitAnalysis/data/Healthy_controls'


# Import person characteristics
bodyweight, height, age, time_since_stroke, comfortable_gait_speed = person_characteristics()

# # Check for normal distribution of the person characteristics
# significance_level = 0.05 # set p-value for significance
# if stats.shapiro(list(bodyweight.values()))[1] < significance_level:
#     print('Bodyweight is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(bodyweight.values())))
# else:
#     print('Bodyweight is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(bodyweight.values())))
# if stats.shapiro(list(height.values()))[1] < significance_level:
#     print('Height is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(height.values())))
# else:
#     print('Height is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(height.values())))
# if stats.shapiro(list(age.values()))[1] < significance_level:
#     print('Age is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(age.values())))
# else:
#     print('Age is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(age.values())))
# if stats.shapiro(list(time_since_stroke.values()))[1] < significance_level:
#     print('Time since stroke onset is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(time_since_stroke.values())))
# else:
#     print('Time since stroke onset is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(time_since_stroke.values())))
# if stats.shapiro(list(comfortable_gait_speed.values()))[1] < significance_level:
#     print('Comfortable gait speed is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(comfortable_gait_speed.values())))
# else:
#     print('Comfortable gait speed is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(comfortable_gait_speed.values())))



# Define corresponding filenames of sensor and vicon data of stroke participants
corresponding_files, triallist = corresponding_filenames()

# Define corresponding filepaths of sensor and vicon data of stroke participants
pathsvicon, pathsxsens = define_filepaths(datafolder_stroke, corresponding_files)
# Define corresponding filepaths of sensor and vicon data of healthy participants
pathsvicon_healthy = define_filepaths_healthy(datafolder_healthy)

# Define variables for vicon data
vicon = dict()
vicon_gait_events = dict()
vicon_spatiotemporals = dict()
analogdata = dict()

# Import vicon data healthy participants
vicon, vicon_gait_events, vicon_spatiotemporals, analogdata = import_vicondata(pathsvicon_healthy, vicon, vicon_gait_events, vicon_spatiotemporals, analogdata)

# Import vicon data stroke participants
vicon, vicon_gait_events, vicon_spatiotemporals, analogdata = import_vicondata(pathsvicon, vicon, vicon_gait_events, vicon_spatiotemporals, analogdata)

# Triallist based on task
trials = dict()
trials['FBIC'] = list()
trials['FBPO'] = list()
trials['2FB'] = list()
trials['1Reg'] = list()
trials['2Reg'] = list()
trials['Healthy controls'] = list()
for key in vicon:
    if "FBIC" in key:
        trials['FBIC'].append(key)
    elif "FBPO" in key:
        trials['FBPO'].append(key)
    elif "2FB" in key:
        if '1019_MR002_2303082FB' in key:
            trials['2Reg'].append(key)
        else:
            trials['2FB'].append(key)
    elif "1Reg" in key:
        trials['1Reg'].append(key)
    elif "2Reg" in key:
        trials['2Reg'].append(key)
    elif "900_V_pp" in key:
        trials['Healthy controls'].append(key)
        
# Group based on trial order
group = group_trialorder()



# # # # # # FOOT STRIKE ANGLE # # # # # #


# Foot kinematics vicon
foot_vicon = foot_kinematics_vicon(vicon)

for trial in foot_vicon:
    ic_left = vicon_gait_events[trial]['Index numbers initial contact left'] [vicon_gait_events[trial]['Index numbers initial contact left'] < vicon_gait_events[trial]['Index numbers terminal contact left'][5]] [-1]
    midstance_left = int( ic_left + 0.5*(vicon_gait_events[trial]['Index numbers terminal contact left'][5] - ic_left) )
    ic_right = vicon_gait_events[trial]['Index numbers initial contact right'] [vicon_gait_events[trial]['Index numbers initial contact right'] < vicon_gait_events[trial]['Index numbers terminal contact right'][5]] [-1]
    midstance_right = int( ic_right + 0.5*(vicon_gait_events[trial]['Index numbers terminal contact right'][5] - ic_left) )
    
    foot_vicon[trial]['Angle left filt'] = foot_vicon[trial]['Angle left filt'] - foot_vicon[trial]['Angle left filt'][ midstance_left ]   
    foot_vicon[trial]['Angle right filt'] = foot_vicon[trial]['Angle right filt'] - foot_vicon[trial]['Angle right filt'][ midstance_right ]   

# Debug figures angle foot/shank
match debugplots:
    case True:
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 16,
                }
        
        # Debugplot foot_angle
        for trial in foot_vicon:
             print(trial)
             time = np.zeros(shape=(len(foot_vicon[trial]['Angle left filt']), 1))
             for i in range(1,len(time)):
                 time[i] = time[i-1]+0.01
             fig = plt.figure()
             plt.title('Flexion-extension angle', fontsize = 17)
             plt.plot(time, foot_vicon[trial]['Angle left filt'], 'k', label='OMCS')
             # plt.plot(time, foot_xsens[trial]['Angle left'], 'r-.', label = 'Sensor')
           
             plt.plot(time[vicon_gait_events[trial]['Index numbers initial contact left']], foot_vicon[trial]['Angle left filt'][vicon_gait_events[trial]['Index numbers initial contact left']], '*r')
             # plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Initial Contact']], foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']], '*r')
             plt.plot(time[vicon_gait_events[trial]['Index numbers terminal contact left']], foot_vicon[trial]['Angle left filt'][vicon_gait_events[trial]['Index numbers terminal contact left']], '*g')
             # plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], '*g')
             plt.ylabel(ylabel='Angle (deg)', fontsize=16)
             plt.yticks(fontsize=14)
             plt.xlabel(xlabel = 'Time (s)', fontsize=16)
             plt.xticks(fontsize=14)
        


# Calculate mean and standard deviation per trial
FSA = dict()
meanFSA = dict()
sdFSA = dict()
for condition in trials:
    FSA[condition] = dict()
    meanFSA[condition] = dict()
    sdFSA[condition] = dict()
    for trial in trials[condition]:
        FSA[condition][trial[0:10]] = np.append(-1*foot_vicon[trial]['Angle left filt'][vicon_gait_events[trial]['Index numbers initial contact left'][0:50]], -1*foot_vicon[trial]['Angle right filt'][vicon_gait_events[trial]['Index numbers initial contact right'][0:50]])
        meanFSA[condition][trial[0:10]] = np.nanmean(FSA[condition][trial[0:10]])
        sdFSA[condition][trial[0:10]] = np.nanstd(FSA[condition][trial[0:10]])

# Put mean FSA per trial in DataFrame
columnnames=list(['participantID'])
columnnames.append('group')
columnnames.append('trialtype')
columnnames.append('FSA')
columnnames.append('time')
FSA_df = pd.DataFrame(columns=columnnames)

for trial in meanFSA:
    for person in meanFSA[trial]:
        if person.startswith('1019'):
            df = pd.DataFrame(columns=columnnames)
            df['participantID'] = [person]
            df['group'] = group[person]
            df['trialtype'] = trial
            df['FSA'] = meanFSA[trial][person]
            if trial == '1Reg':
                df['time'] = 1
            elif trial == 'FBIC' and group[person] == 'FSA':
                df['time'] = 2
            elif trial == 'FBPO' and group[person] == 'PO':
                df['time'] = 2
            elif trial == 'FBPO' and group[person] == 'FSA':
                df['time'] = 3
            elif trial == 'FBIC' and group[person] == 'PO':
                df['time'] = 3
            elif trial == '2FB':
                df['time'] = 4
            elif trial == '2Reg':
                df['time'] = 5
            FSA_df = pd.concat([FSA_df, df], ignore_index=True)

# FSA_df.to_excel("meanFSA.xlsx", index=False)



match debugplots:
    case True:
        conditions=list(trials.keys())
        FSA_df_FSAgroup = FSA_df[FSA_df['group'] =='FSA']
        FSA_df_POgroup = FSA_df[FSA_df['group'] =='PO']
        fig = plt.figure()
        for i in range(0,len(trials)):
            condition = conditions[i]
            plt.plot(i, np.nanmean(FSA_df_FSAgroup['FSA'][ FSA_df_FSAgroup['trialtype'] == condition ]), 'ko', markersize=10)
            plt.errorbar(i, np.nanmean(FSA_df_FSAgroup['FSA'][ FSA_df_FSAgroup['trialtype'] == condition ]) , yerr=np.nanstd(FSA_df_FSAgroup['FSA'][ FSA_df_FSAgroup['trialtype'] == condition ]), ecolor='k', capsize=3)
        
            plt.plot(i, np.nanmean(FSA_df_POgroup['FSA'][ FSA_df_POgroup['trialtype'] == condition ]), 'bo', markersize=10)
            plt.errorbar(i, np.nanmean(FSA_df_POgroup['FSA'][ FSA_df_POgroup['trialtype'] == condition ]) , yerr=np.nanstd(FSA_df_POgroup['FSA'][ FSA_df_POgroup['trialtype'] == condition ]), ecolor='b', capsize=3)
        
        plt.xticks(ticks = np.arange(0,len(conditions)), labels=conditions)
        plt.xlim(left=-0.5,right=4.5)


FSA_df_wide = FSA_df.pivot(index='participantID', columns='trialtype', values='FSA')

groupFSA = list([])
groupPO = list([])

colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
i=-1
fig = plt.figure()
plt.title('FSA per trial', fontsize=22)
for participant in group:
    if participant.startswith('1019'): 
        i+=1
        
        if group[participant] == 'FSA':
        #     markerfig = 'o'
            groupFSA.append(participant)
        #     yvalues = np.array([FSA_df_wide['1Reg'][participant], FSA_df_wide['FBIC'][participant], FSA_df_wide['FBPO'][participant], FSA_df_wide['2FB'][participant], FSA_df_wide['2Reg'][participant]])
        elif group[participant] == 'PO':
        #     markerfig = '^'
            groupPO.append(participant)
        #     yvalues = np.array([FSA_df_wide['1Reg'][participant], FSA_df_wide['FBPO'][participant], FSA_df_wide['FBIC'][participant], FSA_df_wide['2FB'][participant], FSA_df_wide['2Reg'][participant]])
        
        markerfig_all = 'o'
        yvalues = np.array([FSA_df_wide['1Reg'][participant], FSA_df_wide['FBIC'][participant], FSA_df_wide['FBPO'][participant], FSA_df_wide['2FB'][participant], FSA_df_wide['2Reg'][participant]])
        yerr = np.array([]),
        try:
            yerr = np.append(yerr, sdFSA['1Reg'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdFSA['FBIC'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdFSA['FBPO'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdFSA['2FB'][participant])
        except:    
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdFSA['2Reg'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        xvalues = np.array([1, 2, 3, 4, 5], dtype=float)
        
        # plt.plot(xvalues, yvalues, marker=markerfig, color=colors[i], alpha=0.25)
        plt.plot(xvalues, yvalues, marker=markerfig_all, color=colors[i], alpha=0.25)
        plt.errorbar(xvalues, yvalues, yerr=yerr, color=colors[i], ecolor=colors[i], alpha=0.25, capsize=5)
        
# xmeansFSA = np.array([0.95, 1.95, 2.95, 3.95, 4.95])
subset1Reg = dict()
for key in groupFSA:
    try:
        subset1Reg[key] =meanFSA['1Reg'][key]
    except:
        pass
subsetFBIC = dict()
for key in groupFSA:
    try:
        subsetFBIC[key] =meanFSA['FBIC'][key]
    except:
        pass
subsetFBPO = dict()
for key in groupFSA:
    try:
        subsetFBPO[key] =meanFSA['FBPO'][key]
    except:
        pass
subset2FB = dict()
for key in groupFSA:
    try:
        subset2FB[key] =meanFSA['2FB'][key]
    except:
        pass
subset2Reg = dict()
for key in groupFSA:
    try:
        subset2Reg[key] =meanFSA['2Reg'][key]
    except:
        pass
# ymeansFSA = np.array([np.mean(list(subset1Reg.values())), np.mean(list(subsetFBIC.values())), np.mean(list(subsetFBPO.values())), np.mean(list(subset2FB.values())), np.mean(list(subset2Reg.values()))])
# # yerrFSA = np.array([np.std(list(subset1Reg.values())), np.std(list(subsetFBIC.values())), np.std(list(subsetFBPO.values())), np.std(list(subset2FB.values())), np.std(list(subset2Reg.values()))])
# plt.plot(xmeansFSA, ymeansFSA, marker='o', color='k', alpha=1, label = 'First feedback FSA')

# xmeansPO = np.array([1.05, 2.05, 3.05, 4.05, 5.05])
# subset1Reg = dict()
for key in groupPO:
    try:
        subset1Reg[key] =meanFSA['1Reg'][key]
    except:
        pass
# subsetFBIC = dict()
for key in groupPO:
    try:
        subsetFBIC[key] =meanFSA['FBIC'][key]
    except:
        pass
# subsetFBPO = dict()
for key in groupPO:
    try:
        subsetFBPO[key] =meanFSA['FBPO'][key]
    except:
        pass
# subset2FB = dict()
for key in groupPO:
    try:
        subset2FB[key] =meanFSA['2FB'][key]
    except:
        pass
# subset2Reg = dict()
for key in groupPO:
    try:
        subset2Reg[key] =meanFSA['2Reg'][key]
    except:
        pass

xmeans_all = np.array([1.05, 2.05, 3.05, 4.05, 5.05])
ymeans_all = np.array([np.nanmean(list(subset1Reg.values())), np.nanmean(list(subsetFBIC.values())), np.nanmean(list(subsetFBPO.values())), np.nanmean(list(subset2FB.values())), np.nanmean(list(subset2Reg.values()))])
yerr_all = np.array([np.nanstd(list(subset1Reg.values())), np.nanstd(list(subsetFBIC.values())), np.nanstd(list(subsetFBPO.values())), np.nanstd(list(subset2FB.values())), np.nanstd(list(subset2Reg.values()))])
# ymeansPO = np.array([np.mean(list(subset1Reg.values())), np.mean(list(subsetFBPO.values())), np.mean(list(subsetFBIC.values())), np.mean(list(subset2FB.values())), np.mean(list(subset2Reg.values()))])
# # yerrPO = np.array([np.std(list(subset1Reg.values())), np.std(list(subsetFBPO.values())), np.std(list(subsetFBIC.values())), np.std(list(subset2FB.values())), np.std(list(subset2Reg.values()))])
# plt.plot(xmeansPO, ymeansPO, marker='^', color='k', alpha=1, label = 'First feedback propulsion')

plt.plot(xmeans_all, ymeans_all, marker=markerfig_all, markersize=10, linewidth=3, color='k', alpha=1, label='Mean FSA')
# plt.errorbar(xmeans_all, ymeans_all, yerr=yerr_all, color = 'k', ecolor='k', alpha=1, capsize=5)
# plt.errorbar(xmeansFSA, ymeansFSA, yerr=yerrFSA, color = 'k', ecolor='k', alpha=1, capsize=5)
# plt.errorbar(xmeansPO, ymeansPO, yerr=yerrPO, color = 'k', ecolor='k', alpha=1, capsize=5)
plt.xlim((0,6))
plt.xticks(ticks=xmeans_all, labels = list(['1Reg', 'FBIC', 'FBPO', '2FB', '2Reg']), fontsize=20)
plt.ylabel('FSA (degrees)', fontsize=20)
plt.yticks(fontsize=16)
# plt.legend(loc = 'center right', bbox_to_anchor=(1.12, 0.5),
#           ncol=1, fancybox=True, shadow=True)


print('unpaired FSA healthy vs stroke: ', stats.ttest_ind(np.fromiter(meanFSA['Healthy controls'].values(), dtype=float), np.fromiter(meanFSA['1Reg'].values(), dtype=float)) )


# # Put FSA's per trial in DataFrame
# # columnnames=list(FSA.keys())
# columnnames=list(['participantID'])
# columnnames.append('group')
# columnnames.append('trialtype')
# columnnames.append('FSA')
# FSA_df = pd.DataFrame(columns=columnnames)
# for condition in FSA:
#     for trial in FSA[condition]:
#         df = pd.DataFrame(columns=columnnames)
#         df['FSA'] = FSA[condition][trial]
#         df['participantID'] = len(FSA[condition][trial]) * list([trial[0:10]])
#         df['group'] = len(FSA[condition][trial]) * list([group[trial[0:10]]])
#         df['trialtype'] = len(FSA[condition][trial]) * list([condition])
#         FSA_df = pd.concat([FSA_df, df], ignore_index=True)

# from ..distribution_plots.jitter_distribution_figure import jitter_distribution_figure
# jitter_distribution_figure(data=np.array(FSA_df['FSA'][FSA_df['group']=='PO']), cats=np.array(FSA_df['trialtype'][FSA_df['group']=='PO']), YLim=np.array([-15, 55]))
# jitter_distribution_figure(data=np.array(FSA_df['FSA'][FSA_df['group']=='FSA']), cats=np.array(FSA_df['trialtype'][FSA_df['group']=='FSA']), YLim=np.array([-15, 55]))







# # # # # # PROPULSION # # # # # #


for trial in vicon_gait_events:
    vicon_gait_events[trial], vicon_spatiotemporals[trial], analogdata[trial] = propulsion(vicon_gait_events[trial], vicon_spatiotemporals[trial], analogdata[trial], bodyweight=bodyweight[trial[0:10]], debugplot=False, plot_title=trial)


# Debug figure propulsion over time
match debugplots:
    case True:
        fig = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        plt.title('Propulsion', fontsize=20)
        colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
        i=0
        for key in triallist:
            for trial in triallist[key]:
                try:
                    x1 = vicon_spatiotemporals[trial]['Propulsion left'][:,0]
                    y1 = vicon_spatiotemporals[trial]['Propulsion left'][:,2]
                    x2 = vicon_spatiotemporals[trial]['Propulsion right'][:,0]
                    y2 = vicon_spatiotemporals[trial]['Propulsion right'][:,2]
                    
                    # Check for inputname in **kwargs items
                    dataType = str()
                    unit = str()
                        
                    plt.scatter(x1, y1, edgecolor = colors[i], facecolor='None', marker = 'o', label=trial+' left') # SMK green: '#004D43'
                    
                    # plt.axhline(mdL,           color=colors[i], linestyle='--', linewidth=3)
                    # # plt.text(-3, md, md_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
                    # plt.axhline(mdL + 1.96*sdL, color=colors[i], linestyle='--', linewidth=3)
                    # # plt.text(-3, md + 1.96*sd, ub_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
                    # plt.axhline(mdL - 1.96*sdL, color=colors[i], linestyle='--', linewidth=3)
                    # # plt.text(-3, md - 1.96*sd, lb_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
                    
                    plt.scatter(x2, y2, edgecolor = colors[i], facecolor='None', marker = '^', label = trial+' right') # SMK green: '#004D43'
                    
                    # plt.axhline(mdR,           color=colors[i], linestyle='-', linewidth=3)
                    # # plt.text(-3, md, md_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
                    # plt.axhline(mdR + 1.96*sdR, color=colors[i], linestyle='--', linewidth=3)
                    # # plt.text(-3, md + 1.96*sd, ub_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
                    # plt.axhline(mdR - 1.96*sdR, color=colors[i], linestyle='--', linewidth=3)
                    # # plt.text(-3, md - 1.96*sd, lb_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
                    # i=i+1
                except KeyError:
                    print('keyerror on trial: ', trial)            
            i=i+1
        
        plt.xlabel("Timestamp IC "+ unit, fontsize=14)
        plt.ylabel("Propulsion " + unit, fontsize=14) #Difference between measures
        
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.yticks(fontsize=14)
        
        plt.legend(fontsize=6)  



# Calculate mean and standard deviation per trial

PROP = dict()
meanPROP = dict()
sdPROP = dict()
for condition in trials:
    PROP[condition] = dict()
    meanPROP[condition] = dict()
    sdPROP[condition] = dict()
    for trial in trials[condition]:
        PROP[condition][trial[0:10]] = np.append(vicon_spatiotemporals[trial]['Propulsion left'][:,2], vicon_spatiotemporals[trial]['Propulsion right'][:,2])
        meanPROP[condition][trial[0:10]] = np.nanmean(PROP[condition][trial[0:10]])
        sdPROP[condition][trial[0:10]] = np.nanstd(PROP[condition][trial[0:10]])

# Put mean propulsion per trial in DataFrame
# Put mean FSA per trial in DataFrame
columnnames=list(['participantID'])
columnnames.append('group')
columnnames.append('trialtype')
columnnames.append('PROP')
columnnames.append('time')
PROP_df = pd.DataFrame(columns=columnnames)

for trial in meanPROP:
    for person in meanPROP[trial]:
        if person.startswith('1019'):
            df = pd.DataFrame(columns=columnnames)
            df['participantID'] = [person]
            df['group'] = group[person]
            df['trialtype'] = trial
            df['PROP'] = meanPROP[trial][person]
            if trial == '1Reg':
                df['time'] = 1
            elif trial == 'FBIC' and group[person] == 'FSA':
                df['time'] = 2
            elif trial == 'FBPO' and group[person] == 'PO':
                df['time'] = 2
            elif trial == 'FBPO' and group[person] == 'FSA':
                df['time'] = 3
            elif trial == 'FBIC' and group[person] == 'PO':
                df['time'] = 3
            elif trial == '2FB':
                df['time'] = 4
            elif trial == '2Reg':
                df['time'] = 5
            PROP_df = pd.concat([PROP_df, df], ignore_index=True)

# PROP_df.to_excel("meanPROP.xlsx", index=False)


PROP_df_wide = PROP_df.pivot(index='participantID', columns='trialtype', values='PROP')

# groupFSA = list([])
# groupPO = list([])

# colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
i=-1
fig = plt.figure()
plt.title('Propulsion per trial', fontsize=22)
for participant in group:
    if participant.startswith('1019'):
        i+=1
        
        # if group[participant] == 'FSA':
        #     markerfig = 'o'
            
        #     yvalues = np.array([PROP_df_wide['1Reg'][participant], PROP_df_wide['FBIC'][participant], PROP_df_wide['FBPO'][participant], PROP_df_wide['2FB'][participant], PROP_df_wide['2Reg'][participant]])
        #     # groupFSA.append(participant)
        # elif group[participant] == 'PO':
        #     markerfig = '^'
        #     yvalues = np.array([PROP_df_wide['1Reg'][participant], PROP_df_wide['FBPO'][participant], PROP_df_wide['FBIC'][participant], PROP_df_wide['2FB'][participant], PROP_df_wide['2Reg'][participant]])
        #     # groupPO.append(participant)
        
        markerfig_all = 'o'
        yvalues = np.array([PROP_df_wide['1Reg'][participant], PROP_df_wide['FBIC'][participant], PROP_df_wide['FBPO'][participant], PROP_df_wide['2FB'][participant], PROP_df_wide['2Reg'][participant]])
        yerr = np.array([]),
        try:
            yerr = np.append(yerr, sdPROP['1Reg'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdPROP['FBIC'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdPROP['FBPO'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdPROP['2FB'][participant])
        except:    
            yerr = np.append(yerr, np.nan)
        try:
            yerr = np.append(yerr, sdPROP['2Reg'][participant])
        except:
            yerr = np.append(yerr, np.nan)
        xvalues = np.array([1, 2, 3, 4, 5], dtype=float)
        
        # plt.plot(xvalues, yvalues, marker=markerfig, color=colors[i], alpha=0.25)
        plt.plot(xvalues, yvalues, marker=markerfig_all, color=colors[i], alpha=0.25)
        plt.errorbar(xvalues, yvalues, yerr=yerr, color=colors[i], ecolor=colors[i], alpha=0.25, capsize=5)

# xmeansFSA = np.array([0.95, 1.95, 2.95, 3.95, 4.95])
subset1Reg = dict()
for key in groupFSA:
    try:
        subset1Reg[key] =meanPROP['1Reg'][key]
    except:
        pass
subsetFBIC = dict()
for key in groupFSA:
    try:
        subsetFBIC[key] =meanPROP['FBIC'][key]
    except:
        pass
subsetFBPO = dict()
for key in groupFSA:
    try:
        subsetFBPO[key] =meanPROP['FBPO'][key]
    except:
        pass
subset2FB = dict()
for key in groupFSA:
    try:
        subset2FB[key] =meanPROP['2FB'][key]
    except:
        pass
subset2Reg = dict()
for key in groupFSA:
    try:
        subset2Reg[key] =meanPROP['2Reg'][key]
    except:
        pass
# ymeansFSA = np.array([np.mean(list(subset1Reg.values())), np.mean(list(subsetFBIC.values())), np.mean(list(subsetFBPO.values())), np.mean(list(subset2FB.values())), np.mean(list(subset2Reg.values()))])
# yerrFSA = np.array([np.std(list(subset1Reg.values())), np.std(list(subsetFBIC.values())), np.std(list(subsetFBPO.values())), np.std(list(subset2FB.values())), np.std(list(subset2Reg.values()))])
# plt.plot(xmeansFSA, ymeansFSA, marker='o', color='k', alpha=1, label = 'First feedback FSA')

# xmeansPO = np.array([1.05, 2.05, 3.05, 4.05, 5.05])
# subset1Reg = dict()
for key in groupPO:
    try:
        subset1Reg[key] =meanPROP['1Reg'][key]
    except:
        pass
# subsetFBIC = dict()
for key in groupPO:
    try:
        subsetFBIC[key] =meanPROP['FBIC'][key]
    except:
        pass
# subsetFBPO = dict()
for key in groupPO:
    try:
        subsetFBPO[key] =meanPROP['FBPO'][key]
    except:
        pass
# subset2FB = dict()
for key in groupPO:
    try:
        subset2FB[key] =meanPROP['2FB'][key]
    except:
        pass
# subset2Reg = dict()
for key in groupPO:
    try:
        subset2Reg[key] =meanPROP['2Reg'][key]
    except:
        pass

xmeans_all = np.array([1.05, 2.05, 3.05, 4.05, 5.05])
ymeans_all = np.array([np.nanmean(list(subset1Reg.values())), np.nanmean(list(subsetFBIC.values())), np.nanmean(list(subsetFBPO.values())), np.nanmean(list(subset2FB.values())), np.nanmean(list(subset2Reg.values()))])
# ymeansPO = np.array([np.mean(list(subset1Reg.values())), np.mean(list(subsetFBPO.values())), np.mean(list(subsetFBIC.values())), np.mean(list(subset2FB.values())), np.mean(list(subset2Reg.values()))])
# yerrPO = np.array([np.std(list(subset1Reg.values())), np.std(list(subsetFBIC.values())), np.std(list(subsetFBPO.values())), np.std(list(subset2FB.values())), np.std(list(subset2Reg.values()))])
# plt.plot(xmeansPO, ymeansPO, marker='^', color='k', alpha=1, label = 'First feedback propulsion')
plt.plot(xmeans_all, ymeans_all, marker=markerfig_all, markersize=10, linewidth=3, color='k', alpha=1, label = 'Mean propulsion')
# plt.errorbar(xmeansFSA, ymeansFSA, yerr=yerrFSA, color = 'k', ecolor='k', alpha=1, capsize=5)
# plt.errorbar(xmeansPO, ymeansPO, yerr=yerrPO, color = 'k', ecolor='k', alpha=1, capsize=5)
plt.xlim((0,6))
plt.xticks(ticks=xmeans_all, labels = list(['1Reg', 'FBIC', 'FBPO', '2FB', '2Reg']), fontsize=20)
plt.ylabel('FSA (degrees)', fontsize=20)
plt.yticks(fontsize=16)
# plt.legend(loc = 'center right', bbox_to_anchor=(1.12, 0.5),
#           ncol=1, fancybox=True, shadow=True)


print('unpaired PROP healthy vs stroke: ', stats.ttest_ind(np.fromiter(meanPROP['Healthy controls'].values(), dtype=float), np.fromiter(meanPROP['1Reg'].values(), dtype=float)) )






# # # # #  EFFECT OF FEEDBACK ON FSA - PAIRED SAMPLES T-TEST # # # # #

# ttest_fb_FSA = dict()
# ttest_NOfb_FSA = dict()
# meanFSAfeedback = dict()
# sdFSAfeedback = dict()
# meanFSANOfeedback = dict()
# sdFSANOfeedback = dict()
# for person in triallist:
#     try:
#         # Identify trial with feedback on FSA
#         FSAfeedbacktrial = [s for s in triallist[person] if 'FBIC' in s]
#         # Identify first regular walking trial
#         NOfeedbacktrial = [s for s in triallist[person] if '1Reg' in s]
        
#         # Take first 100 strides of both trials
#         FSAfeedback = np.append(foot_vicon[FSAfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers initial contact left'][0:50]], foot_vicon[FSAfeedbacktrial[0]]['Angle right filt'][vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers initial contact right'][0:50]])
#         FSANOfeedback = np.append(foot_vicon[NOfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[NOfeedbacktrial[0]]['Index numbers initial contact left'][0:50]], foot_vicon[NOfeedbacktrial[0]]['Angle right filt'][vicon_gait_events[NOfeedbacktrial[0]]['Index numbers initial contact right'][0:50]])
        
#         # Calculate mean and SD
#         meanFSAfeedback[person] = np.nanmean(FSAfeedback)
#         sdFSAfeedback[person] = np.nanstd(FSAfeedback)
#         meanFSANOfeedback[person] = np.nanmean(FSANOfeedback)
#         sdFSANOfeedback[person] = np.nanstd(FSANOfeedback)
    
#         # Dependent samples t-test for statistical difference between IC angel with and without feedback of the first 80 strides in each trial
#         ttest_fb_FSA[FSAfeedbacktrial[0][0:10]] = FSAfeedback 
#         ttest_NOfb_FSA[NOfeedbacktrial[0][0:10]] = FSANOfeedback
        
#         print('Effect of feedback on FSA for person ', person, 'was: ', stats.ttest_rel(ttest_fb_FSA[FSAfeedbacktrial[0][0:10]], ttest_NOfb_FSA[NOfeedbacktrial[0][0:10]]))
#     except:
#         pass

# # Effect of feedback on FSA for all participants
# ttest_fb_FSA['all'] = np.array([])
# ttest_NOfb_FSA['all'] = np.array([])
# for person in ttest_fb_FSA:
#     ttest_fb_FSA['all'] = np.append(ttest_fb_FSA['all'], ttest_fb_FSA[person])
#     ttest_NOfb_FSA['all'] = np.append(ttest_NOfb_FSA['all'], ttest_NOfb_FSA[person])

# meanFSAfeedback['all'] = np.nanmean(ttest_fb_FSA['all'])
# sdFSAfeedback['all'] = np.nanstd(ttest_fb_FSA['all'])
# meanFSANOfeedback['all'] = np.nanmean(ttest_NOfb_FSA['all'])
# sdFSANOfeedback['all'] = np.nanstd(ttest_NOfb_FSA['all'])
# print('Effect of feedback on FSA across subjects was: ', stats.ttest_rel(ttest_fb_FSA['all'], ttest_NOfb_FSA['all']) )


# # Debug plot of flexion-extension angle in trial with and without feedback
# match debugplots:
#     case True:
#         timeFB = np.zeros(shape=(len(foot_vicon[FSAfeedbacktrial[0]]['Angle left filt']), 1))
#         for i in range(1,len(timeFB)):
#             timeFB[i] = timeFB[i-1]+0.01
#         timeNO = np.zeros(shape=(len(foot_vicon[NOfeedbacktrial[0]]['Angle left filt']), 1))
#         for i in range(1,len(timeNO)):
#             timeNO[i] = timeNO[i-1]+0.01
#         fig = plt.figure()
#         plt.title('Flexion-extension angle', fontsize = 17)
#         plt.plot(timeFB, foot_vicon[FSAfeedbacktrial[0]]['Angle left filt'], 'k', label='OMCS')
#         plt.plot(timeNO, foot_vicon[NOfeedbacktrial[0]]['Angle left filt'], 'b', label='OMCS')
#         plt.plot(timeFB[vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers initial contact left']], foot_vicon[FSAfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers initial contact left']], 'vk')
#         plt.plot(timeNO[vicon_gait_events[NOfeedbacktrial[0]]['Index numbers initial contact left']], foot_vicon[NOfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[NOfeedbacktrial[0]]['Index numbers initial contact left']], 'vb')
#         plt.ylabel(ylabel='Angle (deg)', fontsize=16)
#         plt.yticks(fontsize=14)
#         plt.xlabel(xlabel = 'Time (s)', fontsize=16)
#         plt.xticks(fontsize=14)


# colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
# i=-1
# fig = plt.figure()
# plt.title('FSA per trial with and without feedback', fontsize=22)
# for trial in meanFSAfeedback:
#     if trial.startswith('1019'):
#         i+=1
#         plt.plot(np.array([-1,1]), -1*np.array([meanFSANOfeedback[trial], meanFSAfeedback[trial]]), marker='o', color=colors[i], alpha=0.25)
#         plt.errorbar(-1, -1*meanFSANOfeedback[trial], yerr=sdFSANOfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
#         plt.errorbar(1, -1*meanFSAfeedback[trial], yerr=sdFSAfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
# plt.plot(np.array([-0.95, 0.95]), -1*np.array([meanFSANOfeedback['all'], meanFSAfeedback['all']]), marker='o', color='k', alpha=1)
# plt.errorbar(-0.95, -1*meanFSANOfeedback['all'], yerr=sdFSANOfeedback['all'], ecolor='k', alpha=1, capsize=5)
# plt.errorbar(0.95, -1*meanFSAfeedback['all'], yerr=sdFSAfeedback['all'], ecolor='k', alpha=1, capsize=5)
# plt.xlim((-2,2))
# plt.xticks(ticks=np.array([-1,1]), labels = list(['Without feedback', 'With feedback']), fontsize=20)
# plt.ylabel('FSA (degrees)', fontsize=20)
# plt.yticks(fontsize=16)








# # # # # #  EFFECT OF FEEDBACK ON PROPULSION  - PAIRED SAMPLES T-TEST# # # # #

# ttest_fb_PROP = dict()
# ttest_NOfb_PROP = dict()
# meanPROPfeedback = dict()
# sdPROPfeedback = dict()
# meanPROPNOfeedback = dict()
# sdPROPNOfeedback = dict()
# for person in triallist:
#     try:
#         # Identify trial with feedback on propulsion
#         PROPfeedbacktrial = [s for s in triallist[person] if 'FBPO' in s]
#         # Identify first regular walking trial
#         NOfeedbacktrial = [s for s in triallist[person] if '1Reg' in s]
        
#         # Take as many strides as equally possible of both trials
#         PROPfeedback = np.append(vicon_spatiotemporals[PROPfeedbacktrial[0]]['Propulsion left'][:,2], vicon_spatiotemporals[PROPfeedbacktrial[0]]['Propulsion right'][:,2])
#         PROPNOfeedback = np.append(vicon_spatiotemporals[NOfeedbacktrial[0]]['Propulsion left'][:,2], vicon_spatiotemporals[NOfeedbacktrial[0]]['Propulsion right'][:,2])
#         min_length = np.min([len(PROPfeedback), len(PROPNOfeedback)])
#         PROPfeedback = PROPfeedback[0:min_length]
#         PROPNOfeedback = PROPNOfeedback[0:min_length]
        
#         # Calculate mean and SD
#         meanPROPfeedback[person] = np.nanmean(PROPfeedback)
#         sdPROPfeedback[person] = np.nanstd(PROPfeedback)
#         meanPROPNOfeedback[person] = np.nanmean(PROPNOfeedback)
#         sdPROPNOfeedback[person] = np.nanstd(PROPNOfeedback)
    
#         # Dependent samples t-test for statistical difference between IC angel with and without feedback of the first 80 strides in each trial
#         ttest_fb_PROP[PROPfeedbacktrial[0][0:10]] = PROPfeedback 
#         ttest_NOfb_PROP[NOfeedbacktrial[0][0:10]] = PROPNOfeedback
        
#         print('Effect of feedback on propulsion for person ', person, 'was: ', stats.ttest_rel(ttest_fb_PROP[PROPfeedbacktrial[0][0:10]], ttest_NOfb_PROP[NOfeedbacktrial[0][0:10]]))
#     except:
#         pass

# # Effect of feedback on FSA for all participants
# ttest_fb_PROP['all'] = np.array([])
# ttest_NOfb_PROP['all'] = np.array([])
# for person in ttest_fb_PROP:
#     ttest_fb_PROP['all'] = np.append(ttest_fb_PROP['all'], ttest_fb_PROP[person])
#     ttest_NOfb_PROP['all'] = np.append(ttest_NOfb_PROP['all'], ttest_NOfb_PROP[person])

# meanPROPfeedback['all'] = np.nanmean(ttest_fb_PROP['all'])
# sdPROPfeedback['all'] = np.nanstd(ttest_fb_PROP['all'])
# meanPROPNOfeedback['all'] = np.nanmean(ttest_NOfb_PROP['all'])
# sdPROPNOfeedback['all'] = np.nanstd(ttest_NOfb_PROP['all'])
# print('Effect of feedback on propulsion across subjects was: ', stats.ttest_rel(ttest_fb_PROP['all'], ttest_NOfb_PROP['all']) )

# colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
# i=-1
# fig = plt.figure()
# plt.title('Propulsion per trial with and without feedback', fontsize=22)
# for trial in meanPROPfeedback:
#     if trial.startswith('1019'):
#         i+=1
#         plt.plot(np.array([-1,1]), np.array([meanPROPNOfeedback[trial], meanPROPfeedback[trial]]), marker='o', color=colors[i], alpha=0.25)
#         plt.errorbar(-1, meanPROPNOfeedback[trial], yerr=sdPROPNOfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
#         plt.errorbar(1, meanPROPfeedback[trial], yerr=sdPROPfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
# plt.plot(np.array([-0.95, 0.95]), np.array([meanPROPNOfeedback['all'], meanPROPfeedback['all']]), marker='o', color='k', alpha=1)
# plt.errorbar(-0.95, meanPROPNOfeedback['all'], yerr=sdPROPNOfeedback['all'], ecolor='k', alpha=1, capsize=5)
# plt.errorbar(0.95, meanPROPfeedback['all'], yerr=sdPROPfeedback['all'], ecolor='k', alpha=1, capsize=5)
# plt.xlim((-2,2))
# plt.xticks(ticks=np.array([-1,1]), labels = list(['Without feedback', 'With feedback']), fontsize=20)
# plt.ylabel('Propulsion (N/(kg*s))', fontsize=20)
# plt.yticks(fontsize=16)
