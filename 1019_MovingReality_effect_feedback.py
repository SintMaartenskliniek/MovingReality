# -*- coding: utf-8 -*-
"""
Main script for:
    1019 Moving(g) Reality, feedback study
    
Version - Author:
    24-08-2023 Study effect of feedback on propulsive force and clean up for publication - C.J. Ensink
    02-08-2023 Only analyze 120 seconds of all trials - C.J. Ensink
    18-07-2023 Check correlation propulsion with stride length - C.J. Ensink
    21-06-2023 Updated dependencies - C.J Ensink
    17-05-2023 Debugged calculation of propulsion (only negative area under the curve) - C.J. Ensink
    28-03-2023 Check kinematics and correct propulsion based on Hafer et al. (2020) for study data - C.J. Ensink
    20-12-2022 Check correlation propulsion and kinematics for pilot data - C.J. Ensink
    31-05-2022 Check correlation propulsion and kinematics for all CVA data 0900 - Smarten the clinic, validation study - C.J. Ensink
    06-04-2022 Check correlation propulsion and kinematics - C.J. Ensink
    15-02-2022 Initial script - C.J. Ensink, c.ensink@maartenskliniek.nl

"""

# Import dependencies
# To run this script you need to pull the following repositories:
#    To import and analyze OMCS data: https://github.com/SintMaartenskliniek/OMCS_GaitAnalysis
#    To import and analyze IMU data: https://github.com/SintMaartenskliniek/IMU_GaitAnalysis
import sys
sys.path.insert(0, 'C:/Users/ensinkc.SMK/OneDrive - Sint Maartenskliniek/Documents/GitHub/SMK_IMU_GaitAnalysis')
sys.path.insert(0, 'C:/Users/ensinkc.SMK/OneDrive - Sint Maartenskliniek/Documents/GitHub/OMCS_GaitAnalysis')

import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import numpy as np
from scipy import signal
from scipy import stats
import os
import pingouin
import pandas as pd
from MovingReality_functions import person_characteristics, corresponding_filenames, define_filepaths, import_vicondata, import_xsensdata, foot_kinematics_vicon, foot_kinematics_xsens, bland_altman_plot
from gaitcharacteristics import propulsion

# Define if debugplots should be plotted or not
# debugplots = True
debugplots = False


# Import person characteristics
bodyweight, height, age, time_since_stroke, comfortable_gait_speed = person_characteristics()
# Check for normal distribution of the person characteristics
significance_level = 0.05 # set p-value for significance
if stats.shapiro(list(bodyweight.values()))[1] < significance_level:
    print('Bodyweight is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(bodyweight.values())))
else:
    print('Bodyweight is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(bodyweight.values())))
if stats.shapiro(list(height.values()))[1] < significance_level:
    print('Height is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(height.values())))
else:
    print('Height is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(height.values())))
if stats.shapiro(list(age.values()))[1] < significance_level:
    print('Age is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(age.values())))
else:
    print('Age is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(age.values())))
if stats.shapiro(list(time_since_stroke.values()))[1] < significance_level:
    print('Time since stroke onset is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(time_since_stroke.values())))
else:
    print('Time since stroke onset is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(time_since_stroke.values())))
if stats.shapiro(list(comfortable_gait_speed.values()))[1] < significance_level:
    print('Comfortable gait speed is not normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(comfortable_gait_speed.values())))
else:
    print('Comfortable gait speed is normally distributed, Shapiro-Wilk result is: ', stats.shapiro(list(comfortable_gait_speed.values())))


# Define location of data
datafolder = os.path.abspath('data')

# Define corresponding filenames of sensor and vicon data
corresponding_files, triallist = corresponding_filenames()

# Define corresponding filepaths of sensor and vicon data
pathsvicon, pathsxsens = define_filepaths(datafolder, corresponding_files)

# Import vicon data
vicon, vicon_gait_events, vicon_spatiotemporals, analogdata = import_vicondata(pathsvicon)

# # Import sensordata
# xsens, errors = import_xsensdata(pathsxsens)

# # Triallist based on task
# trials = dict()
# trials['FBIC'] = list()
# trials['FBPO'] = list()
# trials['2FB'] = list()
# trials['1Reg'] = list()
# trials['2Reg'] = list()
# for key in xsens:
#     if "FBIC" in key:
#         trials['FBIC'].append(key)
#     elif "FBPO" in key:
#         trials['FBPO'].append(key)
#     elif "2FB" in key:
#         if '1019_MR002_2303082FB' in key:
#             trials['2Reg'].append(key)
#         else:
#             trials['2FB'].append(key)
#     elif "1Reg" in key:
#         trials['1Reg'].append(key)
#     elif "2Reg" in key:
#         trials['2Reg'].append(key)




# # # # # # FOOT STRIKE ANGLE # # # # # #

# Foot kinematics vicon
foot_vicon = foot_kinematics_vicon(vicon)

# Foot kinematics xsens
# foot_xsens = foot_kinematics_xsens(xsens)

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
           
             # plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Initial Contact']], foot_vicon[trial]['Angle left filt'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']], '*r')
             # plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Initial Contact']], foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']], '*r')
             # plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], foot_vicon[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], '*g')
             # plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], '*g')
             plt.ylabel(ylabel='Angle (deg)', fontsize=16)
             plt.yticks(fontsize=14)
             plt.xlabel(xlabel = 'Time (s)', fontsize=16)
             plt.xticks(fontsize=14)
        



# # # # #  EFFECT OF FEEDBACK ON FSA  # # # # #

ttest_fb_FSA = dict()
ttest_NOfb_FSA = dict()
meanFSAfeedback = dict()
sdFSAfeedback = dict()
meanFSANOfeedback = dict()
sdFSANOfeedback = dict()
for person in triallist:
    try:
        # Identify trial with feedback on FSA
        FSAfeedbacktrial = [s for s in triallist[person] if 'FBIC' in s]
        # Identify first regular walking trial
        NOfeedbacktrial = [s for s in triallist[person] if '1Reg' in s]
        
        # Take first 100 strides of both trials
        FSAfeedback = np.append(foot_vicon[FSAfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers initial contact left'][0:50]], foot_vicon[FSAfeedbacktrial[0]]['Angle right filt'][vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers initial contact right'][0:50]])
        FSANOfeedback = np.append(foot_vicon[NOfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[NOfeedbacktrial[0]]['Index numbers initial contact left'][0:50]], foot_vicon[NOfeedbacktrial[0]]['Angle right filt'][vicon_gait_events[NOfeedbacktrial[0]]['Index numbers initial contact right'][0:50]])
        
        # Calculate mean and SD
        meanFSAfeedback[person] = np.nanmean(FSAfeedback)
        sdFSAfeedback[person] = np.nanstd(FSAfeedback)
        meanFSANOfeedback[person] = np.nanmean(FSANOfeedback)
        sdFSANOfeedback[person] = np.nanstd(FSANOfeedback)
    
        # Dependent samples t-test for statistical difference between IC angel with and without feedback of the first 80 strides in each trial
        ttest_fb_FSA[FSAfeedbacktrial[0][0:10]] = FSAfeedback 
        ttest_NOfb_FSA[NOfeedbacktrial[0][0:10]] = FSANOfeedback
        
        print('Effect of feedback on FSA for person ', person, 'was: ', stats.ttest_rel(ttest_fb_FSA[FSAfeedbacktrial[0][0:10]], ttest_NOfb_FSA[NOfeedbacktrial[0][0:10]]))
    except:
        pass

# Effect of feedback on FSA for all participants
ttest_fb_FSA['all'] = np.array([])
ttest_NOfb_FSA['all'] = np.array([])
for person in ttest_fb_FSA:
    ttest_fb_FSA['all'] = np.append(ttest_fb_FSA['all'], ttest_fb_FSA[person])
    ttest_NOfb_FSA['all'] = np.append(ttest_NOfb_FSA['all'], ttest_NOfb_FSA[person])

meanFSAfeedback['all'] = np.nanmean(ttest_fb_FSA['all'])
sdFSAfeedback['all'] = np.nanstd(ttest_fb_FSA['all'])
meanFSANOfeedback['all'] = np.nanmean(ttest_NOfb_FSA['all'])
sdFSANOfeedback['all'] = np.nanstd(ttest_NOfb_FSA['all'])
print('Effect of feedback on FSA across subjects was: ', stats.ttest_rel(ttest_fb_FSA['all'], ttest_NOfb_FSA['all']) )


# Debug plot of flexion-extension angle in trial with and without feedback
match debugplots:
    case True:
        timeFB = np.zeros(shape=(len(foot_vicon[FSAfeedbacktrial[0]]['Angle left filt']), 1))
        for i in range(1,len(timeFB)):
            timeFB[i] = timeFB[i-1]+0.01
        timeNO = np.zeros(shape=(len(foot_vicon[NOfeedbacktrial[0]]['Angle left filt']), 1))
        for i in range(1,len(timeNO)):
            timeNO[i] = timeNO[i-1]+0.01
        fig = plt.figure()
        plt.title('Flexion-extension angle', fontsize = 17)
        plt.plot(timeFB, foot_vicon[FSAfeedbacktrial[0]]['Angle left filt'], 'k', label='OMCS')
        plt.plot(timeNO, foot_vicon[NOfeedbacktrial[0]]['Angle left filt'], 'b', label='OMCS')
        plt.plot(timeFB[vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers heel strike left']], foot_vicon[FSAfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[FSAfeedbacktrial[0]]['Index numbers heel strike left']], 'vk')
        plt.plot(timeNO[vicon_gait_events[NOfeedbacktrial[0]]['Index numbers heel strike left']], foot_vicon[NOfeedbacktrial[0]]['Angle left filt'][vicon_gait_events[NOfeedbacktrial[0]]['Index numbers heel strike left']], 'vb')
        plt.ylabel(ylabel='Angle (deg)', fontsize=16)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel = 'Time (s)', fontsize=16)
        plt.xticks(fontsize=14)


colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
i=-1
fig = plt.figure()
plt.title('FSA per trial with and without feedback', fontsize=22)
for trial in meanFSAfeedback:
    if trial.startswith('1019'):
        i+=1
        plt.plot(np.array([-1,1]), -1*np.array([meanFSANOfeedback[trial], meanFSAfeedback[trial]]), marker='o', color=colors[i], alpha=0.25)
        plt.errorbar(-1, -1*meanFSANOfeedback[trial], yerr=sdFSANOfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
        plt.errorbar(1, -1*meanFSAfeedback[trial], yerr=sdFSAfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
plt.plot(np.array([-0.95, 0.95]), -1*np.array([meanFSANOfeedback['all'], meanFSAfeedback['all']]), marker='o', color='k', alpha=1)
plt.errorbar(-0.95, -1*meanFSANOfeedback['all'], yerr=sdFSANOfeedback['all'], ecolor='k', alpha=1, capsize=5)
plt.errorbar(0.95, -1*meanFSAfeedback['all'], yerr=sdFSAfeedback['all'], ecolor='k', alpha=1, capsize=5)
plt.xlim((-2,2))
plt.xticks(ticks=np.array([-1,1]), labels = list(['Without feedback', 'With feedback']), fontsize=20)
plt.ylabel('FSA (degrees)', fontsize=20)
plt.yticks(fontsize=16)

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



# # # # #  EFFECT OF FEEDBACK ON PROPULSION  # # # # #

ttest_fb_PROP = dict()
ttest_NOfb_PROP = dict()
meanPROPfeedback = dict()
sdPROPfeedback = dict()
meanPROPNOfeedback = dict()
sdPROPNOfeedback = dict()
for person in triallist:
    try:
        # Identify trial with feedback on propulsion
        PROPfeedbacktrial = [s for s in triallist[person] if 'FBPO' in s]
        # Identify first regular walking trial
        NOfeedbacktrial = [s for s in triallist[person] if '1Reg' in s]
        
        # Take as many strides as equally possible of both trials
        PROPfeedback = np.append(vicon_spatiotemporals[PROPfeedbacktrial[0]]['Propulsion left'][:,2], vicon_spatiotemporals[PROPfeedbacktrial[0]]['Propulsion right'][:,2])
        PROPNOfeedback = np.append(vicon_spatiotemporals[NOfeedbacktrial[0]]['Propulsion left'][:,2], vicon_spatiotemporals[NOfeedbacktrial[0]]['Propulsion right'][:,2])
        min_length = np.min([len(PROPfeedback), len(PROPNOfeedback)])
        PROPfeedback = PROPfeedback[0:min_length]
        PROPNOfeedback = PROPNOfeedback[0:min_length]
        
        # Calculate mean and SD
        meanPROPfeedback[person] = np.nanmean(PROPfeedback)
        sdPROPfeedback[person] = np.nanstd(PROPfeedback)
        meanPROPNOfeedback[person] = np.nanmean(PROPNOfeedback)
        sdPROPNOfeedback[person] = np.nanstd(PROPNOfeedback)
    
        # Dependent samples t-test for statistical difference between IC angel with and without feedback of the first 80 strides in each trial
        ttest_fb_PROP[PROPfeedbacktrial[0][0:10]] = PROPfeedback 
        ttest_NOfb_PROP[NOfeedbacktrial[0][0:10]] = PROPNOfeedback
        
        print('Effect of feedback on propulsion for person ', person, 'was: ', stats.ttest_rel(ttest_fb_PROP[PROPfeedbacktrial[0][0:10]], ttest_NOfb_PROP[NOfeedbacktrial[0][0:10]]))
    except:
        pass

# Effect of feedback on FSA for all participants
ttest_fb_PROP['all'] = np.array([])
ttest_NOfb_PROP['all'] = np.array([])
for person in ttest_fb_PROP:
    ttest_fb_PROP['all'] = np.append(ttest_fb_PROP['all'], ttest_fb_PROP[person])
    ttest_NOfb_PROP['all'] = np.append(ttest_NOfb_PROP['all'], ttest_NOfb_PROP[person])

meanPROPfeedback['all'] = np.nanmean(ttest_fb_PROP['all'])
sdPROPfeedback['all'] = np.nanstd(ttest_fb_PROP['all'])
meanPROPNOfeedback['all'] = np.nanmean(ttest_NOfb_PROP['all'])
sdPROPNOfeedback['all'] = np.nanstd(ttest_NOfb_PROP['all'])
print('Effect of feedback on propulsion across subjects was: ', stats.ttest_rel(ttest_fb_PROP['all'], ttest_NOfb_PROP['all']) )

colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
i=-1
fig = plt.figure()
plt.title('Propulsion per trial with and without feedback', fontsize=22)
for trial in meanPROPfeedback:
    if trial.startswith('1019'):
        i+=1
        plt.plot(np.array([-1,1]), np.array([meanPROPNOfeedback[trial], meanPROPfeedback[trial]]), marker='o', color=colors[i], alpha=0.25)
        plt.errorbar(-1, meanPROPNOfeedback[trial], yerr=sdPROPNOfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
        plt.errorbar(1, meanPROPfeedback[trial], yerr=sdPROPfeedback[trial], ecolor=colors[i], alpha=0.25, capsize=5)
plt.plot(np.array([-0.95, 0.95]), np.array([meanPROPNOfeedback['all'], meanPROPfeedback['all']]), marker='o', color='k', alpha=1)
plt.errorbar(-0.95, meanPROPNOfeedback['all'], yerr=sdPROPNOfeedback['all'], ecolor='k', alpha=1, capsize=5)
plt.errorbar(0.95, meanPROPfeedback['all'], yerr=sdPROPfeedback['all'], ecolor='k', alpha=1, capsize=5)
plt.xlim((-2,2))
plt.xticks(ticks=np.array([-1,1]), labels = list(['Without feedback', 'With feedback']), fontsize=20)
plt.ylabel('Propulsion (N/(kg*s))', fontsize=20)
plt.yticks(fontsize=16)
