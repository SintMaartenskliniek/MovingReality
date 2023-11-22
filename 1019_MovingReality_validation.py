"""
Main script for:
    1019 Moving(g) Reality project, validation of foot strike angle and
    identification of an inidcator for forward propulsion.
    
2023, Carmen Ensink, Sint Maartenksliniek,
c.ensink@maartenskliniek.nl

"""

# Import dependencies
# To run this script you need to pull the following repositories:
#    To import and analyze OMCS data: https://github.com/SintMaartenskliniek/OMCS_GaitAnalysis (Release: version 1.2.0, tag v1.2.0)
#    To import and analyze IMU data: https://github.com/SintMaartenskliniek/IMU_GaitAnalysis (Release: Validation study, tag v1.1.0)
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
from MovingReality_functions import person_characteristics, corresponding_filenames, define_filepaths, import_vicondata, import_xsensdata, foot_kinematics_vicon, foot_kinematics_xsens, shank_kinematics_vicon, shank_kinematics_xsens, bland_altman_plot, calculate_indicative_variables, transform_pcc_input_variables_all, transform_pcc_input_variables_perPerson
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

# Import sensordata
xsens, errors = import_xsensdata(pathsxsens)

# Check on matching data (discard if no data available from both systems)
remove=list()
for key in vicon:
    if key not in xsens.keys():
        remove.append(key)
for key in remove:
    del vicon[key], vicon_gait_events[key], vicon_spatiotemporals[key], analogdata[key]
remove=list()
for key in xsens:
    if key not in vicon.keys():
        remove.append(key)
for key in remove:
    del xsens[key]
    
# Triallist based on task
trials = dict()
trials['FBIC'] = list()
trials['FBPO'] = list()
trials['2FB'] = list()
trials['1Reg'] = list()
trials['2Reg'] = list()
for key in xsens:
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




# # # # # # FOOT STRIKE ANGLE # # # # # #

# Foot kinematics vicon
foot_vicon = foot_kinematics_vicon(vicon, xsens=xsens)

# Foot kinematics xsens
foot_xsens = foot_kinematics_xsens(xsens)

# Shank kinematics vicon
shank_vicon = shank_kinematics_vicon(vicon)

# Shank kinematics xsens
shank_xsens = shank_kinematics_xsens(xsens)


# Debug figures angle foot/shank
match debugplots:
    case True:
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 16,
                }
        
        # Debugplot foot_angle
        for trial in foot_xsens:
             print(trial)
             time = np.zeros(shape=(len(foot_vicon[trial]['Angle left filt']), 1))
             for i in range(1,len(time)):
                 time[i] = time[i-1]+0.01
             fig = plt.figure()
             plt.title('Flexion-extension angle', fontsize = 17)
             plt.plot(time, foot_vicon[trial]['Angle left filt'], 'k', label='OMCS')
             plt.plot(time, foot_xsens[trial]['Angle left'], 'r-.', label = 'Sensor')
           
             plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Initial Contact']], foot_vicon[trial]['Angle left filt'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']], '*r')
             plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Initial Contact']], foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']], '*r')
             plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], foot_vicon[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], '*g')
             plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], '*g')
             plt.ylabel(ylabel='Angle (deg)', fontsize=16)
             plt.yticks(fontsize=14)
             plt.xlabel(xlabel = 'Time (s)', fontsize=16)
             plt.xticks(fontsize=14)
        
        # Debugplot shank angle
        for trial in shank_xsens:
            print(trial)
            time = np.zeros(shape=(len(shank_vicon[trial]['Angle left']), 1))
            for i in range(1,len(time)):
                time[i] = time[i-1]+0.01
            fig = plt.figure()
            plt.title('Shank angle - ' +  trial, fontsize = 17)
            plt.plot(time, shank_vicon[trial]['Angle left'], 'k', label='OMCS')
            plt.plot(time, shank_xsens[trial]['Angle left'], 'r-.', label = 'Sensor')
            
            plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Initial Contact']], shank_vicon[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']], '*r')
            plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Initial Contact']], shank_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']], '*r')
            plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], shank_vicon[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], '*g')
            plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], shank_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Terminal Contact']], '*g')
            plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Mid-Stance Onset']], shank_vicon[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Mid-Stance Onset']], 'kx')
            plt.plot(time[xsens[trial]['Left foot']['Gait Events']['Mid-Stance Onset']], shank_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Mid-Stance Onset']], 'kx')
            plt.ylabel(ylabel='Angle (deg)', fontsize=16)
            plt.yticks(fontsize=14)
            plt.xlabel(xlabel = 'Time (s)', fontsize=16)
            plt.xticks(fontsize=14)    


# # #  ALL EVENTS, ALL TRIALS, ALL PARTICIPANTS  # # #

# Pearson correlation FSA xsens vs vicon - all events of all trials of every participant
pcc_ICangle_xsens = np.array([])
pcc_ICangle_vicon = np.array([])
for trial in foot_vicon:
    pcc_ICangle_vicon = np.append(pcc_ICangle_vicon, foot_vicon[trial]['Angle left filt'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']])
    pcc_ICangle_vicon = np.append(pcc_ICangle_vicon, foot_vicon[trial]['Angle right filt'][xsens[trial]['Right foot']['Gait Events']['Initial Contact']])
for trial in foot_xsens:
    pcc_ICangle_xsens = np.append(pcc_ICangle_xsens, foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']])
    pcc_ICangle_xsens = np.append(pcc_ICangle_xsens, foot_xsens[trial]['Angle right'][xsens[trial]['Right foot']['Gait Events']['Initial Contact']])

rho_ICangle, pval_ICangle = stats.pearsonr(pcc_ICangle_vicon, pcc_ICangle_xsens)

# Intraclass correlation coefficient FSA xsens vs vicon - all events of all trials of every participant
icc_df = pd.DataFrame(columns = ['strideID', 'value', 'system']) 
df_sensor = pd.DataFrame(columns = ['strideID', 'value', 'system'])
df_sensor['strideID'] = np.arange(start=1, stop=len(pcc_ICangle_xsens)+1, step=1)
df_sensor['value'] = pcc_ICangle_xsens
df_sensor['system'] = 'xsens'
df_vicon = pd.DataFrame(columns = ['strideID', 'value', 'system'])
df_vicon['strideID'] = np.arange(start=1, stop=len(pcc_ICangle_vicon)+1, step=1)
df_vicon['value'] = pcc_ICangle_vicon
df_vicon['system'] = 'vicon'
icc_df=icc_df.append(df_sensor)
icc_df=icc_df.append(df_vicon)

icc = pingouin.intraclass_corr(data=icc_df, targets='strideID', raters='system', ratings='value').round(3)
icc.set_index("Type")

# Bland-Altman analysis FSA xsens vs vicon
bland_altman_plot(pcc_ICangle_xsens, pcc_ICangle_vicon) #dataType='Initial contact angle', unit='(deg)'

# Repeatability coefficient
# https://rowannicholls.github.io/python/statistics/agreement/repeatability_coefficient.html
RC_df_SL = pd.DataFrame(columns = ['OMCS','sensor'])
RC_df_SL['OMCS'] = pcc_ICangle_vicon
RC_df_SL['sensor'] = pcc_ICangle_xsens
# Within stride sample variances
var_w = RC_df_SL[['OMCS', 'sensor']].var(axis=1, ddof=1)
# Mean within-stride sample variance
var_mean_w = var_w.mean()
# Within-stride sample standard deviation
sd_w = np.sqrt(var_mean_w)
# Coefficient of repeatability
RC_stridelevel = 1.96*np.sqrt(2) * sd_w



# # #  ACROSS SUBJECTS  # # #

# Pearson correlation foot angle at initial contact xsens vs vicon - all events for each trial
rho_ICangle_trials = dict()
pval_ICangle_trials = dict()

for trial in foot_vicon:
    rho_ICangle_trials[trial], pval_ICangle_trials[trial] = stats.pearsonr(foot_vicon[trial]['Angle left filt'], foot_xsens[trial]['Angle left'])
    

# Bland-Altman person colored - stack FSA for all trials within subject
means_pp = dict()
data_per_person = dict()
fig = plt.subplots()
plt.title('Bland-Altman plot', fontsize=20)
colors = list(['seagreen', 'mediumturquoise', 'mediumblue', 'fuchsia', 'red', 'darkorange', 'gold', 'gray', 'lightgreen', 'deepskyblue', 'palevioletred', 'darkcyan', 'darkorchid'])
i=0
for key in triallist:
    means_pp[key] = np.array([ [], [], [], [], [], [] ]).T
    data_per_person[key] = dict()
    data_per_person[key]['Vicon'] = np.array([])
    data_per_person[key]['Xsens'] = np.array([])
    for trial in triallist[key]:
        try:
            
            dataVL     = np.asarray(foot_vicon[trial]['Angle left filt'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']])
            dataSL     = np.asarray(foot_xsens[trial]['Angle left'][xsens[trial]['Left foot']['Gait Events']['Initial Contact']])
            dataVR     = np.asarray(foot_vicon[trial]['Angle right filt'][xsens[trial]['Right foot']['Gait Events']['Initial Contact']])
            dataSR     = np.asarray(foot_xsens[trial]['Angle right'][xsens[trial]['Right foot']['Gait Events']['Initial Contact']])
            
            meanL      = np.nanmean([dataVL, dataSL], axis=0)
            meanR      = np.nanmean([dataVR, dataSR], axis=0)

            diffL      = (dataSL - dataVL)
            diffR      = (dataSR - dataVR)
            mdL        = np.nanmean(diffL)                  # Mean of the difference
            mdR        = np.nanmean(diffR)
            
            sdL        = np.nanstd(diffL, axis=0)           # Standard deviation of the difference
            sdR        = np.nanstd(diffR, axis=0)           # Standard deviation of the difference
            
            means_pp[key] = np.vstack((means_pp[key], np.array([mdL, mdL-1.96*sdL, mdL+1.96*sdL, mdR, mdR-1.96*sdR, mdR+1.96*sdR])))
            
            data_per_person[key]['Vicon'] = np.append(data_per_person[key]['Vicon'], dataVL)
            data_per_person[key]['Vicon'] = np.append(data_per_person[key]['Vicon'], dataVR)
            data_per_person[key]['Xsens'] = np.append(data_per_person[key]['Xsens'], dataSL)
            data_per_person[key]['Xsens'] = np.append(data_per_person[key]['Xsens'], dataSR)
            
            # Check for inputname in **kwargs items
            dataType = str()
            unit = str()
                
            plt.scatter(meanL, diffL, edgecolor = colors[i], facecolor='None', marker = 'o', label=trial+' left') # SMK green: '#004D43'
            
            plt.scatter(meanR, diffR, edgecolor = colors[i], facecolor='None', marker = '^', label = trial+' right') # SMK green: '#004D43'
            
        except KeyError:
            print('keyerror on trial: ', trial)            
    i=i+1

plt.xlabel("Mean (sensor, optical) "+ unit, fontsize=14)
plt.ylabel("Difference (sensor - optical) " + unit, fontsize=14) #Difference between measures
plt.xticks(fontsize=14)    
plt.yticks(fontsize=14)
plt.legend(fontsize=6)  

# Check difference, mean difference and limits of agreement within each participant
for person in data_per_person:
    data_per_person[person]['Difference'] = data_per_person[person]['Xsens'] - data_per_person[person]['Vicon']
    data_per_person[person]['Mean difference'] = np.nanmean(data_per_person[person]['Difference'])
    data_per_person[person]['LoA'] = np.array([data_per_person[person]['Mean difference']-1.96*np.nanstd(data_per_person[person]['Difference'], axis=0), data_per_person[person]['Mean difference']+1.96*np.nanstd(data_per_person[person]['Difference'], axis=0)])
    data_per_person[person]['LoA interval'] = data_per_person[person]['LoA'][1]-data_per_person[person]['LoA'][0]

# Check mean interval of limits of agreement across subjects (mean and standard deviation of all within subject intervals)
LoAintervals = np.array([])
for person in data_per_person:
    LoAintervals = np.append(LoAintervals, data_per_person[person]['LoA interval'])
meanLoA = np.nanmean(LoAintervals)
sdLoA = np.nanstd(LoAintervals)

# Bland-Altman analysis FSA xsens vs vicon across subjects (mean xsens-based FSA vs mean vicon-based FSA for each person )
BA_acrossSubjects_vicon = np.array([])
BA_acrossSubjects_xsens = np.array([])
for person in data_per_person:
    mas = np.nanmean(data_per_person[person]['Vicon'])
    BA_acrossSubjects_vicon = np.append(BA_acrossSubjects_vicon, mas)
    mas = np.nanmean(data_per_person[person]['Xsens'])
    BA_acrossSubjects_xsens = np.append(BA_acrossSubjects_xsens, mas)

bland_altman_plot(BA_acrossSubjects_xsens, BA_acrossSubjects_vicon, alpha=1) #, dataType='Initial contact angle across subjects', unit='(deg)'

# Repeatability coefficient
# https://rowannicholls.github.io/python/statistics/agreement/repeatability_coefficient.html
RC_df_PL = pd.DataFrame(columns = ['OMCS','sensor'])
RC_df_PL['OMCS'] = BA_acrossSubjects_vicon
RC_df_PL['sensor'] = BA_acrossSubjects_xsens
# Within stride sample variances
var_w = RC_df_PL[['OMCS', 'sensor']].var(axis=1, ddof=1)
# Within-stride sample standard deviation
sd_w = np.sqrt(var_w)
# Coefficient of repeatability
RC_personlevel_mean = np.mean(1.96*np.sqrt(2) * sd_w)
RC_personlevel_sd = np.std(1.96*np.sqrt(2) * sd_w)



# # # # # # PROPULSION # # # # # #

# Caculate propulsion
for trial in vicon_gait_events:
    vicon_gait_events[trial], vicon_spatiotemporals[trial], analogdata[trial] = propulsion(vicon_gait_events[trial], vicon_spatiotemporals[trial], analogdata[trial], bodyweight=bodyweight[trial[0:10]], debugplot=False, plot_title=trial)


# Debug figure propulsion over time colored and marked by person and side
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
                    plt.scatter(x2, y2, edgecolor = colors[i], facecolor='None', marker = '^', label = trial+' right') # SMK green: '#004D43'
                    
                except KeyError:
                    print('keyerror on trial: ', trial)            
            i=i+1
        
        plt.xlabel("Timestamp IC "+ unit, fontsize=14)
        plt.ylabel("Propulsion " + unit, fontsize=14) #Difference between measures
        
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.yticks(fontsize=14)
        
        plt.legend(fontsize=6)  





# Calculate foot angle at TC, shank angle at TC, max angular velocity during stance, max angular acceleration during stance
foot_vicon, shank_vicon, foot_xsens, shank_xsens = calculate_indicative_variables(vicon_gait_events, vicon_spatiotemporals, xsens, foot_vicon, shank_vicon, foot_xsens, shank_xsens)


# For all participants, for all trials, put the left and right kinematic variable and propulsive force in one "pcc_variable" to be able to calculate the Pearson correlation between the kinematic variable and propulsive force
for trial in vicon_spatiotemporals:
    vicon_spatiotemporals[trial]['Peak propulsion left'][:,1] = -1*vicon_spatiotemporals[trial]['Peak propulsion left'][:,1]
    vicon_spatiotemporals[trial]['Peak propulsion right'][:,1] = -1*vicon_spatiotemporals[trial]['Peak propulsion right'][:,1]
pcc_propulsion, pcc_propulsion_peak, pcc_foot_angleTC_vicon, pcc_shank_angleTC_vicon, pcc_foot_maxangvel_vicon, pcc_shank_maxangvel_vicon, pcc_foot_maxangacc_vicon, pcc_shank_maxangacc_vicon, pcc_shank_maxlinacc_vicon, pcc_foot_stridelength_vicon, pcc_foot_angleTC_xsens, pcc_shank_angleTC_xsens, pcc_foot_maxangvel_xsens, pcc_shank_maxangvel_xsens, pcc_foot_maxangacc_xsens, pcc_shank_maxangacc_xsens, pcc_shank_maxlinacc_xsens, pcc_foot_stridelength_xsens = transform_pcc_input_variables_all(vicon_spatiotemporals, foot_vicon, shank_vicon, foot_xsens, shank_xsens)


# # # # # PROPULSION AS AREA UNDER THE CURVE # # # # #

# Assign variable names for the Pearson rho and p-values
rho_prop_foot_angleTC = dict()
pval_prop_foot_angleTC = dict() 
rho_prop_shank_angleTC = dict()
pval_prop_shank_angleTC = dict() 
rho_prop_foot_angvel = dict()
pval_prop_foot_angvel = dict()
rho_prop_shank_angvel = dict()
pval_prop_shank_angvel = dict()
rho_prop_foot_angacc = dict()
pval_prop_foot_angacc = dict()
rho_prop_shank_angacc = dict()
pval_prop_shank_angacc = dict()
rho_prop_shank_linacc = dict()
pval_prop_shank_linacc = dict()
rho_prop_foot_stridelength = dict()
pval_prop_foot_stridelength = dict()

# Calculate Pearson correlation over all trials
# Vicon
rho_prop_foot_angvel['all vicon'], pval_prop_foot_angvel['all vicon'] = stats.pearsonr(pcc_propulsion, pcc_foot_maxangvel_vicon)
rho_prop_foot_angacc['all vicon'], pval_prop_foot_angacc['all vicon'] = stats.pearsonr(pcc_propulsion, pcc_foot_maxangacc_vicon)
rho_prop_foot_angleTC['all vicon'], pval_prop_foot_angleTC['all vicon'] = stats.pearsonr(pcc_propulsion, pcc_foot_angleTC_vicon)
rho_prop_shank_angvel['all vicon'], pval_prop_shank_angvel['all vicon'] = stats.pearsonr(pcc_propulsion, pcc_shank_maxangvel_vicon)
rho_prop_shank_angacc['all vicon'], pval_prop_shank_angacc['all vicon'] = stats.pearsonr(pcc_propulsion, pcc_shank_maxangacc_vicon)
rho_prop_shank_angleTC['all vicon'], pval_prop_shank_angleTC['all vicon'] = stats.pearsonr(pcc_propulsion, pcc_shank_angleTC_vicon)
rho_prop_shank_linacc['all vicon'], pval_prop_shank_linacc['all vicon'] = stats.pearsonr(pcc_propulsion, pcc_shank_maxlinacc_vicon)
rho_prop_foot_stridelength['all vicon'], pval_prop_foot_stridelength['all vicon'] = stats.pearsonr(pcc_propulsion[~np.isnan(pcc_foot_stridelength_vicon)], pcc_foot_stridelength_vicon[~np.isnan(pcc_foot_stridelength_vicon)])

# Xsens
rho_prop_foot_angvel['all xsens'], pval_prop_foot_angvel['all xsens'] = stats.pearsonr(pcc_propulsion, pcc_foot_maxangvel_xsens)
rho_prop_foot_angacc['all xsens'], pval_prop_foot_angacc['all xsens'] = stats.pearsonr(pcc_propulsion, pcc_foot_maxangacc_xsens)
rho_prop_foot_angleTC['all xsens'], pval_prop_foot_angleTC['all xsens'] = stats.pearsonr(pcc_propulsion, pcc_foot_angleTC_xsens)
rho_prop_shank_angvel['all xsens'], pval_prop_shank_angvel['all xsens'] = stats.pearsonr(pcc_propulsion, pcc_shank_maxangvel_xsens)
rho_prop_shank_angacc['all xsens'], pval_prop_shank_angacc['all xsens'] = stats.pearsonr(pcc_propulsion, pcc_shank_maxangacc_xsens)
rho_prop_shank_angleTC['all xsens'], pval_prop_shank_angleTC['all xsens'] = stats.pearsonr(pcc_propulsion, pcc_shank_angleTC_xsens)
rho_prop_shank_linacc['all xsens'], pval_prop_shank_linacc['all xsens'] = stats.pearsonr(pcc_propulsion, pcc_shank_maxlinacc_xsens)
rho_prop_foot_stridelength['all xsens'], pval_prop_foot_stridelength['all xsens'] = stats.pearsonr(pcc_propulsion[~np.isnan(pcc_foot_stridelength_xsens)], pcc_foot_stridelength_xsens[~np.isnan(pcc_foot_stridelength_xsens)])

# Calculate Pearon correlation for each trial
# Vicon
for trial in foot_vicon:
    prop_left = vicon_spatiotemporals[trial]['Propulsion left'][:,2][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])]
    prop_right = vicon_spatiotemporals[trial]['Propulsion right'][:,2][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]
    rho_prop_foot_angleTC[trial+' vicon'], pval_prop_foot_angleTC[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_vicon[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_vicon[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_angleTC[trial+' vicon'], pval_prop_shank_angleTC[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_foot_angvel[trial+' vicon'], pval_prop_foot_angvel[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_vicon[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_vicon[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_angvel[trial+' vicon'], pval_prop_shank_angvel[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_foot_angacc[trial+' vicon'], pval_prop_foot_angacc[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_vicon[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_vicon[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_angacc[trial+' vicon'], pval_prop_shank_angacc[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_linacc[trial+' vicon'], pval_prop_shank_linacc[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Max linear acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Max linear acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    sl_and_prop_true_left = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])), np.argwhere(~np.isnan(foot_vicon[trial]['Stride length left']))).flatten()]
    sl_and_prop_true_right = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])), np.argwhere(~np.isnan(foot_vicon[trial]['Stride length right']))).flatten()]
    rho_prop_foot_stridelength[trial+' vicon'], pval_prop_foot_stridelength[trial+' vicon'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][sl_and_prop_true_left,2], vicon_spatiotemporals[trial]['Propulsion right'][sl_and_prop_true_right,2]), np.append(foot_vicon[trial]['Stride length left'][sl_and_prop_true_left], foot_vicon[trial]['Stride length right'][sl_and_prop_true_right]))
    # rho_prop_foot_stridelength[trial+' vicon'], pval_prop_foot_stridelength[trial+' vicon'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][~np.isnan(foot_vicon[trial]['Stride length left']),2], vicon_spatiotemporals[trial]['Propulsion right'][~np.isnan(foot_vicon[trial]['Stride length right']),2]), np.append(foot_vicon[trial]['Stride length left'][~np.isnan(foot_vicon[trial]['Stride length left'])], foot_vicon[trial]['Stride length right'][~np.isnan(foot_vicon[trial]['Stride length right'])]))


# Xsens
for trial in foot_xsens:
    prop_left = vicon_spatiotemporals[trial]['Propulsion left'][:,2][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])]
    prop_right = vicon_spatiotemporals[trial]['Propulsion right'][:,2][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]
    rho_prop_foot_angleTC[trial+' xsens'], pval_prop_foot_angleTC[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_xsens[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_xsens[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_angleTC[trial+' xsens'], pval_prop_shank_angleTC[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_foot_angvel[trial+' xsens'], pval_prop_foot_angvel[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_xsens[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_xsens[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_angvel[trial+' xsens'], pval_prop_shank_angvel[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_foot_angacc[trial+' xsens'], pval_prop_foot_angacc[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_xsens[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_xsens[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_angacc[trial+' xsens'], pval_prop_shank_angacc[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_prop_shank_linacc[trial+' xsens'], pval_prop_shank_linacc[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Max linear acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Max linear acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    sl_and_prop_true_left = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])), np.argwhere(~np.isnan(foot_xsens[trial]['Stride length left']))).flatten()]
    sl_and_prop_true_right = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])), np.argwhere(~np.isnan(foot_xsens[trial]['Stride length right']))).flatten()]
    rho_prop_foot_stridelength[trial+' xsens'], pval_prop_foot_stridelength[trial+' xsens'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][sl_and_prop_true_left,2], vicon_spatiotemporals[trial]['Propulsion right'][sl_and_prop_true_right,2]), np.append(foot_xsens[trial]['Stride length left'][sl_and_prop_true_left], foot_xsens[trial]['Stride length right'][sl_and_prop_true_right]))
    # rho_prop_foot_stridelength[trial+' xsens'], pval_prop_foot_stridelength[trial+' xsens'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][~np.isnan(foot_xsens[trial]['Stride length left']),2], vicon_spatiotemporals[trial]['Propulsion right'][~np.isnan(foot_xsens[trial]['Stride length right']),2]), np.append(foot_xsens[trial]['Stride length left'][~np.isnan(foot_xsens[trial]['Stride length left'])], foot_xsens[trial]['Stride length right'][~np.isnan(foot_xsens[trial]['Stride length right'])]))
    

# For all participants, for all trials, put the left and right kinematic variable and propulsive force in a "pcc_per_person" to be able to calculate the Pearson correlation between the kinematic variable and propulsive force for each individual
pcc_per_person = transform_pcc_input_variables_perPerson(triallist, vicon_spatiotemporals, foot_vicon, shank_vicon, foot_xsens, shank_xsens)

# Calculate Pearson correlation for each individual
rho_per_person = dict()
rho_per_person['vicon'] = dict()
rho_per_person['xsens'] = dict()
rho_per_person['vicon']['foot angle TC'] = np.array([])
rho_per_person['xsens']['foot angle TC'] = np.array([])
rho_per_person['vicon']['foot angular velocity'] = np.array([])
rho_per_person['xsens']['foot angular velocity'] = np.array([])
rho_per_person['vicon']['foot angular acceleration'] = np.array([])
rho_per_person['xsens']['foot angular acceleration'] = np.array([])
rho_per_person['vicon']['stride length'] = np.array([])
rho_per_person['xsens']['stride length'] = np.array([])
rho_per_person['vicon']['shank angle TC'] = np.array([])
rho_per_person['xsens']['shank angle TC'] = np.array([])
rho_per_person['vicon']['shank angular velocity'] = np.array([])
rho_per_person['xsens']['shank angular velocity'] = np.array([])
rho_per_person['vicon']['shank angular acceleration'] = np.array([])
rho_per_person['xsens']['shank angular acceleration'] = np.array([])
rho_per_person['vicon']['shank linear acceleration'] = np.array([])
rho_per_person['xsens']['shank linear acceleration'] = np.array([])

for param in pcc_per_person['vicon']:
    for person in pcc_per_person['propulsion']:
        if param != 'stride length':
            rho_per_person['vicon'][param] = np.append(rho_per_person['vicon'][param], stats.pearsonr(pcc_per_person['propulsion'][person], pcc_per_person['vicon'][param][person])[0] )
            rho_per_person['xsens'][param] = np.append(rho_per_person['xsens'][param], stats.pearsonr(pcc_per_person['propulsion'][person], pcc_per_person['xsens'][param][person])[0] )
        elif param == 'stride length':
            rho_per_person['vicon'][param] = np.append(rho_per_person['vicon'][param], stats.pearsonr(pcc_per_person['propulsion stridelength vicon'][person], pcc_per_person['vicon'][param][person])[0] )
            rho_per_person['xsens'][param] = np.append(rho_per_person['xsens'][param], stats.pearsonr(pcc_per_person['propulsion stridelength xsens'][person], pcc_per_person['xsens'][param][person])[0] )

# visual inspection of the histogram plots showed normal distribution; thus calculate mean and standard deviation
mean_rho = dict()
mean_rho['vicon'] = dict()
mean_rho['xsens'] = dict()
sd_rho = dict()
sd_rho['vicon'] = dict()
sd_rho['xsens'] = dict()
for param in rho_per_person['vicon']:
    mean_rho['vicon'][param] = np.nanmean(rho_per_person['vicon'][param])
    mean_rho['xsens'][param] = np.nanmean(rho_per_person['xsens'][param])
    sd_rho['vicon'][param] = np.nanstd(rho_per_person['vicon'][param])
    sd_rho['xsens'][param] = np.nanstd(rho_per_person['xsens'][param])



# # # # # PROPULSION AS PEAK AP-GRF # # # # # 

# Assign variable names for the Pearson rho and p-values
rho_peakprop_foot_angleTC = dict()
pval_peakprop_foot_angleTC = dict() 
rho_peakprop_shank_angleTC = dict()
pval_peakprop_shank_angleTC = dict() 
rho_peakprop_foot_angvel = dict()
pval_peakprop_foot_angvel = dict()
rho_peakprop_shank_angvel = dict()
pval_peakprop_shank_angvel = dict()
rho_peakprop_foot_angacc = dict()
pval_peakprop_foot_angacc = dict()
rho_peakprop_shank_angacc = dict()
pval_peakprop_shank_angacc = dict()
rho_peakprop_shank_linacc = dict()
pval_peakprop_shank_linacc = dict()
rho_peakprop_foot_stridelength = dict()
pval_peakprop_foot_stridelength = dict()

# Calculate Pearson correlation over all trials
# Vicon
rho_peakprop_foot_angvel['all vicon'], pval_peakprop_foot_angvel['all vicon'] = stats.pearsonr(pcc_propulsion_peak, pcc_foot_maxangvel_vicon)
rho_peakprop_foot_angacc['all vicon'], pval_peakprop_foot_angacc['all vicon'] = stats.pearsonr(pcc_propulsion_peak, pcc_foot_maxangacc_vicon)
rho_peakprop_foot_angleTC['all vicon'], pval_peakprop_foot_angleTC['all vicon'] = stats.pearsonr(pcc_propulsion_peak, pcc_foot_angleTC_vicon)
rho_peakprop_shank_angvel['all vicon'], pval_peakprop_shank_angvel['all vicon'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_maxangvel_vicon)
rho_peakprop_shank_angacc['all vicon'], pval_peakprop_shank_angacc['all vicon'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_maxangacc_vicon)
rho_peakprop_shank_angleTC['all vicon'], pval_peakprop_shank_angleTC['all vicon'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_angleTC_vicon)
rho_peakprop_shank_linacc['all vicon'], pval_peakprop_shank_linacc['all vicon'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_maxlinacc_vicon)
rho_peakprop_foot_stridelength['all vicon'], pval_peakprop_foot_stridelength['all vicon'] = stats.pearsonr(pcc_propulsion_peak[~np.isnan(pcc_foot_stridelength_vicon)], pcc_foot_stridelength_vicon[~np.isnan(pcc_foot_stridelength_vicon)])

# Xsens
rho_peakprop_foot_angvel['all xsens'], pval_peakprop_foot_angvel['all xsens'] = stats.pearsonr(pcc_propulsion_peak, pcc_foot_maxangvel_xsens)
rho_peakprop_foot_angacc['all xsens'], pval_peakprop_foot_angacc['all xsens'] = stats.pearsonr(pcc_propulsion_peak, pcc_foot_maxangacc_xsens)
rho_peakprop_foot_angleTC['all xsens'], pval_peakprop_foot_angleTC['all xsens'] = stats.pearsonr(pcc_propulsion_peak, pcc_foot_angleTC_xsens)
rho_peakprop_shank_angvel['all xsens'], pval_peakprop_shank_angvel['all xsens'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_maxangvel_xsens)
rho_peakprop_shank_angacc['all xsens'], pval_peakprop_shank_angacc['all xsens'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_maxangacc_xsens)
rho_peakprop_shank_angleTC['all xsens'], pval_peakprop_shank_angleTC['all xsens'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_angleTC_xsens)
rho_peakprop_shank_linacc['all xsens'], pval_peakprop_shank_linacc['all xsens'] = stats.pearsonr(pcc_propulsion_peak, pcc_shank_maxlinacc_xsens)
rho_peakprop_foot_stridelength['all xsens'], pval_peakprop_foot_stridelength['all xsens'] = stats.pearsonr(pcc_propulsion_peak[~np.isnan(pcc_foot_stridelength_xsens)], pcc_foot_stridelength_xsens[~np.isnan(pcc_foot_stridelength_xsens)])

# Calculate Pearon correlation for each trial
# Vicon
for trial in foot_vicon:
    prop_left = vicon_spatiotemporals[trial]['Peak propulsion left'][:,1]
    prop_right = vicon_spatiotemporals[trial]['Peak propulsion right'][:,1]
    rho_peakprop_foot_angleTC[trial+' vicon'], pval_peakprop_foot_angleTC[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_vicon[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_vicon[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_angleTC[trial+' vicon'], pval_peakprop_shank_angleTC[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_foot_angvel[trial+' vicon'], pval_peakprop_foot_angvel[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_vicon[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_vicon[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_angvel[trial+' vicon'], pval_peakprop_shank_angvel[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_foot_angacc[trial+' vicon'], pval_peakprop_foot_angacc[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_vicon[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_vicon[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_angacc[trial+' vicon'], pval_peakprop_shank_angacc[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_linacc[trial+' vicon'], pval_peakprop_shank_linacc[trial+' vicon'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_vicon[trial]['Max linear acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_vicon[trial]['Max linear acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    sl_and_prop_true_left = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])), np.argwhere(~np.isnan(foot_vicon[trial]['Stride length left']))).flatten()]
    sl_and_prop_true_right = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])), np.argwhere(~np.isnan(foot_vicon[trial]['Stride length right']))).flatten()]
    rho_peakprop_foot_stridelength[trial+' vicon'], pval_peakprop_foot_stridelength[trial+' vicon'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][sl_and_prop_true_left,2], vicon_spatiotemporals[trial]['Propulsion right'][sl_and_prop_true_right,2]), np.append(foot_vicon[trial]['Stride length left'][sl_and_prop_true_left], foot_vicon[trial]['Stride length right'][sl_and_prop_true_right]))
    # rho_peakprop_foot_stridelength[trial+' vicon'], pval_peakprop_foot_stridelength[trial+' vicon'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][~np.isnan(foot_vicon[trial]['Stride length left']),2], vicon_spatiotemporals[trial]['Propulsion right'][~np.isnan(foot_vicon[trial]['Stride length right']),2]), np.append(foot_vicon[trial]['Stride length left'][~np.isnan(foot_vicon[trial]['Stride length left'])], foot_vicon[trial]['Stride length right'][~np.isnan(foot_vicon[trial]['Stride length right'])]))


# Xsens
for trial in foot_xsens:
    prop_left = vicon_spatiotemporals[trial]['Peak propulsion left'][:,1]
    prop_right = vicon_spatiotemporals[trial]['Peak propulsion right'][:,1]
    rho_peakprop_foot_angleTC[trial+' xsens'], pval_peakprop_foot_angleTC[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_xsens[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_xsens[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_angleTC[trial+' xsens'], pval_peakprop_shank_angleTC[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Angle at TC left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Angle at TC right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_foot_angvel[trial+' xsens'], pval_peakprop_foot_angvel[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_xsens[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_xsens[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_angvel[trial+' xsens'], pval_peakprop_shank_angvel[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Max angular velocity stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Max angular velocity stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_foot_angacc[trial+' xsens'], pval_peakprop_foot_angacc[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(foot_xsens[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], foot_xsens[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_angacc[trial+' xsens'], pval_peakprop_shank_angacc[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Max angular acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Max angular acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    rho_peakprop_shank_linacc[trial+' xsens'], pval_peakprop_shank_linacc[trial+' xsens'] = stats.pearsonr(np.append(prop_left, prop_right), np.append(shank_xsens[trial]['Max linear acceleration stance phase left'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])], shank_xsens[trial]['Max linear acceleration stance phase right'][~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])]))
    sl_and_prop_true_left = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion left'][:,2])), np.argwhere(~np.isnan(foot_xsens[trial]['Stride length left']))).flatten()]
    sl_and_prop_true_right = np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])) [np.isin(np.argwhere(~np.isnan(vicon_spatiotemporals[trial]['Propulsion right'][:,2])), np.argwhere(~np.isnan(foot_xsens[trial]['Stride length right']))).flatten()]
    rho_peakprop_foot_stridelength[trial+' xsens'], pval_peakprop_foot_stridelength[trial+' xsens'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][sl_and_prop_true_left,2], vicon_spatiotemporals[trial]['Propulsion right'][sl_and_prop_true_right,2]), np.append(foot_xsens[trial]['Stride length left'][sl_and_prop_true_left], foot_xsens[trial]['Stride length right'][sl_and_prop_true_right]))
    # rho_peakprop_foot_stridelength[trial+' xsens'], pval_peakprop_foot_stridelength[trial+' xsens'] = stats.pearsonr(np.append(vicon_spatiotemporals[trial]['Propulsion left'][~np.isnan(foot_xsens[trial]['Stride length left']),2], vicon_spatiotemporals[trial]['Propulsion right'][~np.isnan(foot_xsens[trial]['Stride length right']),2]), np.append(foot_xsens[trial]['Stride length left'][~np.isnan(foot_xsens[trial]['Stride length left'])], foot_xsens[trial]['Stride length right'][~np.isnan(foot_xsens[trial]['Stride length right'])]))
    

# For all participants, for all trials, put the left and right kinematic variable and propulsive force in a "pcc_per_person_peak" to be able to calculate the Pearson correlation between the kinematic variable and propulsive force for each individual
pcc_per_person_peak = transform_pcc_input_variables_perPerson(triallist, vicon_spatiotemporals, foot_vicon, shank_vicon, foot_xsens, shank_xsens)

# Calculate Pearson correlation for each individual
rho_per_person_peak = dict()
rho_per_person_peak['vicon'] = dict()
rho_per_person_peak['xsens'] = dict()
rho_per_person_peak['vicon']['foot angle TC'] = np.array([])
rho_per_person_peak['xsens']['foot angle TC'] = np.array([])
rho_per_person_peak['vicon']['foot angular velocity'] = np.array([])
rho_per_person_peak['xsens']['foot angular velocity'] = np.array([])
rho_per_person_peak['vicon']['foot angular acceleration'] = np.array([])
rho_per_person_peak['xsens']['foot angular acceleration'] = np.array([])
rho_per_person_peak['vicon']['stride length'] = np.array([])
rho_per_person_peak['xsens']['stride length'] = np.array([])
rho_per_person_peak['vicon']['shank angle TC'] = np.array([])
rho_per_person_peak['xsens']['shank angle TC'] = np.array([])
rho_per_person_peak['vicon']['shank angular velocity'] = np.array([])
rho_per_person_peak['xsens']['shank angular velocity'] = np.array([])
rho_per_person_peak['vicon']['shank angular acceleration'] = np.array([])
rho_per_person_peak['xsens']['shank angular acceleration'] = np.array([])
rho_per_person_peak['vicon']['shank linear acceleration'] = np.array([])
rho_per_person_peak['xsens']['shank linear acceleration'] = np.array([])

for param in pcc_per_person_peak['vicon']:
    for person in pcc_per_person_peak['propulsion peak']:
        if param != 'stride length':
            rho_per_person_peak['vicon'][param] = np.append(rho_per_person_peak['vicon'][param], stats.pearsonr(pcc_per_person_peak['propulsion peak'][person], pcc_per_person_peak['vicon'][param][person])[0] )
            rho_per_person_peak['xsens'][param] = np.append(rho_per_person_peak['xsens'][param], stats.pearsonr(pcc_per_person_peak['propulsion peak'][person], pcc_per_person_peak['xsens'][param][person])[0] )
        elif param == 'stride length':
            rho_per_person_peak['vicon'][param] = np.append(rho_per_person_peak['vicon'][param], stats.pearsonr(pcc_per_person_peak['propulsion stridelength vicon'][person], pcc_per_person_peak['vicon'][param][person])[0] )
            rho_per_person_peak['xsens'][param] = np.append(rho_per_person_peak['xsens'][param], stats.pearsonr(pcc_per_person_peak['propulsion stridelength xsens'][person], pcc_per_person_peak['xsens'][param][person])[0] )

# visual inspection of the histogram plots showed normal distribution; thus calculate mean and standard deviation
mean_rho_peak = dict()
mean_rho_peak['vicon'] = dict()
mean_rho_peak['xsens'] = dict()
sd_rho_peak = dict()
sd_rho_peak['vicon'] = dict()
sd_rho_peak['xsens'] = dict()
for param in rho_per_person_peak['vicon']:
    mean_rho_peak['vicon'][param] = np.nanmean(rho_per_person_peak['vicon'][param])
    mean_rho_peak['xsens'][param] = np.nanmean(rho_per_person_peak['xsens'][param])
    sd_rho_peak['vicon'][param] = np.nanstd(rho_per_person_peak['vicon'][param])
    sd_rho_peak['xsens'][param] = np.nanstd(rho_per_person_peak['xsens'][param])









match debugplots:
    case True:
        # Correlation scatter for all variables
        plt_variable = 'Max foot angular velocity'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_foot_maxangvel_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)  
        
        
        plt_variable = 'Max shank angular velocity'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_shank_maxangvel_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)  
        
        
        plt_variable = 'Max foot angular acceleration'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_foot_maxangacc_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)
        
        plt_variable = 'Max shank angular acceleration'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_shank_maxangacc_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)
        
        
        plt_variable = 'Foot angle at terminal contact'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_foot_angleTC_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.scatter(pcc_propulsion, pcc_foot_angleTC_vicon, edgecolor = 'r', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)
        
        
        plt_variable = 'Shank angle at terminal contact'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_shank_angleTC_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.scatter(pcc_propulsion, pcc_shank_angleTC_vicon, edgecolor = 'r', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)
        
        
        plt_variable = 'Shank linear acceleration'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_shank_maxlinacc_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)
        
        plt_variable = 'Stride length'
        fig = plt.subplots()
        plt.title('Correlation scatter '+plt_variable, fontsize=20)
        plt.scatter(pcc_propulsion, pcc_foot_stridelength_xsens, edgecolor = 'k', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.scatter(pcc_propulsion, pcc_foot_stridelength_vicon/1000, edgecolor = 'r', facecolor='None', marker = 'o', label=plt_variable) # SMK green: '#004D43'
        plt.xlabel("Propulsion", fontsize=14)
        plt.xticks(fontsize=14)    
        # set_xticklabels(fontsize=16)
        plt.ylabel(plt_variable, fontsize=14) #Difference between measures
        plt.yticks(fontsize=14)
        # plt.legend(fontsize=10)