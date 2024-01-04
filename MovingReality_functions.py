"""
Function script for:
    1019 Moving(g) Reality, feedback study
    
Version - Author:
    C.J. Ensink, c.ensink@maartenskliniek.nl

"""

import os
import numpy as np
import pyquaternion as pyq
import math
import matplotlib.pyplot as plt
from scipy import signal

from readmarkerdata import readmarkerdata
from gaiteventdetection import gaiteventdetection
from gaitcharacteristics import spatiotemporals, propulsion

from gaittool.helpers.preprocessor import data_filelist, data_preprocessor
from gaittool.feet_processor.processor import process

def person_characteristics(**kwargs):
    
    bodyweight = dict()
    height = dict()
    age = dict()
    time_since_stroke = dict()
    comfortable_gait_speed = dict()
    
    # Bodyweight in kg
    bodyweight['1019_MR001'] = 122
    bodyweight['1019_MR002'] = 68
    bodyweight['1019_MR003'] = 75
    bodyweight['1019_MR004'] = 70
    bodyweight['1019_MR005'] = 80
    bodyweight['1019_MR006'] = 85
    bodyweight['1019_MR007'] = 90
    bodyweight['1019_MR008'] = 91
    bodyweight['1019_MR009'] = 74
    bodyweight['1019_MR010'] = 82
    bodyweight['1019_MR011'] = 104
    bodyweight['1019_MR012'] = 79
    
    # Height in cm
    height['1019_MR001'] = 177.5
    height['1019_MR002'] = 171
    height['1019_MR003'] = 172
    height['1019_MR004'] = 163
    height['1019_MR005'] = 183
    height['1019_MR006'] = 194
    height['1019_MR007'] = 183
    height['1019_MR008'] = 172
    height['1019_MR009'] = 171
    height['1019_MR010'] = 188
    height['1019_MR011'] = 170
    height['1019_MR012'] = 172
    
    # Age in years
    age['1019_MR001'] =  64
    age['1019_MR002'] =  49
    age['1019_MR003'] =  61
    age['1019_MR004'] =  58
    age['1019_MR005'] =  65
    age['1019_MR006'] =  69
    age['1019_MR007'] =  69
    age['1019_MR008'] =  65
    age['1019_MR009'] =  74
    age['1019_MR010'] =  37
    age['1019_MR011'] =  59
    age['1019_MR012'] =  62
    
    # Time since stroke onset in months
    time_since_stroke['1019_MR001'] = 189
    time_since_stroke['1019_MR002'] = 8
    time_since_stroke['1019_MR003'] = 210
    time_since_stroke['1019_MR004'] = 84
    time_since_stroke['1019_MR005'] = 6
    time_since_stroke['1019_MR006'] = 21
    time_since_stroke['1019_MR007'] = 18
    time_since_stroke['1019_MR008'] = 28
    time_since_stroke['1019_MR009'] = 6
    time_since_stroke['1019_MR010'] = 12
    time_since_stroke['1019_MR011'] = 36
    time_since_stroke['1019_MR012'] = 74
    
    # Fixed comfortable gait speed during feeedback trials based on the first self-paced walking trial in m/s
    comfortable_gait_speed['1019_MR001'] =  0.4
    comfortable_gait_speed['1019_MR002'] =  1.6
    comfortable_gait_speed['1019_MR003'] =  0.4
    comfortable_gait_speed['1019_MR004'] =  1.2
    comfortable_gait_speed['1019_MR005'] =  0.9
    comfortable_gait_speed['1019_MR006'] =  1.2
    comfortable_gait_speed['1019_MR007'] =  1.0
    comfortable_gait_speed['1019_MR008'] =  1.2
    comfortable_gait_speed['1019_MR009'] =  0.9
    comfortable_gait_speed['1019_MR010'] =  1.4
    comfortable_gait_speed['1019_MR011'] =  0.8
    comfortable_gait_speed['1019_MR012'] =  1.1

    return bodyweight, height, age, time_since_stroke, comfortable_gait_speed



def group_trialorder(**kwargs):
    # Trialorder PO group: 1Reg - FPPO - FPIC - 2FB - 2Reg
    # Trialorder FSA group: 1Reg - FPIC - FPPO - 2FB - 2Reg

    group=dict()
    group['1019_MR001']='FSA'
    group['1019_MR003']='FSA'
    group['1019_MR005']='FSA'
    group['1019_MR007']='FSA'
    group['1019_MR008']='FSA'
    group['1019_MR009']='FSA'
    group['1019_MR011']='FSA'
    group['1019_MR002']='PO'
    group['1019_MR004']='PO'
    group['1019_MR006']='PO'
    group['1019_MR010']='PO'
    group['1019_MR012']='PO'
    # group['FSA']=list(['1019_MR001', '1019_MR003', '1019_MR005', '1019_MR007', '1019_MR008', '1019_MR009', '1019_MR011'])
    # group['PO']=list(['1019_MR002', '1019_MR004', '1019_MR006', '1019_MR010', '1019_MR012'])
    
    return group



def corresponding_filenames(**kwargs):
    
    corresponding_files = dict()
    triallist = dict()
    
    # 1019_pp01
    triallist['1019_MR001'] = ['1019_MR001_1Reg.c3d', '1019_MR001_FBIC.c3d', '1019_MR001_FBPO.c3d', '1019_MR001_2FB.c3d', '1019_MR001_2Reg.c3d']
    corresponding_files['1019_MR001_1Reg.c3d'] = 'exported000'
    corresponding_files['1019_MR001_FBIC.c3d'] = 'exported006'
    corresponding_files['1019_MR001_FBPO.c3d'] = 'exported001'
    corresponding_files['1019_MR001_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR001_2Reg.c3d'] = 'exported007'
    
    # 1019_pp02
    triallist['1019_MR002'] = ['1019_MR002_Reg.c3d', '1019_MR002_FBIC.c3d', '1019_MR002_FBPO.c3d', '1019_MR002_2FB.c3d', '1019_MR002_2Reg.c3d']
    corresponding_files['1019_MR002_Reg.c3d'] = 'exported000'
    corresponding_files['1019_MR002_FBIC.c3d'] = 'exported001'
    corresponding_files['1019_MR002_FBPO.c3d'] = 'exported002'
    corresponding_files['1019_MR002_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR002_2Reg.c3d'] = 'exported004'
    
    # 1019_pp03
    triallist['1019_MR003'] = ['1019_MR003_1Reg02.c3d', '1019_MR003_FBIC.c3d', '1019_MR003_FBPO.c3d', '1019_MR003_2FB.c3d','1019_MR003_2Reg02.c3d']
    corresponding_files['1019_MR003_1Reg02.c3d'] = 'exported002'
    corresponding_files['1019_MR003_FBPO.c3d'] = 'exported003'
    corresponding_files['1019_MR003_FBIC.c3d'] = '' # Xsens recording error
    corresponding_files['1019_MR003_2FB.c3d'] = 'exported005'
    corresponding_files['1019_MR003_2Reg02.c3d'] = 'exported006'

    # 1019_pp04
    triallist['1019_MR004'] = ['1019_MR004_1Reg.c3d', '1019_MR004_FBIC.c3d', '1019_MR004_FBPO.c3d', '1019_MR004_2FB.c3d','1019_MR004_2Reg02.c3d']
    corresponding_files['1019_MR004_1Reg.c3d'] = 'exported000'
    corresponding_files['1019_MR004_FBIC.c3d'] = 'exported001'
    corresponding_files['1019_MR004_FBPO.c3d'] = 'exported002'
    corresponding_files['1019_MR004_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR004_2Reg02.c3d'] = 'exported004'

    # 1019_pp05
    triallist['1019_MR005'] = ['1019_MR005_1Reg01.c3d', '1019_MR005_FBIC.c3d', '1019_MR005_FBPO.c3d', '1019_MR005_2FB.c3d', '1019_MR005_2Reg.c3d']
    corresponding_files['1019_MR005_1Reg01.c3d'] = 'exported001'
    corresponding_files['1019_MR005_FBIC.c3d'] = 'exported003'
    corresponding_files['1019_MR005_FBPO.c3d'] = 'exported002'
    corresponding_files['1019_MR005_2FB.c3d'] = 'exported004'
    corresponding_files['1019_MR005_2Reg.c3d'] = 'exported005'
    
    # 1019_pp06
    triallist['1019_MR006'] = ['1019_MR006_1Reg.c3d', '1019_MR006_FBPO.c3d', '1019_MR006_2FB.c3d', '1019_MR006_2Reg02.c3d']
    corresponding_files['1019_MR006_1Reg.c3d'] = 'exported001'
    # corresponding_files['1019_MR006_FBIC.c3d'] = 'exported002' # Vicon data not of good quality; no sufficient gold-standard
    corresponding_files['1019_MR006_FBPO.c3d'] = 'exported003'
    corresponding_files['1019_MR006_2FB.c3d'] = 'exported004'
    corresponding_files['1019_MR006_2Reg02.c3d'] = '' # Xsens recording error
    
    # 1019_pp07
    triallist['1019_MR007'] = ['1019_MR007_1Reg02.c3d', '1019_MR007_FBIC.c3d', '1019_MR007_FBPO.c3d', '1019_MR007_2FB.c3d', '1019_MR007_2Reg.c3d']
    corresponding_files['1019_MR007_1Reg02.c3d'] = 'exported000'
    corresponding_files['1019_MR007_FBIC.c3d'] = 'exported002'
    corresponding_files['1019_MR007_FBPO.c3d'] = 'exported001'
    corresponding_files['1019_MR007_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR007_2Reg.c3d'] = 'exported004'
    
    # 1019_pp08
    triallist['1019_MR008'] = ['1019_MR008_1Reg02.c3d', '1019_MR008_FBIC.c3d', '1019_MR008_FBPO.c3d', '1019_MR008_2FB.c3d', '1019_MR008_2Reg.c3d']
    corresponding_files['1019_MR008_1Reg02.c3d'] = 'exported001'
    corresponding_files['1019_MR008_FBIC.c3d'] = 'exported004'
    corresponding_files['1019_MR008_FBPO.c3d'] = 'exported005'
    corresponding_files['1019_MR008_2FB.c3d'] = 'exported006'
    corresponding_files['1019_MR008_2Reg.c3d'] = 'exported007'
    
    # 1019_pp09
    triallist['1019_MR009'] = ['1019_MR009_1Reg.c3d', '1019_MR009_FBIC.c3d', '1019_MR009_FBPO.c3d', '1019_MR009_2FB.c3d', '1019_MR009_2Reg.c3d']
    corresponding_files['1019_MR009_1Reg.c3d'] = 'exported000'
    corresponding_files['1019_MR009_FBIC.c3d'] = 'exported002'
    corresponding_files['1019_MR009_FBPO.c3d'] = 'exported001'
    corresponding_files['1019_MR009_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR009_2Reg.c3d'] = 'exported004'
    
    # 1019_pp10
    triallist['1019_MR010'] = ['1019_MR010_1Reg.c3d', '1019_MR010_FBIC.c3d', '1019_MR010_FBPO.c3d', '1019_MR010_2FB.c3d', '1019_MR010_2Reg.c3d']
    corresponding_files['1019_MR010_1Reg.c3d'] = 'exported000'
    corresponding_files['1019_MR010_FBIC.c3d'] = 'exported001'
    corresponding_files['1019_MR010_FBPO.c3d'] = 'exported002'
    corresponding_files['1019_MR010_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR010_2Reg.c3d'] = 'exported004'
    
    # 1019_pp11
    triallist['1019_MR011'] = ['1019_MR011_1Reg.c3d', '1019_MR011_FBIC.c3d', '1019_MR011_FBPO.c3d', '1019_MR011_2FB.c3d', '1019_MR011_2Reg.c3d']
    corresponding_files['1019_MR011_1Reg.c3d'] = 'exported000'
    corresponding_files['1019_MR011_FBIC.c3d'] = 'exported002'
    corresponding_files['1019_MR011_FBPO.c3d'] = 'exported001'
    corresponding_files['1019_MR011_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR011_2Reg.c3d'] = 'exported004'
    
    # 1019_pp12
    triallist['1019_MR012'] = ['1019_MR012_1Reg.c3d', '1019_MR012_FBIC.c3d', '1019_MR012_FBPO.c3d', '1019_MR012_2FB.c3d', '1019_MR012_2Reg.c3d']
    corresponding_files['1019_MR012_1Reg.c3d'] = 'exported000'
    corresponding_files['1019_MR012_FBIC.c3d'] = 'exported001'
    corresponding_files['1019_MR012_FBPO.c3d'] = 'exported002'
    corresponding_files['1019_MR012_2FB.c3d'] = 'exported003'
    corresponding_files['1019_MR012_2Reg.c3d'] = 'exported004'
        
    return corresponding_files, triallist



def define_filepaths(datafolder, corresponding_files):
    
    pathsvicon = dict()
    pathsxsens = dict()
    for root, dirs, files in os.walk(datafolder):
        for file in files:
            if (file in (list(corresponding_files.keys()))):
                pathsvicon[file] = (os.path.normpath(os.path.join(root,file)))
                pathsvicon[file] = pathsvicon[file].replace("\\", "/")
    for key in pathsvicon:
        pathsxsens[key] = pathsvicon[key][0:pathsvicon[key].find('Vicon/')]+'Xsens/'+corresponding_files[key]
                # ppfoldersxsens[file] = 
    
    
    return pathsvicon, pathsxsens



def import_vicondata(pathsvicon):
    vicon = dict()
    vicon_gait_events = dict()
    vicon_spatiotemporals = dict()
    analogdata = dict()
    for trial in pathsvicon:
        try:
            
            print('Analyzing vicon data of trial: ', trial)
            vicon[trial], fs_markerdata, analogdata[trial], fs_analogdata = readmarkerdata(pathsvicon[trial], analogdata=True)
            # Only analyze last 120 seconds of trial
            for marker in vicon[trial]:
                vicon[trial][marker] = vicon[trial][marker][int(-120*fs_markerdata):,:]
            for key in analogdata[trial]:
                try:
                    analogdata[trial][key] = analogdata[trial][key][int(-120*fs_markerdata*10):,:]
                except IndexError:
                    analogdata[trial][key] = analogdata[trial][key][int(-120*fs_markerdata*10):]
            vicon_gait_events[trial] = gaiteventdetection(vicon[trial], fs_markerdata, algorithmtype='velocity', trialtype='treadmill')
            vicon_spatiotemporals[trial] = spatiotemporals(vicon[trial], vicon_gait_events[trial], sample_frequency=fs_markerdata)
        except:
            print('Vicon data of trial: ', trial, ' could not be analyzed')
        
    return vicon, vicon_gait_events, vicon_spatiotemporals, analogdata



def import_xsensdata(pathsxsens):
    xsens = dict()
    errors = dict()
    
    for trial in pathsxsens:
        try:
            print('Analyzing xsens data of trial: ', trial)
            filepaths, sensortype, sample_frequency = data_filelist(pathsxsens[trial])
            if len(filepaths) > 0:
                # Define data dictionary with all sensordata
                if sample_frequency == False:
                    xsens[trial] = data_preprocessor(filepaths, sensortype)
                else:
                    xsens[trial] = data_preprocessor(filepaths, sensortype, sample_frequency=sample_frequency)
            # Only analyze last 120 seconds of trial
            for var in xsens[trial]['Left foot']['raw']:
                xsens[trial]['Left foot']['raw'][var] = xsens[trial]['Left foot']['raw'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Left shank']['raw']:
                xsens[trial]['Left shank']['raw'][var] = xsens[trial]['Left shank']['raw'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Right foot']['raw']:
                xsens[trial]['Right foot']['raw'][var] = xsens[trial]['Right foot']['raw'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Right shank']['raw']:
                xsens[trial]['Right shank']['raw'][var] = xsens[trial]['Right shank']['raw'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Lumbar']['raw']:
                xsens[trial]['Lumbar']['raw'][var] = xsens[trial]['Lumbar']['raw'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Left foot']['derived']:
                xsens[trial]['Left foot']['derived'][var] = xsens[trial]['Left foot']['derived'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Left shank']['derived']:
                xsens[trial]['Left shank']['derived'][var] = xsens[trial]['Left shank']['derived'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Right foot']['derived']:
                xsens[trial]['Right foot']['derived'][var] = xsens[trial]['Right foot']['derived'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Right shank']['derived']:
                xsens[trial]['Right shank']['derived'][var] = xsens[trial]['Right shank']['derived'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            for var in xsens[trial]['Lumbar']['derived']:
                xsens[trial]['Lumbar']['derived'][var] = xsens[trial]['Lumbar']['derived'][var][-120*xsens[trial]['Sample Frequency (Hz)']:,:]
            xsens[trial]['Timestamp'] =  xsens[trial]['Timestamp'][-120*xsens[trial]['Sample Frequency (Hz)']:]
            xsens[trial]['trialType'] = 'Feedback treadmill'
            # Process the data
            xsens[trial], errors[trial] = process(xsens[trial], showfigure = 'hide')
        except:
            print('Xsens data of trial: ', trial, ' could not be analyzed')
    
    return xsens, errors

            


def relative_orientation(sensorRelative, sensorFixed, relative_to):
    # sensorRelative should include the orientation of the sensor relative to the orientation of another sensor over time.
    # sensorFixed should include the orientation of the sensor to which another sensor's orientation should be calculated over time.
    
    quat_sensorRelative = sensorRelative['raw']['Orientation Quaternion']
    quat_sensorFixed = sensorFixed['raw']['Orientation Quaternion']
    
    phi = np.zeros((len(quat_sensorRelative), 1))
    theta = np.zeros((len(quat_sensorRelative), 1))
    psi = np.zeros((len(quat_sensorRelative), 1))
    
    for j in range(len(quat_sensorFixed)):
        q_fixed = pyq.Quaternion( quat_sensorFixed[j,:] )
        q_relative = pyq.Quaternion( quat_sensorRelative[j,:] )
        
        # Get the 3D difference between these two orientations
        qd = q_relative * q_fixed.conjugate
        qd = qd.normalised
    
        # Calculate Euler angles from this difference quaternion
        phi[j]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) ) )
        theta[j] = np.rad2deg( math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) ) )
        psi[j]   = np.rad2deg( math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) ) )
    
    sensorRelative['derived']['Orientation relative to '+relative_to] = np.array([phi.flatten(), theta.flatten(), psi.flatten()]).T
        
    return sensorRelative



# static rotation about z-axis of sensor frame
def staticRotation(data_uncalibrated):
    stat_rot = np.array(((np.cos(np.pi), -np.sin(np.pi), 0),
                         (np.sin(np.pi), np.cos(np.pi), 0),
                         (0, 0, 1)))
    
    if len(data_uncalibrated) == 0:
        data_calibrated = data_uncalibrated
    else:
        data_calibrated = np.transpose(stat_rot.dot(np.transpose(data_uncalibrated)))

    return data_calibrated



# Foot kinematics vicon data
def foot_kinematics_vicon(vicon, **kwargs):
    
    fs_markerdata = 100 # Hz default vicon data
    for key, value in kwargs.items():
        if key == 'fs_markerdata':
            fs_markerdata = value
        
    # Set second-order low-pass butterworth filter;
    # Cut-off frequency: 15Hz
    fc = 15  # Cut-off frequency of the filter
    omega = fc / (fs_markerdata / 2) # Normalize the frequency
    N = 2 # Order of the butterworth filter
    filter_type = 'lowpass' # Type of the filter
    b, a = signal.butter(N, omega, filter_type)
    
    foot_vicon = dict()
    for trial in vicon:
        foot_vicon[trial] = dict()
        # Vicon footsegment
        foot_vicon[trial]['Segement left'] = vicon[trial]['LTOE'] - vicon[trial]['LHEE']
        foot_vicon[trial]['Segement right'] = vicon[trial]['RTOE'] - vicon[trial]['RHEE']
        # Vicon foot angle
        foot_vicon[trial]['Angle left'] = np.rad2deg(np.unwrap(np.arctan2(foot_vicon[trial]['Segement left'][:,2], foot_vicon[trial]['Segement left'][:,1])))
        foot_vicon[trial]['Angle right'] = np.rad2deg(np.unwrap(np.arctan2(foot_vicon[trial]['Segement right'][:,2], foot_vicon[trial]['Segement right'][:,1])))
        # Correct for gimbal lock
        if np.mean(foot_vicon[trial]['Angle left']) > 100:
            foot_vicon[trial]['Angle left'] = foot_vicon[trial]['Angle left']-180
        if np.mean(foot_vicon[trial]['Angle right']) > 100:
            foot_vicon[trial]['Angle right'] = foot_vicon[trial]['Angle right']-180
        if np.mean(foot_vicon[trial]['Angle left']) < -100:
            foot_vicon[trial]['Angle left'] = foot_vicon[trial]['Angle left']+180
        if np.mean(foot_vicon[trial]['Angle right']) < -100:
            foot_vicon[trial]['Angle right'] = foot_vicon[trial]['Angle right']+180
        
        # Vicon foot angle filtered and corrected for angle during mid-stance of first 10 mid-stance periods
        midstance_left = np.array([], dtype=int)
        midstance_right = np.array([], dtype=int)
        for key, value in kwargs.items():
            if key == 'xsens':
                if trial in value.keys():
                    midstance_left = value[trial]['Left foot']['Gait Phases']['Mid-Stance'][ value[trial]['Left foot']['Gait Phases']['Mid-Stance'] < value[trial]['Left foot']['Gait Events']['Initial Contact'][10] ]
                    midstance_right = value[trial]['Right foot']['Gait Phases']['Mid-Stance'][ value[trial]['Right foot']['Gait Phases']['Mid-Stance'] < value[trial]['Right foot']['Gait Events']['Initial Contact'][10] ]
        foot_vicon[trial]['Angle left filt'] = signal.filtfilt(b, a, foot_vicon[trial]['Angle left']) # Apply filter
        foot_vicon[trial]['Angle right filt'] = signal.filtfilt(b, a, foot_vicon[trial]['Angle right']) # Apply filter
        if len(midstance_left) > 0:
            foot_vicon[trial]['Angle left filt'] = foot_vicon[trial]['Angle left filt'] - np.nanmean( foot_vicon[trial]['Angle left filt'][ midstance_left ] ) # Set mid-stance angle to 0
        if len(midstance_right) >0:
            foot_vicon[trial]['Angle right filt'] = foot_vicon[trial]['Angle right filt'] - np.nanmean( foot_vicon[trial]['Angle right filt'][ midstance_right ] ) # Set mid-stance angle to 0
        # Vicon foot angular velocity
        foot_vicon[trial]['Angular velocity left'] = np.zeros(len(foot_vicon[trial]['Angle left filt']))
        foot_vicon[trial]['Angular velocity right'] = np.zeros(len(foot_vicon[trial]['Angle right filt']))
        foot_vicon[trial]['Angular velocity left'][1:] = np.diff(foot_vicon[trial]['Angle left filt'])
        foot_vicon[trial]['Angular velocity right'][1:] = np.diff(foot_vicon[trial]['Angle right filt'])
        # Foot angular acceleration
        foot_vicon[trial]['Angular acceleration left'] = np.zeros(len(foot_vicon[trial]['Angle left filt']))
        foot_vicon[trial]['Angular acceleration right'] = np.zeros(len(foot_vicon[trial]['Angle right filt']))
        foot_vicon[trial]['Angular acceleration left'][1:] = np.diff(foot_vicon[trial]['Angular velocity left'])
        foot_vicon[trial]['Angular acceleration right'][1:] = np.diff(foot_vicon[trial]['Angular velocity right'])
    
    return foot_vicon



def foot_kinematics_xsens(xsens, **kwargs):
    foot_xsens = dict()
    for trial in xsens:
        foot_xsens[trial] = dict()
        # Foot angle
        foot_xsens[trial]['Angle left'] = staticRotation(xsens[trial]['Left foot']['raw']['Orientation Euler'])[:,1] #Rotate around z-axis to match vicon orientation
        foot_xsens[trial]['Angle right'] = staticRotation(xsens[trial]['Right foot']['raw']['Orientation Euler'])[:,1]
        # Correct gimbal lock
        pks = signal.find_peaks(foot_xsens[trial]['Angle left'], height = 70, prominence=3)[0]
        for i in np.arange(start=0, stop=len(pks)-1, step=1):
            if np.min(foot_xsens[trial]['Angle left'][pks[i]:pks[i+1]]) > 50:
                foot_xsens[trial]['Angle left'][pks[i]:pks[i+1]] = -1*(foot_xsens[trial]['Angle left'][pks[i]:pks[i+1]]) - np.diff([foot_xsens[trial]['Angle left'][pks[i]], -1*foot_xsens[trial]['Angle left'][pks[i]]])
        pks = signal.find_peaks(foot_xsens[trial]['Angle right'], height = 70, prominence=3)[0]
        for i in np.arange(start=0, stop=len(pks)-1, step=1):
            if np.min(foot_xsens[trial]['Angle right'][pks[i]:pks[i+1]]) > 50:
                foot_xsens[trial]['Angle right'][pks[i]:pks[i+1]] = -1*(foot_xsens[trial]['Angle right'][pks[i]:pks[i+1]]) - np.diff([foot_xsens[trial]['Angle right'][pks[i]], -1*foot_xsens[trial]['Angle right'][pks[i]]])
        # Correct for angle during mid-stance of first 10 mid-stance periods (so foot angle defined comparable to vicon)
        foot_xsens[trial]['Angle left'] = foot_xsens[trial]['Angle left'] - np.mean( foot_xsens[trial]['Angle left'][ xsens[trial]['Left foot']['Gait Phases']['Mid-Stance'][ xsens[trial]['Left foot']['Gait Phases']['Mid-Stance'] < xsens[trial]['Left foot']['Gait Events']['Initial Contact'][10] ] ] )
        foot_xsens[trial]['Angle right'] = foot_xsens[trial]['Angle right'] - np.mean( foot_xsens[trial]['Angle right'][ xsens[trial]['Right foot']['Gait Phases']['Mid-Stance'][ xsens[trial]['Right foot']['Gait Phases']['Mid-Stance'] < xsens[trial]['Right foot']['Gait Events']['Initial Contact'][10] ] ] )
        # Foot angular velocity
        foot_xsens[trial]['Angular velocity left'] = np.sqrt(xsens[trial]['Left foot']['raw']['Gyroscope'][:,0]**2 + xsens[trial]['Left foot']['raw']['Gyroscope'][:,1]**2 + xsens[trial]['Left foot']['raw']['Gyroscope'][:,2]**2)
        foot_xsens[trial]['Angular velocity right'] = np.sqrt(xsens[trial]['Right foot']['raw']['Gyroscope'][:,0]**2 + xsens[trial]['Right foot']['raw']['Gyroscope'][:,1]**2 + xsens[trial]['Right foot']['raw']['Gyroscope'][:,2]**2)
        # Foot angular acceleration
        foot_xsens[trial]['Angular acceleration left'] = np.zeros(len(foot_xsens[trial]['Angular velocity left']))
        foot_xsens[trial]['Angular acceleration left'][1:] = np.diff(foot_xsens[trial]['Angular velocity left'])
        foot_xsens[trial]['Angular acceleration right'] = np.zeros(len(foot_xsens[trial]['Angular velocity right']))
        foot_xsens[trial]['Angular acceleration right'][1:] = np.diff(foot_xsens[trial]['Angular velocity right'])
        
    return foot_xsens



def shank_kinematics_xsens(xsens, **kwargs):
    shank_xsens = dict()
    for trial in xsens:
        shank_xsens[trial] = dict()
        
        # Shank angle
        shank_xsens[trial]['Angle left'] = xsens[trial]['Left shank']['raw']['Orientation Euler'][:,1] #staticRotation(xsens[trial]['Left shank']['raw']['Orientation Euler'])[:,1] #Rotate around z-axis to match vicon orientation
        shank_xsens[trial]['Angle right'] = xsens[trial]['Right shank']['raw']['Orientation Euler'][:,1] #staticRotation(xsens[trial]['Right shank']['raw']['Orientation Euler'])[:,1]
        
        # Set second-order low-pass butterworth filter;
        # Cut-off frequency: 15Hz
        fc = 15  # Cut-off frequency of the filter
        omega = fc / (xsens[trial]['Sample Frequency (Hz)'] / 2) # Normalize the frequency
        N = 2 # Order of the butterworth filter
        filter_type = 'lowpass' # Type of the filter
        b, a = signal.butter(N, omega, filter_type)
        
        # Shank angle filtered
        shank_xsens[trial]['Angle left'] = signal.filtfilt(b, a, xsens[trial]['Left shank']['raw']['Orientation Euler'][:,1]) # Apply filter
        shank_xsens[trial]['Angle right'] = signal.filtfilt(b, a, xsens[trial]['Right shank']['raw']['Orientation Euler'][:,1]) # Apply filter
        
        # Correct for 90 degree angle definition (so absolute vertical is 0 degrees)
        if np.nanmean(shank_xsens[trial]['Angle left']) < -60:
            shank_xsens[trial]['Angle left'] = shank_xsens[trial]['Angle left'] + 90
        elif np.nanmean(shank_xsens[trial]['Angle left']) > 60:
            shank_xsens[trial]['Angle left'] = shank_xsens[trial]['Angle left'] - 90
        if np.nanmean(shank_xsens[trial]['Angle right']) < -60:
            shank_xsens[trial]['Angle right'] = shank_xsens[trial]['Angle right'] + 90
        elif np.nanmean(shank_xsens[trial]['Angle right']) > 60:
            shank_xsens[trial]['Angle right'] = shank_xsens[trial]['Angle right'] - 90
        
        # Correct gimbal lock left
        pos_pks = signal.find_peaks(shank_xsens[trial]['Angle left'], prominence=3)[0]
        # Check if there is actually gimbal lock
        if np.nanstd(shank_xsens[trial]['Angle left'][pos_pks]) > np.abs(0.20 * np.nanmean(shank_xsens[trial]['Angle left'][pos_pks])): # If true; correct
            pos_pks = pos_pks[shank_xsens[trial]['Angle left'][pos_pks]>(np.nanmean(shank_xsens[trial]['Angle left'][pos_pks]))]
            neg_pks = signal.find_peaks(-shank_xsens[trial]['Angle left'], prominence=1)[0]
            
            for i in range(0,len(pos_pks)):
                after_pks = neg_pks[neg_pks>pos_pks[i]]
                before_pks = neg_pks[neg_pks<pos_pks[i]]
                if np.any(after_pks) == True:
                    after_pks = after_pks[0]
                elif np.any(after_pks) == False and i == len(pos_pks)-1:
                    after_pks = len(shank_xsens[trial]['Angle left'])
                if np.any(before_pks) == True:
                    before_pks = before_pks[-1]
                if np.any(before_pks) == False and i == 0:
                    after_pks = 0
                try:
                    shank_xsens[trial]['Angle left'][before_pks:after_pks] = -1*(shank_xsens[trial]['Angle left'][before_pks:after_pks]) - np.diff([shank_xsens[trial]['Angle left'][before_pks], -1*shank_xsens[trial]['Angle left'][before_pks]])
                except:
                    pass
            # Flip the signal
            shank_xsens[trial]['Angle left'] = -1*shank_xsens[trial]['Angle left']
        
        # Correct gimbal lock right
        pos_pks = signal.find_peaks(shank_xsens[trial]['Angle right'], prominence=3)[0]
        # Check if there is actually gimbal lock
        if np.nanstd(shank_xsens[trial]['Angle right'][pos_pks]) > np.abs(0.20 * np.nanmean(shank_xsens[trial]['Angle right'][pos_pks])): # If true; correct
            pos_pks = pos_pks[shank_xsens[trial]['Angle right'][pos_pks]>(np.nanmean(shank_xsens[trial]['Angle right'][pos_pks]))]
            neg_pks = signal.find_peaks(-shank_xsens[trial]['Angle right'], prominence=1)[0]
            
            for i in range(0,len(pos_pks)):
                after_pks = neg_pks[neg_pks>pos_pks[i]]
                before_pks = neg_pks[neg_pks<pos_pks[i]]
                if np.any(after_pks) == True:
                    after_pks = after_pks[0]
                elif np.any(after_pks) == False and i == len(pos_pks)-1:
                    after_pks = len(shank_xsens[trial]['Angle right'])
                if np.any(before_pks) == True:
                    before_pks = before_pks[-1]
                if np.any(before_pks) == False and i == 0:
                    after_pks = 0
                try:
                    shank_xsens[trial]['Angle right'][before_pks:after_pks] = -1*(shank_xsens[trial]['Angle right'][before_pks:after_pks]) - np.diff([shank_xsens[trial]['Angle right'][before_pks], -1*shank_xsens[trial]['Angle right'][before_pks]])
                except:
                    pass
            # Flip the signal
            shank_xsens[trial]['Angle right'] = -1*shank_xsens[trial]['Angle right']
            
        # Correct for 90 degree angle definition (so absolute vertical is 0 degrees)
        if np.nanmean(shank_xsens[trial]['Angle left']) < -55:
            shank_xsens[trial]['Angle left'] = shank_xsens[trial]['Angle left'] + 90
        elif np.nanmean(shank_xsens[trial]['Angle left']) > 55:
            shank_xsens[trial]['Angle left'] = shank_xsens[trial]['Angle left'] - 90
        if np.nanmean(shank_xsens[trial]['Angle right']) < -55:
            shank_xsens[trial]['Angle right'] = shank_xsens[trial]['Angle right'] + 90
        elif np.nanmean(shank_xsens[trial]['Angle right']) > 55:
            shank_xsens[trial]['Angle right'] = shank_xsens[trial]['Angle right'] - 90
                
            
        # Correct for angle during mid-stance of first 10 mid-stance periods (so foot angle defined comparable to vicon)
        # Shank angular velocity
        shank_xsens[trial]['Angular velocity left'] = xsens[trial]['Left shank']['raw']['Gyroscope'][:,1] #staticRotation(xsens[trial]['Left shank']['raw']['Gyroscope'])[:,1] #Rotate around z-axis to match vicon orientation
        shank_xsens[trial]['Angular velocity right'] = xsens[trial]['Right shank']['raw']['Gyroscope'][:,1] #staticRotation(xsens[trial]['Right shank']['raw']['Gyroscope'])[:,1]
        # Shank angular acceleration
        shank_xsens[trial]['Angular acceleration left'] = np.zeros(len(shank_xsens[trial]['Angular velocity left']))
        shank_xsens[trial]['Angular acceleration left'][1:] = np.diff(shank_xsens[trial]['Angular velocity left'])
        shank_xsens[trial]['Angular acceleration right'] = np.zeros(len(shank_xsens[trial]['Angular velocity right']))
        shank_xsens[trial]['Angular acceleration right'][1:] = np.diff(shank_xsens[trial]['Angular velocity right'])
        # Shank magnitude of linear acceleration
        shank_xsens[trial]['Acceleration left'] = np.sqrt(xsens[trial]['Left shank']['raw']['Accelerometer Earth Frame'][:,0]**2 + xsens[trial]['Left shank']['raw']['Accelerometer Earth Frame'][:,1]**2 + xsens[trial]['Left shank']['raw']['Accelerometer Earth Frame'][:,2]**2)
        shank_xsens[trial]['Acceleration right'] = np.sqrt(xsens[trial]['Right shank']['raw']['Accelerometer Earth Frame'][:,0]**2 + xsens[trial]['Right shank']['raw']['Accelerometer Earth Frame'][:,1]**2 + xsens[trial]['Right shank']['raw']['Accelerometer Earth Frame'][:,2]**2)
        
    return shank_xsens





def shank_kinematics_vicon(vicon, **kwargs):
    shank_vicon = dict()
    sample_frequency = 100 # Hz, default for vicon markerdata
    for key, value in kwargs.items():
        if key == 'fs':
            sample_frequency = value
    for trial in vicon:
        shank_vicon[trial] = dict()
        # Shank angle
        shank_vicon[trial]['Angle left'] = np.rad2deg(np.arctan((vicon[trial]['LKNE'][:,2] - vicon[trial]['LANK'][:,2]) / (vicon[trial]['LKNE'][:,1] - vicon[trial]['LANK'][:,1])))
        shank_vicon[trial]['Angle right'] = np.rad2deg(np.arctan((vicon[trial]['RKNE'][:,2] - vicon[trial]['RANK'][:,2]) / (vicon[trial]['RKNE'][:,1] - vicon[trial]['RANK'][:,1])))
        # Correct for 90 degree angle definition (so absolute vertical is 0 degrees)
        shank_vicon[trial]['Angle left'] = shank_vicon[trial]['Angle left'] + 90 
        shank_vicon[trial]['Angle right'] = shank_vicon[trial]['Angle right'] + 90 
        # Correct gimbal lock
        shank_vicon[trial]['Angle left'][np.argwhere(shank_vicon[trial]['Angle left'] > 100)] = shank_vicon[trial]['Angle left'][np.argwhere(shank_vicon[trial]['Angle left'] > 100)]-180
        shank_vicon[trial]['Angle right'][np.argwhere(shank_vicon[trial]['Angle right'] > 100)] = shank_vicon[trial]['Angle right'][np.argwhere(shank_vicon[trial]['Angle right'] > 100)]-180
        # Shank angular velocity
        shank_vicon[trial]['Angular velocity left'] = np.zeros(len(shank_vicon[trial]['Angle left']))
        shank_vicon[trial]['Angular velocity left'][1:] = np.diff(shank_vicon[trial]['Angle left'])
        shank_vicon[trial]['Angular velocity right'] = np.zeros(len(shank_vicon[trial]['Angle right']))
        shank_vicon[trial]['Angular velocity right'][1:] = np.diff(shank_vicon[trial]['Angle right'])
        # Shank angular acceleration
        shank_vicon[trial]['Angular acceleration left'] = np.zeros(len(shank_vicon[trial]['Angular velocity left']))
        shank_vicon[trial]['Angular acceleration left'][1:] = np.diff(shank_vicon[trial]['Angular velocity left'])
        shank_vicon[trial]['Angular acceleration right'] = np.zeros(len(shank_vicon[trial]['Angular velocity right']))
        shank_vicon[trial]['Angular acceleration right'][1:] = np.diff(shank_vicon[trial]['Angular velocity right'])
        # Shank magnitude of linear acceleration
        shank_vicon[trial]['Acceleration left'] = np.append(np.array([0]), np.diff(np.append(np.array([0]), np.diff( np.sqrt( (vicon[trial]['LKNE'][:,1]-vicon[trial]['LANK'][:,1])**2)))))
        shank_vicon[trial]['Acceleration right'] = np.append(np.array([0]), np.diff(np.append(np.array([0]), np.diff( np.sqrt( (vicon[trial]['RKNE'][:,1]-vicon[trial]['RANK'][:,1])**2)))))
        
    return shank_vicon





# Bland-Altman plot
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.nanmean([data1, data2], axis=0)
    diff      = (data1 - data2)                   # Difference between data1 and data2
    md        = np.nanmean(diff)                  # Mean of the difference
    md_string = 'mean of difference: ' + round(md, 2).astype(str)
    sd        = np.nanstd(diff, axis=0)           # Standard deviation of the difference
    ub_string = '+ 1.96*SD: ' + round(md + 1.96*sd, 2).astype(str)
    lb_string = '- 1.96*SD: ' + round(md - 1.96*sd, 2).astype(str)
    
    # Check for inputname in **kwargs items
    dataType = str()
    unit = str()
    alpha = 0.25
    for key, value in kwargs.items():
        if key == 'dataType':
            dataType = value
        if key == 'unit':
            unit = value
        if key == 'alpha':
            alpha = value
    
    fig = plt.subplots()
    plt.title('Bland-Altman analysis, ' + dataType, fontsize=20)
    
    plt.scatter(mean, diff, edgecolor = 'none', facecolor='black', alpha=alpha, marker = 'o') # SMK green: '#004D43'
    
    plt.axhline(md,           color='gray', linestyle='--', linewidth=3)
    # plt.text(7, md, md_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=3)
    # plt.text(7, md + 1.96*sd, ub_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=3)
    # plt.text(7, md - 1.96*sd, lb_string, fontsize=14) #max(mean)-0.15*np.abs(np.nanmean(mean))
    
    plt.xlabel("Mean (IMU, OMCS) "+ unit, fontsize=14)
    plt.ylabel("Difference (IMU - OMCS) " + unit, fontsize=14) #Difference between measures
    
    plt.xticks(fontsize=14)    
    # set_xticklabels(fontsize=16)
    plt.yticks(fontsize=14)
    plt.ylim((-10, 10))
    plt.legend(fontsize=14)  
        
    return




def calculate_indicative_variables(vicon_gait_events, vicon_spatiotemporals, xsens, foot_vicon, shank_vicon, foot_xsens, shank_xsens):
    for trial in vicon_spatiotemporals:
        # Vicon
        # Left
        foot_vicon[trial]['Angle at TC left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_vicon[trial]['Angle at TC left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        foot_vicon[trial]['Max angular velocity stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_vicon[trial]['Max angular velocity stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        foot_vicon[trial]['Max angular acceleration stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_vicon[trial]['Max angular acceleration stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_vicon[trial]['Max linear acceleration stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        foot_vicon[trial]['Stride length left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        TOL = vicon_gait_events[trial]['Index numbers terminal contact left']
        HSL = vicon_gait_events[trial]['Index numbers initial contact left']
        
        for i in range(len(vicon_spatiotemporals[trial]['Propulsion left'])):
            stance_start = int( HSL[ HSL <= vicon_spatiotemporals[trial]['Propulsion left'][i,0]+10 ][-1] )
            stance_stop = int( TOL[ TOL >= vicon_spatiotemporals[trial]['Propulsion left'][i,1]-10 ][0] )
            foot_vicon[trial]['Angle at TC left'][i] = foot_vicon[trial]['Angle left filt'][stance_stop]
            shank_vicon[trial]['Angle at TC left'][i] = shank_vicon[trial]['Angle left'][stance_stop]
            foot_vicon[trial]['Max angular velocity stance phase left'][i] = np.nanmax(foot_vicon[trial]['Angular velocity left'][ stance_start : stance_stop])
            shank_vicon[trial]['Max angular velocity stance phase left'][i] = np.nanmax(shank_vicon[trial]['Angular velocity left'][ stance_start : stance_stop])
            foot_vicon[trial]['Max angular acceleration stance phase left'][i] = np.nanmax(foot_vicon[trial]['Angular acceleration left'][ stance_start : stance_stop])
            shank_vicon[trial]['Max angular acceleration stance phase left'][i] = np.nanmax(shank_vicon[trial]['Angular acceleration left'][ stance_start : stance_stop])
            shank_vicon[trial]['Max linear acceleration stance phase left'][i] = np.nanmax(shank_vicon[trial]['Acceleration left'][ stance_start : stance_stop])
            try:
                foot_vicon[trial]['Stride length left'][i] = vicon_spatiotemporals[trial]['Stridelength left (mm)'][vicon_spatiotemporals[trial]['Stridelength left (mm)'][:,0]==stance_stop, 2]
            except:
                foot_vicon[trial]['Stride length left'][i] = np.nan
            
        # Right
        foot_vicon[trial]['Angle at TC right'] = np.zeros(len(vicon_spatiotemporals[trial]['Propulsion right']))
        shank_vicon[trial]['Angle at TC right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        foot_vicon[trial]['Max angular velocity stance phase right'] = np.zeros(len(vicon_spatiotemporals[trial]['Propulsion right']))
        shank_vicon[trial]['Max angular velocity stance phase right'] = np.zeros(len(vicon_spatiotemporals[trial]['Propulsion right']))
        foot_vicon[trial]['Max angular acceleration stance phase right'] = np.zeros(len(vicon_spatiotemporals[trial]['Propulsion right']))
        shank_vicon[trial]['Max angular acceleration stance phase right'] = np.zeros(len(vicon_spatiotemporals[trial]['Propulsion right']))
        shank_vicon[trial]['Max linear acceleration stance phase right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        foot_vicon[trial]['Stride length right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        TOR = vicon_gait_events[trial]['Index numbers terminal contact right']
        HSR = vicon_gait_events[trial]['Index numbers initial contact right']
        for i in range(len(vicon_spatiotemporals[trial]['Propulsion right'])):
            
            stance_start = int( HSR[ HSR <= vicon_spatiotemporals[trial]['Propulsion right'][i,0]+10 ][-1] )
            stance_stop = int( TOR[ TOR >= vicon_spatiotemporals[trial]['Propulsion right'][i,1]-10 ][0] )
            foot_vicon[trial]['Angle at TC right'][i] = foot_vicon[trial]['Angle right filt'][stance_stop]
            shank_vicon[trial]['Angle at TC right'][i] = shank_vicon[trial]['Angle right'][stance_stop]
            foot_vicon[trial]['Max angular velocity stance phase right'][i] = np.nanmax(foot_vicon[trial]['Angular velocity right'][ stance_start : stance_stop])
            shank_vicon[trial]['Max angular velocity stance phase right'][i] = np.nanmax(shank_vicon[trial]['Angular velocity right'][ stance_start : stance_stop])
            foot_vicon[trial]['Max angular acceleration stance phase right'][i] = np.nanmax(foot_vicon[trial]['Angular acceleration right'][ stance_start : stance_stop])  
            shank_vicon[trial]['Max angular acceleration stance phase right'][i] = np.nanmax(shank_vicon[trial]['Angular acceleration right'][ stance_start : stance_stop])
            shank_vicon[trial]['Max linear acceleration stance phase right'][i] = np.nanmax(shank_vicon[trial]['Acceleration right'][ stance_start : stance_stop])
            try:
                foot_vicon[trial]['Stride length right'][i] = vicon_spatiotemporals[trial]['Stridelength right (mm)'][vicon_spatiotemporals[trial]['Stridelength right (mm)'][:,0]==stance_stop, 2]
            except:
                foot_vicon[trial]['Stride length right'][i] = np.nan
        
        # Xsens
        # Left
        foot_xsens[trial]['Angle at TC left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_xsens[trial]['Angle at TC left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        foot_xsens[trial]['Max angular velocity stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_xsens[trial]['Max angular velocity stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        foot_xsens[trial]['Max angular acceleration stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_xsens[trial]['Max angular acceleration stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        shank_xsens[trial]['Max linear acceleration stance phase left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        foot_xsens[trial]['Stride length left'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion left'])[0])
        # TOL = vicon_gait_events[trial]['Index numbers terminal contact left']
        TOL = xsens[trial]['Left foot']['Gait Events']['Terminal Contact']
        # HSL = vicon_gait_events[trial]['Index numbers initial contact left']
        HSL = xsens[trial]['Left foot']['Gait Events']['Initial Contact']
        for i in range(len(vicon_spatiotemporals[trial]['Propulsion left'])):
            
            stance_start = int( HSL[ HSL <= vicon_spatiotemporals[trial]['Propulsion left'][i,0]+10 ][-1] )
            stance_stop = int( TOL[ TOL >= vicon_spatiotemporals[trial]['Propulsion left'][i,1]-10 ][0] )
            foot_xsens[trial]['Angle at TC left'][i] = foot_xsens[trial]['Angle left'][stance_stop]
            shank_xsens[trial]['Angle at TC left'][i] = shank_xsens[trial]['Angle left'][stance_stop]
            foot_xsens[trial]['Max angular velocity stance phase left'][i] = np.nanmax(foot_xsens[trial]['Angular velocity left'][ stance_start : stance_stop])
            shank_xsens[trial]['Max angular velocity stance phase left'][i] = np.nanmax(shank_xsens[trial]['Angular velocity left'][ stance_start : stance_stop])
            foot_xsens[trial]['Max angular acceleration stance phase left'][i] = np.nanmax(foot_xsens[trial]['Angular acceleration left'][ stance_start : stance_stop])
            shank_xsens[trial]['Max angular acceleration stance phase left'][i] = np.nanmax(shank_xsens[trial]['Angular acceleration left'][ stance_start : stance_stop])
            shank_xsens[trial]['Max linear acceleration stance phase left'][i] = np.nanmax(shank_xsens[trial]['Acceleration left'][ stance_start : stance_stop])
            try:
                foot_xsens[trial]['Stride length left'][i] = xsens[trial]['Left foot']['derived']['Gait speed per stride (m/s)'][xsens[trial]['Left foot']['derived']['Gait speed per stride (m/s)'][:,0]==stance_stop, 2]
            except:
                foot_xsens[trial]['Stride length left'][i] = np.nan
        
        # Right
        foot_xsens[trial]['Angle at TC right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        shank_xsens[trial]['Angle at TC right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        foot_xsens[trial]['Max angular velocity stance phase right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        shank_xsens[trial]['Max angular velocity stance phase right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        foot_xsens[trial]['Max angular acceleration stance phase right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        shank_xsens[trial]['Max angular acceleration stance phase right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        shank_xsens[trial]['Max linear acceleration stance phase right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        foot_xsens[trial]['Stride length right'] = np.zeros(np.shape(vicon_spatiotemporals[trial]['Propulsion right'])[0])
        # TOR = vicon_gait_events[trial]['Index numbers toe off right']
        TOR = xsens[trial]['Right foot']['Gait Events']['Terminal Contact']
        # HSR = vicon_gait_events[trial]['Index numbers heel strike right']
        HSR = xsens[trial]['Right foot']['Gait Events']['Initial Contact']
        for i in range(len(vicon_spatiotemporals[trial]['Propulsion right'])):
            
            stance_start = int( HSR[ HSR <= vicon_spatiotemporals[trial]['Propulsion right'][i,0]+10 ][-1] )
            stance_stop = int( TOR[ TOR >= vicon_spatiotemporals[trial]['Propulsion right'][i,1]-10 ][0] )
            foot_xsens[trial]['Angle at TC right'][i] = foot_xsens[trial]['Angle right'][stance_stop]
            shank_xsens[trial]['Angle at TC right'][i] = shank_xsens[trial]['Angle right'][stance_stop]
            foot_xsens[trial]['Max angular velocity stance phase right'][i] = np.nanmax(foot_xsens[trial]['Angular velocity right'][ stance_start : stance_stop])
            shank_xsens[trial]['Max angular velocity stance phase right'][i] = np.nanmax(shank_xsens[trial]['Angular velocity right'][ stance_start : stance_stop])
            foot_xsens[trial]['Max angular acceleration stance phase right'][i] = np.nanmax(foot_xsens[trial]['Angular acceleration right'][ stance_start : stance_stop])    
            shank_xsens[trial]['Max angular acceleration stance phase right'][i] = np.nanmax(shank_xsens[trial]['Angular acceleration right'][ stance_start : stance_stop])    
            shank_xsens[trial]['Max linear acceleration stance phase right'][i] = np.nanmax(shank_xsens[trial]['Acceleration right'][ stance_start : stance_stop])
            try:
                foot_xsens[trial]['Stride length right'][i] = xsens[trial]['Right foot']['derived']['Gait speed per stride (m/s)'][xsens[trial]['Right foot']['derived']['Gait speed per stride (m/s)'][:,0]==stance_stop, 2]
            except:
                foot_xsens[trial]['Stride length right'][i] = np.nan
            
    return foot_vicon, shank_vicon, foot_xsens, shank_xsens





def transform_pcc_input_variables_all(vicon_spatiotemporals, foot_vicon, shank_vicon, foot_xsens, shank_xsens):
    # Vicon
    pcc_propulsion = np.array([])
    pcc_propulsion_peak = np.array([])
    pcc_foot_angleTC_vicon = np.array([])
    pcc_shank_angleTC_vicon = np.array([])
    pcc_foot_maxangvel_vicon = np.array([])
    pcc_shank_maxangvel_vicon = np.array([])
    pcc_foot_maxangacc_vicon = np.array([])
    pcc_shank_maxangacc_vicon = np.array([])
    pcc_shank_maxlinacc_vicon = np.array([])
    pcc_foot_stridelength_vicon = np.array([])
    for trial in foot_vicon:
        pcc_propulsion = np.append(pcc_propulsion, vicon_spatiotemporals[trial]['Propulsion left'][:,2])
        pcc_propulsion = np.append(pcc_propulsion, vicon_spatiotemporals[trial]['Propulsion right'][:,2])
        pcc_propulsion_peak = np.append(pcc_propulsion_peak, vicon_spatiotemporals[trial]['Peak propulsion left'][:,1])
        pcc_propulsion_peak = np.append(pcc_propulsion_peak, vicon_spatiotemporals[trial]['Peak propulsion right'][:,1])
        pcc_foot_angleTC_vicon = np.append(pcc_foot_angleTC_vicon, foot_vicon[trial]['Angle at TC left'])
        pcc_foot_angleTC_vicon = np.append(pcc_foot_angleTC_vicon, foot_vicon[trial]['Angle at TC right'])
        pcc_shank_angleTC_vicon = np.append(pcc_shank_angleTC_vicon, shank_vicon[trial]['Angle at TC left'])
        pcc_shank_angleTC_vicon = np.append(pcc_shank_angleTC_vicon, shank_vicon[trial]['Angle at TC right'])
        pcc_foot_maxangvel_vicon = np.append(pcc_foot_maxangvel_vicon, foot_vicon[trial]['Max angular velocity stance phase left'])
        pcc_foot_maxangvel_vicon = np.append(pcc_foot_maxangvel_vicon, foot_vicon[trial]['Max angular velocity stance phase right'])
        pcc_shank_maxangvel_vicon = np.append(pcc_shank_maxangvel_vicon, shank_vicon[trial]['Max angular velocity stance phase left'])
        pcc_shank_maxangvel_vicon = np.append(pcc_shank_maxangvel_vicon, shank_vicon[trial]['Max angular velocity stance phase right'])
        pcc_foot_maxangacc_vicon = np.append(pcc_foot_maxangacc_vicon, foot_vicon[trial]['Max angular acceleration stance phase left'])
        pcc_foot_maxangacc_vicon = np.append(pcc_foot_maxangacc_vicon, foot_vicon[trial]['Max angular acceleration stance phase right'])
        pcc_shank_maxangacc_vicon = np.append(pcc_shank_maxangacc_vicon, shank_vicon[trial]['Max angular acceleration stance phase left'])
        pcc_shank_maxangacc_vicon = np.append(pcc_shank_maxangacc_vicon, shank_vicon[trial]['Max angular acceleration stance phase right'])
        pcc_shank_maxlinacc_vicon = np.append(pcc_shank_maxlinacc_vicon, shank_vicon[trial]['Max linear acceleration stance phase left'])
        pcc_shank_maxlinacc_vicon = np.append(pcc_shank_maxlinacc_vicon, shank_vicon[trial]['Max linear acceleration stance phase right'])
        pcc_foot_stridelength_vicon = np.append(pcc_foot_stridelength_vicon, foot_vicon[trial]['Stride length left'])
        pcc_foot_stridelength_vicon = np.append(pcc_foot_stridelength_vicon, foot_vicon[trial]['Stride length right'])
        
    # Xsens
    pcc_foot_angleTC_xsens = np.array([])
    pcc_shank_angleTC_xsens = np.array([])
    pcc_foot_maxangvel_xsens = np.array([])
    pcc_shank_maxangvel_xsens = np.array([])
    pcc_foot_maxangacc_xsens = np.array([])
    pcc_shank_maxangacc_xsens = np.array([])
    pcc_shank_maxlinacc_xsens = np.array([])
    pcc_foot_stridelength_xsens = np.array([])
    for trial in foot_xsens:
        pcc_foot_angleTC_xsens = np.append(pcc_foot_angleTC_xsens, foot_xsens[trial]['Angle at TC left'])
        pcc_foot_angleTC_xsens = np.append(pcc_foot_angleTC_xsens, foot_xsens[trial]['Angle at TC right'])
        pcc_shank_angleTC_xsens = np.append(pcc_shank_angleTC_xsens, shank_xsens[trial]['Angle at TC left'])
        pcc_shank_angleTC_xsens = np.append(pcc_shank_angleTC_xsens, shank_xsens[trial]['Angle at TC right'])
        pcc_foot_maxangvel_xsens = np.append(pcc_foot_maxangvel_xsens, foot_xsens[trial]['Max angular velocity stance phase left'])
        pcc_foot_maxangvel_xsens = np.append(pcc_foot_maxangvel_xsens, foot_xsens[trial]['Max angular velocity stance phase right'])
        pcc_shank_maxangvel_xsens = np.append(pcc_shank_maxangvel_xsens, shank_xsens[trial]['Max angular velocity stance phase left'])
        pcc_shank_maxangvel_xsens = np.append(pcc_shank_maxangvel_xsens, shank_xsens[trial]['Max angular velocity stance phase right'])
        pcc_foot_maxangacc_xsens = np.append(pcc_foot_maxangacc_xsens, foot_xsens[trial]['Max angular acceleration stance phase left'])
        pcc_foot_maxangacc_xsens = np.append(pcc_foot_maxangacc_xsens, foot_xsens[trial]['Max angular acceleration stance phase right'])
        pcc_shank_maxangacc_xsens = np.append(pcc_shank_maxangacc_xsens, shank_xsens[trial]['Max angular acceleration stance phase left'])
        pcc_shank_maxangacc_xsens = np.append(pcc_shank_maxangacc_xsens, shank_xsens[trial]['Max angular acceleration stance phase right'])
        pcc_shank_maxlinacc_xsens = np.append(pcc_shank_maxlinacc_xsens, shank_xsens[trial]['Max linear acceleration stance phase left'])
        pcc_shank_maxlinacc_xsens = np.append(pcc_shank_maxlinacc_xsens, shank_xsens[trial]['Max linear acceleration stance phase right'])
        pcc_foot_stridelength_xsens = np.append(pcc_foot_stridelength_xsens, foot_xsens[trial]['Stride length left'])
        pcc_foot_stridelength_xsens = np.append(pcc_foot_stridelength_xsens, foot_xsens[trial]['Stride length right'])
        
    
    return pcc_propulsion[~np.isnan(pcc_propulsion)], pcc_propulsion_peak, pcc_foot_angleTC_vicon[~np.isnan(pcc_propulsion)], pcc_shank_angleTC_vicon[~np.isnan(pcc_propulsion)], pcc_foot_maxangvel_vicon[~np.isnan(pcc_propulsion)], pcc_shank_maxangvel_vicon[~np.isnan(pcc_propulsion)], pcc_foot_maxangacc_vicon[~np.isnan(pcc_propulsion)], pcc_shank_maxangacc_vicon[~np.isnan(pcc_propulsion)], pcc_shank_maxlinacc_vicon[~np.isnan(pcc_propulsion)], pcc_foot_stridelength_vicon[~np.isnan(pcc_propulsion)], pcc_foot_angleTC_xsens[~np.isnan(pcc_propulsion)], pcc_shank_angleTC_xsens[~np.isnan(pcc_propulsion)], pcc_foot_maxangvel_xsens[~np.isnan(pcc_propulsion)], pcc_shank_maxangvel_xsens[~np.isnan(pcc_propulsion)], pcc_foot_maxangacc_xsens[~np.isnan(pcc_propulsion)], pcc_shank_maxangacc_xsens[~np.isnan(pcc_propulsion)], pcc_shank_maxlinacc_xsens[~np.isnan(pcc_propulsion)], pcc_foot_stridelength_xsens[~np.isnan(pcc_propulsion)]
    




def transform_pcc_input_variables_perPerson(triallist, vicon_spatiotemporals, foot_vicon, shank_vicon, foot_xsens, shank_xsens):
    pcc_per_person = dict()
    pcc_per_person['propulsion'] = dict()
    pcc_per_person['propulsion peak'] = dict()
    pcc_per_person['propulsion stridelength vicon'] = dict()
    pcc_per_person['propulsion stridelength xsens'] = dict()
    pcc_per_person['vicon'] = dict()
    pcc_per_person['xsens'] = dict()
    pcc_per_person['vicon']['foot angle TC'] = dict()
    pcc_per_person['xsens']['foot angle TC'] = dict()
    pcc_per_person['vicon']['foot angular velocity'] = dict()
    pcc_per_person['xsens']['foot angular velocity'] = dict()
    pcc_per_person['vicon']['foot angular acceleration'] = dict()
    pcc_per_person['xsens']['foot angular acceleration'] = dict()
    pcc_per_person['vicon']['stride length'] = dict()
    pcc_per_person['xsens']['stride length'] = dict()

    pcc_per_person['vicon']['shank angle TC'] = dict()
    pcc_per_person['xsens']['shank angle TC'] = dict()
    pcc_per_person['vicon']['shank angular velocity'] = dict()
    pcc_per_person['xsens']['shank angular velocity'] = dict()
    pcc_per_person['vicon']['shank angular acceleration'] = dict()
    pcc_per_person['xsens']['shank angular acceleration'] = dict()
    pcc_per_person['vicon']['shank linear acceleration'] = dict()
    pcc_per_person['xsens']['shank linear acceleration'] = dict()

    for person in triallist:
        pcc_per_person['propulsion'][person] = np.array([])
        pcc_per_person['propulsion peak'][person] = np.array([])
        pcc_per_person['propulsion stridelength vicon'][person] = np.array([])
        pcc_per_person['propulsion stridelength xsens'][person] = np.array([])
        pcc_per_person['vicon']['foot angle TC'][person] = np.array([])
        pcc_per_person['xsens']['foot angle TC'][person] = np.array([])
        pcc_per_person['vicon']['foot angular velocity'][person] = np.array([])
        pcc_per_person['xsens']['foot angular velocity'][person] = np.array([])
        pcc_per_person['vicon']['foot angular acceleration'][person] = np.array([])
        pcc_per_person['xsens']['foot angular acceleration'][person] = np.array([])
        pcc_per_person['vicon']['stride length'][person] = np.array([])
        pcc_per_person['xsens']['stride length'][person] = np.array([])

        pcc_per_person['vicon']['shank angle TC'][person] = np.array([])
        pcc_per_person['xsens']['shank angle TC'][person] = np.array([])
        pcc_per_person['vicon']['shank angular velocity'][person] = np.array([])
        pcc_per_person['xsens']['shank angular velocity'][person] = np.array([])
        pcc_per_person['vicon']['shank angular acceleration'][person] = np.array([])
        pcc_per_person['xsens']['shank angular acceleration'][person] = np.array([])
        pcc_per_person['vicon']['shank linear acceleration'][person] = np.array([])
        pcc_per_person['xsens']['shank linear acceleration'][person] = np.array([])
        for trial in triallist[person]:
            try:
                pcc_per_person['propulsion'][person] = np.append(pcc_per_person['propulsion'][person], np.append(vicon_spatiotemporals[trial]['Propulsion left'][:,2], vicon_spatiotemporals[trial]['Propulsion right'][:,2]) )
                pcc_per_person['propulsion peak'][person] = np.append(pcc_per_person['propulsion peak'][person], np.append(vicon_spatiotemporals[trial]['Peak propulsion left'][:,1], vicon_spatiotemporals[trial]['Peak propulsion right'][:,1]) )
                pcc_per_person['propulsion stridelength vicon'][person] = np.append(pcc_per_person['propulsion stridelength vicon'][person], np.append(vicon_spatiotemporals[trial]['Propulsion left'][~np.isnan(foot_vicon[trial]['Stride length left']),2], vicon_spatiotemporals[trial]['Propulsion right'][~np.isnan(foot_vicon[trial]['Stride length right']),2]) )
                pcc_per_person['propulsion stridelength xsens'][person] = np.append(pcc_per_person['propulsion stridelength xsens'][person], np.append(vicon_spatiotemporals[trial]['Propulsion left'][~np.isnan(foot_xsens[trial]['Stride length left']),2], vicon_spatiotemporals[trial]['Propulsion right'][~np.isnan(foot_xsens[trial]['Stride length right']),2]) )
                pcc_per_person['vicon']['foot angle TC'][person] = np.append(pcc_per_person['vicon']['foot angle TC'][person], np.append(foot_vicon[trial]['Angle at TC left'], foot_vicon[trial]['Angle at TC right']))
                pcc_per_person['xsens']['foot angle TC'][person] = np.append(pcc_per_person['xsens']['foot angle TC'][person], np.append(foot_xsens[trial]['Angle at TC left'], foot_xsens[trial]['Angle at TC right']))
                pcc_per_person['vicon']['foot angular velocity'][person] = np.append(pcc_per_person['vicon']['foot angular velocity'][person], np.append(foot_vicon[trial]['Max angular velocity stance phase left'], foot_vicon[trial]['Max angular velocity stance phase right']))
                pcc_per_person['xsens']['foot angular velocity'][person] = np.append(pcc_per_person['xsens']['foot angular velocity'][person], np.append(foot_xsens[trial]['Max angular velocity stance phase left'], foot_xsens[trial]['Max angular velocity stance phase right']))
                pcc_per_person['vicon']['foot angular acceleration'][person] = np.append(pcc_per_person['vicon']['foot angular acceleration'][person], np.append(shank_vicon[trial]['Max angular acceleration stance phase left'], shank_vicon[trial]['Max angular acceleration stance phase right']))
                pcc_per_person['xsens']['foot angular acceleration'][person] = np.append(pcc_per_person['xsens']['foot angular acceleration'][person], np.append(foot_xsens[trial]['Max angular acceleration stance phase left'], foot_xsens[trial]['Max angular acceleration stance phase right']))
                pcc_per_person['vicon']['stride length'][person] = np.append(pcc_per_person['vicon']['stride length'][person], np.append(foot_vicon[trial]['Stride length left'][~np.isnan(foot_vicon[trial]['Stride length left'])], foot_vicon[trial]['Stride length right'][~np.isnan(foot_vicon[trial]['Stride length right'])]))
                pcc_per_person['xsens']['stride length'][person] = np.append(pcc_per_person['xsens']['stride length'][person], np.append(foot_xsens[trial]['Stride length left'][~np.isnan(foot_xsens[trial]['Stride length left'])], foot_xsens[trial]['Stride length right'][~np.isnan(foot_xsens[trial]['Stride length right'])]))
                pcc_per_person['vicon']['shank angle TC'][person] = np.append(pcc_per_person['vicon']['shank angle TC'][person], np.append(shank_vicon[trial]['Angle at TC left'], shank_vicon[trial]['Angle at TC right']))
                pcc_per_person['xsens']['shank angle TC'][person] = np.append(pcc_per_person['xsens']['shank angle TC'][person], np.append(shank_xsens[trial]['Angle at TC left'], shank_xsens[trial]['Angle at TC right']))
                pcc_per_person['vicon']['shank angular velocity'][person] = np.append(pcc_per_person['vicon']['shank angular velocity'][person], np.append(shank_vicon[trial]['Max angular velocity stance phase left'], shank_vicon[trial]['Max angular velocity stance phase right']))
                pcc_per_person['xsens']['shank angular velocity'][person] = np.append(pcc_per_person['xsens']['shank angular velocity'][person], np.append(shank_xsens[trial]['Max angular velocity stance phase left'], shank_xsens[trial]['Max angular velocity stance phase right']))
                pcc_per_person['vicon']['shank angular acceleration'][person] = np.append(pcc_per_person['vicon']['shank angular acceleration'][person], np.append(shank_vicon[trial]['Max angular acceleration stance phase left'], shank_vicon[trial]['Max angular acceleration stance phase right']))
                pcc_per_person['xsens']['shank angular acceleration'][person] = np.append(pcc_per_person['xsens']['shank angular acceleration'][person], np.append(shank_xsens[trial]['Max angular acceleration stance phase left'], shank_xsens[trial]['Max angular acceleration stance phase right']))
                pcc_per_person['vicon']['shank linear acceleration'][person] = np.append(pcc_per_person['vicon']['shank linear acceleration'][person], np.append(shank_vicon[trial]['Max linear acceleration stance phase left'], shank_vicon[trial]['Max linear acceleration stance phase right']))
                pcc_per_person['xsens']['shank linear acceleration'][person] = np.append(pcc_per_person['xsens']['shank linear acceleration'][person], np.append(shank_xsens[trial]['Max linear acceleration stance phase left'], shank_xsens[trial]['Max linear acceleration stance phase right']))
            except:
                pass
    
    idx_true = dict()
    for key in pcc_per_person:
        if key == 'propulsion':
            for person in pcc_per_person[key]:
                idx_true[person] = np.argwhere(~np.isnan(pcc_per_person[key][person])).flatten()
            
    for key in pcc_per_person:
        if key == 'propulsion':
            for person in pcc_per_person[key]:
                pcc_per_person[key][person] = pcc_per_person[key][person][idx_true[person]]
        if key == 'vicon' or key=='xsens':
            for param in pcc_per_person[key]:
                if param != 'stride length': 
                    for person in pcc_per_person[key][param]:
                        pcc_per_person[key][param][person] = pcc_per_person[key][param][person][idx_true[person]]
    
    idx_true_sl_v = dict()
    for person in pcc_per_person['propulsion stridelength vicon']:
        idx_true_sl_v[person] = np.argwhere(~np.isnan(pcc_per_person['propulsion stridelength vicon'][person])).flatten()
        pcc_per_person['propulsion stridelength vicon'][person] = pcc_per_person['propulsion stridelength vicon'][person][idx_true_sl_v[person]]
        pcc_per_person['vicon']['stride length'][person] = pcc_per_person['vicon']['stride length'][person][idx_true_sl_v[person]]
    idx_true_sl_x = dict()
    for person in pcc_per_person['propulsion stridelength xsens']:
        idx_true_sl_x[person] = np.argwhere(~np.isnan(pcc_per_person['propulsion stridelength xsens'][person])).flatten()
        pcc_per_person['propulsion stridelength xsens'][person] = pcc_per_person['propulsion stridelength xsens'][person][idx_true_sl_x[person]]
        pcc_per_person['xsens']['stride length'][person] = pcc_per_person['xsens']['stride length'][person][idx_true_sl_x[person]]
        
    return pcc_per_person