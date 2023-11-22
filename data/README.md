The data folder has the following structure:

- 1019_pp## (folder for participant based on study ID)
	- Vicon (folder with vicon, gold standard, data)
		*.c3d file for each trial. Trialnames have te following configuration:
			1019_MR**studyIDnumber**_**trialtype**.c3d*
		*trialtypes: 
			1Reg: First trial regular walking
			FBIC: Feedback on initial contact angle
			FBPO: Feedback on push off force
			2FB: Feedback on both initial contact angle and push off force
			2Reg: Second trial regular walking
	- Xsens (folder with IMU sensordata)
		- exported###
			*.txt file per sensor, filenames have the following configuration:
				XsensAwindaMasterstationID_trialnumber_XsensSensorID.txt*
			*sensorspec.json file with specifications of which sensor (XsensSensorID) was attached to which body segment*
- ...

The vicon data is saved as .c3d files
The xsens sensordata is exported with MTManager to .txt files in the different "exported###" folders for each trial (see "MT Manager export settings.PNG" figure for the export settings).
