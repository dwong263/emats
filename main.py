import sys
import os
import datetime
import time
import glob

from PyQt5 import QtCore, QtGui, QtWidgets, uic

import numpy as np
import cv2

from collections import defaultdict

qtCreatorFile = "ui/emats.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class Mouse:
	def __init__(self):

		self.id = ''

		self.video_infile = ''
		self.video_outfile = ''
		self.video_spf = 0

		# measured position
		self.center = []

		# tracking counters
		self.CA_entries = 0
		self.CA_frames = 0
		self.CA_latency = 0

		self.OA_entries = 0
		self.OA_frames = 0
		self.OA_latency = 0

		self.center_entries = 0
		self.center_frames = 0
		self.center_latency = 0

		self.CA_isActive = False
		self.OA_isActive = False
		self.center_isActive = False

		self.CA_firstTime = True
		self.OA_firstTime = True
		self.center_firstTime = True

	def getPathLength(self, pixel_to_cm):
		path_length = 0
		path = self.center

		for i in range(1,len(path)):
			path_length = path_length + np.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2)*pixel_to_cm

		return path_length

	def getClosestPoint(self, apparatus_points):
		distances = []
		indices = []
		for (i, point) in enumerate(apparatus_points):
			distances.append(np.sqrt((point[0]-self.center[-1][0])**2+(point[1]-self.center[-1][1])**2))
			indices.append(i)
		sorted_by_distance = sorted(zip(distances, indices), key=lambda x: x[0], reverse=True)
		return sorted_by_distance[0][1]

	def setActive(self, region, frame_count):
		if region == 'closed arm':
			
			self.OA_isActive = False
			self.center_isActive = False

			if not(self.CA_isActive):
				# switching from inactive to active
				self.CA_entries = self.CA_entries + 1	# increment number of times this region has been visited
				self.CA_isActive = True					# indicate this region is active
				if self.CA_firstTime:
					self.CA_latency = frame_count-2
					self.CA_firstTime = False			# this region has been entered, no longer the first time

			# always increment number of active frames
			self.CA_frames = self.CA_frames + 1

		elif region == 'open arm':

			self.CA_isActive = False
			self.center_isActive = False

			if not(self.OA_isActive):
				# switching from inactive to active
				self.OA_entries = self.OA_entries + 1	# increment number of times this region has been visited
				self.OA_isActive = True					# indicate this region is active
				if self.OA_firstTime:
					self.OA_latency = frame_count-2
					self.OA_firstTime = False			# this region has been entered, no longer the first time

			# always increment number of active frames
			self.OA_frames = self.OA_frames + 1

		elif region == 'center':

			self.CA_isActive = False
			self.OA_isActive = False

			if not(self.center_isActive):
				# switching from inactive to active
				self.center_entries = self.center_entries + 1	# increment number of times this region has been visited
				self.center_isActive = False					# indicate this region is active
				if self.center_firstTime: 
					self.center_latency = frame_count-2
					self.center_firstTime = False				# this region has been entered, no longer the first time

			# always increment number of active frames
			self.center_frames = self.center_frames + 1

class videoAnalysis(QtCore.QObject):

	postToConsole = QtCore.pyqtSignal(str)
	postToError = QtCore.pyqtSignal(str)
	setcurrentPoolProgressBarMax = QtCore.pyqtSignal(int)
	updatecurrentPoolProgressBar = QtCore.pyqtSignal()
	outputResults = QtCore.pyqtSignal(object)
	finished = QtCore.pyqtSignal(int)

	def __init__(self, thread_num, mouse, BLUR_KERNEL_HEIGHT, BLUR_KERNEL_WIDTH, DILATE_KERNEL_HEIGHT, DILATE_KERNEL_WIDTH, CLOSED_ARM_LOCATION):
		
		QtCore.QObject.__init__(self)

		self.thread_num = thread_num

		self.BLUR_KERNEL_HEIGHT = BLUR_KERNEL_HEIGHT
		self.BLUR_KERNEL_WIDTH = BLUR_KERNEL_WIDTH

		self.DILATE_KERNEL_HEIGHT = DILATE_KERNEL_HEIGHT
		self.DILATE_KERNEL_WIDTH = DILATE_KERNEL_WIDTH

		self.CLOSED_ARM_LOCATION = CLOSED_ARM_LOCATION

		self.mouse = mouse

	def process(self):

		try:

			self.postToConsole.emit('Processing ' + self.mouse.id + ' ...')
			print 'Processing ' + self.mouse.id + ' ...'

			frame_count = 0

			# open video
			cap = cv2.VideoCapture(str(self.mouse.video_infile))

			# get video information
			fps = cap.get(cv2.CAP_PROP_FPS)
			self.mouse.video_spf = (1/fps)

			frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
			self.setcurrentPoolProgressBarMax.emit(frames_total)

			# setup output video format
			fourcc = cv2.VideoWriter_fourcc(*'XVID')

			while(cap.isOpened()):
				ret, frame = cap.read()
				frame_count = frame_count + 1

				self.updatecurrentPoolProgressBar.emit()

				if ret == True:

					output = frame.copy()

					# create output file (do once)
					if frame_count == 1:
						rows, cols, color = frame.shape
						out = cv2.VideoWriter(str(self.mouse.video_outfile), fourcc, fps, (cols, rows))

					# convert current frame to grayscale
					frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

					# blur current frame
					frame_blur = cv2.blur(frame_gray, (self.BLUR_KERNEL_WIDTH, self.BLUR_KERNEL_HEIGHT))

					# threshold blurred frame
					ret_binary, frame_binary = cv2.threshold(frame_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

					if frame_count == 1:
						# floodfill the background to segment apparatus
						frame_floodfill = frame_binary.copy()
						h, w = frame_binary.shape[:2]
						mask = np.zeros((h+2, w+2), np.uint8)
						cv2.floodFill(frame_floodfill,mask,(0,0),255)

						# generate a mask from the segmented apparatus
						EPM_mask = cv2.bitwise_not(cv2.bitwise_not(frame_floodfill) | frame_binary)

						# improve the mask by creating an EPM template to find the center of the apparatus
						#	--> helps a lot if mouse is the apparatus already in the first frame
						EPM_template = np.ones((h, w), np.uint8)
						for i in range(h/2-int(0.4*h),h/2+int(0.4*h)+1):
							for j in range(w/2-int(0.015*w), w/2+int(0.015*w)+1):
								EPM_template[i][j] = 255
						for j in range(w/2-int(0.4*h),w/2+int(0.4*h)+1):
							for i in range(h/2-int(0.015*w), h/2+int(0.015*w)+1):
								EPM_template[i][j] = 255
						EPM_template = EPM_template[h/2-int(0.4*h):h/2+int(0.4*h), w/2-int(0.4*h):w/2+int(0.4*h)]
						EPM_template = cv2.bitwise_not(EPM_template)

						EPM_convolved = cv2.matchTemplate(EPM_mask, EPM_template, cv2.TM_CCOEFF_NORMED)
						EPM_convolved_shifted = np.zeros((h,w), np.uint8)
						h2, w2 = EPM_convolved.shape
						for i in range(h/2-int(h2/2), h/2+int(h2/2)+1):
							for j in range(w/2-int(w2/2), w/2+int(w2/2)+1):
								if EPM_convolved[i-h/2-int(h2/2)][j-w/2-int(w2/2)] > 0.15:
									EPM_convolved_shifted[i][j] = 255
								else:
									EPM_convolved_shifted[i][j] = 0

						# apply template to improve mask
						EPM_mask = cv2.bitwise_not(cv2.bitwise_not(EPM_mask) | EPM_convolved_shifted)
						EPM_mask = cv2.dilate(EPM_mask, np.ones((self.DILATE_KERNEL_WIDTH, self.DILATE_KERNEL_HEIGHT),np.uint8), iterations=1)

						# find the contour of the segmented apparatus
						EPM_contour_image = EPM_mask.copy()
						image, contours, hierarchy = cv2.findContours(EPM_contour_image,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

						areas = []
						for cnt in contours:
							areas.append(cv2.contourArea(cnt))

						sorted_by_area = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
						EPM_contour = sorted_by_area[1][1]

						# find the ends of the apparatus arms
						rect = cv2.minAreaRect(EPM_contour)
						box = cv2.boxPoints(rect)
						EPM_extreme_points = np.int0(box)					

						angles = []
						for point in EPM_extreme_points:
							angle = np.arctan2(point[1]-int(h/2), point[0]-int(w/2))
							angles.append(angle-np.pi/2)
						sorted_by_angle = sorted(zip(angles, EPM_extreme_points), key=lambda x: x[0], reverse=False)

						# find the center region
						EPM_hull = cv2.convexHull(EPM_contour,returnPoints=False)
						EPM_defects = cv2.convexityDefects(EPM_contour, EPM_hull)

						potential_center_points = []
						min_distances = []
						for i in range(EPM_defects.shape[0]):
							s, e, f, d = EPM_defects[i,0]
							far = cnt[f][0]
							potential_center_points.append(far)

							distances = []
							for point in EPM_extreme_points:
								distances.append(np.sqrt((point[0]-far[0])**2+(point[1]-far[1])**2))

							min_distances.append(np.amin(distances))

						sorted_by_distance = sorted(zip(min_distances, potential_center_points), key=lambda x: x[0], reverse=True)
						EPM_center_points = []
						EPM_center_points.append(sorted_by_distance[0][1])
						EPM_center_points.append(sorted_by_distance[1][1])
						EPM_center_points.append(sorted_by_distance[2][1])
						EPM_center_points.append(sorted_by_distance[3][1])
						
						EPM_center_points = np.int0(EPM_center_points)
						EPM_center_points = cv2.minAreaRect(EPM_center_points)
						EPM_center_points = cv2.boxPoints(EPM_center_points)
						EPM_center_points = np.int0(EPM_center_points)

					# apply apparatus mask to allow segmentation of the mouse
					frame_masked = cv2.bitwise_or(frame_binary, EPM_mask)

					if frame_count != 1:
						# draw apparatus
						output = cv2.drawContours(output, EPM_contour, -1, (0, 255, 0), 2)

						# label apparatus
						labels_NS = ['CA', 'OA', 'CA', 'OA']
						labels_WE = ['OA', 'CA', 'OA', 'CA']

						if self.CLOSED_ARM_LOCATION == 'N' or self.CLOSED_ARM_LOCATION == 'S':
							for (i, point) in enumerate(EPM_extreme_points):
								output = cv2.putText(output, labels_NS[i], (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
						elif self.CLOSED_ARM_LOCATION == 'W' or self.CLOSED_ARM_LOCATION == 'E':
							for (i, point) in enumerate(EPM_extreme_points):
								output = cv2.putText(output, labels_WE[i], (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)

						output = cv2.drawContours(output, [EPM_center_points], 0, (0,0,255),2)

						# segment mouse
						mouse_contour_image = frame_masked.copy()
						mouse_contour_image = cv2.dilate(mouse_contour_image, np.ones((self.DILATE_KERNEL_WIDTH, self.DILATE_KERNEL_HEIGHT),np.uint8), iterations=1)
						image, contours, hierarchy = cv2.findContours(mouse_contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

						areas = []
						for cnt in contours:
							areas.append(cv2.contourArea(cnt))

						# if largest contour can be found (i.e. mouse is present), segment mouse and update center position
						if len(areas) > 1:

							# find largest contour (which is usually the mouse)
							sorted_by_area = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
							mouse_contour = sorted_by_area[1][1]

							# calculate mouse centroid
							M = cv2.moments(mouse_contour)
							mouse_x = int(M['m10']/M['m00'])
							mouse_y = int(M['m01']/M['m00'])

							self.mouse.center.append((mouse_x, mouse_y))

							# draw mouse position
							output = cv2.drawContours(output, mouse_contour, -1, (255,0,0), 2)
							output = cv2.circle(output, self.mouse.center[-1], 4, (255,0,0), -1)

							# now that we have position, find which area the mouse is in
							inCenter = cv2.pointPolygonTest(EPM_center_points,self.mouse.center[-1],True)
							if inCenter >= 0:
								# mouse is in the center
								output = cv2.putText(output, "In Center", (mouse_x+15, mouse_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
								self.mouse.setActive('center', frame_count)
							else:
								# mouse is in one of the arms
								if self.CLOSED_ARM_LOCATION == 'N' or self.CLOSED_ARM_LOCATION == 'S':
									if self.mouse.getClosestPoint(EPM_extreme_points) == 0 or self.mouse.getClosestPoint(EPM_extreme_points) == 2:
										# mouse is in the closed arm
										output = cv2.putText(output, "In Closed Arm", (mouse_x+15, mouse_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
										self.mouse.setActive('closed arm', frame_count)
									else:
										# mouse is in the open arm
										output = cv2.putText(output, "In Open Arm", (mouse_x+15, mouse_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
										self.mouse.setActive('open arm', frame_count)
								elif self.CLOSED_ARM_LOCATION == 'W' or self.CLOSED_ARM_LOCATION == 'E':
									if self.mouse.getClosestPoint(EPM_extreme_points) == 0 or self.mouse.getClosestPoint(EPM_extreme_points) == 2:
										# mouse is in the open arm
										output = cv2.putText(output, "In Open Arm", (mouse_x+15, mouse_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
										self.mouse.setActive('open arm', frame_count)
									else:
										# mouse is in the closed arm
										output = cv2.putText(output, "In Closed Arm", (mouse_x+15, mouse_y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 1)
										self.mouse.setActive('closed arm', frame_count)

					# cv2.imshow(self.mouse.id, output)
					out.write(output)

					# cv2.waitKey(1)

				else:
					break

			# SEND SAVE DATA SIGNAL
			self.outputResults.emit(self.mouse)

			cap.release()
			out.release()
			# cv2.destroyAllWindows()

			self.postToConsole.emit('Completed ' + self.mouse.id + '!')
			self.finished.emit(self.thread_num)
		
		except:
			e = str(sys.exc_info()[0])
			self.postToConsole.emit('Error when processing ' + self.mouse.id + '!')
			self.postToConsole.emit(e)
			self.postToError.emit(str(self.mouse.id))
			self.finished.emit(self.thread_num)

class MyApp(QtWidgets.QWidget, Ui_MainWindow):

	def __init__(self):

		super(self.__class__, self).__init__()
		self.setupUi(self)

		self.setWorkingDirectoryButton.clicked.connect(self.setWorkingDirectory)
		self.setWorkingDirectoryButton.setEnabled(True)

		self.loadVideosButton.clicked.connect(self.loadVideos)
		self.loadVideosButton.setEnabled(False)

		self.confirmMiceButton.clicked.connect(self.confirmMice)
		self.confirmMiceButton.setEnabled(False)

		self.confirmParametersButton.clicked.connect(self.confirmParameters)
		self.confirmParametersButton.setEnabled(False)

		self.runVideoAnalysisButton.clicked.connect(self.runVideoAnalysis)
		self.runVideoAnalysisButton.setEnabled(False)

	def setWorkingDirectory(self):

		self.workingDirectory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Set Working Directory', os.path.expanduser('~')))
		if self.workingDirectory == '':
			self.workingDirectory = os.path.expanduser('~')

		self.consoleOutputText.append('Working directory set to:')
		self.consoleOutputText.append(' >> ' + str(self.workingDirectory))
		self.consoleOutputText.append('')

		self.setWorkingDirectoryButton.setEnabled(False)
		self.loadVideosButton.setEnabled(True)

	def loadVideos(self):

		self.videoList = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Video File', self.workingDirectory, 'Video Files (*.avi)')[0]
		
		if len(self.videoList) > 0:
			
			self.consoleOutputText.append('The following videos were loaded:')
			for (i, video) in enumerate(self.videoList):
				self.videoList[i] = str(video)
				self.consoleOutputText.append(' >> ' + str(video))
			self.consoleOutputText.append('')

			self.loadVideosButton.setEnabled(False)
			self.confirmMiceButton.setEnabled(True)

		else:
			
			self.consoleOutputText.append('No videos selected ... try again.')
			self.consoleOutputText.append('')
			self.loadVideosButton.setEnabled(True)

	def confirmMice(self):

		self.mouseList = self.mouseIDsTextEdit.toPlainText().split('\n')
		for (i, mouse) in enumerate(self.mouseList):
			self.mouseList[i] = str(mouse)

		if len(self.mouseList) != len(self.videoList):
			self.consoleOutputText.append('Number of mice IDs do not equal number of videos ... specify the IDs again.')
			self.consoleOutputText.append('')
		else:
			self.consoleOutputText.append('The following mice were selected:')
			self.consoleOutputText.append(' >> ' + str(self.mouseList))
			self.consoleOutputText.append('')
			self.confirmMiceButton.setEnabled(False)
			self.confirmParametersButton.setEnabled(True)

	def confirmParameters(self):

		self.BLUR_KERNEL_WIDTH = int(self.BLUR_KERNEL_WIDTH_lineEdit.text())
		self.BLUR_KERNEL_HEIGHT = int(self.BLUR_KERNEL_HEIGHT_lineEdit.text())

		self.DILATE_KERNEL_WIDTH = int(self.DILATE_KERNEL_WIDTH_lineEdit.text())
		self.DILATE_KERNEL_HEIGHT = int(self.DILATE_KERNEL_HEIGHT_lineEdit.text())

		if self.northRadioButton.isChecked():
			self.CLOSED_ARM_LOCATION = 'N'
		elif self.southRadioButton.isChecked():
			self.CLOSED_ARM_LOCATION = 'S'
		elif self.westRadioButton.isChecked():
			self.CLOSED_ARM_LOCATION = 'W'
		elif self.eastRadioButton.isChecked():
			self.CLOSED_ARM_LOCATION = 'E'

		self.consoleOutputText.append('The following parameters were selected:')
		self.consoleOutputText.append(' >> BLUR_KERNEL_WIDTH    ' + str(self.BLUR_KERNEL_WIDTH))
		self.consoleOutputText.append(' >> BLUR_KERNEL_HEIGHT   ' + str(self.BLUR_KERNEL_HEIGHT))
		self.consoleOutputText.append(' >> DILATE_KERNEL_WIDTH  ' + str(self.DILATE_KERNEL_WIDTH))
		self.consoleOutputText.append(' >> DILATE_KERNEL_HEIGHT ' + str(self.DILATE_KERNEL_HEIGHT))
		self.consoleOutputText.append(' >> CLOSED_ARM_LOCATION  ' + str(self.CLOSED_ARM_LOCATION))
		self.consoleOutputText.append('')

		self.confirmParametersButton.setEnabled(False)
		self.runVideoAnalysisButton.setEnabled(True)

	def runVideoAnalysis(self):

		self.currentPoolProgressBar.setMaximum(0)
		self.currentPoolProgressBar.setValue(0)

		self.thread_pool = self.tree()
		self.analysis_objects = self.tree()

		self.youngest_thread = 0
		self.threads_processed = 0

		self.errors = []

		# create a new sessions file
		session_output_filename = 'sessions/EMATS_session_' + datetime.datetime.now().strftime("%Y_%m_%d")
		session_output_filename = self.incrementFilename(session_output_filename, 0, 'csv')
		self.session_output_file = open(session_output_filename, 'w')
		heading = 'Mouse ID,Closed Arm Entries,Closed Arm Time,Latency to Closed Arm,Open Arm Entries,Open Arm Time,Latency to Open Arm,Center Time'
		self.session_output_file.write(heading + '\n')

		for (i, ID) in enumerate(self.mouseList):
			
			# create a new mouse object
			mouse = Mouse()
			mouse.id = ID
			mouse.video_infile = self.videoList[i]
			mouse.video_outfile = 'results/' + mouse.id + '_tracked.avi'

			# create a new thread to analyse video
			analysis_thread = QtCore.QThread()
			self.thread_pool[i] = analysis_thread

			analysis = videoAnalysis(i, mouse, self.BLUR_KERNEL_HEIGHT, self.BLUR_KERNEL_WIDTH, self.DILATE_KERNEL_HEIGHT, self.DILATE_KERNEL_WIDTH, self.CLOSED_ARM_LOCATION)
			self.analysis_objects[i] = analysis

			# events output to console
			self.analysis_objects[i].postToConsole.connect(self.postToConsole)

			# events to record error
			self.analysis_objects[i].postToError.connect(self.postToError)

			# events to update current video progress bar
			self.analysis_objects[i].setcurrentPoolProgressBarMax.connect(self.setcurrentPoolProgressBarMax)
			self.analysis_objects[i].updatecurrentPoolProgressBar.connect(self.updatecurrentPoolProgressBar)

			# events to output results
			self.analysis_objects[i].outputResults.connect(self.outputResults)

			# event to signal processing complete
			self.analysis_objects[i].finished.connect(self.analysisFinished)

			self.analysis_objects[i].moveToThread(self.thread_pool[i])
			self.thread_pool[i].started.connect(self.analysis_objects[i].process)

		self.startThreads()

		self.overallProgressBar.setMaximum(len(self.thread_pool))
		self.overallProgressBar.setValue(0)
		self.runVideoAnalysisButton.setEnabled(False)

	def tree(self): return defaultdict(self.tree)

	def startThreads(self):
		if self.youngest_thread + 3 > len(self.mouseList)-1:
			for i in range(self.youngest_thread, len(self.mouseList)):
				self.thread_pool[i].start()
		else:
			self.thread_pool[self.youngest_thread + 3].start()
			self.thread_pool[self.youngest_thread + 2].start()
			self.thread_pool[self.youngest_thread + 1].start()
			self.thread_pool[self.youngest_thread + 0].start()

	def analysisFinished(self, thread_num):		
		self.thread_pool[thread_num].quit()
		self.threads_processed = self.threads_processed + 1
		self.overallProgressBar.setValue(self.overallProgressBar.value() + 1)

		if self.overallProgressBar.value() == self.overallProgressBar.maximum():
			self.consoleOutputText.append('Saving data ...')
			self.consoleOutputText.append('')
			time.sleep(2)
			self.session_output_file.close()
			self.consoleOutputText.append('Analysis finished.')
			if len(self.errors) > 0:
				self.consoleOutputText.append(' >> The following videos had errors:')
				self.consoleOutputText.append(' >> ' + str(self.errors))
			else:
				self.consoleOutputText.append(' >> No errors detected.')
			self.runVideoAnalysisButton.setEnabled(False)
			self.setWorkingDirectoryButton.setEnabled(True)
		else:
			if self.threads_processed == 4:
				self.youngest_thread = self.youngest_thread + 4
				
				self.currentPoolProgressBar.setMaximum(0)
				self.currentPoolProgressBar.setValue(0)
				
				self.threads_processed = 0
				self.startThreads()

	def incrementFilename(self, filename, num, extension):
		
		filename = filename.replace('.'+extension,'').replace('_('+str(num-1)+')','')

		if num == 0:
			filename = filename + '.' + extension
		else:
			filename = filename + '_(' + str(num) + ').' + extension
		
		if os.path.isfile(filename):
			num = num + 1
			filename = self.incrementFilename(filename, num, extension)

		return filename

	def postToConsole(self, string):
		self.consoleOutputText.append(string)

	def postToError(self, string):
		self.errors.append(string)

	def setcurrentPoolProgressBarMax(self, max_num):
		self.currentPoolProgressBar.setMaximum(self.currentPoolProgressBar.maximum() + max_num)

	def updatecurrentPoolProgressBar(self):
		self.currentPoolProgressBar.setValue(self.currentPoolProgressBar.value()+1)

	def outputResults(self, mouse):

		spf = mouse.video_spf

		result = str(mouse.id) + ',' + str(mouse.CA_entries) + ',' + str(mouse.CA_frames*spf) + ',' + str(mouse.CA_latency*spf) + ',' + str(mouse.OA_entries) + ',' + str(mouse.OA_frames*spf) + ',' + str(mouse.OA_latency*spf) + ',' + str(mouse.center_frames*spf)
		self.session_output_file.write(result + '\n')

# launch application
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())