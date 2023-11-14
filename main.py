import pandas as pd
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit('Please declare the environment variable SUMO_HOME')
    # Can declare using export SUMO_HOME="/usr/local/Cellar/sumo" (default location)
import traci

import subprocess
import xml.etree.ElementTree as ET


if __name__ == '__main__':
	print('Start')
	
	result = []

	next_states = [
		[1,3,5,7],
		[3,5,7],
		[1,5,7],
		[1,3,7],
		[1,3,5],
		[5,7],
		[3,7],
		[3,5],
		[1,7],
		[1,5],
		[1,3],
		[1],
		[3],
		[5],
		[7],
		[]
	]
	
	# Loop through possible values
	for max_gap in range(1, 3):
		for min_dur in range(11, 13):
			for max_dur in range(51, 53):
				for next_state in next_states:
					# print(f'max_gap: {max_gap}')
					# print(f'min_dur: {min_dur}')
					# print(f'max_dur: {max_dur}')
					# print(f'next_state: {next_state}')

					net_tree = ET.parse('./TwoLoops/TwoLoops-default.net.xml')
					net_root = net_tree.getroot()

					# Modify the net xml
					for param in net_root.findall('./tlLogic/param'):
						if param.attrib['key'] == 'max-gap':
							param.attrib['value'] = str(float(max_gap))

					# Modify the net xml
					for idx, phase in enumerate(net_root.findall('./tlLogic/phase')):
						if phase.attrib['duration'] != '3':
							phase.attrib['minDur'] = str(min_dur)
							phase.attrib['maxDur'] = str(max_dur)
						elif phase.attrib['duration'] == '3':
							if idx not in next_state:
								phase.attrib.pop('next')

					# Write back to replace the xml file so that the changes can take effect
					net_tree.write('./TwoLoops/TwoLoops.net.xml')

					# Similar to running on command line
					subprocess.run(['sumo', 'TwoLoops/TwoLoops.sumocfg'])
					
					# Read the statistics log xml
					statistics_tree = ET.parse('./TwoLoops/TwoLoops-statistics.log.xml')
					statistics_root = statistics_tree.getroot()
					waiting_time = statistics_root.findall('./vehicleTripStatistics')[0].attrib['waitingTime']
					# print(f'waiting_time: {waiting_time}')

					# Append result as a dict to a list
					result.append({
						'max_gap': max_gap,
						'min_dur': min_dur,
						'max_dur': max_dur,
						'next_state': next_state,
						'waiting_time': waiting_time
					})
	
	df = pd.DataFrame(result)
	df.to_csv('./result.csv')

	print('End')
