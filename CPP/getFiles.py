import os

baseDir = '/scratch/rmmartin/tcdat'

for season in os.listdir(baseDir):
	for basin in os.listdir('{}/{}'.format(baseDir,season)):
		for storm in os.listdir('{}/{}/{}/'.format(baseDir,season,basin)):
			for image in os.listdir('{}/{}/{}/{}/ir/geo/1km_bw/'.format(baseDir,season,basin,storm)):
				#os.rename('{}/{}/{}/{}/ir/geo/1km_bw/{}'.format(baseDir,season,basin,storm,image),'{}/{}/{}/{}/ir/geo/1km_bw/{}'.format(baseDir,season,basin,storm,image[:-1]))
				print('{}/{}/{}/{}/ir/geo/1km_bw/{}'.format(baseDir,season,basin,storm,image))