sats = ['05.gms6','06.gms6','07.gms6','08.gms6','09.gms6','10.gms6',
		'06.goes11','07.goes11','08.goes11','09.goes11','10.goes11','11.goes11',
		'06.goes12','07.goes12','08.goes12','09.goes12',
		'10.goes13','11.goes13','12.goes13','13.goes13','14.goes13','15.goes13','16.goes13','17.goes13',
		'12.goes14','13.goes14',
		'12.goes15','15.goes15','14.goes15','15.goes15','16.goes15','17.goes15','18.goes15',
		'15.himawari8','16.himawari8','17.himawari8','18.himawari8',
		'06.meteo5','07.meteo5',
		'07.meteo7','08.meteo7','09.meteo7','10.meteo7','11.meteo7','12.meteo7','13.meteo7','14.meteo7','15.meteo7','16.meteo7',
		'06.msg1','07.msg1','08.msg1','09.msg1','13.msg1','14.msg1','15.msg1','16.msg1',
		'06.msg2','07.msg2','08.msg2','09.msg2','10.msg2','11.msg2','12.msg2',
		'13.msg3','14.msg3','15.msg3','16.msg3','17.msg3',
		'18.msg4'
		'11.mtsat1r','12.mtsat1r','13.mtsat1r','14.mtsat1r',
		'10.mtsat2','11.mtsat2','12.mtsat2',
		'12.mtsat-2','13.mtsat-2','14.mtsat-2','15.mtsat-2']

with open('irbwimages.csv','r') as f:
	images = f.readlines()

approved = []
for i in images:
	a = i.rstrip().split('/')
	season = a[4]
	info = a[-1].split('.')
	sat = info[2]
	trackInfo = info[6].split('-')
	try:
		wind = int(trackInfo[0][:-3])
	except ValueError:
		print(i)
	if wind>137 and season[-2:]+'.'+sat in sats:
		approved.append(i)

with open('cat5.txt','w') as f:
	for i in approved:
		f.write(i)