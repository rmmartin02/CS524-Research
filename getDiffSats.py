with open('irbwimages.csv','r') as f:
	lines = f.readlines()

hashes = []
images = []
for l in lines:
	a = l.rstrip().split('/')
	season = a[4]
	b = a[-1].split('.')
	sat = b[2]
	hash = season + sat
	if hash not in hashes:
		hashes.append(hash)
		images.append(l.rstrip())

with open('examples.txt','w') as f:
	for i in range(len(hashes)):
		f.write('{}\n'.format(images[i]))