def write2file(filename, string):
	with open(filename, 'w') as f:
		f.write(string)
		f.close()