data = open('Canon G3.txt').read()

data1 = data.split('\n')

data2 = [asdf.split('##',1)[-1] for asdf in data1]

data3 = ''

for asdf in data2:
	data3 = data3 + asdf

data4 = data3.split('[t]')