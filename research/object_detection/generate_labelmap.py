import os, sys, pandas as pd, shutil, getopt

def get_class_list(csv_path):
	dframe = pd.read_csv(csv_path)
	clist = dframe['class'].unique()
	print(len(clist), 'unique classes found.')
	return clist

def generate_labelmap(output_path, clist):
	labelmap_path = os.path.join(output_path, 'labelmap.pbtxt')
	file = open(labelmap_path, 'w')
	
	counter = 1
	for item in clist:
		entry = "item {\n\tid: " + str(counter) + "\n\tname: '" + item + "'\n}\n\n"
		file.write(entry)
		counter += 1
	
	file.close()
	print('Label map generated at:', labelmap_path)
	
def main(argv):
	csv_path = ''	
	output_path = ''
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["csv_path=","output_path="])
	except getopt.GetoptError:
		print ('Exception: Syntax Error test.py -i <csv_path> -o <output_path>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print ('Syntax: test.py -i <csv_path> -o <output_path>')
			sys.exit()
		elif opt in ("-i", "--csv_path"):
			csv_path = arg
		elif opt in ("-o", "--output_path"):
			output_path = arg
			
	if csv_path == '' or output_path == '':
		print('Exception: One or more directories are not specified')
		sys.exit(2)
	
	clist = get_class_list(csv_path)
	
	
	generate_labelmap(output_path, clist)
	
	

if __name__ == "__main__":
   main(sys.argv[1:])
	