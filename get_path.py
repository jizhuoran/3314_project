import pickle, os, glob
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from scipy import misc
from scipy import spatial
import numpy as np

class_id = 0
path_book = []


dis = 8
lian_path = "/home/zrji/haichenglian_dataset/filter_" + str(dis) + "/"

os.chdir(lian_path)
print(lian_path)
for person in glob.glob("*"):

	print("gg")
	os.chdir(lian_path + person)


	if len(glob.glob("*.png")) < 30 or len(glob.glob("*.png")) > 75:
		continue
	for tu in glob.glob("*.png"):
		path_book.append((lian_path + person + "/" + tu,class_id))

	class_id += 1


os.chdir("/home/zrji/haichenglian")

with open("reference_point_dict","rb") as f:
	reference_point_dict = pickle.load(f)

with open("path_"+str(dis), 'w') as f:
	for item in path_book:
		try:
			refernece_point = reference_point_dict[item[0].replace("filter_8", "no_filtered")]
			tmp = item[0] + " " + str(item[1])
			for i in range(5):
				tmp += " " + refernece_point[i] + " " + refernece_point[i+5]
			f.write("%s\n" % tmp)
		except Exception as e:
			pass


		

