import pickle, os, glob
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from scipy import misc
from scipy import spatial
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


class_id = 0
path_book = []

count1 = defaultdict(int)
count2 = defaultdict(int)

dis = 10
lian_path2 = "/home/zrji/haichenglian_dataset/filter_" + str(dis) + "/"
lian_path1 = "/home/zrji/CASIA-WebFace/"

os.chdir(lian_path1)

for person in glob.glob("*"):

	# print("gg")
	os.chdir(lian_path1 + person)

	count1[len(glob.glob("*.jpg"))] += 1

	# if len(glob.glob("*.png")) < 25 or len(glob.glob("*.png")) > 90:
	# 	continue
	# for tu in glob.glob("*.png"):
	# 	path_book.append((lian_path + person + "/" + tu,class_id))

	# class_id += 1


os.chdir(lian_path2)

for person in glob.glob("*"):

	# print("gg")
	os.chdir(lian_path2 + person)

	count2[len(glob.glob("*.png"))] += 1



tmp1 = sorted(list(count1.items()), key = lambda x: x[0])[:100]
tmp2 = sorted(list(count2.items()), key = lambda x: x[0])

x1 = np.array([i[0] for i in tmp1])
y1 = np.array([i[1] for i in tmp1])

x2 = np.array([i[0] for i in tmp2])
y2 = np.array([i[1] for i in tmp2])

print(x2)
print(y2)
os.chdir("/home/zrji/haichenglian")


# print(x)
# print(y)

# the histogram of the data
# n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)

# # add a 'best fit' line
# l = plt.plot(bins, y, 'r--', linewidth=1)

# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
plt.plot(x1, y1, 'r--')
plt.plot(x2, y2, 'b--')
plt.savefig('temp.png')

# os.chdir("/home/zrji/haichenglian")

# with open("reference_point_dict","rb") as f:
# 	reference_point_dict = pickle.load(f)

# with open("path_"+str(dis), 'w') as f:
# 	for item in path_book:
# 		try:
# 			refernece_point = reference_point_dict[item[0].replace("filter_8", "no_filtered")]
# 			tmp = item[0] + " " + str(item[1])
# 			for i in range(5):
# 				tmp += " " + refernece_point[i] + " " + refernece_point[i+5]
# 			f.write("%s\n" % tmp)
# 		except Exception as e:
# 			pass


		

