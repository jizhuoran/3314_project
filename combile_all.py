import pickle, os, glob
from collections import defaultdict, Counter

dir_path = "/home/zrji/haichenglian_no_filtered/"
os.chdir(dir_path)

for dir_id in glob.glob("*"):
	# print('mv ' + dir_path + dir_id + "/aligned/*" + "../haichenglian_dataset_no_filtered/")
	os.system('mv ' + dir_path + dir_id + "/aligned/*" + " ../haichenglian_dataset_no_filtered/")