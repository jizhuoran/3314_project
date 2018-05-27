import pickle, os, glob
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from scipy import misc
from scipy import spatial
import numpy as np
import shutil

embedding_path = "/home/zrji/haichenglian_dataset/embedding"


    
with open(embedding_path, 'rb') as f:
    emb_array = pickle.load(f)


# with open("embedding_test", 'rb') as f:
    # emb_array = pickle.load(f)

# print(emb_array[0])

# exit(0)
emb_dict = []
path_dict = []

emb_dict = defaultdict(list)
path_dict = defaultdict(list)
count = defaultdict(int)


for emb, path in emb_array:
    name = path.split("/")[-2]
    emb_dict[name].append((emb, path))

emb_list = list(emb_dict.values())
result = []

# sum_dcit = [0 for i in range(0, 120)]

thisis = 0
for item in emb_list:

    print(thisis, "   ",len(emb_list))
    thisis+=1
    p = [i[1] for i in item]
    e = np.array([i[0] for i in item])
    # db = DBSCAN(eps=1 ,min_samples=5, n_jobs = 36)
    # db.fit_predict(e)
    # print(list(zip(p, db.labels_)))
    # print(p[1])
    eps = 1

    pdist=spatial.distance.squareform(spatial.distance.pdist(e))

    for i in range(0, len(pdist)):
        pdist[i] = np.array(list(map(lambda x: 1 if x < eps else 0, pdist[i])))
        # print(pdist[i])

    # print(np.sum(pdist, axis = 1))
    # print(pdist[np.argmax(np.sum(pdist, axis=1))])
    # if(int(np.sum(pdist[np.argmax(np.sum(pdist, axis=1))])) == 1):
        # print(p)
        # exit(0)
    # count[int(np.sum(pdist[np.argmax(np.sum(pdist, axis=1))]))] += 1
    for index, belongsto in zip(p, pdist[np.argmax(np.sum(pdist, axis=1))]):
        if belongsto > 0.5:
            path, name = os.path.split(index)
            path1 = path.replace("no_filtered","filter_10")
            if not os.path.exists(path1):
                os.mkdir(path1)
            try:
                shutil.copy(path+"/"+name, path1+"/"+name)
            except Exception as e:
                print(e)
    # exit(0)
# 
    # for coor_path, group in list(zip(p, db.labels_)):
        
    #     if not group == 0:
    #         try:
    #             os.remove(coor_path)
    #         except Exception as e:
    #             print(e)

    # result.append(list(zip(p, db.labels_)))
    # print(db.labels_)
    #print(db.core_sample_indices_)
#  print(len(result))
# print(sum_dcit)
# print(count)
    #print(len(label_tmp))
    #print(len(db.labels_))

    #for i in list(zip(label_tmp, db.labels_)):
    #   print(i)

