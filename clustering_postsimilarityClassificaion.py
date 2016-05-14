#here going to classify a post according to clusters

# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import scipy as sp
import sys
import nltk.stem
import sklearn.datasets
# 	def build_analyzer(self):
# 		analyzer = super(StemmedCountVectorizer,self).build_analyzer()
# 		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedTfidfVectorizer,self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def dist_raw(v1, v2):
	#calculting here the normalized distance between the two vectors
	v1_normalized = v1/sp.linalg.norm(v1.toarray())
	v2_normalized = v2/sp.linalg.norm(v2.toarray())
	delta = v1_normalized-v2_normalized
	# print delta
	return sp.linalg.norm(delta.toarray())
	#norm calculates euclidean norm
#Countvectorizer count the words and vectorize them

english_stemmer = nltk.stem.SnowballStemmer('english')

vectorizer = StemmedTfidfVectorizer(stop_words='english', decode_error='ignore')
# vectorizer = CountVectorizer(min_df=1,stop_words = "english")
# content = ["How to format my hard disk", " Hard disk format problems "]
# X = vectorizer.fit_transform(content)
#now turning this to vectorized form
# print(vectorizer.get_feature_names())
# print(X.toarray().transpose())
#this has turned content to vectorized form
os.environ["MLCOMP_DATASETS_HOME"] = "/home/ritesh/R/machinelearning/machinelearningCodes/data/"
# DIR = "/home/ritesh/R/machinelearning/machinelearningCodes/data"
print( sklearn.datasets.get_data_home())
# all_data = sklearn.datasets.fetch_20newsgroups(subset='all')
# groups = ['comp.graphics', 'comp.os.ms-windows.misc',
# 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
# 'comp.windows.x', 'sci.space']
posts = sklearn.datasets.load_mlcomp("20news-18828",set_='train')
# print(all_data)	

print("here")
# posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
print(posts.target_names)
print(len(posts.filenames))



#now fitting these posts to CountVectorizer
X_train = vectorizer.fit_transform(posts.data[1:1222])

num_samples, num_features = X_train.shape
print(vectorizer.get_feature_names())
print(X_train.toarray().transpose())
print("Samples %d Words %d"%(num_samples,num_features))
#to transform new post we do
# print(X_train)

# ##########################
# ###### Now going to comment this out
# ##########################
# new_post = "imaging databases"
# new_post_vec = vectorizer.transform([new_post])
# print("Samples %d Words %d"%(num_samples,num_features))
# #that tables updated is the same but the samples dataset is different
# # print(X_train.getrow(1))
# # print enumerate(num_samples)

# ###now comparing this newpost with other to find the most similar one
# best_doc = None
# best_dist = sys.maxint
# best_i = None
# post = None
# for i in range(0,num_samples):
# 	post = posts[i]
#  	if post == new_post:
#  		continue
#  	post_vec = X_train.getrow(i)
#  	d = dist_raw(post_vec, new_post_vec)
#  	print("=== Post %i with dist=%.2f: %s"%(i, d, post))
#  	if d<best_dist:
#  		best_dist = d
#  		best_i = i
# print("Best post is %i with dist=%.2f"%(best_i, best_dist))
# print("Surce post is %s"%(new_post))

#  #as seen this has Some problems on the classification

num_clusters = 50
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters, init='random', n_init=1,verbose=1, random_state=3)
km.fit(X_train)

print(km.labels_)

new_post = "Disk drive problems. Hi, I have a problem with my hard disk.After 1 year it is working only sporadically now.I tried to format it, but now it doesn't boot any more.Any ideas? Thanks."
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]

#fetching posts with same label
print(new_post_label)
similar_indices = (km.labels_== new_post_label).nonzero()[0]

#now printing similarity scores
similar = []
for i in similar_indices:
	dist = sp.linalg.norm((new_post_vec - X_train[i]).toarray())
	similar.append((dist, posts.data[i]))
similar = sorted(similar)
# print(similar)
