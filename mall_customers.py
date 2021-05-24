import pandas as pd
import sklearn.cluster as sk_cluster
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk_preproc
import numpy as np

df = pd.read_csv('data/Mall_Customers.csv')
print(df)
le = sk_preproc.LabelEncoder()
le.fit(df[['Genre']])
xx = le.transform(df[['Genre']])
df['Genre'] = xx
#rename gnre to gender
X = df[['Genre','Age','Annual Income (k$)','Spending Score (1-100)']]

kmeans_pp_cluster_model = sk_cluster.KMeans(n_clusters=5,n_init=11)
kmeans_pp_cluster_model.fit(X)
kmeans_pp_clasters = kmeans_pp_cluster_model.predict(X)
plt.scatter(df['CustomerID'], df['Spending Score (1-100)'],c= kmeans_pp_clasters,cmap='gist_rainbow')#color, grounded on claster
kmeans_pp_centroids = kmeans_pp_cluster_model.cluster_centers_



#pivot table -- for showing results
df['class']=kmeans_pp_cluster_model.labels_

pivot_score_kmeans = df.pivot_table(
                                    index='class',
                                    values=['Genre','Age','Annual Income (k$)','Spending Score (1-100)'],
                                    aggfunc={'Genre':np.mean, 'Age': len,
                                             'Annual Income (k$)':np.mean,'Spending Score (1-100)':np.mean
                                             }
                                    )
print(pivot_score_kmeans)




aggl_pp_cluster_model = sk_cluster.AgglomerativeClustering(n_clusters=5,linkage='ward')#n_init=11)
# kmeans_pp_cluster_model.fit(X)
aggl_pp_clasters = aggl_pp_cluster_model.fit_predict(X)
df['class'] = aggl_pp_cluster_model.labels_
print('\n\nAGGL CLUSTER\n\n')
pivot_score_aggl = df.pivot_table(
                                    index='class',
                                    values=['Genre','Age','Annual Income (k$)','Spending Score (1-100)'],
                                    aggfunc={'Genre':np.mean, 'Age': len,
                                             'Annual Income (k$)':np.mean,'Spending Score (1-100)':np.mean
                                             }
                                    )
print(pivot_score_aggl)
fig2 = plt.figure()
plt.scatter(df['CustomerID'], df['Spending Score (1-100)'],c= aggl_pp_clasters,cmap='gist_rainbow')#color, grounded on claster

plt.show()


