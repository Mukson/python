# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:50:46 2020

@author: Maksim
"""
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, zero_one_loss
import csv
import numpy as np
import pandas as pd
import os
import time
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import itertools
import seaborn as sn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score,homogeneity_score, v_measure_score,accuracy_score

def check_dict(x, attack_mapping):
    if x in attack_mapping:
        return attack_mapping[x]
    else:
        return attack_mapping['unknown']
    return

def create_attac_mapping(atack_types_file):
    category = defaultdict(list)
    with open(atack_types_file) as at:
        types = csv.reader(at, delimiter=",")
        for my_type in types:
            attack, cat = my_type
            category[cat].append(attack)
    #print("cat ",category.keys())        
    at_mapping = dict((v,k) for k in category for v in category[k])
    return at_mapping, category

def find_cat_feature(data_file):
    cat_list = list()
    for col_name in data_file.columns:
        if data_file[col_name].dtypes == 'object' :
            unique_cat = len(data_file[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
            cat_list.append(col_name)
    return cat_list

def plot_top_features(train_x,train_Y):
    RFC = RandomForestClassifier()
    RFC.fit(train_x, train_Y)
    score = np.round(RFC.feature_importances_,3)
    importances = pd.DataFrame({'feature':train_x.columns,'importance':score})
    importances = importances.sort_values('importance',ascending=False).set_index('feature')
    plt.rcParams['figure.figsize'] = (30, 10)
    importances.plot.bar(fontsize = 20)
    plt.show()
    return
def get_top_features(train_x,train_Y ):
    RFC = RandomForestClassifier()
    rfe = RFE(RFC, n_features_to_select=50)
    rfe = rfe.fit(train_x, train_Y)
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
    selected_features = [v for i, v in feature_map if i==True]
    return selected_features


dataset_dir = "D:/NSL_KDD-master"
train_file = os.path.join(dataset_dir,"KDDTrain+.csv")
test_file = os.path.join(dataset_dir,"KDDTest+.csv")
atack_types = os.path.join(dataset_dir,"Attack Types.csv")
 
header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
                    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                    'num_failed_logins', 'logged_in', 'num_compromised',
                    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                    'num_shells', 'num_access_files', 'num_outbound_cmds', 
                    'is_host_login', 'is_guest_login', 'count', 'srv_count', 
                    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
                    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                    'dst_host_srv_rerror_rate', 'attack_type', 'success_pred'] 

labels2 = ['normal', 'attack']
labels5 = ['normal', 'dos', 'probe', 'r2l', 'u2r']

attack_mapping, category = create_attac_mapping(atack_types)
print("ATTACK MAPPING: ",attack_mapping)
   
    #Generating and analyzing train and test sets
train_df = pd.read_csv(train_file, names=header_names)

train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
    
test_df = pd.read_csv(test_file, names=header_names)

test_df['attack_category'] = test_df['attack_type'].map(lambda x: check_dict(x, attack_mapping))
    

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()
test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

train_df['labels2'] = train_df.apply(lambda x: 'normal' if 'normal' in x['attack_type'] else 'attack', axis=1)
test_df['labels2'] = test_df.apply(lambda x: 'normal' if 'normal' in x['attack_type'] else 'attack', axis=1)

train_df.drop(['success_pred','attack_type','attack_category'], axis=1, inplace=True)
test_df.drop(['success_pred','attack_type','attack_category'], axis=1, inplace=True)

train_Y_bin = train_df['labels2'].apply(lambda x: 0 if x is 'normal' else 1)
test_Y_bin = test_df['labels2'].apply(lambda x: 0 if x is 'normal' else 1)

print(train_df)

# определяем категориальные фичи 
train_cat_feature = find_cat_feature(train_df)
print('Symbolic features in train dataset: ',train_cat_feature)
dummies_train_df = pd.get_dummies(train_df, columns = train_cat_feature, drop_first = True)
dummies_test_df = pd.get_dummies(test_df, columns = train_cat_feature, drop_first = True)
print(find_cat_feature(test_df))
    
    #определяем отличающиеся категории в тренировочном и тестовом датасете
differ = dummies_train_df.columns.difference(dummies_test_df.columns)
print("Difference in: ",differ)
    
    # добавляем отличающиеся категории в тестовый датасет
dummies_test_df = pd.concat([dummies_test_df, pd.DataFrame(columns=differ)],axis = 1)
dummies_test_df[differ] = 0

## do feature selection
plot_top_features(dummies_train_df, train_Y_bin)
features_to_use = get_top_features(dummies_train_df, train_Y_bin)
print("Selected feature: ", features_to_use)
##########################3

train_df_trimmed = dummies_train_df[features_to_use]
test_df_trimmed = dummies_test_df[features_to_use]

#### do normalization
cols = train_df_trimmed.columns
standard_scaler = StandardScaler()
train_df_trimmed[cols] =standard_scaler.fit_transform(train_df_trimmed[cols]) 
test_df_trimmed[cols] = standard_scaler.fit_transform(test_df_trimmed[cols])
######

kmeans = KMeans(n_clusters=8, random_state=23)
kmeans.fit(train_df_trimmed)
kmeans_train_y = kmeans.labels_

print("##############",pd.crosstab(kmeans_train_y, train_Y_bin))

train_df['kmeans_y'] = kmeans_train_y
train_df_trimmed['kmeans_y'] = kmeans_train_y

kmeans_test_y = kmeans.predict(test_df_trimmed)
test_df['kmeans_y'] = kmeans_test_y

pca8 = PCA(n_components=2)
train_df_trimmed_pca8 = pca8.fit_transform(train_df_trimmed)

plt.figure(figsize=(15,10))

colors8 = ['navy', 'turquoise', 'darkorange', 'red', 'purple', 'green', 'magenta', 'black']
labels8 = [0,1,2,3,4,5,6,7]

for color, cat in zip(colors8, labels8):
    plt.scatter(train_df_trimmed_pca8[train_df.kmeans_y==cat, 0], train_df_trimmed_pca8[train_df.kmeans_y==cat, 1],
                color=color, alpha=.8, lw=2, label=cat)
    
print(pd.crosstab(test_df.kmeans_y, test_df.labels2))

#Cluster 0 - Random Forest Classifier (Strategy Option 3)

train_y0 = train_df[train_df.kmeans_y==0]
test_y0 = test_df[test_df.kmeans_y==0]
rfc = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=17).fit(train_y0.drop(['labels2', 'kmeans_y'], axis=1), train_y0['labels2'])
pred_y0 = rfc.predict(test_y0.drop(['labels2',  'kmeans_y'], axis=1))
print(confusion_matrix(test_y0['labels2'], pred_y0))
    
#Cluster 1 - Dominant Label Zero (Strategy Option 2)

print(confusion_matrix(test_df[test_df.kmeans_y==1]['labels2'], np.zeros(len(test_df[test_df.kmeans_y==1]))))

#Cluster 2 - Dominant Label Zero (Strategy Option 2)

print(confusion_matrix(test_df[test_df.kmeans_y==2]['labels2'], np.zeros(len(test_df[test_df.kmeans_y==2]))))

#Cluster 3 - Empty Cluster
#Cluster 4 - Random Forest Classifier (Strategy Option 3)

train_y0 = train_df[train_df.kmeans_y==4]
test_y0 = test_df[test_df.kmeans_y==4]
rfc = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=17).fit(train_y0.drop([ 'kmeans_y'], axis=1), train_y0['labels2'])
pred_y0 = rfc.predict(test_y0.drop(['kmeans_y'], axis=1))
print("cluster {} score is {}, {}".format(4, accuracy_score(pred_y0, test_y0['labels2']), accuracy_score(pred_y0, test_y0['labels2'], normalize=False)))

print(confusion_matrix(test_y0['labels2'], pred_y0))

#Cluster 5 - Outlier/Attack (Strategy Option 1)

print(confusion_matrix(test_df[test_df.kmeans_y==5]['labels2'], np.ones(len(test_df[test_df.kmeans_y==5]))))

#Cluster 6 - Outlier/Attack (Strategy Option 1)

print(confusion_matrix(test_df[test_df.kmeans_y==6]['labels2'], np.ones(len(test_df[test_df.kmeans_y==6]))))

#Cluster 7 - Dominant Label Zero (Strategy Option 2)

print(confusion_matrix(test_df[test_df.kmeans_y==7]['labels2'], np.zeros(len(test_df[test_df.kmeans_y==7]))))

#Combined Results: k-means + Random Forest Classifier ensembling with AR feature selection
# combined results:
num_samples = 22544
false_pos = 3177 + 22 + 1 + 8
false_neg = 152 + 87 + 6 + 11 + 5

print('True positive %: {}'.format(1-(false_pos/num_samples)))
print('True negative %: {}'.format(1-(false_neg/num_samples)))
