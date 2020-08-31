# -*- coding: utf-8 -*-
"""
Created on Sun May 31 18:02:07 2020

@author: Maksim Miskov
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
from sklearn.metrics import completeness_score,homogeneity_score, v_measure_score

def check_dict(x, attack_mapping):
    if x in attack_mapping:
        return attack_mapping[x]
    else:
        return attack_mapping['unknown']
    return

def find_cat_feature(data_file):
    cat_list = list()
    for col_name in data_file.columns:
        if data_file[col_name].dtypes == 'object' :
            unique_cat = len(data_file[col_name].unique())
            print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))
            cat_list.append(col_name)
    return cat_list

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
    rfe = RFE(RFC, n_features_to_select=40)
    rfe = rfe.fit(train_x, train_Y)
    feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
    selected_features = [v for i, v in feature_map if i==True]
    return selected_features

def plot_pca(_pca,_category, _train_x_pca_cont, _buf_test, _colors):
    
    plt.figure(figsize=(15,10))
    for color, cat in zip(_colors, _category.keys()):
        plt.scatter(_train_x_pca_cont[_buf_test['attack_category']==cat, 0],
                    _train_x_pca_cont[_buf_test['attack_category']==cat, 1],
                    color=color, alpha=.8, lw=2, label=cat)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("Test dataset visualization")   
    plt.show()
    return

def plot_res_ml(pca,train_x_pca_cont, pred_y, buf_test, colors, category, _ml_type):
    # Plot in 2d 
   
    plt.figure(figsize=(15,10))
    for color, cat in zip(colors,category.keys()):
        plt.scatter(train_x_pca_cont[pred_y==cat, 0],
                    train_x_pca_cont[pred_y==cat, 1],
                    color=color, alpha=.8, lw=2, label=cat)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("{_ml_type} result ".format(_ml_type = _ml_type )) 
    plt.show()
    return

def classify_by_dec_tree(train_x,train_Y,test_x,test_Y,category ):
    
    classifier = DecisionTreeClassifier(max_depth =7, random_state=25)
    classifier.fit(train_x, train_Y)
    pred_y = classifier.predict(test_x)
    
    results = confusion_matrix(test_Y, pred_y)
    error = zero_one_loss(test_Y, pred_y)
    accuracy = metrics.accuracy_score(test_Y,pred_y)
    classification = metrics.classification_report(test_Y,pred_y)
    
    #Plot in 2d 
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple','green']
    pca = PCA(n_components=2)
    train_x_pca_cont = pca.fit_transform(test_x)
    plt.figure(figsize=(15,10))
    for color, cat in zip(colors, category.keys()):
        plt.scatter(train_x_pca_cont[pred_y==cat, 0],
                    train_x_pca_cont[pred_y==cat, 1],
                    color=color, alpha=.8, lw=2, label=cat)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(" DecisionTree result visualization")    
    plt.show()
    
    plt.figure(figsize = (10,7))
    sn.heatmap(results, annot=True, fmt='d')
    plt.title("DecisionTree confusion matrix ")
    
    print("DecisionTree confusion matrix: \n",results)
    print("Error: ",error*100,"%")
    print("Accuracy: ", accuracy*100, "%")
    print("Classification report:" "\n", classification)
    
    return pred_y

def classify_by_KNeighbors(train_x,train_Y,test_x,test_Y, colors, category):
    
    classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    classifier.fit(train_x, train_Y)

    pred_y = classifier.predict(test_x)
    results = confusion_matrix(test_Y, pred_y)
    
    error = zero_one_loss(test_Y, pred_y)
    accuracy = metrics.accuracy_score(test_Y,pred_y)
    classification = metrics.classification_report(test_Y,pred_y)
    
    pca = PCA(n_components=2)
    train_x_pca_cont = pca.fit_transform(test_x)
    plt.figure(figsize=(15,10))
    for color, cat in zip(colors, category.keys()):
        plt.scatter(train_x_pca_cont[pred_y==cat, 0],
                    train_x_pca_cont[pred_y==cat, 1],
                    color=color, alpha=.8, lw=2, label=cat)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(" KNeibors result visualization")    
    plt.show()
    
    plt.figure(figsize = (10,7))
    sn.heatmap(results, annot=True, fmt='d')
    plt.title("KNeighbors confusion matrix: \n")
    
    print("KNeighbors confusion matrix: ",results)
    print("Error: ", error*100,'%')
    print("Accuracy: ", accuracy*100, "%")
    print("Classification report:" "\n", classification)
    return pred_y

### PCA+ Kmeans
def classify_by_Kmeans(_dummies_test_df, _buf_test, _train_x_pca_cont, colors):
    
    pca = PCA(n_components=2)
    _train_x_pca_cont = pca.fit_transform(_dummies_test_df)
# Fit the training data to a k-means clustering estimator model
    kmeans = KMeans(n_clusters=6, random_state=23).fit(_train_x_pca_cont)
    kmeans_y = kmeans.labels_
    
    # Plot in 2d 
    plt.figure(figsize=(15,10))
    for color, cat in zip(colors, range(6)):
        plt.scatter(_train_x_pca_cont[kmeans_y==cat, 0],
                    _train_x_pca_cont[kmeans_y==cat, 1],
                    color=color, alpha=.8, lw=2, label=cat)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("Kmeans result")    
    plt.show()
    
    print('Completeness: {}'.format(completeness_score(_buf_test['attack_category'], kmeans_y)))
    print('Homogeneity: {}'.format(homogeneity_score(_buf_test['attack_category'], kmeans_y)))
    print('V-measure: {}'.format(v_measure_score(_buf_test['attack_category'], kmeans_y)))
    return 
def linearsVC():
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    clf = SVC(gamma='auto')
    clf.fit(dummies_train_df[selected_features], buf_train['attack_category'])
    predictedsvc = clf.predict(dummies_test_df[selected_features])
    print (metrics.accuracy_score(buf_test['attack_category'], predictedsvc))
    print (metrics.confusion_matrix(buf_test['attack_category'], predictedsvc))
    print( metrics.classification_report(buf_test['attack_category'], predictedsvc))
    return
def main():
    """Entry point"""
    dataset_dir = "D:/NSL_KDD-master"
    train_file = os.path.join(dataset_dir,"KDDTrain+.csv")
    test_file = os.path.join(dataset_dir,"KDDTest+.csv")
    atack_types = os.path.join(dataset_dir,"Attack Types.csv")
    Field_Names = os.path.join(dataset_dir,"Field Names.csv")
    
    #Reading and processing dataset
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
   
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple','green']
    
    attack_mapping, category = create_attac_mapping(atack_types)
    print("ATTACK MAPPING: ",attack_mapping)
   
    #Generating and analyzing train and test sets
    train_df = pd.read_csv(train_file, names=header_names)
    train_df.drop(['success_pred'], axis=1, inplace=True)
    train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
    
    test_df = pd.read_csv(test_file, names=header_names)
    test_df.drop(['success_pred'], axis=1, inplace=True)
    test_df['attack_category'] = test_df['attack_type'].map(lambda x: check_dict(x, attack_mapping))
    
    train_attack_types = train_df['attack_type'].value_counts()
    train_attack_cats = train_df['attack_category'].value_counts()
   
    train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)
    plt.title('Train attack types',fontsize=20)
    plt.show()
    
    train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)
    plt.title('Train attack categories',fontsize=20)
    plt.show()
    
    test_attack_types = test_df['attack_type'].value_counts()
    test_attack_cats = test_df['attack_category'].value_counts()
    
    test_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)
    plt.title('Test attack types',fontsize=20)
    plt.show()
    
    test_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)
    plt.title('Test attack categories',fontsize=20)
    plt.show()
    
    ################### DATA PREPROCESSING ######################
    
    buf_train, buf_test = train_df[[ 'attack_type','attack_category']], test_df[['attack_type','attack_category' ]]
    train_df.drop(['attack_type','attack_category'], axis=1, inplace=True)
    test_df.drop(['attack_type','attack_category'], axis=1, inplace=True)
    
    # определяем категориальные фичи 
    train_cat_feature = find_cat_feature(train_df)
    print('\n Symbolic features in train dataset: ',train_cat_feature)
    dummies_train_df = pd.get_dummies(train_df, columns = train_cat_feature, drop_first = True)
    dummies_test_df = pd.get_dummies(test_df, columns = train_cat_feature, drop_first = True)
    print("\n Category features: ", find_cat_feature(test_df))
    
    #определяем отличающиеся категории в тренировочном и тестовом датасете
    differ = dummies_train_df.columns.difference(dummies_test_df.columns)
    print("\n Difference in: ",differ)
    
    # добавляем отличающиеся категории в тестовый датасет
    dummies_test_df = pd.concat([dummies_test_df, pd.DataFrame(columns=differ)],axis = 1)
    dummies_test_df[differ] = 0
    
    ####  NORMALIZATION ########
    cols = dummies_train_df.columns
    standard_scaler = StandardScaler()
    dummies_train_df[cols] =standard_scaler.fit_transform(dummies_train_df[cols]) 
    dummies_test_df[cols] = standard_scaler.fit_transform(dummies_test_df[cols])
    
   ################################################
    pca = PCA(n_components=2)
    train_x_pca_cont = pca.fit_transform(dummies_test_df)
    plot_pca(pca, category, train_x_pca_cont, buf_test, colors)
    
    ################# FEATURE SELECTION ##################
    plot_top_features(dummies_train_df, buf_train['attack_category'])
    selected_features = get_top_features(dummies_train_df, buf_train['attack_category'])
    print("\n Selected feature: ", selected_features)
    ####################################################
    
    print("#########Classification by Decision tree#########")
    pred_y = classify_by_dec_tree(dummies_train_df[selected_features], buf_train['attack_category'],
                                  dummies_test_df[selected_features], buf_test['attack_category'], category)
    
    print("#########Classification by Kneighbors#########")
    classify_by_KNeighbors(dummies_train_df[selected_features], buf_train['attack_category'],
                           dummies_test_df[selected_features], buf_test['attack_category'], colors, category)
    print("##############################################")
    
    print("#########Classification by Kneighbors#########")
    classify_by_Kmeans(dummies_test_df[selected_features], buf_test, train_x_pca_cont, colors)
    print("##############################################")
    
    ##### Kmeans+PCA
    #classify_by_Kmeans(dummies_test_df[selected_features], buf_test, train_x_pca_cont, colors)
    """
    classify_by_dec_tree(dummies_train_df, buf_train['attack_category'],dummies_test_df, buf_test['attack_category'])
    dummies_test_df = pd.concat([dummies_test_df, buf_test], axis=1 )
    dummies_train_df = pd.concat([dummies_train_df, buf_train], axis=1 )
     """
    return 0

if __name__ == "__main__":
    sys.exit(main())