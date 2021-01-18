# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:05:55 2021

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
from sklearn.metrics import completeness_score,homogeneity_score, v_measure_score

from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras import regularizers
from keras.utils.data_utils import get_file
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
import matplotlib.pyplot as plt

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
    rfe = RFE(RFC, n_features_to_select=15)
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

def getModel(x):
    inp = Input(shape=(x.shape[1],))
    d1=Dropout(0.5)(inp)
    encoded = Dense(8, activation='relu', activity_regularizer=regularizers.l2(10e-5))(d1)
    decoded = Dense(x.shape[1], activation='relu')(encoded)
    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def calculate_losses(x,preds):
    losses=np.zeros(len(x))
    for i in range(len(x)):
        losses[i]=((preds[i] - x[i]) ** 2).mean(axis=None)
        
    return losses

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def classify_by_autoencoder(x, y, x_test, y_test, y0, y0_test, category, ZcategoryZ ):
    autoencoder=getModel(x)
    history=autoencoder.fit(x[np.where(y0==0)],x[np.where(y0==0)],
               epochs=10,
                batch_size=100,
                shuffle=True,
                validation_split=0.1)
    # We set the threshold equal to the training loss of the autoencoder
    threshold=history.history["loss"][-1]

    testing_set_predictions=autoencoder.predict(x_test)
    test_losses=calculate_losses(x_test,testing_set_predictions)
    """
    #####
    #Plot in 2d 
    colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple','green']
    pca = PCA(n_components=2)
    train_x_pca_cont = pca.fit_transform(x_test)
    plt.figure(figsize=(15,10))
    for color, cat in zip(colors, ZcategoryZ.keys()):
        plt.scatter(train_x_pca_cont[testing_set_predictions==cat, 0],
                    train_x_pca_cont[testing_set_predictions==cat, 1],
                    color=color, alpha=.8, lw=2, label=cat)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(" Autoencoder result visualization")    
    plt.show()
    
    results = confusion_matrix(y0_test,testing_set_predictions)
    plt.figure(figsize = (10,7))
    sn.heatmap(results, annot=True, fmt='d')
    plt.title("Autoencoder confusion matrix ")
    ####
    """
    testing_set_predictions=np.zeros(len(test_losses))
    testing_set_predictions[np.where(test_losses>threshold)]=1
    accuracy=accuracy_score(y0_test,testing_set_predictions)
    recall=recall_score(y0_test,testing_set_predictions)
    precision=precision_score(y0_test,testing_set_predictions)
    f1=f1_score(y0_test,testing_set_predictions)
    print("Performance over the testing data set \n")
    print("Accuracy : {} , Recall : {} , Precision : {} , F1 : {}\n".format(accuracy,recall,precision,f1 ))

    for class_ in category:
        print(class_+" Detection Rate : {}".format(len(np.where(np.logical_and(testing_set_predictions==1 , y_test==class_))[0])/len(np.where(y_test==class_)[0])))
 
   # c = confusion_matrix(y0_test,testing_set_predictions)
   # plot_confusion_matrix(c,["Normal","Attack"])
    
    """
    plt.ylabel('Loss')
    plt.xticks(np.arange(0,5), category)
    plt.violinplot([test_losses[np.where(y_test==class_)] for class_ in category],np.arange(0,len(category)),showmeans =True )
    plt.axhline(y=threshold,c='r',label="Threshold Value")
    plt.legend();
    """
    
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
    cats = ['dos', 'u2r', 'r2l', 'probe', 'normal', 'unknown']
    
    
    #Generating and analyzing train and test sets
    train_df = pd.read_csv(train_file, names=header_names)
    train_df.drop(['success_pred'], axis=1, inplace=True)
    train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping[x])
    
    test_df = pd.read_csv(test_file, names=header_names)
    test_df.drop(['success_pred'], axis=1, inplace=True)
    test_df['attack_category'] = test_df['attack_type'].map(lambda x: check_dict(x, attack_mapping))
    
    ###### строим бары#######
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
    
    x,y=dummies_train_df[selected_features],buf_train.pop("attack_category").values
    x=x.values
    x_test,y_test=dummies_test_df[selected_features],buf_test.pop("attack_category").values
    x_test=x_test.values
    y0=np.ones(len(y),np.int8)
    y0[np.where(y==cats[4])]=0
    y0_test=np.ones(len(y_test),np.int8)
    y0_test[np.where(y_test==cats[4])]=0
    
    
    print("######### AUTOENCODER #########")
    classify_by_autoencoder(x, y, x_test, y_test, y0, y0_test, cats, category)
  

    return 0

if __name__ == "__main__":
    sys.exit(main())