#!/usr/bin/env python
# coding: utf-8

# In[561]:


from sklearn import linear_model
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


# In[642]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# In[563]:


Dataset_pd = pd.read_csv('/Users/jhasija9/Desktop/808L/Final Project/archive/Star99999_raw.csv')
Dataset = Dataset_pd.values
Dataset


# In[711]:


Dataset_pd.info()
Dataset_pd.shape


# In[565]:


# Columns:
# Vmag - Visual Apparent Magnitude of the Star (m)

# Plx - Distance Between the Star and the Earth (d)

# e_Plx - Standard Error of Plx

# B-V - B-V color index. (A hot star has a B-V color index close to 0 or negative, while a cool star has a B-V color index close to 2.0. Other stars are somewhere in between.)

# SpType - Stellar classification.



# In[566]:


print(Dataset_pd.isnull().sum())
# Dataset_pd = Dataset_pd.dropna(how='all')
# Dataset_pd


# In[567]:


Dataset_pd.dropna(how='all')
print(Dataset_pd.isnull().sum())


# In[568]:


raw = Dataset_pd.dropna()
raw.shape
(96742, 5)
raw.isnull().sum()


# In[569]:


# Assuming 'raw' is your DataFrame
raw = raw.rename(columns={'B-V': 'B_V'})
raw.Vmag = pd.to_numeric(raw.Vmag, downcast='float', errors ='coerce')
raw.Plx = pd.to_numeric(raw.Plx, downcast='float', errors ='coerce')
#raw.e_Plx = pd.to_numeric(raw.e_Plx, downcast='float', errors ='coerce')
raw['B_V'] = pd.to_numeric(raw['B_V'], downcast='float', errors ='coerce')


# In[570]:


raw.loc[:, 'Vmag'] = pd.to_numeric(raw['Vmag'], downcast='float', errors='coerce')
raw.loc[:, 'Plx'] = pd.to_numeric(raw['Plx'], downcast='float', errors='coerce')
raw.loc[:, 'B-V'] = pd.to_numeric(raw['B-V'], downcast='float', errors='coerce')


# In[571]:


raw.query("Plx == 0")


# In[572]:


raw = raw.query('Plx != 0')
raw.shape


# In[573]:


raw.query('Plx == 0')
raw.isnull().sum()


# In[574]:


raw = Dataset_pd.dropna()
raw.shape


# In[575]:


raw.isnull().sum()


# In[576]:


# Convert 'Vmag' column to numeric (float)
raw['Vmag'] = pd.to_numeric(raw['Vmag'], errors='coerce')

# Convert 'Plx' column to numeric (float)
raw['Plx'] = pd.to_numeric(raw['Plx'], errors='coerce')

# Perform the calculation after ensuring 'Plx' column contains numeric values
raw['Amag'] = raw['Vmag'].astype(str) + ' ' + (5 * (np.log10(raw['Plx'].astype(float)) + 1)).astype(str)
raw.head()


# In[577]:


summary = raw.describe().T
summary['IQR']=summary['75%']-summary['25%']
summary.head()

summary['IQR']=summary['75%']-summary['25%']
summary


# In[578]:


summary['cutoff']=round(summary.IQR*1.6, 3)
summary.head()


# In[579]:


summary['lw']=round(summary['25%']-summary.cutoff, 3)
summary['rw']=round(summary['75%']+summary.cutoff, 3)
summary.head()


# In[580]:


# create a df with outliers
outliers=pd.DataFrame(columns=raw.columns)

#loop to detect outliers in each column
for col in summary.index:
    lower=summary.at[col,'lw'] #get lower whisker for this column
    upper=summary.at[col,'rw'] #get upper whisker for this column
    results=raw[(raw[col]<lower)|
        (raw[col]>upper)].copy() #get the dataframe
    results['Outlier']=col #to be able to identify in which column we obtained outliers
    outliers=outliers.append(results) #save them

outliers.shape


# In[581]:


df = raw.drop(outliers.index)
print('Shape after dropping changed:', df.shape)


df.describe().T


# In[582]:


df.reset_index(drop=True, inplace=True)
df.info()


# In[583]:


df.isnull().sum()


# In[584]:


new_data=df.dropna()


# In[585]:


new_data.isnull().sum()


# In[586]:


new_data.head()


# In[707]:


# Based on the Roman numeral convention, stars labeled with Roman numerals greater than or equal to 'III' 
# are classified as giants. Those labeled with Roman numerals less than 'III' are classified as dwarfs. 
# Information collected from Dataset.
def label_gen_stars(star):
    dwarf = ['D','VI', 'VII', 'V']
    giant = ['IV', 'III', 'II', 'Ib', 'Ia', 'Ia-O']
    for i in dwarf :
        if i in star:
            return 'Dwarf'
    for i in giant:
        if i in star:
            return 'Giant'
    return 'Other'
new_data['Star_Type'] = new_data.SpType.apply(label_gen_stars)


# In[708]:


new_data.head()


# In[709]:


new_data['Target'] = np.where(new_data.Star_Type == 'Giant', 0,1)
new_data.Target.value_counts()


# In[590]:


new_data.head()


# In[591]:


new_data.Star_Type.value_counts()


# In[592]:


new_data = new_data.query('Star_Type != "Other"')

new_data.query('Star_Type == "Other"')

new_data.Star_Type.value_counts()


# In[593]:


new_data.isnull().sum()
new_data.dropna()


# In[594]:


new_data.isnull().sum()


# In[595]:


new_data.dropna()


# In[596]:


new_data.isnull().sum()


# In[597]:


new_data.reset_index(drop=True, inplace=True)
new_data.info()


# In[598]:


new_data['Target'] = np.where(new_data.Star_Type == 'Giant', 0,1)
new_data.Target.value_counts()


# In[599]:


new_data.rename(columns={'B-V':'B_V'}, inplace = True)


# In[600]:


new_data.head()


# In[601]:


new_data.drop('SpType', axis = 1, inplace = True)
new_data.drop('Star_Type', axis = 1, inplace = True)


# In[602]:


new_data.head()


# In[603]:


#new_data['Temperature'] = 7090/new_data.B_V + 0.72
#new_data['Temperature'] = 7090 / new_data['B_V'] + 0.72


# In[604]:


new_data['B_V'] = pd.to_numeric(new_data['B_V'], errors='coerce')

# Calculate 'Temperature' based on the updated 'B_V' column
new_data['Temperature'] = 7090 / new_data['B_V'] + 0.72


# In[605]:


new_data.head()


# In[606]:


new_data = new_data.loc[:, ['Unnamed: 0','Vmag','Plx','e_Plx','B_V','Amag','Temperature','Target']]  # Reorder columns using column names


# In[607]:


new_data.head()


# In[608]:


new_data.isnull().sum()


# In[609]:


new_data.dropna()


# In[610]:


# Assuming 'column_name' is the name of the column containing the problematic values
new_data['Amag'] = new_data['Amag'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) else x)
new_data.head()


# In[611]:


summary = new_data.describe().T
summary['IQR']=summary['75%']-summary['25%']
summary.head()

summary['IQR']=summary['75%']-summary['25%']
summary


# In[612]:


summary['cutoff']=round(summary.IQR*1.6, 3)
summary.head()


# In[613]:


summary['lw']=round(summary['25%']-summary.cutoff, 3)
summary['rw']=round(summary['75%']+summary.cutoff, 3)
summary.head()


# In[614]:


# create a df with outliers
outliers=pd.DataFrame(columns=new_data.columns)

#loop to detect outliers in each column
for col in summary.index:
    lower=summary.at[col,'lw'] #get lower whisker for this column
    upper=summary.at[col,'rw'] #get upper whisker for this column
    results=new_data[(new_data[col]<lower)|
        (new_data[col]>upper)].copy() #get the dataframe
    results['Outlier']=col #to be able to identify in which column we obtained outliers
    outliers=outliers.append(results) #save them

outliers.shape


# In[615]:


new_data = new_data.drop(outliers.index)
print('Shape after dropping changed:', new_data.shape)


new_data.describe().T


# In[616]:


new_data.drop('Unnamed: 0', axis = 1, inplace = True)
new_data.head()


# In[619]:


new_data.head()


# In[620]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(new_data.isnull(), cbar=False, cmap='viridis')
plt.title('NaN Values in DataFrame')
plt.show()


# In[627]:


new_data.isnull().sum()
new_data.dropna(how='all')


# In[636]:


new_data.isnull().sum()
final=new_data.dropna()


# In[712]:


final.isnull().sum()
final.shape


# In[687]:


X_train, X_test, y_train, y_test = train_test_split(final.drop('Target', axis = 1),
                                                   final.Target, test_size = 0.25,
                                                   random_state = 42,
                                                   stratify = final.Target)


# In[688]:


scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler()
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[698]:


model_logistic = LogisticRegression()
# max_iter = 1e5

model_logistic.fit(X_train_scaled, y_train)


y_pred1 = model_logistic.predict(X_test_scaled)

print("Logistic Regression on the dataset")
cf= confusion_matrix(y_test, y_pred1)
print(cf)
plt.matshow(cf)
plt.title('Confusion matrix for Logistic Regression')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print('Accuracy score is : {0:.2%}'.format(accuracy_score(y_test, y_pred1)))
print('Precision score is : {0:.2%}'.format(precision_score(y_test, y_pred1)))
print('Recall score is : {0:.2%}'.format(recall_score(y_test, y_pred1)))
print('F1 score is :{0:.2%}'.format(f1_score(y_test, y_pred1)))
plt.show()

Metrics=metrics.classification_report(y_pred1,y_test)
lr_fpr, lr_tpr, threshold1 = metrics.roc_curve(y_test, y_pred1)
lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)
print(Metrics)


# In[702]:


model_tree = DecisionTreeClassifier()
model_tree.fit(X_train_scaled, y_train)
y_pred2 = model_tree.predict(X_test_scaled)


print("Decision Tree on the dataset")

cf= confusion_matrix(y_test, y_pred2)
print(cf)
plt.matshow(cf)
plt.title('Confusion matrix for decision tree')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print('Accuracy score is : {0:.2%}'.format(accuracy_score(y_test, y_pred2)))
print('Precision score is : {0:.2%}'.format(precision_score(y_test, y_pred2)))
print('Recall score is : {0:.2%}'.format(recall_score(y_test, y_pred2)))
print('F1 score is :{0:.2%}'.format(f1_score(y_test, y_pred2)))


Metrics=metrics.classification_report(y_pred2,y_test)
dt_fpr, dt_tpr, threshold1 = metrics.roc_curve(y_test, y_pred2)
dt_roc_auc = metrics.auc(dt_fpr, dt_tpr)
print(Metrics)


# In[703]:


from sklearn.model_selection import cross_val_score
# Define a range of k values
k_values = np.arange(1, 21)

# Store mean cross-validation scores for each k
mean_scores = []
# Perform k-NN for each value of k
best_accuracy = 0
best_k = 0

print(" K-nearest neighbors  on the dataset")


# Perform k-NN for each value of k and store the mean scores
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn,X_train_scaled, y_train, cv=5)
    mean_scores.append(np.mean(scores))
    accuracy = np.mean(scores)
    
    # Check if this k gives better accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best k: {best_k} with accuracy: {best_accuracy}")


# Plotting the graph
plt.figure(figsize=(8, 6))
plt.plot(k_values, mean_scores, marker='o', linestyle='-', color='b')
plt.title('k-NN Performance for Different Values of k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# Train a k-NN classifier with the best k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)

# Predict using the best k-NN classifier
y_pred3 = best_knn.predict(X_test_scaled)

# Calculate precision for the best k
precision = precision_score(y_test, y_pred3, average='weighted')
print(f'Precision for k={best_k}: {precision}')

# Confusion matrix for the best k
conf_matrix = confusion_matrix(y_test, y_pred3)
print(f'Confusion matrix for k={best_k}:\n{conf_matrix}')
plt.matshow(conf_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

Metrics=metrics.classification_report(y_pred3,y_test)
knn_fpr, knn_tpr, threshold1 = metrics.roc_curve(y_test, y_pred3)
knn_roc_auc = metrics.auc(knn_fpr, knn_tpr)
print(Metrics)


# In[704]:


model_NB = GaussianNB()
model_NB.fit(X_train_scaled, y_train)
y_pred4 = model_NB.predict(X_test_scaled)

print(" Gaussian Naive Bayes on the dataset")

cf_matrix= confusion_matrix(y_test, y_pred5)
print(cf_matrix)
plt.matshow(cf_matrix)
plt.title('Confusion matrix for Gaussian Naive Bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print('Accuracy score is : {0:.2%}'.format(accuracy_score(y_test, y_pred4)))
print('Precision score is : {0:.2%}'.format(precision_score(y_test, y_pred4)))
print('Recall score is : {0:.2%}'.format(recall_score(y_test, y_pred4)))
print('F1 score is :{0:.2%}'.format(f1_score(y_test, y_pred4)))


Metrics=metrics.classification_report(y_pred4,y_test)
gnb_fpr, gnb_tpr, threshold1 = metrics.roc_curve(y_test, y_pred4)
gnb_roc_auc = metrics.auc(gnb_fpr, gnb_tpr)
print(Metrics)


# In[705]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Linear Discriminant Analysis Classifier
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train_scaled, y_train)
y_pred5 = lda_classifier.predict(X_test_scaled)

print(" Linear Discriminant Analysis on the dataset")


cf_matrix= confusion_matrix(y_test, y_pred5)
print(cf_matrix)
plt.matshow(cf_matrix)
plt.title('Confusion matrix for Linear Discriminant Analysis')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print('Accuracy score is : {0:.2%}'.format(accuracy_score(y_test, y_pred5)))
print('Precision score is : {0:.2%}'.format(precision_score(y_test, y_pred5)))
print('Recall score is : {0:.2%}'.format(recall_score(y_test, y_pred5)))
print('F1 score is :{0:.2%}'.format(f1_score(y_test, y_pred5)))



Metrics=metrics.classification_report(y_pred5,y_test)
lda_fpr, lda_tpr, threshold1 = metrics.roc_curve(y_test, y_pred5)
lda_roc_auc = metrics.auc(lda_fpr, lda_tpr)
print(Metrics)


# In[706]:


print(" Roc Curves ")

plt. figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_roc_auc:.2f})', linestyle='-', marker='o', color='purple')
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_roc_auc:.2f})',  linestyle='--', marker='s', color='blue')
plt.plot(knn_fpr, knn_tpr, label=f'KNN (AUC = {knn_roc_auc:.2f})', color='cyan')
plt.plot(gnb_fpr, gnb_tpr, label=f'Gaussian Naive Bayes(AUC = {gnb_roc_auc:.2f})', color='brown',  linestyle= '--')
plt.plot(lda_fpr, lda_tpr, label=f'Linear Discriminant Analysis(AUC = {lda_roc_auc:.2f})')
plt. plot ([0, 1], [0, 1], 'k--')
plt.ylabel( 'True Positive Rate')
plt.xlabel("False Positive Rate")
plt. title( 'ROC Curves')
plt. legend (loc='best')
plt.grid (True)
plt.show()


# In[ ]:




