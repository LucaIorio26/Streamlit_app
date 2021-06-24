# Moduli usati
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.stats import chi2_contingency

import statsmodels.api as sm
from statsmodels.formula.api import logit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


#Url del dataset caricato su github
url = 'https://raw.githubusercontent.com/LucaIorio26/Streamlit_app/main/heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(url,sep=';')

# Rinominazione delle colonne in italiano
df = df.rename(columns={'age': 'Eta','anaemia': 'Anemia', 'creatinine_phosphokinase': 'CPK','diabetes': 'Diabete', 
                          'ejection_fraction': 'Frazione_di_Eiezione','high_blood_pressure': 'Ipertensione', 'platelets': 'Piastrine',
                          'serum_creatinine': 'Creatinina','serum_sodium': 'Sodio',
                          'sex': 'Genere','smoking': 'Fumatore','time': 'Giorni','DEATH_EVENT': 'Decesso'})

# Assenza di dati mancanti
df.isna().any()


# Sproporzione tra decessi e sopravvissuti
df['Decesso' ].value_counts() 

# Correzioni di Eta e Piastrine
data = pd.DataFrame(df)
data.loc[data['Eta'] == 60667, 'Eta'] = 61

data.loc[data['Piastrine'] == 26335803, 'Piastrine'] = 263358



#Controllo delle correzioni
data.describe()
data.isna().any()



# Matrice di correlazione
data_plot = data[['Eta','CPK','Frazione_di_Eiezione','Piastrine','Creatinina','Sodio','Giorni']]

corr = data_plot.corr(method='pearson')
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True,cmap='viridis')
plt.show()


# Coppia di colori usati per i plot
colors = ["#450085", "#00846b"]
sns.set_palette(sns.color_palette(colors))
sns.set_theme(style="whitegrid")

# Grafico densità di Frazione_di_Eiezione
plt.figure(figsize=(10,8))
sns.displot(data=data,x='Giorni',hue='Decesso',palette=colors,multiple="stack",kind="kde")
plt.show()

# Grafico densità di Giorni
plt.figure(figsize=(10,8))
sns.displot(data=data,x='Frazione_di_Eiezione',hue='Decesso',palette=colors,multiple="stack",kind="kde")
plt.show()

# Creazione della variabile Rischio
conditions = [
    (data['Giorni'] <= 70),
    (data['Giorni'] > 71)]
values = [1,0]
data['Rischio'] = np.select(conditions, values)


#Box plot Eta e Decesso
plt.figure(figsize=(10,8))
sns.catplot(kind='box',data=data,y='Eta', x='Decesso', col='Rischio',palette=colors)
plt.show()


#Test sulla differenza dei livelli medi per le variabili quantitative

stats.f_oneway(data['Frazione_di_Eiezione'][data['Decesso'] == 0],
               data['Frazione_di_Eiezione'][data['Decesso'] == 1]) #SIGNIFICATIVA 
stats.f_oneway(data['Creatinina'][data['Decesso'] == 0],
               data['Creatinina'][data['Decesso'] == 1]) #SIGNIFICATIVA 
stats.f_oneway(data['Sodio'][data['Decesso'] == 0],
               data['Sodio'][data['Decesso'] == 1]) #SIGNIFICATIVA 
stats.f_oneway(data['Giorni'][data['Decesso'] == 0],
               data['Giorni'][data['Decesso'] == 1]) #SIGNIFICATIVA 
stats.f_oneway(data['CPK'][data['Decesso'] == 0],
               data['CPK'][data['Decesso'] == 1]) #NON SIGNIFICATIVA 
stats.f_oneway(data['Eta'][data['Decesso'] == 0],
               data['Eta'][data['Decesso'] == 1]) #SIGNIFICATIVA 
stats.f_oneway(data['Piastrine'][data['Decesso'] == 0],
               data['Piastrine'][data['Decesso'] == 1]) #NON SIGNIFICATIVA 


# Creazione della variabile Mese
conditions = [
    (data['Giorni'] <= 30),
    (data['Giorni'] >=31) & (data['Giorni'] <= 60),
    (data['Giorni'] >=61) & (data['Giorni'] <= 90),
    (data['Giorni'] >=91) & (data['Giorni'] <= 120),
    (data['Giorni'] >=121) & (data['Giorni'] <= 150),
    (data['Giorni'] >=151) & (data['Giorni'] <= 180),
    (data['Giorni'] >=181) & (data['Giorni'] <= 210),
    (data['Giorni'] >=211) & (data['Giorni'] <= 240),
    (data['Giorni'] >=241) & (data['Giorni'] <= 270),
    (data['Giorni'] >=271)]
values = [1,2,3,4,5,6,7,8,9,10]
data['Mese'] = np.select(conditions, values)

#Grafico a barre di Mese per la variabile di risposta
plt.figure(figsize=(10,8))
sns.countplot(data=data, x='Mese',hue='Decesso',palette=colors)
plt.show()

#Point plot di Mese per la  variabile di risposta
plt.figure(figsize=(10,8))
sns.pointplot(data=data,x='Mese',y='Decesso',ci=None,hue='Genere',palette=colors)
plt.show()


#Test sull'indipendenza per le variabili qualitative in relazione con la variabile di risposta
chi2, p, dof, expected = chi2_contingency((pd.crosstab(data['Decesso'], data['Ipertensione']).values))
print (f'Chi-square Statistic Decesso-Ipertensione : {chi2} ,p-value: {p}')

chi2, p, dof, expected = chi2_contingency((pd.crosstab(data['Decesso'], data['Fumatore']).values))
print (f'Chi-square Statistic Decesso-Fumatore : {chi2} ,p-value: {p}')

chi2, p, dof, expected = chi2_contingency((pd.crosstab(data['Decesso'], data['Diabete']).values))
print (f'Chi-square Statistic Decesso-Diabete : {chi2} ,p-value: {p}')

chi2, p, dof, expected = chi2_contingency((pd.crosstab(data['Decesso'], data['Genere']).values))
print (f'Chi-square Statistic Decesso-Genere : {chi2} ,p-value: {p}')

chi2, p, dof, expected = chi2_contingency((pd.crosstab(data['Decesso'], data['Anemia']).values))
print (f'Chi-square Statistic Decesso-Anemia : {chi2} ,p-value: {p}')

chi2, p, dof, expected = chi2_contingency((pd.crosstab(data['Decesso'], data['Rischio']).values))
print (f'Chi-square Statistic Decesso-Rischio : {chi2} ,p-value: {p}')


#Scaling e Feature Extraction
quantitative = data[['Eta','Frazione_di_Eiezione','Creatinina','Sodio','Giorni','CPK','Piastrine']]
qualitative = data[['Genere','Fumatore','Diabete','Rischio']]
y = data['Decesso']
scaler = StandardScaler()
scaler.fit(quantitative)
quantitative_scale = scaler.transform(quantitative)


X_scale = pd.DataFrame(quantitative_scale)
X_scale = X_scale.join(qualitative)
X_scale=X_scale.rename(columns={0: 'Eta', 1: 'Frazione_di_Eiezione',
                                2: 'Creatinina',3: 'Sodio',4: 'Giorni',5:'CPK',6:'Piastrine','Genere':'Genere',
                                'Fumatore':'Fumatore','Diabete':'Diabete','Rischio':'Rischio'})

#Train test split 
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, random_state=1234, stratify=y, test_size=.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


#Feature Importance con Random Forests
classifier = RandomForestClassifier(random_state=1234,n_estimators=300)
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
score = metrics.accuracy_score(y_test,predictions)
print(score)

# Data frame per il plot
feature_importance = classifier.feature_importances_
features = ['Eta','Frazione_di_Eiezione','Creatinina','Sodio','Giorni','CPK','Piastrine','Genere','Fumatore','Diabete','Rischio']
data_plot = pd.DataFrame(features)
data_plot['feature_importance']=feature_importance
data_plot=data_plot.rename(columns={0:'Feature','feature_importance':'Importanza'})

#Bar plot importanza features
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,8))
sns.catplot(kind='bar',data=data_plot,y='Feature',x='Importanza',height=6,aspect=2,palette='Dark2')
plt.axvline(0.09, 0,color='black',linestyle='--')


# Si decide di prendere le prime 4 variabili : RISCHIO, GIORNI, FRAZIONE DI EIEZIONE E CREATININA
X_train = X_train[['Frazione_di_Eiezione','Rischio','Giorni','Creatinina']]
X_test = X_test[['Frazione_di_Eiezione','Rischio','Giorni','Creatinina']]

# ENSEMBLE

log_reg = LogisticRegression(random_state=1234)
knn = KNeighborsClassifier()
tree = DecisionTreeClassifier(random_state=1234)
rand_forest = RandomForestClassifier(random_state=1234)
svm = SVC(random_state=1234)
naive = GaussianNB()
gb = GradientBoostingClassifier(random_state=1234)


tree.fit(X_train, y_train)
pred_tree = tree.predict(X_test)

log_reg.fit(X_train, y_train)
pred_log_reg = log_reg.predict(X_test)

naive.fit(X_train, y_train)
pred_naive = naive.predict(X_test)

svm.fit(X_train, y_train)
pred_svm = svm.predict(X_test)

knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)

gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)

rand_forest.fit(X_train, y_train)
pred_rand_forest = rand_forest.predict(X_test)

#Matrici di confusione degli algoritmi
plt.figure(figsize=(10,8))
cm = metrics.confusion_matrix(y_test, pred_knn)
sns.heatmap(cm,annot=True,square=True,cmap='viridis')
plt.ylabel('ACTUAL LABEL')
plt.xlabel('PREDICTED LABEL')
plt.title('KNN CONFUSION MATRIX')
plt.show()

plt.figure(figsize=(10,8))
cm = metrics.confusion_matrix(y_test, pred_log_reg)
sns.heatmap(cm,annot=True,square=True,cmap='viridis')
plt.ylabel('ACTUAL LABEL')
plt.xlabel('PREDICTED LABEL')
plt.title('LOGISTIC CONFUSION MATRIX')
plt.show()

plt.figure(figsize=(10,8))
cm = metrics.confusion_matrix(y_test, pred_tree)
sns.heatmap(cm,annot=True,square=True,cmap='viridis')
plt.ylabel('ACTUAL LABEL')
plt.xlabel('PREDICTED LABEL')
plt.title('DECISION TREE CONFUSION MATRIX')
plt.show()

plt.figure(figsize=(10,8))
cm = metrics.confusion_matrix(y_test, pred_rand_forest)
sns.heatmap(cm,annot=True,square=True,cmap='viridis')
plt.ylabel('ACTUAL LABEL')
plt.xlabel('PREDICTED LABEL')
plt.title('RANDOM FOREST CONFUSION MATRIX')
plt.show()


plt.figure(figsize=(10,8))
cm = metrics.confusion_matrix(y_test, pred_svm)
sns.heatmap(cm,annot=True,square=True,cmap='viridis')
plt.ylabel('ACTUAL LABEL')
plt.xlabel('PREDICTED LABEL')
plt.title('SVM CONFUSION MATRIX')
plt.show()

plt.figure(figsize=(10,8))
cm = metrics.confusion_matrix(y_test, pred_naive)
sns.heatmap(cm,annot=True,square=True,cmap='viridis')
plt.ylabel('ACTUAL LABEL')
plt.xlabel('PREDICTED LABEL')
plt.title('NAIVE BAYES CONFUSION MATRIX')
plt.show()

plt.figure(figsize=(10,8))
cm = metrics.confusion_matrix(y_test, pred_gb)
sns.heatmap(cm,annot=True,square=True,cmap='viridis')
plt.ylabel('ACTUAL LABEL')
plt.xlabel('PREDICTED LABEL')
plt.title('GRADIENT BOOSTING CONFUSION MATRIX')
plt.show()



#Calcolo delle metriche usate
#RECALL 
recall_tree = metrics.recall_score(y_test,pred_tree)
recall_log_reg = metrics.recall_score(y_test,pred_log_reg)
recall_naive = metrics.recall_score(y_test,pred_naive)
recall_gb = metrics.recall_score(y_test,pred_gb)
recall_knn = metrics.recall_score(y_test,pred_knn)
recall_svm = metrics.recall_score(y_test,pred_svm)
recall_random_forest = metrics.recall_score(y_test,pred_rand_forest)

#ACCURACY
accuracy_score_tree = metrics.accuracy_score(y_test,pred_tree)
accuracy_score_log_reg = metrics.accuracy_score(y_test,pred_log_reg)
accuracy_score_naive = metrics.accuracy_score(y_test,pred_naive)
accuracy_score_gb = metrics.accuracy_score(y_test,pred_gb)
accuracy_score_knn = metrics.accuracy_score(y_test,pred_knn)
accuracy_score_svm = metrics.accuracy_score(y_test,pred_svm)
accuracy_score_random_forest = metrics.accuracy_score(y_test,pred_rand_forest)

#PRECISION
precision_score_tree = metrics.precision_score(y_test,pred_tree)
precision_score_log_reg = metrics.precision_score(y_test,pred_log_reg)
precision_score_naive = metrics.precision_score(y_test,pred_naive)
precision_score_gb = metrics.precision_score(y_test,pred_gb)
precision_score_knn = metrics.precision_score(y_test,pred_knn)
precision_score_svm = metrics.precision_score(y_test,pred_svm)
precision_score_random_forest = metrics.precision_score(y_test,pred_rand_forest)

#F1 SCORE
f1_score_tree = metrics.f1_score(y_test,pred_tree)
f1_score_log_reg = metrics.f1_score(y_test,pred_log_reg)
f1_score_naive = metrics.f1_score(y_test,pred_naive)
f1_score_gb = metrics.f1_score(y_test,pred_gb)
f1_score_knn = metrics.f1_score(y_test,pred_knn)
f1_svm = metrics.f1_score(y_test,pred_svm)
f1_random_forest = metrics.f1_score(y_test,pred_rand_forest)


# Bar plot della recall
recall_score = [recall_tree, recall_naive, recall_knn, recall_svm, recall_log_reg, recall_gb, recall_random_forest]
 
algorithm = ['DECISION TREE', 'NAIVE',  'KNN','SVM','LOGISTIC',  'GRADIENT BOOSTING', 'RANDOM FOREST', ]
data_plot = pd.DataFrame(recall_score,algorithm)

sns.catplot(data=data_plot, x=recall_score, y=algorithm, kind='bar',height=6,aspect=2,palette='Dark2')
plt.show()


#CURVE ROC



plt.figure(figsize=(10, 8))


models = [{
    'label': 'Logistic Regression',
    'model': LogisticRegression(),},{
    'label': 'Gradient Boosting',
    'model': GradientBoostingClassifier(),},
    {
    'label': 'SVM',
    'model': SVC(probability=True),},
    {
    'label': 'Naive Bayes',
    'model': GaussianNB(),},
    {
    'label': 'Decisiom Tree',
    'model': DecisionTreeClassifier(),},
    {
    'label': 'Random Forest',
    'model': RandomForestClassifier(),},
    {
    'label': 'KNN',
    'model': KNeighborsClassifier(),}]

for m in models:
    model = m['model'] 
    model.fit(X_train, y_train) 
    y_pred=model.predict(X_test) 

    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
    auc = metrics.roc_auc_score(y_test,model.predict(X_test))
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))

import matplotlib.style
import matplotlib as mpl 
matplotlib.style.use('seaborn-whitegrid')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificità(False Positive Rate)')
plt.ylabel('Sensitività(True Positive Rate)')
plt.title('Curve ROC')
plt.legend(loc="lower right", prop={'size': 13})
plt.show() 


#DECISION TREE PLOT
from sklearn.tree import export_graphviz

clf=DecisionTreeClassifier(random_state=1234,max_depth=3)
clf.fit(X_train,y_train)
predictions_clf = clf.predict(X_test)
score_clf = metrics.accuracy_score(y_test,predictions_clf)

plt.style.use('classic')
fig = plt.figure(figsize=(200,200))
plot_albero= tree.plot_tree(clf,feature_names=['Frazione_di_Eiezione','Creatinina','Giorni','Rischio'],
                   class_names=['Sopravvissuti','Deceduti'],filled=True)



# GRID SEARCH CON 5-FOLD CROSS VALIDATION

# IPERPARAMETRI
grid_log_reg = {'C': [0.01, 0.1, 1]}

grid_svm = [{'kernel': ['rbf'],'C': [0.1, 0.5, 1,10,100]},
            {'kernel': ['linear', 'poly'],  'C': [0.1, 0.5, 1, 10, 100]}]

grid_rand_forest = {
    'max_depth': [80, 90, 100],
    'max_features': [2,3,4,5],
    'n_estimators': [100,200,300,400,500]}


grid_knn = {'n_neighbors':[2,3,4,5,6,7,8,9,10,12,15,18,20,25], 'metric':['manhattan','euclidean']}

grid_tree = {'criterion':['gini','entropy'],'max_depth':[3,4,5,6,7,8,9,10,20,50,75,100]}

grid_gb = {"criterion": ["friedman_mse",  "mae"],
              "loss":["deviance","exponential"],
              "max_features":["log2","sqrt"],
              'learning_rate': [0.01,0.05,0.1],
              'max_depth': [3,4,5],
              'min_samples_leaf': [4,5,6],
              'subsample': [0.6,0.7,0.8],
              'n_estimators': [50,100,200]}


# 5 Fold cross validation con grid search

log_reg = LogisticRegression(random_state=1234)
knn = KNeighborsClassifier()
tree = DecisionTreeClassifier(random_state=1234)
rand_forest = RandomForestClassifier(random_state=1234)
svm = SVC(random_state=1234)
gb = GradientBoostingClassifier(random_state=1234)


log_reg_cv = GridSearchCV(log_reg, grid_log_reg, cv=5)
log_reg_cv.fit(X_train,y_train)
pred_log_reg_cv = log_reg_cv.predict(X_test)
score_log_reg_cv = log_reg_cv.score(X_test,y_test)



knn_cv = GridSearchCV(knn, grid_knn, cv=5)
knn_cv.fit(X_train,y_train)
pred_knn_cv = knn_cv.predict(X_test)
score_knn_cv = knn_cv.score(X_test,y_test)


svm_cv = GridSearchCV(svm, grid_svm, cv=5)
svm_cv.fit(X_train,y_train)
pred_svm_cv = svm_cv.predict(X_test)
score_svm_cv = svm_cv.score(X_test,y_test)

tree_cv = GridSearchCV(tree, grid_tree, cv=5)
tree_cv.fit(X_train,y_train)
pred_tree_cv = tree_cv.predict(X_test)
score_tree_cv = tree_cv.score(X_test,y_test)


rand_forest_cv = GridSearchCV(rand_forest, grid_rand_forest, cv=5)
rand_forest_cv.fit(X_train, y_train)
pred_rand_forest_cv = rand_forest_cv.predict(X_test)
score_rand_forest_cv = rand_forest_cv.score(X_test, y_test)


gb_cv = GridSearchCV(gb, grid_gb, cv=5)
gb_cv.fit(X_train, y_train)
pred_gb_cv = gb_cv.predict(X_test)
score_gb_cv = gb_cv.score(X_test, y_test)


# METRICHE DOPO LA GRID SEARCH
#recall
recall_log_reg_cv = metrics.recall_score(y_test,pred_log_reg_cv)
recall_svm_cv = metrics.recall_score(y_test,pred_svm_cv)
recall_random_forest_cv = metrics.recall_score(y_test,pred_rand_forest_cv)
recall_gb_cv = metrics.recall_score(y_test,pred_gb_cv)
recall_knn_cv = metrics.recall_score(y_test,pred_knn_cv)
recall_tree_cv = metrics.recall_score(y_test,pred_tree_cv)

#accuracy
accuracy_log_reg_cv = metrics.accuracy_score(y_test,pred_log_reg_cv)
accuracy_svm_cv = metrics.accuracy_score(y_test,pred_svm_cv)
accuracy_random_forest_cv = metrics.accuracy_score(y_test,pred_rand_forest_cv)
accuracy_gb_cv = metrics.accuracy_score(y_test,pred_gb_cv)
accuracy_knn_cv = metrics.accuracy_score(y_test,pred_knn_cv)
accuracy_tree_cv = metrics.accuracy_score(y_test,pred_tree_cv)

#PRECISION
precision_score_tree_cv = metrics.precision_score(y_test,pred_tree_cv)
precision_score_log_reg_cv = metrics.precision_score(y_test,pred_log_reg_cv)
precision_score_gb_cv = metrics.precision_score(y_test,pred_gb_cv)
precision_score_knn_cv = metrics.precision_score(y_test,pred_knn_cv)
precision_score_svm_cv= metrics.precision_score(y_test,pred_svm_cv)
precision_score_random_forest_cv = metrics.precision_score(y_test,pred_rand_forest_cv)

#F1 SCORE
f1_score_tree_cv = metrics.f1_score(y_test,pred_tree_cv)
f1_score_log_reg_cv = metrics.f1_score(y_test,pred_log_reg_cv)
f1_score_gb_cv = metrics.f1_score(y_test,pred_gb_cv)
f1_score_knn_cv= metrics.f1_score(y_test,pred_knn_cv)
f1_score_svm_cv = metrics.f1_score(y_test,pred_svm_cv)
f1_score_random_forest_cv = metrics.f1_score(y_test,pred_rand_forest_cv)


# Bar plot con tutte le metriche usate
accuracy = [accuracy_log_reg_cv,accuracy_knn_cv,
            accuracy_tree_cv, accuracy_random_forest_cv,
            accuracy_svm_cv, accuracy_gb_cv]

precision = [precision_score_log_reg_cv,precision_score_knn_cv,
            precision_score_tree_cv, precision_score_random_forest_cv,
            precision_score_svm_cv, precision_score_gb_cv]

recall = [recall_log_reg_cv,recall_knn_cv,
            recall_tree_cv, recall_random_forest_cv,
            recall_svm_cv, recall_gb_cv]

f1  = [f1_score_log_reg_cv,f1_score_knn_cv,
            f1_score_tree_cv, f1_score_random_forest_cv,
            f1_score_svm_cv, f1_score_gb_cv]


algorithm = ['LOGISTIC CV','KNN CV', 'DECISION TREE CV', 'RANDOM FOREST CV',
             'SVM CV', 'GRADIENT BOOSTING CV']

data_plot = pd.DataFrame(accuracy,index=algorithm,columns=['Accuracy'])
data_plot['Recall'] = recall
data_plot['Precision'] = precision
data_plot['F1 Score'] = f1
data_plot['Algoritmi'] = algorithm

data_plot = pd.melt(data_plot, id_vars = "Algoritmi")

plt.figure(figsize=(15,13))
colors = ["#450085", "#00846b"]
sns.set_palette(sns.color_palette(colors))
sns.set_theme(style="whitegrid")
sns.barplot(y = "Algoritmi", x='value', hue = 'variable',data=data_plot, orient="h",palette='viridis')
plt.show()


#MIGLIORI IPERPARAMETRI PER GRADIENT BOOSTING (verrà usato in seguito)
classifier = gb_cv.best_estimator_



#Digressione sulla regressione logistica

import statsmodels.api as sm
from statsmodels.formula.api import logit


dataset = data[['Decesso','Eta','CPK','Frazione_di_Eiezione','Piastrine','Creatinina','Sodio','Giorni',
                'Genere','Anemia','Fumatore','Diabete','Ipertensione','Rischio']]

train, test = train_test_split(dataset, random_state=1234,stratify=dataset['Decesso'], test_size=.25)

formula = ('Decesso ~ Eta + Frazione_di_Eiezione + Giorni + Rischio + Sodio ')

model = logit(formula = formula, data=train).fit()

model.summary() 
