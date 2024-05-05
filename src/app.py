from utils import db_connect
#engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pickle import dump  

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv')
df

df = df.drop_duplicates().reset_index(drop=True)

X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train

plt.figure(figsize=(15,15))
pd.plotting.parallel_coordinates(df,'Outcome',colormap='viridis')

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

fig = plt.figure(figsize=(15,15))
tree.plot_tree(model,feature_names=list(X_train.columns),filled=True)
plt.show()

y_pred = model.predict(X_test)

accuracy_score(y_test,y_pred)

hyperparametros = {
    'criterion':['gini','entropy'],
    'max_depth':[None,5,10,20],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

grid = GridSearchCV(model,hyperparametros,scoring='accuracy',cv=5)
grid


#Ajustamos la grilla
grid.fit(X_train,y_train)
print(f'Mejores parámetros: {grid.best_params_}')


#Crear el modelo
mejor_modelo = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_leaf=2,min_samples_split=2,random_state=42)
mejor_modelo.fit(X_train,y_train)
y_pred = mejor_modelo.predict(X_test)
accuracy_score(y_pred,y_test)

dump(mejor_modelo, open("../models/mejor_modelo_arbol_decision.sav","wb"))