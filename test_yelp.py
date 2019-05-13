import simplejson as json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Oeffnen der JSON-Datei 'y_business.json':
with open('y_business.json', encoding='utf-8') as f:
    data = (line.strip() for line in f) 
    data_json = "[{0}]".format(','.join(data))

# Mit den Daten aus 'y_business.json' einen neuen DataFrame 'df_business' erzeugen:
data_js = json.loads(data_json)
df_business = pd.DataFrame.from_dict(data_js)

print("---------------------------------------------------------------------------------------------------")
print("Start:")
print(df_business.head())
print("")
print("-----------")
print("")
print("Info DF business:")
print(df_business.info(verbose = True, null_counts = True))
print("")
print("-----------")
print("")
print("-----------")
print(df_business.shape)
print("")
print("")

# Spalten, die fuer die Datenanalyse nicht verwendet werden aus 'df_business' loeschen (durch inplace = True):
df_business.drop(['address'], axis = 1, inplace = True)
df_business.drop(['business_id'], axis = 1, inplace = True)
df_business.drop(['city'], axis = 1, inplace = True)
df_business.drop(['latitude'], axis = 1, inplace = True)
df_business.drop(['longitude'], axis = 1, inplace = True)
df_business.drop(['hours'], axis = 1, inplace = True)
df_business.drop(['state'], axis = 1, inplace = True)
df_business.drop(['postal_code'], axis = 1, inplace = True)
df_business.drop(['name'], axis = 1, inplace = True)
df_business.drop(['review_count'], axis = 1, inplace = True)
df_business.drop(['is_open'], axis = 1, inplace = True)

# Alle Zeilen in 'df_business' loeschen, in denen der Eintrag 'nan' enthalten ist:
df_business.dropna(inplace = True)
df_business.reset_index(drop = True, inplace = True)

"""
print(df_business.head())
print("")
print("Info DF business:")
print(df_business.info(verbose = True, null_counts = True))
print("")
"""

# Geschaeftsbetriebe auswaehlen, die Restaurants sind.
# Nur Zeilen in 'df_business' auswaehlen, die Daten ueber Restaurants enthalten. Dabei wird in der Spalte 'categories', die vom Typ 'object' ist,
# nach dem String 'Restaurants' gesucht:
df_business = df_business[df_business.categories.str.contains('Restaurants')]
df_business.reset_index(drop=True, inplace=True)

"""
print(df_business.head())
print("")
print("Info DF business:")
print(df_business.info(verbose = True, null_counts = True))
print("")
"""

# ALle Features, welche die Sternebewertung eines Restaurants beeinflussen, muessen entpackt und danach 
# im DataFrame df_attributes'  gespeichert werden:
dict_attributes = [{'attributes': x} for x in df_business['attributes'].values]
df_attributes = pd.io.json.json_normalize(dict_attributes)

#pd.set_option('display.max_columns', 21)
#print("DF attributes:")
print(df_attributes.describe())
#print(df_attributes.head(3))
#print("Attributes shape = ", df_attributes.shape)
print("")

# In dieser Datenanalyse soll die Bewertung eines Restaurants in Abhaengigkeit der Features 'attributes.RestaurantsPriceRange2' ,
# 'attributes.GoodForKids' und 'attributes.RestaurantsGoodForGroups' untersucht werden. Dazu werden 'attributes.RestaurantsPriceRange2', 
# 'attributes.GoodForKids' und 'attributes.RestaurantsGoodForGroups' aus dem DataFrame 'df_attributes' in einem neuen
# DataFrame 'df_attributes_new' und in 'PriceRange', 'GoodForGroups' und 'GoodForKids' umbenannt:
attr_pricerange = df_attributes['attributes.RestaurantsPriceRange2']
attr_goodforkids = df_attributes['attributes.GoodForKids']
attr_goodforgroups = df_attributes['attributes.RestaurantsGoodForGroups']
df_attributes_new = pd.DataFrame(list(zip(attr_pricerange, attr_goodforgroups, attr_goodforkids)), columns = ['PriceRange', 'GoodForGroups', 'GoodForKids']) 



print("df_attributes: ")
#print(df_attributes_new.head())
print("")
print("Info DF business:")
#print(df_attributes_new.info(verbose = True, null_counts = True))
print("")

# Neuen DataFrame aus 'df_attributes_new' und 'df_business' zusammensetzen:
df_new = pd.concat([df_attributes_new, df_business], axis = 1)

# Jetzt ueberfluessige Spalten 'attributes' und 'categories' aus 'df_new' loeschen:
df_new.drop(['attributes'], axis = 1, inplace = True)
df_new.drop(['categories'], axis = 1, inplace = True)


# Alle Zeilen, in denen ein Eintrag 'na' ist loeschen und den Index von 'df_new' anpassen:
df_new.dropna(inplace = True)
df_new.reset_index(drop = True, inplace = True)


#print("DF df_new:")
#print(df_new.head(12))
print("Attributes shape = ", df_new.shape)
print("")
print("Info DF df_new:")
print(df_new.info(verbose = True, null_counts = True))
print("")

# Pruefen, ob es Eintraege mit 'nan' vorhanden sind:
#print("sum nan = ", np.sum(pd.isnull(df_new)))
#print("")

#print("DF df_new mit 'stars':")
#print(df_new.head(12))
print("Attributes shape = ", df_new.shape)
print("")
print("Info DF df_new:")
print(df_new.info(verbose = True, null_counts = True))
print("")


# Alle Zeilen in 'df_new' loeschen, in denen 'None' vorkommt und anschliessend wieder den Index von 'df_new' anpassen:
df_new = df_new[df_new['PriceRange'] != 'None']
df_new = df_new[df_new['GoodForGroups'] != 'None']
df_new = df_new[df_new['GoodForKids'] != 'None']
df_new.reset_index(drop = True, inplace = True)

# Spalten 'PriceRange' und 'stars' vom Datentyp 'object' in Datentyp 'int' konvertieren:
df_new['PriceRange'] = df_new['PriceRange'].astype('int64')
df_new['stars'] = df_new['stars'].astype('int64')


# Labels (Restaurant-Bewertungen) aus DataFrame 'df_new' extrahieren und danach die entsprechende Spalte in 'df_new' loeschen:
y = df_new['stars']
df_new.drop(['stars'], axis = 1, inplace = True)
print("")

#print("y = ")
#print(y)


print("DF df_new ohne 'stars':")
#print(df_new.head(12))
print("Attributes shape = ", df_new.shape)
print("")
print("Info DF df_new:")
print(df_new.info(verbose = True, null_counts = True))
print("")


# Neuen DataFrame 'df_enc' erzeugen und dabei die in 'df_new' beinhalteten Daten vom 
# Typ 'object' mit one-hot-encoding transformieren:
df_enc = pd.get_dummies(df_new)


#print("DF df_new:")
#print(df_enc.head(37))
print("Attributes shape = ", df_new.shape)
print("")
print("Info DF df_new:")
print(df_enc.info(verbose = True, null_counts = True))
print("")


# Erzeugung der Features-Matrix X aus dem DataFrame 'df_enc':
X = df_enc.values

print(X)
print("")


from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Daten aus X und y in Trainings- und Testdatensatz aufteilen:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.25)


print("X_train shape = ", X_train.shape)
print("y_train shape = ", y_train.shape)
print("X_testshape = ", X_test.shape)
print("y_test shape = ", y_test.shape)
print("")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score



# -----------------------------------------------------------
# Code fuer das RandomForestClassifier-Modell:

forest = RandomForestClassifier(random_state = 1)

param_g = {'n_estimators': [1000, 1500, 2000],  'max_features': [2, 3], 'max_depth': [10, 20, 30]}
grid = RandomizedSearchCV(forest, param_distributions = param_g, n_iter = 15, scoring = "accuracy",  cv = 5, n_jobs = -1)
grid.fit(X_train, y_train)
grid.predict(X_train)
print("Best Grid  score = ", grid.best_score_)
print("Best Grid  params = ", grid.best_params_)
print("")


# Mit den via RandomizedSearchCV gefundenen Parametern neuen RandomForestClassifier mit dem gesamten Trainingsdatensatz trainieren:
n_estimators_f = grid.best_params_["n_estimators"]
max_features_f = grid.best_params_["max_features"]
max_depth_f = grid.best_params_["max_depth"]

forest_final = RandomForestClassifier(n_estimators = n_estimators_f, max_depth = max_depth_f, max_features = max_features_f, random_state = 1, n_jobs = -1)
forest_final.fit(X_train, y_train)

print("Random Forest:")
print("Training Set Accuracy: {:.3f}".format(forest_final.score(X_train, y_train)))
print("Test Set Accuracy: {:.3f}".format(forest_final.score(X_test, y_test)))
print("")
# ----------------------------------------------------------



"""
# ---------------------------------------------------------
# Code fuer das GradientBoostingClassifier-Modell
gbc =  GradientBoostingClassifier(random_state = 42)

param_g = {'n_estimators': [1000, 1500, 2000],  'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [2, 3, 4, 5]}
grid = RandomizedSearchCV(gbc, param_distributions = param_g, n_iter = 20, scoring = "accuracy",  cv = 5, n_jobs = -1)
grid.fit(X_train, y_train)
grid.predict(X_train)
print("Best Grid  score = ", grid.best_score_)
print("Best Grid  params = ", grid.best_params_)
print("")

# Mit den via RandomizedSearchCV gefundenen Parametern neuen GradientBoostingClassifier mit dem gesamten Trainingsdatensatz trainieren:
n_estimators_f = grid.best_params_["n_estimators"]
learning_rate_f = grid.best_params_["learning_rate"]
max_depth_f = grid.best_params_["max_depth"]


gbc_final = GradientBoostingClassifier(n_estimators = n_estimators_f, learning_rate = learning_rate_f, max_depth = max_depth_f, random_state = 42)
gbc_final.fit(X_train, y_train)


print("GradientBoostingClassifier:")
print("Training Set Accuracy: {:.3f}".format(gbc_final.score(X_train, y_train)))
print("Test Set Accuracy: {:.3f}".format(gbc_final.score(X_test, y_test)))
print("")
# ------------------------------------------------------------

"""

