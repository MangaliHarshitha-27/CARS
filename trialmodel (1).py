import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

def train(df):
    df=df[["brand","year","km_driven","fuel_type","price"]]

    df["brand"] = df["brand"].astype('category').cat.codes
    df["fuel_type"] = df["fuel_type"].astype('category').cat.codes

    X = df[["brand","year","km_driven","fuel_type"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

   models={
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor()
   }
   scores = {}
   for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = model.score(X_test, y_test)
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return best_model_name, scores[best_model_name]