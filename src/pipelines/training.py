import os
import json
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_data(train_path='./../../data/train.csv', store_path='./../../data/store.csv'):
    train_data = pd.read_csv(train_path)
    store_data = pd.read_csv(store_path)
    return pd.merge(train_data, store_data, on='Store')

def preprocess_data(df):
    df = df[df['Open'] == 1]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days

    fill_values = {
        'CompetitionDistance': df['CompetitionDistance'].median(),
        'CompetitionOpenSinceMonth': 1,
        'CompetitionOpenSinceYear': df['Year'],
        'Promo2SinceWeek': 1,
        'Promo2SinceYear': df['Year'],
        'PromoInterval': 'None'
    }
    df.fillna(fill_values, inplace=True)

    df['Promo2Since'] = 52 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek'])
    df['Promo2Since'] = df['Promo2Since'].apply(lambda x: x if x > 0 else 0)

    df['CompetitionOpenSince'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: x if x > 0 else 0)

    return df

def engineer_features(df):
    store_group = df.groupby('Store')
    df.sort_values(by=['Store', 'Date'], inplace=True)
   
    df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 'PromoInterval'], drop_first=True, dtype=int)

    df['Sales_lag_1'] = store_group['Sales'].shift(1)
    df['Sales_lag_7'] = store_group['Sales'].shift(7)
    df['Sales_lag_14'] = store_group['Sales'].shift(14)
    df.dropna(subset=['Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14'], inplace=True)

    df['Sales_Mean_3'] = df.groupby('Store')['Sales'].rolling(window=3).mean().reset_index(0, drop=True)
    df['Sales_Mean_7'] = df.groupby('Store')['Sales'].rolling(window=7).mean().reset_index(0, drop=True)
    df['Sales_Mean_14'] = df.groupby('Store')['Sales'].rolling(window=14).mean().reset_index(0, drop=True)
    df.dropna(subset=['Sales_Mean_3', 'Sales_Mean_7', 'Sales_Mean_14'], inplace=True)

    df['IsBeforePromo'] = store_group['Promo'].shift(-1).fillna(0)
    df['IsAfterPromo'] = store_group['Promo'].shift(1).fillna(0)
    df['Sales_DOW_Mean'] = df.groupby(['Store', 'DayOfWeek'])['Sales'].transform('mean')
    df['Sales_DOW_Deviation'] = df['Sales'] - df['Sales_DOW_Mean']
    df['PromoDuringHoliday'] = ((df['Promo'] == 1) & (df['SchoolHoliday_1'] == 1)).astype(int)
    df['CompetitionIntensity'] = df['CompetitionDistance'] / (1 + df['CompetitionOpenSince'])

    month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    df['MonthStr'] = df['Month'].map(month_map)
    df['IsPromo2Active'] = 0
    
    promo_interval_cols = [col for col in df.columns if 'PromoInterval_' in col]
    for col in promo_interval_cols:
        interval_months = col.split('_')[1].split(',')
        for month_abbr in interval_months:
            df.loc[(df['Promo2'] == 1) & (df[col] == 1) & (df['MonthStr'] == month_abbr), 'IsPromo2Active'] = 1


    df.drop(columns=['MonthStr', 'Open', 'Customers'], inplace=True)


    return df

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=260,
        max_depth=7,
        learning_rate=0.068,
        subsample=0.73,
        colsample_bytree=0.98,
        gamma=0.42,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

#Entrainement avec DMatrix
def train_model_dmatrix(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle XGBoost en utilisant l’API bas-niveau avec DMatrix.
    Permet l’early stopping si un set de validation est fourni.
    """
    dtrain = xgb.DMatrix(X_train, label=y_train)    
    dtest  = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 7,
        'learning_rate': 0.068,
        'subsample': 0.73,
        'colsample_bytree': 0.98,
        'gamma': 0.42,
        'verbosity': 0
    }

    # 🔹 Liste de suivi pour l'évaluation
    watchlist = [(dtrain, "train"), (dtest, "eval")]

    # Entraînement du modèle
    print("🔄 Entraînement du modèle XGBoost avec DMatrix...")
    model = xgb.train(params, dtrain, num_boost_round=1500, evals=watchlist, early_stopping_rounds=50)
    return model

def evaluate_and_save(model, X_test, y_test, save_model=True, save_metrics=True, metrics_log_path='metrics_log.json'):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)    

    print(f"✅ Réentraînement terminé — RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    if save_model:
        joblib.dump(model, 'xgboost_model.pkl')
        print("📦 Modèle sauvegardé sous 'xgboost_model.pkl'")

    if save_metrics:
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'rmse': round(rmse, 2),
            'mae': round(mae, 2)
        }
        if os.path.exists(metrics_log_path):
            with open(metrics_log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(record)
        with open(metrics_log_path, 'w') as f:
            json.dump(logs, f, indent=2)
        print(f"📈 Métriques ajoutées à '{metrics_log_path}'")

def reentrainement_pipeline():
    df = load_data()
    df = preprocess_data(df)
    df = engineer_features(df)

    selected_features = [
    'Store', 'DayOfWeek', 'Promo', 'CompetitionDistance',
    'CompetitionOpenSinceMonth', 'Day','StoreType_b','StoreType_c','StoreType_d',
    'WeekOfYear', 'Promo2Since', 'CompetitionOpenSince', 'DaysSinceStart', 'IsPromo2Active',
    'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14', 'Sales_Mean_3', 'Sales_Mean_7', 'Sales_Mean_14',
    'CompetitionIntensity', 'PromoDuringHoliday', 'IsAfterPromo', 'IsBeforePromo', 'Sales_DOW_Mean', 'Sales_DOW_Deviation'
    ]

    X = df[selected_features]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model_dmatrix(X_train, y_train, X_test, y_test)
    evaluate_and_save(model, X_test, y_test)

# 🔁 Lancer la pipeline
reentrainement_pipeline()