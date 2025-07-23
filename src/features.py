import joblib


FEATURES = [
    'DayOfWeek', 'Promo', 'CompetitionDistance',
    'CompetitionOpenSinceMonth', 'Day','StoreType_b','StoreType_c','StoreType_d',
    'WeekOfYear', 'Promo2Since', 'CompetitionOpenSince', 'DaysSinceStart', 'IsPromo2Active',
    'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14', 'Sales_Mean_3', 'Sales_Mean_7', 'Sales_Mean_14',
    'CompetitionIntensity', 'PromoDuringHoliday', 'IsAfterPromo', 'IsBeforePromo', 'Sales_DOW_Mean', 'Sales_DOW_Deviation'
]

def get_features():
    return FEATURES

def load_model():
    return joblib.load("models/xgboost_model.pkl")