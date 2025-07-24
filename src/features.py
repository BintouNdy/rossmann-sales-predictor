import joblib
import pandas as pd
import os


FEATURES = [
    'DayOfWeek', 'Promo','Day', 'WeekOfYear',
    'Sales_Mean_3', 'Sales_Mean_7', 'Sales_Mean_14',
    'CompetitionIntensity', 'Sales_DOW_Mean', 'Sales_DOW_Deviation'
]


def get_features(df):
    """Select the best features for training"""
    selected_features = FEATURES
    print(f"🔍 Sélection des features: {selected_features}")
    # Check which features are actually available
    available_features = [f for f in selected_features if f in df.columns]
    missing_features = [f for f in selected_features if f not in df.columns]    
    if missing_features:
        print(f"⚠️  Features manquantes: {missing_features}")
    
    print(f"✅ Utilisation de {len(available_features)} features sur {len(selected_features)} sélectionnées")
    return available_features


def load_model():
    return joblib.load("models/xgboost_model1.pkl")

def load_data(train_path='data/train.csv', store_path='data/store.csv'):
    """Load and merge training data with store information"""
    # Get the absolute path from project root (src is one level down)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_full_path = os.path.join(project_root, train_path)
    store_full_path = os.path.join(project_root, store_path)
    
    print(f"📁 Chargement train data: {train_full_path}")
    print(f"📁 Chargement store data: {store_full_path}")
    
    train_data = pd.read_csv(train_full_path)
    store_data = pd.read_csv(store_full_path)
    return pd.merge(train_data, store_data, on='Store')


def preprocess_data(df):
    """Preprocess the raw data"""
    # Keep only open stores
    df = df[df['Open'] == 1]
    
    # Convert date and extract time features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    # Fill missing values
    fill_values = {
        'CompetitionDistance': df['CompetitionDistance'].median(),
        'CompetitionOpenSinceMonth': 1,
        'CompetitionOpenSinceYear': df['Year'],
        'Promo2SinceWeek': 1,
        'Promo2SinceYear': df['Year'],
        'PromoInterval': 'None'
    }
    df.fillna(fill_values, inplace=True)

    df['CompetitionOpenSince'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
    df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: x if x > 0 else 0)

    return df


def engineer_features(df):
    """Create engineered features"""
    # Sort data for lag features
    store_group = df.groupby('Store')
    df.sort_values(by=['Store', 'Date'], inplace=True)   

    # Create rolling mean features
    df['Sales_Mean_3'] = df.groupby('Store')['Sales'].rolling(window=3).mean().reset_index(0, drop=True)
    df['Sales_Mean_7'] = df.groupby('Store')['Sales'].rolling(window=7).mean().reset_index(0, drop=True)
    df['Sales_Mean_14'] = df.groupby('Store')['Sales'].rolling(window=14).mean().reset_index(0, drop=True)
    df.dropna(subset=['Sales_Mean_3', 'Sales_Mean_7', 'Sales_Mean_14'], inplace=True)

    # Create additional features    
    df['Sales_DOW_Mean'] = df.groupby(['Store', 'DayOfWeek'])['Sales'].transform('mean')
    df['Sales_DOW_Deviation'] = df['Sales'] - df['Sales_DOW_Mean']    
    df['CompetitionIntensity'] = df['CompetitionDistance'] / (1 + df['CompetitionOpenSince'])

    # Clean up unnecessary columns
    columns_to_drop = ['Open', 'StoreType', 'Assortment', 'StateHoliday', 'SchoolHoliday', 'PromoInterval']
    if 'Customers' in df.columns:
        columns_to_drop.append('Customers')
    
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    return df