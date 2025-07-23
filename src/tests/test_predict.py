from predict import predict_sales

def test_predict_sales():
    sample_input = {
        'DayOfWeek': 2, 'Promo': 1, 'CompetitionDistance': 500.0, 'CompetitionOpenSinceMonth': 6, 'Day': 15,
        'StoreType_b': 0, 'StoreType_c': 1, 'StoreType_d': 0, 'WeekOfYear': 22, 'Promo2Since': 50,
        'CompetitionOpenSince': 36, 'DaysSinceStart': 1000, 'IsPromo2Active': 1,
        'Sales_lag_1': 6000, 'Sales_lag_7': 6100, 'Sales_lag_14': 6200,
        'Sales_Mean_3': 6100, 'Sales_Mean_7': 6150, 'Sales_Mean_14': 6180,
        'CompetitionIntensity': 14.5, 'PromoDuringHoliday': 0,
        'IsAfterPromo': 0, 'IsBeforePromo': 1,
        'Sales_DOW_Mean': 6050, 'Sales_DOW_Deviation': 30
    }

    result = predict_sales(sample_input)
    assert isinstance(result, float)
    assert result > 0  # Vente attendue > 0
