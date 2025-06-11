import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def clean_data(df):
    # xoa cac gia tri  trung
    df = df.drop_duplicates()

    keep_columns = [
        'work_year', 'experience_level', 'employment_type', 'job_title',
        'salary_in_usd', 'employee_residence', 'remote_ratio',
        'company_location', 'company_size'
    ]
    
    df = df[keep_columns]

    return df

def encode_features(df, encoder: OneHotEncoder = None, fit_encoder=True):
    categorical_cols = [
        'experience_level', 'employment_type', 'job_title',
        'employee_residence', 'company_location', 'company_size'
    ]
    X_cat = df[categorical_cols]

    if fit_encoder:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_encoded = encoder.fit_transform(X_cat)
    else:
        X_encoded = encoder.transform(X_cat)

    encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

    # Thêm cột numeric vào
    numeric_cols = ['work_year', 'remote_ratio']
    X_numeric = df[numeric_cols]

    X = pd.concat([X_numeric, encoded_df], axis=1)
    return X, encoder

def select_features(df):
    X = df.drop('salary_in_usd', axis=1)
    y = df['salary_in_usd']
    return X, y
    
