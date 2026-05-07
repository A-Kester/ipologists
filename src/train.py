from src.pipeline import DataPrepPipeline

def preprocess(df):
    df_clean = df.dropna(subset=['underpriced'])
    X_df = df_clean.drop(columns=[
        'underpriced', 'Offer To 1st Close',
        'Pricing Date', 'Issuer Name', 'ticker',
        'Primary Exchange', 'Instit Owner (% Shares Out)',
        'Industry Sector', 'lead_bookrunner'
    ])
    y_df = df_clean['underpriced']

    X_train_df, y_train_df, X_test_df, y_test_df = test_train_split(X_df, y_df)

    pipeline = DataPrepPipeline(X_train_df.columns.tolist())
    pipeline.fit(X_train_df)

    X_train = pipeline.transform(X_train_df)
    X_test  = pipeline.transform(X_test_df)
    y_train = y_train_df.values
    y_test  = y_test_df.values

    return X_train, y_train, X_test, y_test


def _test_train_split(X_df, y_df, train_size=0.8, random_state=42):
    train_ix = X_df.sample(frac=0.8, random_state=42).index
    test_ix = X_df.drop(train_ix).index

    X_train_df = X_df.loc[train_ix]
    y_train_df = y_df.loc[train_ix]

    X_test_df  = X_df.loc[test_ix]
    y_test_df  = y_df.loc[test_ix]

    return X_train_df, y_train_df, X_test_df, y_test_df