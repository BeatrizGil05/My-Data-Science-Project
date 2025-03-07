import pandas as pd

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df

def clean_data(df):
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values 
    df = df.dropna()  
    
    return df

def create_features(df):
    # Age Group Feature
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 45, 65, 100], labels=['Young', 'Adult', 'Middle_Aged', 'Senior'])
    
    # Work Hours Category
    df['work_hours_category'] = df['hours_per_week'].apply(lambda x: 'Part-time' if x < 35 else ('Full-time' if x <= 50 else 'Overtime'))
    
    # Capital Gain/Loss Interaction Term
    df['net_capital'] = df['capital_gain'] - df['capital_loss']
    
    return df

def save_data(df, output_path):
    #Saves the processed dataset to a new CSV file.
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load the datasets
    train_data, test_data = load_data("train.csv", "test.csv")
    
    # Clean the datasets
    cleaned_train = clean_data(train_data)
    cleaned_test = clean_data(test_data)
    
    # Create new features
    processed_train = create_features(cleaned_train)
    processed_test = create_features(cleaned_test)
    
    # Save the new datasets
    save_data(processed_train, "processed_train.csv")
    save_data(processed_test, "processed_test.csv")
    
    print("Feature engineering complete. Processed train and test datasets saved.")
