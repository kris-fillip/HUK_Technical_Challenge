import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from xgboost import XGBClassifier
import lightgbm as lgb
import joblib


def get_and_combine_data():
    age_gender_df = pd.read_csv("./data/alter_geschlecht.csv", delimiter=',')

    interest_df = pd.read_csv("./data/interesse.csv", delimiter=',')

    rest_df = pd.read_csv("./data/rest.csv", delimiter=';')

    combined_df = pd.merge(age_gender_df, rest_df, on="id", how="inner")
    combined_df = pd.merge(combined_df, interest_df, on="id", how="inner")
    return combined_df


def handle_gender(data):
    data_one_hot = pd.get_dummies(data['Geschlecht']).astype(int)
    data = pd.concat([data, data_one_hot], axis=1)
    return data


def handle_age(data):
    bins = [19, 29, 39, 49, 59, 69, 79, 89]
    labels = ["Age_20-29", "Age_30-39", "Age_40-49",
              "Age_50-59", "Age_60-69", "Age_70-79", "Age_80-89"]

    data['Altersgruppen'] = pd.cut(
        data['Alter'], bins=bins, labels=labels)

    data_one_hot = pd.get_dummies(data['Altersgruppen']).astype(int)
    data = pd.concat([data, data_one_hot], axis=1)

    return data


def handle_car_age(data):
    data_one_hot = pd.get_dummies(data['Alter_Fzg']).astype(int)
    data_one_hot.columns = data_one_hot.columns.str.replace(
        ' ', '_').str.replace('>', 'gt').str.replace('<', 'lt')
    data = pd.concat([data, data_one_hot], axis=1)
    return data


def handle_pre_damage(data):
    data_one_hot = pd.get_dummies(
        data['Vorschaden'], prefix='Vorschaden').astype(int)
    data = pd.concat([data, data_one_hot], axis=1)
    return data


def handle_yearly_premium(data):
    data['Dominant_Premium'] = (data['Jahresbeitrag'] == 2630.0).astype(int)
    data['Jahresbeitrag_Log'] = np.log1p(data['Jahresbeitrag'])
    return data


def handle_sales_channel(data):
    top_5_sales_channels = data['Vertriebskanal'].value_counts().nlargest(
        5).index
    one_hot_encoded = pd.get_dummies(
        data['Vertriebskanal'], prefix='Vertriebskanal').astype(int)

    top_5_columns = [
        f'Vertriebskanal_{channel}' for channel in top_5_sales_channels]
    data = pd.concat([data, one_hot_encoded[top_5_columns]], axis=1)
    return data


def handle_customer_loyalty(data):
    scaler = StandardScaler()
    data['Kundentreue_scaled'] = scaler.fit_transform(data[['Kundentreue']])
    return data


def handle_interest(data):
    data["Interesse"] = data["Interesse"].astype(int)
    return data


def preprocess_data(data):
    data = handle_gender(data)
    data = handle_age(data)
    # Fahrerlaubnis already binary
    # Vorversicherung already binary
    data = handle_car_age(data)
    data = handle_pre_damage(data)
    data = handle_yearly_premium(data)
    data = handle_sales_channel(data)
    data = handle_customer_loyalty(data)
    data = handle_interest(data)

    return data


def calculate_scale_pos_weight(data):
    # Calculate scale_pos_weight
    neg, pos = data.value_counts()
    scale_pos_weight = neg / pos
    return scale_pos_weight


def train(data, model_name):
    X = data.drop(columns=["Interesse", "Geschlecht", "Alter",
                  "Alter_Fzg", "Vorschaden", "Jahresbeitrag", "Altersgruppen"])
    y = data["Interesse"]

    target_encoder = ce.TargetEncoder(cols=['Regional_Code', 'Vertriebskanal'])

    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for train_idx, valid_idx in tqdm(strat_kfold.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # Apply target encoding to the training set
        X_train_encoded = target_encoder.fit_transform(X_train, y_train)

        # Apply the same encoding to the validation set (using the learned mappings from training)
        X_valid_encoded = target_encoder.transform(X_valid)

        # Handle unseen categories: Replace with global mean of the target in the training set
        global_mean = y_train.mean()
        X_valid_encoded = X_valid_encoded.fillna(global_mean)

        scale_pos_weight = calculate_scale_pos_weight(y_train)

        if model_name == "rf":
            # Random Forest Classifier
            model = RandomForestClassifier(
                random_state=42, class_weight='balanced')

        elif model_name == "lgbm":
            # LightGBM Classifier
            model = lgb.LGBMClassifier(
                random_state=42, scale_pos_weight=scale_pos_weight, n_estimators=70)
        elif model_name == "xgb":
            # XGBoost Classifier
            model = XGBClassifier(
                random_state=42, scale_pos_weight=scale_pos_weight)
        print(X_train_encoded.columns.tolist())
        model.fit(X_train_encoded, y_train)

        y_pred = model.predict(X_valid_encoded)

        accuracy_scores.append(accuracy_score(y_valid, y_pred))
        f1_scores.append(f1_score(y_valid, y_pred))
        precision_scores.append(precision_score(y_valid, y_pred))
        recall_scores.append(recall_score(y_valid, y_pred))

    print(f'Average Accuracy: {np.mean(accuracy_scores)}')
    print(f'Average F1-Score: {np.mean(f1_scores)}')
    print(f'Average Precision: {np.mean(precision_scores)}')
    print(f'Average Recall: {np.mean(recall_scores)}')
    return model


def save_model(model, model_name):
    joblib.dump(model, f"./models/{model_name}.pkl")


def main():
    data = get_and_combine_data()
    data = preprocess_data(data)
    model_name = "lgbm"
    model = train(data, model_name)
    save_model(model, model_name)


if __name__ == "__main__":
    main()
