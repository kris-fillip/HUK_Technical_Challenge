import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    neg, pos = data.value_counts()
    scale_pos_weight = neg / pos
    return scale_pos_weight


def train(data, model_name):
    X = data.drop(columns=["Interesse", "Geschlecht", "Alter",
                  "Alter_Fzg", "Vorschaden", "Jahresbeitrag", "Altersgruppen"])
    y = data["Interesse"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    target_encoding_cols = ['Regional_Code', 'Vertriebskanal']
    target_encoder = ce.TargetEncoder(cols=target_encoding_cols)

    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for train_idx, val_idx in tqdm(strat_kfold.split(X_train, y_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx].copy(
        ), X_train.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        X_train_fold.loc[:, target_encoding_cols] = target_encoder.fit_transform(
            X_train_fold[target_encoding_cols], y_train_fold)

        X_val_fold.loc[:, target_encoding_cols] = target_encoder.transform(
            X_val_fold[target_encoding_cols])

        overall_mean = y_train_fold.mean()
        X_val_fold.loc[:, target_encoding_cols] = X_val_fold[target_encoding_cols].fillna(
            overall_mean)

        scale_pos_weight = calculate_scale_pos_weight(y_train_fold)

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

        model.fit(X_train_fold, y_train_fold)
        y_val_pred = model.predict(X_val_fold)

        accuracy_scores.append(accuracy_score(y_val_fold, y_val_pred))
        precision_scores.append(precision_score(
            y_val_fold, y_val_pred, zero_division=1))
        recall_scores.append(recall_score(
            y_val_fold, y_val_pred, zero_division=1))
        f1_scores.append(f1_score(y_val_fold, y_val_pred, zero_division=1))

    print(
        f'cross-validation Accuracies: {[round(float(el), 4) for el in accuracy_scores]}')
    print(
        f'cross-validation F1-Scores: {[round(float(el), 4) for el in f1_scores]}')
    print(
        f'cross-validation Precisions: {[round(float(el), 4) for el in precision_scores]}')
    print(
        f'cross-validation Recalls: {[round(float(el), 4) for el in recall_scores]}')

    print(f'Average cross-validation Accuracy: {np.mean(accuracy_scores)}')
    print(f'Average cross-validation F1-Score: {np.mean(f1_scores)}')
    print(f'Average cross-validation Precision: {np.mean(precision_scores)}')
    print(f'Average cross-validation Recall: {np.mean(recall_scores)}')

    X_train.loc[:, target_encoding_cols] = target_encoder.fit_transform(
        X_train[target_encoding_cols], y_train)

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    if model_name == "rf":
        # Random Forest Classifier
        final_model = RandomForestClassifier(
            random_state=42, class_weight='balanced')

    elif model_name == "lgbm":
        # LightGBM Classifier
        final_model = lgb.LGBMClassifier(
            random_state=42, scale_pos_weight=scale_pos_weight, n_estimators=70)
    elif model_name == "xgb":
        # XGBoost Classifier
        final_model = XGBClassifier(
            random_state=42, scale_pos_weight=scale_pos_weight)

    final_model.fit(X_train, y_train)

    X_test.loc[:, target_encoding_cols] = target_encoder.transform(
        X_test[target_encoding_cols])

    overall_mean = y_train.mean()
    X_test.loc[:, target_encoding_cols] = X_test[target_encoding_cols].fillna(
        overall_mean)

    y_test_pred = final_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=1)
    test_recall = recall_score(y_test, y_test_pred, zero_division=1)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=1)

    print(f"Final model Accuracy on hold-out test set: {test_accuracy:.4f}")
    print(f"Final model F1 score on hold-out test set: {test_f1:.4f}")
    print(f"Final model Precision on hold-out test set: {test_precision:.4f}")
    print(f"Final model Recall on hold-out test set: {test_recall:.4f}")
    return final_model


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
