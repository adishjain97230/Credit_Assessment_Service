from config import switch_properties, logging_config, constants
import pandas as pd
from models import split
from sklearn import linear_model, metrics, preprocessing
import numpy as np
import joblib
from data_cleaning import dataCleaning
from pathlib import Path

logger = logging_config.get_logger(__name__)
module_properties = switch_properties.SWITCH_PROPERTIES[constants.models]

def loadModel():
    if Path(module_properties[constants.logistic_regression][constants.model_path]):
        return joblib.load(module_properties[constants.logistic_regression][constants.model_path])
    else:
        return {}

logistic_regression_model = loadModel()

def oheEncode(df, model):
    ohe_df = model[constants.ohe_enc].transform(df[model[constants.ohe_cols]].astype(str))
    ohe_cols_names = model[constants.ohe_enc].get_feature_names_out(model[constants.ohe_cols])
    ohe_df = pd.DataFrame(ohe_df, columns=ohe_cols_names, index=df.index)
    return pd.concat([df.drop(columns=model[constants.ohe_cols]), ohe_df], axis=1).copy()

def meEncode(df, model):
    for col in model[constants.me_cols]:
        df[f"{col}_enc"] = df[col].map(model[constants.me_mappings][col]).fillna(model[constants.global_mean]).astype(float)
        df = df.drop(columns=col)
    
    return df

def predict(df, model = logistic_regression_model):
    df = dataCleaning.convertStringColumnsToNumeric(df)

    if (len(df) <= 0):
        logger.warning("Textual columns are not in correct format")
        return None, None, ValueError("Textual Columns are not in correct format")

    df = oheEncode(df, model)

    df = meEncode(df, model)[model[constants.all_cols]]

    df = model[constants.scaler].transform(df)

    return model[constants.model].predict_proba(df)[:, 1], model[constants.p_threshold], None


def main():
    logger.info("Logistic Regression Model")
    
    df = pd.read_parquet(module_properties[constants.dataset_path])

    split_data = split.SplitData(df, validation_set=True, validation_size=0.2, test_size=0.2)

    split_data.encodeCategoricalColumns()

    split_data.standardizeColumns()

    model = split_data.getBestModelWithC()

    logistic_regression_model[constants.model] = model

    threshold = split_data.getBestThreshold(model)

    logistic_regression_model[constants.p_threshold] = threshold

    p_default_test = model.predict_proba(split_data.X_test)[:, 1]
    y_pred = (p_default_test >= threshold).astype(int)

    logger.info("AUC-ROC: %f", metrics.roc_auc_score(split_data.y_test, p_default_test))
    logger.info("Accuracy: %f", metrics.accuracy_score(split_data.y_test, y_pred))
    logger.info("Precision: %f", metrics.precision_score(split_data.y_test, y_pred))
    logger.info("Recall: %f", metrics.recall_score(split_data.y_test, y_pred))
    logger.info("F1 Score: %f", metrics.f1_score(split_data.y_test, y_pred))
    logger.info("Confusion Matrix:\n%s", metrics.confusion_matrix(split_data.y_test, y_pred))
    logger.info("Classification Report:\n%s", metrics.classification_report(split_data.y_test, y_pred))

    joblib.dump(logistic_regression_model, module_properties[constants.logistic_regression][constants.model_path])


if __name__ == "__main__":
    main()