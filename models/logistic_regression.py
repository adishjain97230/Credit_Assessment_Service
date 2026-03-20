from config import switch_properties, logging_config, constants
import pandas as pd
from models import split
from sklearn import linear_model, metrics, preprocessing
import numpy as np

logger = logging_config.get_logger(__name__)
module_properties = switch_properties.SWITCH_PROPERTIES[constants.models]

def main():
    logger.info("Logistic Regression Model")
    
    df = pd.read_parquet(module_properties[constants.dataset_path])

    split_data = split.SplitData(df, validation_set=True, validation_size=0.2, test_size=0.2)

    split_data.encodeCategoricalColumns()

    split_data.standardizeColumns()

    model = split_data.getBestModelWithC()

    threshold = split_data.getBestThreshold(model)

    p_default_test = model.predict_proba(split_data.X_test)[:, 1]
    y_pred = (p_default_test >= threshold).astype(int)

    logger.info("AUC-ROC: %f", metrics.roc_auc_score(split_data.y_test, p_default_test))
    logger.info("Accuracy: %f", metrics.accuracy_score(split_data.y_test, y_pred))
    logger.info("Precision: %f", metrics.precision_score(split_data.y_test, y_pred))
    logger.info("Recall: %f", metrics.recall_score(split_data.y_test, y_pred))
    logger.info("F1 Score: %f", metrics.f1_score(split_data.y_test, y_pred))
    logger.info("Confusion Matrix:\n%s", metrics.confusion_matrix(split_data.y_test, y_pred))
    logger.info("Classification Report:\n%s", metrics.classification_report(split_data.y_test, y_pred))


if __name__ == "__main__":
    main()