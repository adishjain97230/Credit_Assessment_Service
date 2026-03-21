from sklearn import model_selection, preprocessing, linear_model, metrics
import pandas as pd
from config import logging_config, switch_properties, constants
import numpy as np
from models import logistic_regression

logger = logging_config.get_logger(__name__)
module_properties = switch_properties.SWITCH_PROPERTIES[constants.models]

class SplitData:
    def __init__(self, df, test_size=0.2, random_state=42, validation_set=False, validation_size=0.2):

        y = df["y"]
        X = df.drop(columns=["y"])

        self.X = X
        self.y = y

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.validation_set = validation_set

        if (validation_set):
            self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(self.X_train, self.y_train, test_size=validation_size, random_state=42, stratify=self.y_train)

    
    def encodeCategoricalColumns(self):

        cat_cols = self.X.select_dtypes(include=["object", "string", "category"]).columns

        unique_values = {}

        for i in cat_cols:
            unique_values[i] = len(self.X[i].unique())
        
        ohe_columns_threshold = module_properties[constants.logistic_regression][constants.ohe_columns_threshold]
        
        ohe_cols = [i for i in cat_cols if unique_values[i] < ohe_columns_threshold]
        me_cols = [i for i in cat_cols if unique_values[i] >= ohe_columns_threshold]

        logistic_regression.logistic_regression_model[constants.ohe_cols] = ohe_cols
        logistic_regression.logistic_regression_model[constants.me_cols] = me_cols

        datasets = [self.X, self.X_train, self.X_test]
        if (self.validation_set):
            datasets.append(self.X_val)

        enc = preprocessing.OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
        enc.fit(self.X_train[ohe_cols].astype(str))
        ohe_cols_names = enc.get_feature_names_out(ohe_cols)
        logistic_regression.logistic_regression_model[constants.ohe_enc] = enc

        for i in range(len(datasets)):
            ohe_cols_df = enc.transform(datasets[i][ohe_cols].astype(str))
            ohe_cols_df = pd.DataFrame(ohe_cols_df, columns=ohe_cols_names, index=datasets[i].index)
            datasets[i] = pd.concat([datasets[i].drop(columns=ohe_cols), ohe_cols_df], axis=1).copy()
        
        if (self.validation_set):
            self.X_val = datasets[3].copy()
        
        self.X_train = datasets[1].copy()
        self.X_test = datasets[2].copy()
        self.X = datasets[0].copy()
        
        train_df = self.X_train.copy()
        train_df["__y__"] = self.y_train.values
        global_mean = float(self.y_train.mean())

        logistic_regression.logistic_regression_model[constants.global_mean] = global_mean

        mappings = {}

        for col in me_cols:
            mapping = train_df.groupby(col)["__y__"].mean()
            counts = train_df.groupby(col).size()
            mapping = mapping.where(counts >= 30, other=global_mean)
            mappings[col] = mapping
            for i in range(len(datasets)):
                datasets[i] = datasets[i].copy()
                datasets[i][f"{col}_enc"] = datasets[i][col].map(mapping).fillna(global_mean).astype(float)
                datasets[i] = datasets[i].drop(columns=col)
        
        logistic_regression.logistic_regression_model[constants.me_mappings] = mappings
        
        self.X = datasets[0].copy()
        self.X_train = datasets[1].copy()
        self.X_test = datasets[2].copy()

        if (self.validation_set):
            self.X_val = datasets[3].copy()

    def standardizeColumns(self):

        scaler = preprocessing.StandardScaler()
        logistic_regression.logistic_regression_model[constants.all_cols] = self.X_train.columns.to_list()
        self.X_train = scaler.fit_transform(self.X_train)
        logistic_regression.logistic_regression_model[constants.scaler] = scaler
        self.X_test = scaler.transform(self.X_test)
        self.X = scaler.transform(self.X)

        if (self.validation_set):
            self.X_val = scaler.transform(self.X_val)
    
    def getBestModelWithC(self):
        best_model = None
        best_auc = 0
        best_c = 0
        for c in [0.01, 0.1, 1.0, 10.0]:
            model = linear_model.LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
            model.fit(self.X_train, self.y_train)
            p = model.predict_proba(self.X_val)[:, 1]
            auc = metrics.roc_auc_score(self.y_val, p)
            if (auc > best_auc):
                best_model = model
                best_auc = auc
                best_c = c
        
        logger.info("Selected Model with C={best_c:>6}  AUC={best_auc:.4f}")
        return best_model

    def getBestThreshold(self, model):
        p_default_val = model.predict_proba(self.X_val)[:, 1]
        thresholds = np.arange(0.05, 0.96, 0.05)
        best_f1 = 0
        for t in thresholds:
            y_val_pred = (p_default_val >= t).astype(int)
            f1 = metrics.f1_score(self.y_val, y_val_pred, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        
        return best_t