from sklearn import model_selection, preprocessing
import pandas as pd

class SplitData:
    def __init__(self, X, y, test_size=0.2, random_state=42, stratify=None):
        self.X = X
        self.y = y
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def encodeCategoricalColumns(self):

        cat_cols = self.X.select_dtypes(include=["object", "string", "category"]).columns

        unique_values = {}

        for i in cat_cols:
            unique_values[i] = len(self.X[i].unique())
        
        ohe_cols = [i for i in cat_cols if unique_values[i] < 20]
        me_cols = [i for i in cat_cols if unique_values[i] >= 20]


        datasets = [self.X, self.X_train, self.X_test]

        enc = preprocessing.OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        enc.fit(self.X[ohe_cols].astype(str))
        ohe_cols_names = enc.get_feature_names_out(ohe_cols)

        for i in range(len(datasets)):
            ohe_cols_df = enc.transform(datasets[i][ohe_cols].astype(str))
            ohe_cols_df = pd.DataFrame(ohe_cols_df, columns=ohe_cols_names, index=datasets[i].index)
            datasets[i] = pd.concat([datasets[i].drop(columns=ohe_cols), ohe_cols_df], axis=1).copy()
        
        train_df = self.X_train.copy()
        train_df["__y__"] = self.y_train.values
        global_mean = float(self.y_train.mean())

        for col in me_cols:
            datasets[i] = datasets[i].copy()
            mapping = train_df.groupby(col)["__y__"].mean()
            counts = train_df.groupby(col).size()
            mapping = mapping.where(counts >= 1000, other=global_mean)
            for i in range(len(datasets)):
                datasets[i][f"{col}_enc"] = datasets[i][col].map(mapping).fillna(global_mean).astype(float)
                datasets[i] = datasets[i].drop(columns=col)
        
        self.X = datasets[0].copy()
        self.X_train = datasets[1].copy()
        self.X_test = datasets[2].copy()
