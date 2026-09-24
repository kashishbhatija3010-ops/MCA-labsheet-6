# MCA-labsheet-6
Mini Projects on Supervised Learning Applications
# Project 1: Student Performance Prediction System
# 1. Imports
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# 2. Load data 
student_df = pd.read_csv("student-mat.csv", sep=";")
print("Shape:", student_df.shape)
print(student_df.head())

# 3. Data understanding
student_df.info()
print("Missing values:", student_df.isnull().sum().sum())
print("Duplicates:", student_df.duplicated().sum())
print(student_df.describe().T)

# 4. Cleaning and descriptive names
student_df = student_df.rename(columns={
    "absences": "absences_count",           # attendance
    "G1": "internal_marks_g1",              # internal marks
    "G2": "previous_result_g2",             # previous semester result
    "studytime": "weekly_study_time_band",  # study hours
    "G3": "final_grade",                    # target
})
student_df = student_df.drop_duplicates().dropna().reset_index(drop=True)
print("After cleaning:", student_df.shape)

# 5. EDA
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(student_df["final_grade"], bins=20, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Final grade distribution")
sns.scatterplot(data=student_df, x="previous_result_g2", y="final_grade",
                ax=axes[0, 1], alpha=0.6)
axes[0, 1].set_title("Previous result (G2) vs final grade")
sns.boxplot(data=student_df, x="weekly_study_time_band", y="final_grade",
            ax=axes[1, 0])
axes[1, 0].set_title("Study time vs final grade")
sns.scatterplot(data=student_df, x="absences_count", y="final_grade",
                ax=axes[1, 1], alpha=0.6, color="tomato")
axes[1, 1].set_title("Absences vs final grade")
plt.tight_layout()
plt.show()

numeric_only = student_df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 9))
sns.heatmap(numeric_only.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            annot_kws={"size": 7})
plt.title("Correlation heatmap")
plt.show()

# 6. Preprocessing and split 
X_all = student_df.drop(columns=["final_grade"])
y_reg = student_df["final_grade"]

categorical_columns = X_all.select_dtypes(include="object").columns.tolist()
numeric_columns = X_all.select_dtypes(exclude="object").columns.tolist()

def build_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_reg, test_size=0.2, random_state=RANDOM_STATE)

# 7. Regression: compare 3 algorithms 
regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}
trained_regressors, rows = {}, []
for name, estimator in regression_models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2 (test)": r2_score(y_test, pred),
        "R2 (5-fold CV)": cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean(),
    })
    trained_regressors[name] = pipe

results_df = pd.DataFrame(rows).sort_values("R2 (test)", ascending=False).reset_index(drop=True)
print(results_df.round(3))
best_model_name = results_df.loc[0, "Model"]
best_regressor = trained_regressors[best_model_name]
print("Best regression model:", best_model_name)

# 8. Regression visualisation 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
results_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", ax=axes[0], rot=15)
axes[0].set_title("Error comparison (lower is better)")

best_pred = best_regressor.predict(X_test)
axes[1].scatter(y_test, best_pred, alpha=0.6)
axes[1].plot([0, 20], [0, 20], "r--")
axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
axes[1].set_title(f"Actual vs predicted ({best_model_name})")

sns.histplot(y_test - best_pred, kde=True, ax=axes[2], color="purple")
axes[2].set_title("Residuals")
plt.tight_layout()
plt.show()

# Feature importance (Random Forest)
rf_pipe = trained_regressors["Random Forest"]
names = rf_pipe.named_steps["prep"].get_feature_names_out()
importances = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names)
importances.sort_values().tail(12).plot(kind="barh", figsize=(9, 6), color="teal")
plt.title("Top 12 features (Random Forest)")
plt.show()

# 9. Classification: Pass (final_grade >= 10) / Fail
pass_label = (student_df["final_grade"] >= 10).astype(int)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_all, pass_label, test_size=0.2, random_state=RANDOM_STATE, stratify=pass_label)

classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
}
trained_classifiers, crow = {}, []
for name, estimator in classification_models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    pipe.fit(Xc_train, yc_train)
    pred = pipe.predict(Xc_test)
    crow.append({"Model": name,
                 "Accuracy": accuracy_score(yc_test, pred),
                 "Precision": precision_score(yc_test, pred),
                 "Recall": recall_score(yc_test, pred),
                 "F1-score": f1_score(yc_test, pred)})
    trained_classifiers[name] = pipe

class_df = pd.DataFrame(crow).sort_values("F1-score", ascending=False).reset_index(drop=True)
print(class_df.round(3))
best_classifier = trained_classifiers[class_df.loc[0, "Model"]]

print(classification_report(yc_test, best_classifier.predict(Xc_test),
                            target_names=["Fail", "Pass"]))
ConfusionMatrixDisplay(confusion_matrix(yc_test, best_classifier.predict(Xc_test)),
                       display_labels=["Fail", "Pass"]).plot(cmap="Blues")
plt.title("Confusion matrix")
plt.show()

# 10. Save models and predict
joblib.dump(best_regressor, "student_performance_regressor.joblib")
joblib.dump(best_classifier, "student_performance_classifier.joblib")

reg = joblib.load("student_performance_regressor.joblib")
clf = joblib.load("student_performance_classifier.joblib")
sample = X_test.iloc[[0]]
print("Predicted final grade:", round(float(reg.predict(sample)[0]), 2), "/ 20")
print("Actual final grade   :", y_test.iloc[0])
print("Predicted result     :", "Pass" if clf.predict(sample)[0] == 1 else "Fail")


# Project 2: House Price Prediction System
# 1. Imports
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# 2. Load data
house_df = pd.read_csv("Housing.csv")
print("Shape:", house_df.shape)
print(house_df.head())

# 3. Data understanding
house_df.info()
print("Missing values:", house_df.isnull().sum().sum())
print("Duplicates:", house_df.duplicated().sum())
print(house_df.describe().T)

# 4. Cleaning and preprocessing
house_df = house_df.drop_duplicates().dropna().reset_index(drop=True)

yes_no_columns = ["mainroad", "guestroom", "basement",
                  "hotwaterheating", "airconditioning", "prefarea"]
for column in yes_no_columns:
    house_df[column] = house_df[column].str.strip().str.lower().map({"yes": 1, "no": 0})

# Outlier removal on price using the IQR rule
q1, q3 = house_df["price"].quantile([0.25, 0.75])
upper_limit = q3 + 1.5 * (q3 - q1)
rows_before = len(house_df)
house_df = house_df[house_df["price"] <= upper_limit].reset_index(drop=True)
print(f"Removed {rows_before - len(house_df)} price outliers. Rows left: {len(house_df)}")

# 5. EDA 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(house_df["price"], bins=30, kde=True, ax=axes[0, 0], color="darkgreen")
axes[0, 0].set_title("Price distribution")
sns.scatterplot(data=house_df, x="area", y="price", hue="bathrooms",
                palette="viridis", ax=axes[0, 1])
axes[0, 1].set_title("Area vs price")
sns.boxplot(data=house_df, x="bedrooms", y="price", ax=axes[1, 0])
axes[1, 0].set_title("Bedrooms vs price")
sns.boxplot(data=house_df, x="prefarea", y="price", ax=axes[1, 1])
axes[1, 1].set_title("Preferred area (0 = no, 1 = yes) vs price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(house_df.select_dtypes(include=np.number).corr(),
            annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()

# 6. Split and preprocessing pipeline
X = house_df.drop(columns=["price"])
y = house_df["price"]

categorical_columns = X.select_dtypes(include="object").columns.tolist()
numeric_columns = X.select_dtypes(exclude="object").columns.tolist()

def build_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

# 7. Compare 6 regression algorithms
regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0, max_iter=10000),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}
trained_models, rows = {}, []
for name, estimator in regression_models.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2 (test)": r2_score(y_test, pred),
        "R2 (5-fold CV)": cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean(),
    })
    trained_models[name] = pipe

comparison_df = pd.DataFrame(rows).sort_values("R2 (test)", ascending=False).reset_index(drop=True)
print(comparison_df.round(3))

# 8. Hyperparameter tuning
tuning_pipe = Pipeline([("prep", build_preprocessor()),
                        ("model", RandomForestRegressor(random_state=RANDOM_STATE))])
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 6, 10],
    "model__min_samples_leaf": [1, 3],
}
grid = GridSearchCV(tuning_pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
tuned_forest = grid.best_estimator_
print("Best parameters:", grid.best_params_)
trained_models["Random Forest (tuned)"] = tuned_forest

# 9. Visualise results 
final_scores = {n: r2_score(y_test, m.predict(X_test)) for n, m in trained_models.items()}
best_model_name = max(final_scores, key=final_scores.get)
best_model = trained_models[best_model_name]
best_pred = best_model.predict(X_test)
print("Best model:", best_model_name, "| R2 =", round(final_scores[best_model_name], 3))
print("MAE :", round(mean_absolute_error(y_test, best_pred)))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, best_pred))))

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
pd.Series(final_scores).sort_values().plot(kind="barh", ax=axes[0], color="cornflowerblue")
axes[0].set_title("R2 on test set (higher is better)")
axes[1].scatter(y_test, best_pred, alpha=0.6)
lims = [y_test.min(), y_test.max()]
axes[1].plot(lims, lims, "r--")
axes[1].set_xlabel("Actual price"); axes[1].set_ylabel("Predicted price")
axes[1].set_title(f"Actual vs predicted ({best_model_name})")
sns.histplot(y_test - best_pred, kde=True, ax=axes[2], color="orange")
axes[2].set_title("Residuals")
plt.tight_layout()
plt.show()

feature_names = tuned_forest.named_steps["prep"].get_feature_names_out()
importances = pd.Series(tuned_forest.named_steps["model"].feature_importances_,
                        index=feature_names)
importances.sort_values().tail(12).plot(kind="barh", figsize=(9, 6), color="seagreen")
plt.title("Feature importance (tuned Random Forest)")
plt.show()

# 10. Save model and predict a new house
joblib.dump(best_model, "house_price_model.joblib")
loaded_model = joblib.load("house_price_model.joblib")

new_house = pd.DataFrame([{
    "area": 6000, "bedrooms": 3, "bathrooms": 2, "stories": 2,
    "mainroad": 1, "guestroom": 0, "basement": 1, "hotwaterheating": 0,
    "airconditioning": 1, "parking": 2, "prefarea": 1,
    "furnishingstatus": "semi-furnished",
}])
print("Predicted price: {:,.0f}".format(loaded_model.predict(new_house)[0]))
