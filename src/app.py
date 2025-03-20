from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from pickle import dump

datos = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv", sep = ";")

datos = datos.drop_duplicates().reset_index(drop = True)

datos.isnull().sum()

datos["job_n"] = pd.factorize(datos["job"])[0]
datos["marital_n"] = pd.factorize(datos["marital"])[0]
datos["education_n"] = pd.factorize(datos["education"])[0]
datos["default_n"] = pd.factorize(datos["default"])[0]
datos["housing_n"] = pd.factorize(datos["housing"])[0]
datos["loan_n"] = pd.factorize(datos["loan"])[0]
datos["contact_n"] = pd.factorize(datos["contact"])[0]
datos["month_n"] = pd.factorize(datos["month"])[0]
datos["day_of_week_n"] = pd.factorize(datos["day_of_week"])[0]
datos["poutcome_n"] = pd.factorize(datos["poutcome"])[0]
datos["y_n"] = pd.factorize(datos["y"])[0]
num_variables = ["job_n", "marital_n", "education_n", "default_n", "housing_n", "loan_n", "contact_n", "month_n", "day_of_week_n", "poutcome_n",
                 "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y_n"]

scaler = MinMaxScaler()
scal_features = scaler.fit_transform(datos[num_variables])
datos_scal = pd.DataFrame(scal_features, index = datos.index, columns = num_variables)

X = datos_scal.drop("y_n", axis = 1)
y = datos_scal["y_n"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

selection_model = SelectKBest(chi2, k = 5)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel["y_n"] = list(y_train)
X_test_sel["y_n"] = list(y_test)
X_train_sel.to_csv("data/processed/train_limpio.csv", index = False)
X_test_sel.to_csv("data/processed/test_limpio.csv", index = False)

datos_train = pd.read_csv("data/processed/train_limpio.csv")
datos_test = pd.read_csv("data/processed/test_limpio.csv")

X_train = datos_train.drop(["y_n"], axis = 1)
y_train = datos_train["y_n"]
X_test = datos_test.drop(["y_n"], axis = 1)
y_test = datos_test["y_n"]

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

hiperparametros = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grilla = GridSearchCV(model, hiperparametros, scoring = "accuracy", cv = 10)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

grilla.fit(X_train, y_train)

modelo = LogisticRegression(C = 0.1, penalty = "l2", solver = "lbfgs")
modelo.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

dump(model, open("models/logistic_regression_C-0.1_penalty-l2_solver-lbfgs_42.sav", "wb"))