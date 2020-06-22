import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=2)  # number of decimal places


class Dataset:
    def __init__(self, filename,
                 x_start=0,
                 y_start=-1,
                 missing_data=False,
                 onehot=False,
                 split=True,
                 test_size=0.2,
                 feat_scale=False,
                 label=False,
                 y_scale=False,
                 graph=False,
                 poly=False,
                 degree=1,
                 regressor='linear'):
        dataset = pd.read_csv(filename)
        self.X = dataset.iloc[:, x_start:y_start].values
        self.y = dataset.iloc[:, y_start].values
        self.X_train = self.X
        self.X_test = self.X
        self.y_train = self.y
        self.y_test = self.y
        self.y_pred = self.y_test
        self.x_offset = 0
        self.graph = graph

        from sklearn.preprocessing import PolynomialFeatures
        self.poly_reg = PolynomialFeatures(degree=degree)
        self.poly = poly
        self.X_poly = self.X_train

        from sklearn.preprocessing import StandardScaler
        self.x_sc = False
        self.y_sc = False
        self.xscalar = StandardScaler()
        self.yscalar = StandardScaler()

        if onehot:
            self.oneHotEncode()
        if missing_data:
            self.fillMissingData()
        if split:
            self.split(test_size=test_size)
        if feat_scale:
            self.featScale()
        if poly:
            self.polyScale()
        if y_scale:
            self.yScale()
        elif label:
            self.labelEncode()

        reg = Regressor()
        self.regressor = reg.getReg(regressor=regressor)
        self.fit()
        self.y_pred = self.getFullPrediction(self.X_test)

    def fillMissingData(self):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.X[:, :])  # avoid string columns
        self.X[:, :] = imputer.transform(self.X[:, :])

    def oneHotEncode(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        temp = np.array(ct.fit_transform(self.X))  # into np array format
        self.x_offset = len(temp[0]) - len(self.X[0]) + 1
        self.X = temp

    def labelEncode(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()  # no array transform, y is a vector
        self.y = le.fit_transform(self.y)  # no/yes -> 0/1

    def split(self, test_size):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=1)

    def featScale(self):
        self.X_train[:, self.x_offset:] = self.xscalar.fit_transform(self.X_train[:, self.x_offset:])
        self.X_test[:, self.x_offset:] = self.xscalar.transform(self.X_test[:, self.x_offset:])
        self.x_sc = True

    def polyScale(self):
        self.X_poly = self.poly_reg.fit_transform(self.X_train)

    def yScale(self):
        self.y_train = self.yscalar.fit_transform(self.y_train)
        self.y_test = self.yscalar.transform(self.y_test)
        self.y_sc = True

    def fit(self):
        if self.poly:
            self.regressor.fit(self.X_poly, self.y_train)
        else:
            self.regressor.fit(self.X_train, self.y_train)

        # must be coded as onehot in advance
    def getSinglePrediction(self, argArray):  # args is a 1d array of features
        args = [argArray]
        if self.poly:
            args = self.poly_reg.fit_transform(args)
        if self.x_sc & self.y_sc:
            # scale_X input, then inv_scale_y the predicted output
            return self.yscalar.inverse_transform(self.regressor.predict(self.xscalar.transform(args)))
        elif self.x_sc:
            return self.regressor.predict(self.xscalar.transform(args))
        elif self.y_sc:
            return self.yscalar.inverse_transform(self.regressor.predict(args))
        else:
            return self.regressor.predict(args)

    def getFullPrediction(self, X):
        if self.x_sc & self.y_sc:
            # scale_X input, then inv_scale_y the predicted output
            return self.yscalar.inverse_transform(self.regressor.predict(self.xscalar.transform(X)))
        elif self.x_sc:
            return self.regressor.predict(self.xscalar.transform(X))
        elif self.y_sc:
            return self.yscalar.inverse_transform(self.regressor.predict(X))
        else:
            return self.regressor.predict(X)

    def plotReg(self, X, y, highres=False, title='Regression Model', xlabel='feature', ylabel='output'):
        if self.graph:
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            X_base = self.shapeInputX(X)
            y_base = self.shapeInputy(y)
            plt.scatter(X_base, y_base, color='red')
            if highres:
                X_grid = np.arange(min(X_base), max(X_base), 0.01)  # higher resolution and smoother curve
                X_grid = X_grid.reshape((len(X_grid), 1))
                if self.poly:
                    X_poly_grid = self.poly_reg.fit_transform(X_grid)
                    plt.plot(X_grid, self.getFullPrediction(X_poly_grid), color='blue')
                else:
                    plt.plot(X_grid, self.getFullPrediction(X_grid), color='blue')
            else:
                if self.poly:
                    X_poly_base = self.shapeInputX(self.X_poly)
                    plt.plot(X_base, self.getFullPrediction(X_poly_base), color='blue')
                else:
                    plt.plot(X_base, self.getFullPrediction(X_base), color='blue')
            plt.show()
        else:
            print('too many features to graph')

    def shapeInputX(self, X):
        if self.x_sc:
            X = self.xscalar.inverse_transform(X)
        return X

    def shapeInputy(self, y):
        if self.y_sc:
            y = self.xscalar.inverse_transform(y)
        return y

    def getR2(self):
        from sklearn.metrics import r2_score
        return r2_score(self.y_test, self.y_pred)


class Regressor:
    def __init__(self, graph=False):
        self.graph = graph

    def getReg(self, regressor='linear'):
        method = getattr(self, regressor, lambda: "Invalid Regressor")
        return method()

    def linear(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        return regressor

    def svr(self):
        from sklearn.svm import SVR
        regressor = SVR(kernel='rbf')
        return regressor

    def dec_tree(self):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state=0)
        return regressor

    def rand_tree(self):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        return regressor
