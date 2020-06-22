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
                 ml_type='regression',
                 regressor='linear',
                 classifier='logistic'):
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

        if ml_type == 'regression':
            reg = Regressor()
            self.trainer = reg.getReg(regressor=regressor)
        elif ml_type == 'classifier':
            cls = Classifier()
            self.trainer = cls.getClassifier(classifier=classifier)
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
            self.trainer.fit(self.X_poly, self.y_train)
        else:
            self.trainer.fit(self.X_train, self.y_train)

        # must be coded as onehot in advance
    def getSinglePrediction(self, argArray):  # args is a 1d array of features
        args = [argArray]
        if self.poly:
            args = self.poly_reg.fit_transform(args)
        if self.x_sc & self.y_sc:
            # scale_X input, then inv_scale_y the predicted output
            return self.yscalar.inverse_transform(self.trainer.predict(self.xscalar.transform(args)))
        elif self.x_sc:
            return self.trainer.predict(self.xscalar.transform(args))
        elif self.y_sc:
            return self.yscalar.inverse_transform(self.trainer.predict(args))
        else:
            return self.trainer.predict(args)

    def getFullPrediction(self, X):
        if self.x_sc & self.y_sc:
            # scale_X input, then inv_scale_y the predicted output
            return self.yscalar.inverse_transform(self.trainer.predict(self.xscalar.transform(X)))
        elif self.x_sc:
            return self.trainer.predict(self.xscalar.transform(X))
        elif self.y_sc:
            return self.yscalar.inverse_transform(self.trainer.predict(X))
        else:
            return self.trainer.predict(X)

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

    def confusionMatrix(self):
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(self.y_test, self.y_pred)
        # print(cm)  # [[#neg , #falsePos],[#falseneg, #Pos]]
        acc = accuracy_score(self.y_test, self.y_pred)
        # print(acc)
        return cm, acc

    def printComparedOutputs(self):
        print(np.concatenate((self.y_pred.reshape(len(self.y_pred), 1),
                              self.y_test.reshape(len(self.y_test), 1)), 1))


class Regressor:
    def __init__(self):
        pass

    def getReg(self, regressor='linear',  kernel='rbf', n_estimators=10):
        self.kernel = kernel
        self.n_estimators = n_estimators
        method = getattr(self, regressor, lambda: "Invalid Regressor")
        return method()

    def linear(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        return regressor

    def svr(self):
        from sklearn.svm import SVR
        regressor = SVR(kernel=self.kernel)
        return regressor

    def dec_tree(self):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state=0)
        return regressor

    def rand_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
        return regressor

class Classifier:
    def __init__(self):
        pass

    def getClassifier(self, classifier='logistic', kernel='rbf', criterion='entropy', n_estimators=10):
        self.kernel = kernel
        self.criterion = criterion
        self.n_estimators = n_estimators
        method = getattr(self, classifier, lambda: "Invalid Classifier")
        return method()

    def logistic(self):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0, C=1)
        return classifier

    def knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
        return classifier

    def svm(self):
        from sklearn.svm import SVC
        classifier = SVC(kernel=self.kernel, random_state=0)
        return classifier

    def bayes(self):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB(priors=None, var_smoothing=1e-9)
        return classifier

    def dec_tree(self):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(random_state=0, criterion=self.criterion)
        return classifier

    def rand_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0, criterion=self.criterion)
        return classifier

