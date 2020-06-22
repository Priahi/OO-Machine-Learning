import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=2)  # number of decimal places


class RegOrClassDataset:
    def __init__(self, filename,
                 x_start=0,
                 y_start=-1,
                 missing_data=False,
                 onehot=False,
                 onehotcols=[0],
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
            self.__oneHotEncode(onehotcols=onehotcols)
        if missing_data:
            self.__fillMissingData()
        if split:
            self.__split(test_size=test_size)
        if feat_scale:
            self.__featScale()
        if poly:
            self.__polyScale()
        if y_scale:
            self.__yScale()
        elif label:
            self.__labelEncode()

        if ml_type == 'regression':
            reg = Regressor()
            self.trainer = reg.getReg(regressor=regressor)
        elif ml_type == 'classifier':
            cls = Classifier()
            self.trainer = cls.getClassifier(classifier=classifier)
        self.__fit()
        self.y_pred = self.__getFullPrediction(self.X_test)

    def __fillMissingData(self):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.X[:, :])  # avoid string columns
        self.X[:, :] = imputer.transform(self.X[:, :])

    def __oneHotEncode(self, onehotcols):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), onehotcols)], remainder='passthrough')
        temp = np.array(ct.fit_transform(self.X))  # into np array format
        self.x_offset = len(temp[0]) - len(self.X[0]) + 1
        self.X = temp

    def __labelEncode(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()  # no array transform, y is a vector
        self.y = le.fit_transform(self.y)  # no/yes -> 0/1

    def __split(self, test_size):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=1)

    def __featScale(self):
        self.X_train[:, self.x_offset:] = self.xscalar.fit_transform(self.X_train[:, self.x_offset:])
        self.X_test[:, self.x_offset:] = self.xscalar.transform(self.X_test[:, self.x_offset:])
        self.x_sc = True

    def __polyScale(self):
        self.X_poly = self.poly_reg.fit_transform(self.X_train)

    def __yScale(self):
        self.y_train = self.yscalar.fit_transform(self.y_train)
        self.y_test = self.yscalar.transform(self.y_test)
        self.y_sc = True

    def __fit(self):
        if self.poly:
            self.trainer.__fit(self.X_poly, self.y_train)
        else:
            self.trainer.__fit(self.X_train, self.y_train)

        # must be coded as onehot in advance

    def getSinglePrediction(self, argArray):  # args is a 1d array of features
        args = [argArray]
        if self.poly:
            args = self.poly_reg.fit_transform(args)
        return self.__getFullPrediction(args)

    def __getFullPrediction(self, X):
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
            X_base, y_base = self.__shapeInputX(X), self.__shapeInputy(y)
            plt.scatter(X_base, y_base, color='red')
            if highres:
                X_grid = np.arange(min(X_base), max(X_base), 0.01)  # higher resolution and smoother curve
                X_grid = X_grid.reshape((len(X_grid), 1))
                if self.poly:
                    X_poly_grid = self.poly_reg.fit_transform(X_grid)
                    plt.plot(X_grid, self.__getFullPrediction(X_poly_grid), color='blue')
                else:
                    plt.plot(X_grid, self.__getFullPrediction(X_grid), color='blue')
            else:
                if self.poly:
                    X_poly_base = self.__shapeInputX(self.X_poly)
                    plt.plot(X_base, self.__getFullPrediction(X_poly_base), color='blue')
                else:
                    plt.plot(X_base, self.__getFullPrediction(X_base), color='blue')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
        else:
            print('too many features to graph')

    def __shapeInputX(self, X):
        if self.x_sc:
            X = self.xscalar.inverse_transform(X)
        return X

    def __shapeInputy(self, y):
        if self.y_sc:
            y = self.xscalar.inverse_transform(y)
        return y

    def plotCls(self, X, y, f1_index=0, f1_margin=10, f1_step=1, f2_index=1, f2_margin=10, f2_step=1,
                title='Classification Model', xlabel='feature 1', ylabel='feature 2'):
        if self.graph:
            from matplotlib.colors import ListedColormap
            X_set, y_set = self.__shapeInputX(X), self.__shapeInputy(y)
            X1, X2 = np.meshgrid(
                np.arange(start=X_set[:, f1_index].min() - f1_margin, stop=X_set[:, f1_index].max() + f1_margin,
                          step=f1_step),
                np.arange(start=X_set[:, f2_index].min() - f2_margin, stop=X_set[:, f2_index].max() + f2_margin,
                          step=f2_step))
            plt.contourf(X1, X2, self.__getFullPrediction(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                         alpha=0.75, cmap=ListedColormap(('red', 'green')))
            plt.xlim(X1.min(), X1.max())
            plt.ylim(X2.min(), X2.max())
            for i, j in enumerate(np.unique(y_set)):
                plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            plt.show()

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

    def getReg(self, regressor='linear', kernel='rbf', n_estimators=10):
        self.kernel = kernel
        self.n_estimators = n_estimators
        method_name = "__" + regressor
        method = getattr(self, method_name, lambda: "Invalid Regressor")
        return method()

    def __linear(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        return regressor

    def __svr(self):
        from sklearn.svm import SVR
        regressor = SVR(kernel=self.kernel)
        return regressor

    def __dec_tree(self):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state=0)
        return regressor

    def __rand_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
        return regressor


class Classifier:
    def __init__(self):
        self.kernel = 'rbf'
        self.criterion = 'entropy'
        self.n_estimators = 100

    def getClassifier(self, classifier='logistic', kernel='rbf', criterion='entropy', n_estimators=10):
        self.kernel = kernel
        self.criterion = criterion
        self.n_estimators = n_estimators
        method_name = "__" + classifier
        method = getattr(self, method_name, lambda: "Invalid Classifier")
        return method()

    def __logistic(self):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0, C=1)
        return classifier

    def __knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
        return classifier

    def __svm(self):
        from sklearn.svm import SVC
        classifier = SVC(kernel=self.kernel, random_state=0)
        return classifier

    def __bayes(self):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB(priors=None, var_smoothing=1e-9)
        return classifier

    def __dec_tree(self):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(random_state=0, criterion=self.criterion)
        return classifier

    def __rand_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0, criterion=self.criterion)
        return classifier
