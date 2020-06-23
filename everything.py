
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
                 onehotcols=None,
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
                 classifier='logistic',
                 dimreduction=False,
                 reducer='pca',
                 n_dims=2,
                 kernel='rbf'):
        if onehotcols is None:
            onehotcols = [0]
        dataset = pd.read_csv(filename)
        self.X = dataset.iloc[:, x_start:y_start].values
        self.y = dataset.iloc[:, y_start].values
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.x_offset = 0
        self.graph = graph
        self.kernel = kernel

        self.poly = poly
        self.degree = degree
        self.poly_reg = None
        self.X_poly = None

        self.x_sc = False
        self.y_sc = False
        self.xscalar = None
        self.yscalar = None

        if onehot:
            self._oneHotEncode(onehotcols=onehotcols)
        if missing_data:
            self._fillMissingData()
        if split:
            self._split(test_size=test_size)
        if feat_scale:
            self._featScale()
        if poly:
            self._polyScale()
        if y_scale:
            self._yScale()
        elif label:
            self._labelEncode()

        if dimreduction:
            dim = DimReducer()
            self.dimreducer = dim.getDimReducer(dimreducer=reducer, kernel=self.kernel, n_components=n_dims)
            if reducer == 'lda':
                self._fit_lda()
            else:
                self._fit_pca()
            if n_dims <= 2:
                self.graph = True

        if ml_type == 'regression':
            reg = Regressor()
            self.trainer = reg.getReg(regressor=regressor, kernel=self.kernel)
        elif ml_type == 'classifier':
            cls = Classifier()
            self.trainer = cls.getClassifier(classifier=classifier, kernel=self.kernel)
        self._fit()
        self.y_pred = self._getFullPrediction(self.X_test)

    def _fillMissingData(self):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.X[:, :])  # avoid string columns
        self.X[:, :] = imputer.transform(self.X[:, :])

    def _oneHotEncode(self, onehotcols):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), onehotcols)], remainder='passthrough')
        temp = np.array(ct.fit_transform(self.X))  # into np array format
        self.x_offset = len(temp[0]) - len(self.X[0]) + 1
        self.X = temp

    def _labelEncode(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()  # no array transform, y is a vector
        self.y = le.fit_transform(self.y)  # no/yes -> 0/1

    def _split(self, test_size):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=1)

    def _featScale(self):
        from sklearn.preprocessing import StandardScaler
        self.xscalar = StandardScaler()
        self.X_train[:, self.x_offset:] = self.xscalar.fit_transform(self.X_train[:, self.x_offset:])
        self.X_test[:, self.x_offset:] = self.xscalar.transform(self.X_test[:, self.x_offset:])
        self.x_sc = True

    def _polyScale(self):
        from sklearn.preprocessing import PolynomialFeatures
        self.poly_reg = PolynomialFeatures(degree=self.degree)
        self.X_poly = self.poly_reg.fit_transform(self.X_train)

    def _yScale(self):
        from sklearn.preprocessing import StandardScaler
        self.yscalar = StandardScaler()
        self.y_train = self.yscalar.fit_transform(self.y_train)
        self.y_test = self.yscalar.transform(self.y_test)
        self.y_sc = True

    def _fit(self):
        if self.poly:
            self.trainer.fit(self.X_poly, self.y_train)
        else:
            self.trainer.fit(self.X_train, self.y_train)

    def _fit_lda(self):
        self.dimreducer.fit_transform(self.X_train, self.y_train)
        self._dim_reduce_transform()

    def _fit_pca(self):
        self.dimreducer.fit_transform(self.X_train)
        self._dim_reduce_transform()

    def _dim_reduce_transform(self):
        self.dimreducer.transform(self.X_test)

    def getSinglePrediction(self, argArray):  # argArray is a 1d array of features
        # must be coded as onehot in advance
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

    def _getFullPrediction(self, X):
        if self.y_sc:
            return self.yscalar.inverse_transform(self.trainer.predict(X))
        else:
            return self.trainer.predict(X)

    def plotReg(self, X, y, highres=False, title='Regression Model', xlabel='feature', ylabel='output'):
        if self.graph:
            X_base, y_base = self._shapeInputX(X), self._shapeInputy(y)
            plt.scatter(X_base, y_base, color='red')
            if highres:
                X_grid = np.arange(min(X_base), max(X_base), 0.01)
                X_grid = X_grid.reshape((len(X_grid), 1))
                if self.poly:
                    X_poly_grid = self.poly_reg.fit_transform(X_grid)
                    plt.plot(X_grid, self._getFullPrediction(X_poly_grid), color='blue')
                else:
                    plt.plot(X_grid, self._getFullPrediction(X_grid), color='blue')
            else:
                if self.poly:
                    X_poly_base = self._shapeInputX(self.X_poly)
                    plt.plot(X_base, self._getFullPrediction(X_poly_base), color='blue')
                else:
                    plt.plot(X_base, self._getFullPrediction(X_base), color='blue')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
        else:
            print('too many features to graph')

    def _shapeInputX(self, X):
        if self.x_sc:
            X = self.xscalar.inverse_transform(X)
        return X

    def _shapeInputy(self, y):
        if self.y_sc:
            y = self.xscalar.inverse_transform(y)
        return y

    def plotCls(self, X, y, f1_index=0, f1_margin=10, f1_step=1, f2_index=1, f2_margin=10, f2_step=1,
                title='Classification Model', xlabel='feature 1', ylabel='feature 2'):
        if self.graph:
            from matplotlib.colors import ListedColormap
            X_set, y_set = self._shapeInputX(X), self._shapeInputy(y)
            X1, X2 = np.meshgrid(
                np.arange(start=X_set[:, f1_index].min() - f1_margin, stop=X_set[:, f1_index].max() + f1_margin,
                          step=f1_step),
                np.arange(start=X_set[:, f2_index].min() - f2_margin, stop=X_set[:, f2_index].max() + f2_margin,
                          step=f2_step))
            plt.contourf(X1, X2, self._getFullPrediction(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
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

    def k_folds_cross(self, cv=10):
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(estimator=self.trainer,
                                     X=self.X_train,
                                     y=self.y_train,
                                     cv=cv)  # 10 different folds
        print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
        print("Standard Deviation: {:.2f}".format(accuracies.std() * 100))

    def grid_search_cv(self, cv=10, parameters=None):
        from sklearn.model_selection import GridSearchCV
        #               regularization param values to test
        if parameters is None:
            parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
                          {'C': [0.25, 0.5, 0.75, 1], 'kernel': [self.kernel], 'gamma': [.1, .2, .3, .4, .5, .6, .7, .8, .9]}]
        grid_search = GridSearchCV(estimator=self.trainer, param_grid=parameters, scoring='accuracy', cv=cv, n_jobs=-1)
        grid_search.fit(X=self.X_train, y=self.y_train)
        best_acc = grid_search.best_score_
        best_param = grid_search.best_params_
        print("Best Accuracy: {:.2f} %".format(best_acc.mean() * 100))
        print("Best Params: ", best_param)


class Regressor:
    def __init__(self):
        self.kernel = None
        self.n_estimators = None

    def getReg(self, regressor='linear', kernel='rbf', n_estimators=10):
        self.kernel = kernel
        self.n_estimators = n_estimators
        method_name = "_" + regressor
        method = getattr(self, method_name, lambda: "Invalid Regressor")
        return method()

    def _linear(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        return regressor

    def _svr(self):
        from sklearn.svm import SVR
        regressor = SVR(kernel=self.kernel)
        return regressor

    def _dec_tree(self):
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state=0)
        return regressor

    def _rand_forest(self):
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators=self.n_estimators, random_state=0)
        return regressor

    # def _xgb(self):
    #     from xgboost import XGBRegressor
    #     regressor = XGBRegressor()
    #     return regressor


class Classifier:
    def __init__(self):
        self.kernel = None
        self.criterion = None
        self.n_estimators = None

    def getClassifier(self, classifier='logistic', kernel='rbf', criterion='entropy', n_estimators=10):
        self.kernel = kernel
        self.criterion = criterion
        self.n_estimators = n_estimators
        method_name = "_" + classifier
        method = getattr(self, method_name, lambda: "Invalid Classifier")
        return method()

    def _logistic(self):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0, C=1)
        return classifier

    def _knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
        return classifier

    def _svm(self):
        from sklearn.svm import SVC
        classifier = SVC(kernel=self.kernel, random_state=0)
        return classifier

    def _bayes(self):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB(priors=None, var_smoothing=1e-9)
        return classifier

    def _dec_tree(self):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(random_state=0, criterion=self.criterion)
        return classifier

    def _rand_forest(self):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0, criterion=self.criterion)
        return classifier

    # def _xgb(self):
    #     from xgboost import XGBClassifier
    #     classifier = XGBClassifier()
    #     return classifier


class DimReducer:
    def __init__(self):
        self.n_components = None
        self.kernel = None

    def getDimReducer(self, dimreducer='pca', kernel='rbf', n_components=2):
        self.n_components = n_components
        self.kernel = kernel
        method_name = "_" + dimreducer
        method = getattr(self, method_name, lambda: "Invalid Classifier")
        return method()

    def _pca(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_components)
        return pca

    def _lda(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components=self.n_components)
        return lda

    def _kernel_pca(self):
        from sklearn.decomposition import KernelPCA
        kpca = KernelPCA(n_components=self.n_components, kernel=self.kernel)
        return kpca

class clusterModeller:
    def __init__(self, filename,
                 x_start=0,
                 x_end=1,
                 kmeans=True,
                 init='k-means++',
                 method='ward',
                 metric='euclidean'):
        dataset = pd.read_csv(filename)
        self.X = dataset.iloc[:, [x_start, x_end]].values
        self.y = None
        self.kmeans = kmeans
        self.clusterer = None
        if self.kmeans:
            self.init = init
            self._wcss(self.init, maxclusters=10)
        else:
            self.method = method
            self.metric = metric
            self._dendrogram(method=self.method, metric=self.metric)

    def _wcss(self, init='k-means++', maxclusters=10):
        from sklearn.cluster import KMeans
        wcss = []
        for i in range(1, maxclusters+1):
            kmeans = KMeans(n_clusters=i, init=init, random_state=42)
            kmeans.fit(self.X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, maxclusters+1), wcss)
        plt.title('Elbow Method')
        plt.xlabel('number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    def _dendrogram(self, method='ward', metric='euclidean', xlabel='units'):
        import scipy.cluster.hierarchy as sch
        dendrogram = sch.dendrogram(sch.linkage(self.X, method=method, metric=metric))
        plt.title('Dendrogram')
        plt.xlabel(xlabel)
        plt.ylabel(metric + 'distances')
        plt.show()

    def kmean_model(self, n_clusters=3):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, init=self.init, random_state=42)
        self.y = kmeans.fit_predict(self.X)
        self.clusterer = kmeans

    def hc_model(self, n_clusters=3):
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters=n_clusters, affinity=self.metric, linkage=self.method)
        self.y = hc.fit_predict(self.X)
        self.clusterer = hc

    def plot(self, n_cluster=3, title='Clusters', xlabel='feature 1', ylabel='feature 2'):
        from random import random
        for i in range(n_cluster):
            r, g, b = random.random(), random.random(), random.random()
            color = (r, g, b)
            plt.scatter(self.X[self.y == i, 0], self.X[self.y == i, 1], s=100, c=color, label='Cluster' + str(i + 1))
        if self.kmeans:
            plt.scatter(self.clusterer.cluster_centers_[:, 0], self.clusterer.cluster_centers_[:, 1], s=300, c='yellow',
                        label='Centroids')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()


class AssociativeLearning:
    def __init__(self, filename, apriori=True, support=0.003, conf=0.2, lift=3, minlen=2, maxdelta=0):
        dataset = pd.read_csv(filename, header=0)
        transactions = []
        for i in range(0, len(dataset)):
            transactions.append([str(dataset.values[i, j]) for j in range(0, len(dataset[0]))])

        from apyori import apriori
        rules = apriori(transactions, min_support=support, min_confidence=conf, min_lift=lift,
                        min_length=minlen, max_length=minlen+maxdelta)
        self.results = list(rules)
        self.apriori = apriori
        self.printResults()

    def printResults(self, n_largest=10):
        if self.apriori:
            columns = ["LHS", "RHS", 'Support', 'Confidence', 'Lifts']
        else:
            columns = ["LHS", "RHS", 'Support']
        resultsInDataFrame = pd.DataFrame(self._inspect(), columns=columns)
        resultsInDataFrame.nlargest(n=n_largest, columns=columns[-1])
        print(resultsInDataFrame)

    def _inspect(self):
        lhs = [tuple(result[2][0][0])[0] for result in self.results]
        rhs = [tuple(result[2][0][1])[0] for result in self.results]
        support = [result[1] for result in self.results]
        if self.apriori:
            confidence = [result[2][0][2] for result in self.results]
            lifts = [result[2][0][3] for result in self.results]
            return list(zip(lhs, rhs, support, confidence, lifts))
        else:
            return list(zip(lhs, rhs, support))


class Reinforcement:
    def __init__(self, filename, ucb=True, N=1000, numChoices=10,
                 xlabel='feature1', ylabel='feature2'):
        self.dataset = pd.read_csv(filename)
        self.N = N
        self.d = numChoices
        self.ads_selected = []
        self.total_reward = 0
        if ucb:
            self.num_selections = [0] * self.d
            self.sum_rewards = [0] * self.d
            self._ucb()
        else:
            self.numbers_of_rewards_1 = [0] * self.d  # successes
            self.numbers_of_rewards_0 = [0] * self.d  # fails
            self._thompson()
        self._plotHistogram(xlabel=xlabel, ylabel=ylabel)

    def _ucb(self):
        import math
        for n in range(0, self.N):
            ad = 0
            max_upper_bound = 0
            for i in range(0, self.d):
                if self.num_selections[i]:
                    avg_reward = self.sum_rewards[i] / self.num_selections[i]
                    delta_i = math.sqrt(1.5 * math.log(n + 1) / self.num_selections[i])
                    ucb = avg_reward + delta_i
                else:
                    ucb = 1e400

                if ucb > max_upper_bound:
                    max_upper_bound = ucb
                    ad = i
            self.ads_selected.append(ad)
            self.num_selections[ad] += 1
            reward = self.dataset.values[n, ad]
            self.sum_rewards[ad] += reward
            self.total_reward += reward

    def _thompson(self):
        import random
        for n in range(0, self.N):
            ad = 0
            max_random = 0
            for i in range(0, self.d):
                random_beta = random.betavariate(self.numbers_of_rewards_1[i] + 1, self.numbers_of_rewards_0[i] + 1)
                if random_beta > max_random:
                    max_random = random_beta
                    ad = i
            self.ads_selected.append(ad)
            reward = self.dataset.values[n, ad]
            if reward:
                self.numbers_of_rewards_1[ad] += 1
            else:
                self.numbers_of_rewards_0[ad] += 1
            self.total_reward += reward

    def _plotHistogram(self, xlabel='feature1', ylabel='feature2'):
        plt.hist(self.ads_selected)
        plt.title('Histogram')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


class NaturalLanguageProcessing:
    def __init__(self, tsv, max_words=1000):
        self.dataset = pd.read_csv(tsv, delimiter='\t', quoting=3)
        self.corpus = None
        self.X = None
        self.y = None
        self.y_test = None
        self.y_pred = None
        import nltk
        nltk.download('stopwords')
        self._fillCorpus()
        self._bagOfWords(max=max_words)
        self._splitAndFit()
        
    def _fillCorpus(self):
        import re
        from nltk.corpus import stopwords
        from nltk.stem.porter import PorterStemmer
        self.corpus = []
        for i in range(0, 1000):
            review = re.sub('[^a-zA-Z]', ' ', self.dataset['Review'][i])  # replace punctuation with spaces
            review = review.lower()  # make it lower case
            review = review.split()  # split into separate words
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word) for word in review if not word in set(all_stopwords)]  # removing root of word
            #  to combine i.e. like and liked if not in stopwords, removing those
            review = ' '.join(review)  # join words with spaces in between
            self.corpus.append(review)

    def _bagOfWords(self, max=1000):
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=max)
        self.X = cv.fit_transform(self.corpus).toarray()
        self.y = self.dataset.iloc[:, -1].values

    def _splitAndFit(self):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=1/3, random_state=0)

        cls = Classifier()
        classifier = cls.getClassifier(classifier='bayes')
        classifier.fit(X_train, y_train)
        self.y_pred = classifier.predict(X_test)

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
        print(np.concatenate((self.y_pred, self.y_test)))
