import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz


class DataManager():
    def __init__(self, validation_size=0.25):
        assert(isinstance(validation_size, float))
        assert(0.0 < validation_size < 1.0)
        self.path = '../dataset/'
        self.__raw_training_data = self.__load_data('train')
        self.__raw_testing_data = self.__load_data('test')
        self.__raw_xtrain = self.__raw_training_data[self.__raw_training_data.columns[:-1]].values
        self.__raw_ytrain = self.__raw_training_data[self.__raw_training_data.columns[-1]].map({'A': 0, 'B': 1, 'C': 2, 'D': 3,
                                                                                                'E': 4}).astype(int).values
        self.validation_size = validation_size

    def __load_data(self, ds):
        '''
        Load dataset.

        Args:
            ds (str): dataset accepts the name 'train' or 'test'.

        Returns:
            pd.Dataframe()
        '''
        assert isinstance(ds, str)
        assert ds == 'train' or ds == 'test'

        file_name = 'pml-training.csv' if ds == 'train' else 'pml-testing.csv'
        data = pd.read_csv(self.path + file_name, dtype='unicode')

        # drop N/A and empty columns
        data.dropna(axis=1, inplace=True)

        # drop useless data
        drops = ['Unnamed: 0', 'user_name', 'raw_timestamp_part_1',
                 'raw_timestamp_part_2', 'cvtd_timestamp', 'new_window', 'num_window']

        if ds == 'test':
            drops.append('problem_id')

        return data.drop(columns=drops, axis=1)

    def get_train_and_valid(self):
        '''
        Split training data to training set and validation set.

        Returns:
            xtrain (np.ndarray): training samples
            ytrain (np.ndarray): training labels
            xvalid (np.ndarray): validation samples
            yvalid (np.ndarray): validation labels
        '''
        xtrain, xvalid, ytrain, yvalid = train_test_split(
            self.__raw_xtrain, self.__raw_ytrain, test_size=self.validation_size, random_state=1)

        return xtrain, ytrain, xvalid, yvalid

    def get_xtest(self):
        '''
        Get testing samples.

        Returns:
            xtest (nd.array)
        '''
        return self.__raw_testing_data.values


class RandomForest():
    def __init__(self, dm, n_estimators=50, criterion='entropy', max_depth=None, class_weight=None):
        self.__dm = dm
        self.__model = None
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.__xtrain, self.__ytrain = None, None
        self.__xvalid, self.__yvalid = None, None
        self.__prediction = []

    def train(self, predict=True):
        '''
        Train on training set.

        Args:
            predict (bool): implement prediction or not.

        Returns:
            model
        '''
        assert(isinstance(predict, bool))

        self.__xtrain, self.__ytrain, self.__xvalid, self.__yvalid = self.__dm.get_train_and_valid()

        model = RandomForestClassifier(
            n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth, class_weight=self.class_weight)

        model.fit(self.__xtrain, self.__ytrain)

        if predict:
            prediction = model.predict(self.__dm.get_xtest()).tolist()
            self.__prediction = [chr(i+65) for i in prediction]

        self.__model = model

        return model

    def accuracy(self, n=1):
        '''
        Test on validation set, and get accuracy after n-fold cross validation.

        Args:
            n (int): n-fold cross validation

        Returns:
            accuracy (float)
        '''
        assert(isinstance(n, int))
        assert(n >= 1)

        accuracy = []
        ytest = []
        for i in range(n):
            model = self.__model if i == 0 else self.train(predict=False)
            accuracy.append(model.score(self.__xvalid, self.__yvalid))

        return sum(accuracy) / len(accuracy)

    def prediction(self):
        '''
        Get class prediction for the testing data.
        '''
        return self.__prediction

    def report(self):
        '''
        Detailed performance on the validation set.
        '''
        prediction = self.__model.predict(self.__xvalid)
        print("Random Forest Classifier report \n",
              classification_report(prediction, self.__yvalid))


class SVM():
    def __init__(self, dm, kernel='rbf', C=2, gamma='scale'):
        self.__dm = dm
        self.__model = None
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.__xtrain, self.__ytrain = None, None
        self.__xvalid, self.__yvalid = None, None
        self.__prediction = []

    def train(self, predict=True):
        '''
        Train on training set.

        Args:
            predict (bool): implement prediction or not.

        Returns:
            model
        '''
        assert(isinstance(predict, bool))

        self.__xtrain, self.__ytrain, self.__xvalid, self.__yvalid = self.__dm.get_train_and_valid()

        model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)

        model.fit(self.__xtrain, self.__ytrain)

        if predict:
            prediction = model.predict(self.__dm.get_xtest()).tolist()
            self.__prediction = [chr(i+65) for i in prediction]

        self.__model = model

        return model

    def accuracy(self, n=1):
        '''
        Test on validation set, and get accuracy after n-fold cross validation.

        Args:
            n (int): n-fold cross validation

        Returns:
            accuracy (float)
        '''
        assert(isinstance(n, int))
        assert(n >= 1)

        accuracy = []
        ytest = []
        for i in range(n):
            model = self.__model if i == 0 else self.train(predict=False)
            accuracy.append(model.score(self.__xvalid, self.__yvalid))

        return sum(accuracy) / len(accuracy)

    def prediction(self):
        '''
        Get class prediction for the testing data.
        '''
        return self.__prediction

    def report(self):
        '''
        Detailed performance on the validation set.
        '''
        prediction = self.__model.predict(self.__xvalid)
        print("SVM report \n",
              classification_report(prediction, self.__yvalid))


def grid_search(x, y, model):
    '''
    Computer the most optimal hyperparameter for the model.

    Args:
        x: training samples
        y: training labels
        model (str): machine learning model (SVM or Random Forest Classifier)
    '''
    assert(isinstance(model, str))

    if model == 'randomforest':
        params = {'n_estimators': [10, 100, 200, 500], 'max_depth': [
            250, 500, None], 'class_weight': ['balanced', None]}
        clf = RandomForestClassifier()

    elif model == 'svm':
        params = {'C': (1, 2, 5, 10), 'gamma':('scale', 'auto')}
        clf = SVC()

    else:
        raise SyntaxError("Only 'svm' or 'randomforest' is accepted.")

    grid = GridSearchCV(clf, params, cv=2, verbose=1)
    grid.fit(x, y)
    print(grid.best_params_)
