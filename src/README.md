# File Descriptions

## `utils.py`

Here are functions that preprocess the dataset and implement classification.

* class `DataManager()`: Preprocess the data.
  * `get_train_and_valid()`
  * `get_xtest()`
* calss `RandomForest()`: Random Forest Classifier model.
  * `train()`
  * `accuracy()`
  * `prediction()`
  * `report()`
* class `SVM()`: Support Vector Machine model.
  * `train()`
  * `accuracy()`
  * `prediction()`
  * `report()`
* function `grid_search()`: Obtain the optimal hyperparameters for the model.

## `Human_Activity_Recognition.ipynb`

Details regarding how the functions work and show the performance of models as well.  

PS: If github cannot load the jupyter file successfully, please copy the url of the file and then go to [https://nbviewer.jupyter.org/](https://nbviewer.jupyter.org/).