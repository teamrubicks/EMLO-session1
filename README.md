# Install DVC on windows

Create a Virtual Environment
- conda create --name dvc_env
- to activate :  conda activate dvc_env

### Download data using DVC
```
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip

unzip -q data.zip
rm -f data.zip
```


Get Started with DVC on WSL

### Install DVC 
- Install Conda
- conda create --name dvc_env

### Download data using DVC
```
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip

unzip -q data.zip
rm -f data.zip
```



### Train the model and Let's capture the current state of this dataset
```
python train.py
dvc init
dvc add model.h5
dvc add data
```
It tells Git to ignore the directory and puts it into the cache (while keeping a file link to it in the workspace, so you can continue working the same way as before). This is achieved by creating a tiny, human-readable .dvc file that serves as a pointer to the cache

Let's commit the current state:
```
git add data.dvc model.h5.dvc metrics.csv .gitignore
git commit -m "First model, trained with 1000 images"
git tag -a "v1.0" -m "model v1.0, 1000 images"
```


### Second model version

Lets double the size of our training data but keep the validation data same
```
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/new-labels.zip
unzip -q new-labels.zip
rm -f new-labels.zip
```

### Retrain the new model and add the new data to track using dvc
```
dvc add data
python train.py
dvc add model.h5
```

### Now lets commit with the second model after training

```
git add data.dvc model.h5.dvc metrics.csv
git commit -m "Second model, trained with 2000 images"
git tag -a "v2.0" -m "model v2.0, 2000 images"
```

# Writing a python test file
- Create a script for writing test cases (using pytest library)
- Create functions to evaluate multiple test cases.  such as:
If you want to check if a file exists in the working directory
```
def test_existance(filename):
    files = os.listdir()
    assert filename not in files
```
Each test function name should have test  as the starting word in it. 


### How to execute on local?
- Create the above script into a folder with the dependencies
- Then Run on this from the command line. '-v' is to make the logs verbose

```
 pytest -v <filename>
```


