## <h1 align="center">Facial Recognition</h1>

<p align="center">
  <img width="600" src="https://cdn.nextgov.com/media/img/cd/2019/11/15/NGbiometrics20191115/860x394.jpg?1618355364" alt="Facial Recognition">
 </p>


# Introduction
### Class: W207
### Team Name: Choo-Choo-Train

### Team Members:
1. Jordan Meyer
2. Amit Thiruvengadam
3. Sweeta Bee
4. Leon Gutierrez


# Instruction to run code

## Preparation

* In the data folder unzip "facial-keypoints-detection.zip"
* This will create a folder called "facial-keypoints-detection". Move the contents of this folder, namely, IdlookupTable.csv, SampleSubmission.csv, train.csv, test.csv into the "data"" folder
* Next unzip training.zip and test.zip and move training.csv and test.csv into the data folder
* Your folder structure should now look like this:

```{}
Choo-Choo-Train
      |_data
           |_IdLookupTable.csv
           |_SampleSubmission.csv
           |_training.csv
           |_test.csv
           |_models (empty folder)
           |_analysis (folder with some content)
           |_augment (empty folder)
      |_notebooks
           |_ main.ipynb
           |_ image_analysis.ipynb
      |_README.md (this file)
      |___init__.py
```

## Note: Please rename the main folder to cct. It should now look like this

```{}
cct
  |_data
       |_IdLookupTable.csv
       |_SampleSubmission.csv
       |_training.csv
       |_test.csv
       |_models (empty folder)
       |_analysis (folder with some content)
       |_augment (empty folder)
  |_notebooks
       |_ main.ipynb
       |_ image_analysis.ipynb
  |_README.md (this file)
  |___init__.py
```

## Creating models

* Models take ~15hrs to generate on a GPU and must be run on a High RAM machine
* Move the entire folder structure to Google Drive and place it in your Colab Notebook folder. It should now look like this:

```{}
|_drive
      |_MyDrive
            |_Colab Notebooks
                          |_cct
```

* open main.ipynb from the notebooks folder and follow instruction on first cell
* It creates a file called CNNMySubmission.csv that can be submitted for the Kaggle competition

```{}
cct
  |_data
       |_CNNMySubmission.csv
```

## Analysis

* open image_analysis.ipynb from the notebooks folder and run all cells

     
