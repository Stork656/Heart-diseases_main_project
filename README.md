# **Heart-diseases**
## **General Info**

**Version:** 1.0

**Date:** 2026

**Data source:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

**Goal:** To identify the best preprocessing pipeline and model for predicting heart failure.

## **Table of contents**
1. [General Info](#General-Info)
2. [Data Description](#Data-Description)
3. [Project Description](#Project-Description)
4. [Results](#Results)
5. [Project Structure](#Project-Structure)
6. [Full ML Pipeline](#Full-ML-Pipeline)
7. [Notes](#Notes)
8. [Technologies Used](#Technologies-Used)
9. [How To Run](#How-To-Run)

## **Data Description**

The Data Set contains 918 records and 12 **attributes**:

(*See the full analysis notebooks\eda*)

1. **Age** - age of the patient (years)
2. **Sex** - sex of the patient
    * M - male
    * F - female

3. **ChestPainType** - chest pain type
    * TA - Typical Angina
    * ATA - Atypical Angina
    * NAP - Non-Anginal Pain
    * ASY - Asymptomatic

4. **RestingBP** - resting blood pressure (mm Hg)
5. **Cholesterol** - serum cholesterol (mm/dl)
6. **FastingBS** - fasting blood sugar
    * 1 - if FastingBS > 120 mg/dl
    * 0 - otherwise

7. **RestingECG** - resting electrocardiogram results
    * Normal - normal
    * ST - having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    * LVH - showing probable or definite left ventricular hypertrophy by Estes' criteria
  
8. **MaxHR** - maximum heart rate achieved
    * Numeric value between 60 and 202

9. **ExerciseAngina** - exercise-induced angina
    * Y - Yes
    * N - No
  
10. **Oldpeak** -  ST depression induced by exercise relative to rest (numeric)
11. **ST_Slope** - the slope of the peak exercise ST segment
    * Up - upsloping
    * Flat - flat
    * Down - downsloping

12. **HeartDisease** - output class
    * 1 - heart disease
    * 0 - Normal


## **Project Description**
The project performs an EDA analysis, based on which the necessary data preprocessing steps are determined and 3 methods are implemented that differ in the methods of performing these steps:
1. **Simple pipeline**

    Contains minimal processing steps:

    - Rows with missing values are removed
    - Outliers are filtered using a predefined threshold
    - No feature scaling
    - One-Hot Encoding using Pandas

2. **Standard pipeline**

    Includes common preprocessing steps:

    - Missing values are replaced with mean (for numerical) and mode (for categorical)
    - Outliers are removed based on percentiles (IQR: 75 - 25)
    - One-Hot Encoding using Scikit-Learn
    - Feature scaling with StandardScaler

3. **Advanced pipeline**

    Applies more sophisticated preprocessing:

    - Categorical values are encoded by frequency
    - Missing values are imputed using KNNImputer
    - Feature scaling with RobustScaler
    - Outliers are removed using IsolationForest

Each pipeline produces a separate train-test split and corresponding trained **models**:

1. LogisticRegression
2. svm.SVC
3. KNeighborsClassifier
4. RandomForestClassifier
5. GradientBoostingClassifier
6. AdaBoostClassifier

The performance of each pipeline and model was evaluated using the following **metrics**:
1. Accuracy
2. Precision
3. Recall
4. F2
Additionally, confusion matrices, ROC curves, and PR curves were analyzed.

## **Results**  
(*For full analysis, see notebooks/evaluation_analysis.ipynb. The results after repeated training may differ from those indicated in the analysis*)

The best and the worst results in the table

|Pipeline|Model|Recall|Precision|F2|
|-------|-----|-------|-------|-----|
|Advanced Preprocessing|Gradient Boosting|0.91|0.84|0.90|
|Simple Preprocessing|KNN|0.74|0.66|0.73|


## **Project Structure**
![Project_Structure](assets/Project_Structure.jpg)

## **Full ML Pipeline**
* **main.py** - the orchestrator
* models, metrics, and logs are configured via yaml
  
![ML Pipeline](assets/ML_Pipeline.jpg)

## **Notes**
1. Pipelines are fully configurable via YAML files.
2. The tests currently cover part of the project:
   * src\preprocessing
   * src\utils\validator
3. The validator currently covers part of the project:
   * src\preprocessing
   * loader
   * main
5. Improvements are planned in future versions.

## **Technologies Used**
Developed and tested using Python 3.14

Libraries:
* numpy == 2.3.5
* pandas == 2.3.3
* scikit-learn == 1.8.0
* joblib == 1.5.3
* PyYAML == 6.0.3
* matplotlib == 3.10.8
* seaborn == 0.13.2
* pytest == 9.0.2

Analysis performed using Jupyter Notebook.

## **How to run**
**1. Python Installation**
* If Python is not installed, download and install it from [Python.org](https://www.python.org/)

* To check if Python is installed, open **cmd** and run:
    ```
    python --version
    ```

  If Python is installed, you'll see a version number.
  
  For example:

    ```
    Python 3.14.0
    ```
**2. Downloading the Project**
* Go to the project: [Project](https://github.com/Stork656/Heart-diseases_main_project)
* Click: **Code → Download ZIP**

**3. Installation**

* Unpack the archive.

* Open **cmd** and navigate to the **root folder** of the project:

    ```
    cd path_to_unpacked_folder
    ```
* Create a virtual environment:

    ```
    python -m venv venv
    ```
* Activate the virtual environment:

    ```
    venv\Scripts\activate
    ```
* Install dependencies:

    ```
    pip install -r requirements.txt
    ```
    Wait for the installation to complete.

**4. Running the Project**

* To start training run:
    ```
    python main.py
    ```
    Wait until the training process finishes.

**5. Viewing Results**

* Quick results:
    ```
    cd src\utils
    python see_results.py
    ```
* For full analysis of the results, you will need an IDE, such as Jupyter: [jupyter.org](https://jupyter.org/try)
* Open `notebooks/evaluation_analysis.ipynb` to explore the detailed results. 

    **Note:** the results after repeated training may differ from those indicated in the analysis.
* You can also explore the Exploratory Data Analysis (EDA) by running `eda.ipynb`

## **Contact**

Thank you for your interest in this project!

If you have any questions or suggestions, please open an issue on GitHub.
