# Scripts and dataset to build/evaluate 3D models  
This folder contains scripts and dataset to build full three-dimensional models for the 150 proteins in the test dataset. Assuming that your distance prediction training is complete and that you have a trained model, this README file provides the steps to build models using [DISTFOLD](https://github.com/badriadhikari/DISTFOLD) and evaluate them using [TMscore](https://zhanglab.ccmb.med.umich.edu/TM-score/).

| File | Description|
| --- | --- |
| predict-distances.py | Script to predict and write distance files (in RR format) using a trained deep learning model |
| build-3d-models.sh | Script to build full atom 3D models with distance files (.rr format) and predicted secondary structures |
| evaluate-3d-models.sh | For each protein in the test dataset, this script compares all the predicted 20 models with the true structures |
| my-rr-predictions.tar.gz | Predictions obtained by the 'predict-distances.py'. This must be replaced by your predictions. |
| pdb.tar.gz | True/Correct 3D structures for the 150 proteins in the test dataset |
| ss.tar.gz | Secondary structures predicted for the proteins in the test dataset using [SSpro](http://scratch.proteomics.ics.uci.edu/) |

# Steps  
### Step 1: Predict top L long-range distances
The script "predict-distances.py" can be used to write distance matrix (map) files and top L shortest long-range distances to a file in the standard [CASP RR file format](http://predictioncenter.org/casp8/index.cgi?page=format#RR). The first step is to update the "predict-distances.py" with the correct path to your model.
```bash
mkdir predictions
python predict-distances.py
```
This will write '*.rr' files. These predictions should be in the format similar to the files in 'my-rr-predictions.tar.gz'

### Step 2: Verify that you have native structures & secondary structure files
Download and unzip 'ss.tar.gz' and 'pdb.tar.gz'.

### Step 3: Download DISTFOLD
[DISTFOLD](https://github.com/badriadhikari/DISTFOLD) is an extremely fast tool to build 3D models. Follow the instructions to download and test it.

### Step 4: Build 3D models
The script "build-3d-models.sh" can be used to build 3D models using the top L long-range distances predicted and the predicted secondary structures. Correctly configure the paths in this script and run it. Since these are 150 jobs, you may not want to run them all parallely. This script builds models only for the first 10 proteins.
```bash
./build-3d-models.sh 
```
### Step 5: Evaluate predicted models
The script 'evaluate-3d-models.sh' uses TMscore to automatically score the predicted models against the true structures.
```bash
./evaluate-3d-models.sh
```

### [Optional] Step 6: Visualize predicted models
Predicted models and true structures may be visualized using [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/).
