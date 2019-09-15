# ICMLA_protein_challenge_2019
ICMLA protein challenge 2019
* Download the dataset from the official host https://console.cloud.google.com/storage/browser/protein-distance
* Put the downloaded file directory in the same directory (Other wise it will give your errors)
* Execute run.sh file which will 
  * fetch files from downloaded directory and make npy files which are of 8 channels
  * After, it will train the model (Make sure npy files generated are in same folder)
  * Finally it will test model (Make sure npy files generated are in same folder)
  * If you are proceeding to TM score check, edit the paths of distfold.pl and evaluate-3d-models.sh. For this, it required path of predictions folder where our pdb files are saved.
