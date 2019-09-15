#!/bin/bash
python3 data_preprocess.py
echo "Data prepared successfully"
sleep 5
# python3 train.py
echo "Model Trained!!!!!"
sleep 5
python3 test.py
echo "Model Inference done !!"
sleep 5
read -p "Do you want to check TM Score also?(y/n) "  option
if [ $option == 'y' ]
then 
cd how_to_build_models	
bash evaluate-3d-models.sh
elif [ $option == 'n' ] 
then 
echo "Metric TM score is not printed"
else
echo "select valid option"
fi
echo "Done!!!!"
