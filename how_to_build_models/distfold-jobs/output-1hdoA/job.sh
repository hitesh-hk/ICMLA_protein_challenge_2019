#!/bin/bash                                       
echo "starting cns.."                           
touch iam.running                                 
# CNS-CONFIGURATION                               
source /home/SSD/protien_ICMLA/code_4/IEEE-ICMLA-2019-PIDP-Challenge/how_to_build_models/cns_solve_1.3/cns_solve_env.sh                
export KMP_AFFINITY=none                          
export CNS_CUSTOMMODULE=/home/SSD/protien_ICMLA/code_4/IEEE-ICMLA-2019-PIDP-Challenge/how_to_build_models/distfold-jobs/output-1hdoA/                 
/home/SSD/protien_ICMLA/code_4/IEEE-ICMLA-2019-PIDP-Challenge/how_to_build_models/cns_solve_1.3/intel-x86_64bit-linux/bin/cns_solve < dgsa.inp > dgsa.log 
if [ -f "1hdoA_20.pdb" ]; then           
   rm iam.running                                 
   echo "trial structures written."             
   rm *embed*                                     
   exit                                           
fi                                                
if [ -f "1hdoAa_20.pdb" ]; then 
   rm iam.running                                 
   echo "accepted structures written."          
   rm *embed*                                     
   exit                                           
fi                                                
tail -n 30 dgsa.log                              
echo "ERROR! Final structures not found!"       
echo "CNS FAILED!"                              
mv iam.running iam.failed                         
