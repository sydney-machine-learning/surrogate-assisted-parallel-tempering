
#!/bin/sh 
echo Running all 	 
 

python surrogate_pt_classifier_master_truepos.py 6 50000 0.5  
python surrogate_pt_classifier_master_truepos.py 7 50000 0.5
python surrogate_pt_classifier_master_truepos.py 8 50000 0.5 
python surrogate_pt_classifier_master_truepos.py 5 50000 0.5 
  