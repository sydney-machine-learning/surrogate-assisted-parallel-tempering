
#!/bin/sh 
echo Running all 	 
 

python surrogate_pt_classifier_master.py 5 50000  0.5  0.1  res_master/
python surrogate_pt_classifier_master.py 6 50000 0.5 0.1 res_master/

python surrogate_pt_classifier_master.py 7 50000  0.5  0.1  res_master/
python surrogate_pt_classifier_master.py 8 50000 0.5 0.1 res_master/