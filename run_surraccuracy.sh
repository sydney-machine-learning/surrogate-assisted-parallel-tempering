
#!/bin/sh 
echo Running all 	 
  
python surrogate_pt_classifier_rw.py 3 10000  0.5  0.05  surr_accuracy/   
python surrogate_pt_classifier_rw.py 3 50000   0.5  0.05  surr_accuracy/  
python surrogate_pt_classifier_rw.py 4 10000   0.5  0.05  surr_accuracy/ 
python surrogate_pt_classifier_rw.py 4 50000  0.5 0.05 surr_accuracy/ 
python surrogate_pt_classifier_rw.py 5 10000   0.5  0.05   surr_accuracy/
python surrogate_pt_classifier_rw.py 5 50000  0.5 0.05 surr_accuracy/  
python surrogate_pt_classifier_rw.py 6 10000  0.5  0.05  surr_accuracy/   
python surrogate_pt_classifier_rw.py 6 50000   0.5  0.05  surr_accuracy/  
python surrogate_pt_classifier_rw.py 7 10000   0.5  0.05  surr_accuracy/ 
python surrogate_pt_classifier_rw.py 7 50000  0.5 0.05 surr_accuracy/ 
python surrogate_pt_classifier_rw.py 8 10000   0.5  0.05   surr_accuracy/
python surrogate_pt_classifier_rw.py 8 50000  0.5 0.05 surr_accuracy/

