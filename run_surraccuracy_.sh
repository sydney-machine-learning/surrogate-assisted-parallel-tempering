
#!/bin/sh 
echo Running all 	 
  
python surrogate_pt_classifier_rw.py 3 20000  0.5  0.05  surr_accuracy_/   
python surrogate_pt_classifier_rw.py 3 50000   0.5  0.05  surr_accuracy_/  
python surrogate_pt_classifier_rw.py 4 20000   0.5  0.05  surr_accuracy_/ 
python surrogate_pt_classifier_rw.py 4 50000  0.5 0.05 surr_accuracy_/ 
python surrogate_pt_classifier_rw.py 5 20000   0.5  0.05   surr_accuracy_/
python surrogate_pt_classifier_rw.py 5 50000  0.5 0.05 surr_accuracy_/  
python surrogate_pt_classifier_rw.py 6 20000  0.5  0.05  surr_accuracy_/   
python surrogate_pt_classifier_rw.py 6 50000   0.5  0.05  surr_accuracy_/  
python surrogate_pt_classifier_rw.py 7 20000   0.5  0.05  surr_accuracy_/ 
python surrogate_pt_classifier_rw.py 7 50000  0.5 0.05 surr_accuracy_/ 
python surrogate_pt_classifier_rw.py 8 20000   0.5  0.05   surr_accuracy_/
python surrogate_pt_classifier_rw.py 8 50000  0.5 0.05 surr_accuracy_/

