
#!/bin/sh 
echo Running all 	 
  
python surrogate_pt_classifier_rw.py 3 5000  0.5  0.05  surr_accuracy/   
python surrogate_pt_classifier_rw.py 4 5000   0.5  0.05  surr_accuracy/  
python surrogate_pt_classifier_rw.py 5 5000   0.5  0.05  surr_accuracy/ 
python surrogate_pt_classifier_rw.py 6 5000  0.5 0.05 surr_accuracy/ 
python surrogate_pt_classifier_rw.py 7 5000   0.5  0.05   surr_accuracy/
python surrogate_pt_classifier_rw.py 8 5000  0.5 0.05 surr_accuracy/


python surrogate_pt_classifier_rw.py 3 5000  0.5  0.1  surr_accuracy/   
python surrogate_pt_classifier_rw.py 4 5000   0.5  0.1  surr_accuracy/  
python surrogate_pt_classifier_rw.py 5 5000   0.5  0.1  surr_accuracy/ 
python surrogate_pt_classifier_rw.py 6 5000  0.5 0.1 surr_accuracy/ 
python surrogate_pt_classifier_rw.py 7 5000   0.5  0.1   surr_accuracy/
python surrogate_pt_classifier_rw.py 8 5000  0.5 0.1 surr_accuracy/


python surrogate_pt_classifier_rw.py 3 50000  0.5  0.05  surr_accuracy/   
python surrogate_pt_classifier_rw.py 4 50000   0.5  0.05  surr_accuracy/  
python surrogate_pt_classifier_rw.py 5 50000   0.5  0.05  surr_accuracy/ 
python surrogate_pt_classifier_rw.py 6 50000  0.5 0.05 surr_accuracy/ 
python surrogate_pt_classifier_rw.py 7 50000   0.5  0.05   surr_accuracy/
python surrogate_pt_classifier_rw.py 8 50000  0.5 0.05 surr_accuracy/

