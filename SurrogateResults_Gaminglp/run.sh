
#!/bin/sh 
echo Running all 	 
  
python surrogate_pt_classifier_keras.py 6 50000 0.5  
python surrogate_pt_classifier_keras.py 7 50000 0.5
python surrogate_pt_classifier_keras.py 8 50000 0.5 
python surrogate_pt_classifier_keras.py 5 50000 0.5   


python surrogate_pt_classifier_keras.py 6 50000 0.75  
python surrogate_pt_classifier_keras.py 7 50000 0.75
python surrogate_pt_classifier_keras.py 8 50000 0.75 
python surrogate_pt_classifier_keras.py 5 50000 0.75 