#!/bin/sh 
echo Running all 	 
  

#python surrogate_pt_classifier_langevingrad.py 4 50000   0.5  0.05  num_replicares/ 4 
#python surrogate_pt_classifier_langevingrad.py 4 50000   0.5  0.05  num_replicares/ 4 
 #python surrogate_pt_classifier_langevingrad.py 4 50000   0.5  0.05  num_replicares/ 6 
#python surrogate_pt_classifier_langevingrad.py 4 50000   0.5  0.05  num_replicares/ 8 
#python surrogate_pt_classifier_langevingrad.py 4 50000   0.5  0.05  num_replicares/ 10

#python surrogate_pt_classifier_langevingrad.py 5 50000   0.25  0.05  num_replicares/
  #python surrogate_pt_classifier_langevingrad.py 5 5000   0.5  0.05  num_replicares/ 6
 #python surrogate_pt_classifier_langevingrad.py 5 5000   0.5  0.05  num_replicares/ 10


python surrogate_pt_classifier_rw.py 5 50000 0.5 0.05 num_replicares_/ 2
python surrogate_pt_classifier_rw.py 5 50000    0.5  0.05  num_replicares_/ 4 
python surrogate_pt_classifier_rw.py 5 50000 0.5 0.05 num_replicares_/ 6
python surrogate_pt_classifier_rw.py 5 50000   0.5  0.05  num_replicares_/ 8
python surrogate_pt_classifier_rw.py 5 50000   0.5  0.05  num_replicares_/ 10

#python surrogate_pt_classifier_langevingrad.py 7 5000   0.5  0.05  num_replicares/ 4

python surrogate_pt_classifier_rw.py 6 50000 0.5 0.05 num_replicares/ 2
python surrogate_pt_classifier_rw.py 6 5000  0.5 0.05 num_replicares/  4
python surrogate_pt_classifier_rw.py 6 5000  0.5 0.05 num_replicares/  6
python surrogate_pt_classifier_rw.py 6 5000  0.5 0.05 num_replicares/  8
python surrogate_pt_classifier_rw.py 6 5000  0.5 0.05 num_replicares/  10
python surrogate_pt_classifier_langevingrad.py 6 5000  0.5 0.05 num_replicares/  10
#python surrogate_pt_classifier_langevingrad.py 7 5000  0.5 0.05 num_replicares/  8
#python surrogate_pt_classifier_langevingrad.py 7 5000  0.5 0.05 num_replicares/  10



#python surrogate_pt_classifier_langevingrad.py 8 50000   0.5  0.05  num_replicares/ 2
#python surrogate_pt_classifier_langevingrad.py 8 50000 0.5 0.05 num_replicares/  4

#python surrogate_pt_classifier_langevingrad.py 3 50000   0.25  0.05  num_replicares/ 
#python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  num_replicares/
#python surrogate_pt_classifier_langevingrad.py 5 50000   0.25  0.05  num_replicares/
#python surrogate_pt_classifier_langevingrad.py 6 50000 0.25 0.05 num_replicares/ 
#python surrogate_pt_classifier_langevingrad.py 7 50000   0.25  0.05  num_replicares/

#python surrogate_pt_classifier_langevingrad.py 7 50000   0.25  0.05  num_replicares/
#python surrogate_pt_classifier_langevingrad.py 8 50000 0.25 0.05 num_replicares/  