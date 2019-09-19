#!/bin/sh 
echo Running all 	 
 
 
python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  surrlangevingrad/ 2
python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  surrlangevingrad/ 4 
python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  surrlangevingrad/ 6 
python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  surrlangevingrad/ 8 
python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  surrlangevingrad/ 10

#python surrogate_pt_classifier_langevingrad.py 5 50000   0.25  0.05  surrlangevingrad/
#python surrogate_pt_classifier_langevingrad.py 5 50000   0.25  0.05  surrlangevingrad/


python surrogate_pt_classifier_langevingrad.py 6 50000 0.5 0.05 surrlangevingrad/ 2
python surrogate_pt_classifier_langevingrad.py 6 50000   0.25  0.05  surrlangevingrad/ 4 
python surrogate_pt_classifier_langevingrad.py 6 50000 0.5 0.05 surrlangevingrad/ 6
python surrogate_pt_classifier_langevingrad.py 6 50000   0.25  0.05  surrlangevingrad/ 8
python surrogate_pt_classifier_langevingrad.py 6 50000   0.25  0.05  surrlangevingrad/ 10

python surrogate_pt_classifier_langevingrad.py 7 50000   0.5  0.05  surrlangevingrad/ 2
python surrogate_pt_classifier_langevingrad.py 7 50000 0.25 0.05 surrlangevingrad/  4



python surrogate_pt_classifier_langevingrad.py 8 50000   0.5  0.05  surrlangevingrad/ 2
python surrogate_pt_classifier_langevingrad.py 8 50000 0.25 0.05 surrlangevingrad/  4

#python surrogate_pt_classifier_langevingrad.py 3 50000   0.25  0.05  surrlangevingrad/ 
#python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  surrlangevingrad/
#python surrogate_pt_classifier_langevingrad.py 5 50000   0.25  0.05  surrlangevingrad/
#python surrogate_pt_classifier_langevingrad.py 6 50000 0.25 0.05 surrlangevingrad/ 
#python surrogate_pt_classifier_langevingrad.py 7 50000   0.25  0.05  surrlangevingrad/

#python surrogate_pt_classifier_langevingrad.py 7 50000   0.25  0.05  surrlangevingrad/
#python surrogate_pt_classifier_langevingrad.py 8 50000 0.25 0.05 surrlangevingrad/  