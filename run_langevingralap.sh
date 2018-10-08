
#!/bin/sh 
echo Running all 	 
 
 
#python surrogate_pt_classifier_langevingrad.py 3 50000   0.25  0.05  surrlangevingrad_lap/ 
#python surrogate_pt_classifier_langevingrad.py 4 50000   0.25  0.05  surrlangevingrad_lap/
#python surrogate_pt_classifier_langevingrad.py 5 10000   0.5  0.05  surrlangevingrad_lap/

python surrogate_pt_classifier_langevingrad.py 5 10000   0.5  0.05  surrlangevingrad_lap/

python surrogate_pt_classifier_langevingrad.py 6 10000   0.5  0.05  surrlangevingrad_lap/








#python surrogate_pt_classifier_langevingrad.py 6 50000 0.25 0.05 surrlangevingrad_lap/ 
#python surrogate_pt_classifier_langevingrad.py 7 5000   0.5  0.05 surrlangevingrad_lap/
#python surrogate_pt_classifier_langevingrad.py 8 50000 0.25 0.05 surrlangevingrad_lap/   