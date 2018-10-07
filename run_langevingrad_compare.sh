
#!/bin/sh 
echo Running all 	 
 
 
python surrogate_pt_classifier_langevingrad_.py 3 50000   0.25  0.05  surrlangevingrad_compare/ 
python surrogate_pt_classifier_langevingrad_.py 4 50000   0.25  0.05  surrlangevingrad_compare/
python surrogate_pt_classifier_langevingrad_.py 5 50000   0.25  0.05  surrlangevingrad_compare/
python surrogate_pt_classifier_langevingrad_.py 6 50000 0.25 0.05 surrlangevingrad_compare/ 
python surrogate_pt_classifier_langevingrad_.py 7 50000  0.25  0.05  surrlangevingrad_compare/
python surrogate_pt_classifier_langevingrad_.py 8 50000 0.25 0.05 surrlangevingrad_compare/  

python surrogate_pt_classifier_langevingrad_.py 3 50000   0.5  0.05  surrlangevingrad_compare/ 
python surrogate_pt_classifier_langevingrad_.py 4 50000   0.5  0.05  surrlangevingrad_compare/
python surrogate_pt_classifier_langevingrad_.py 5 50000   0.5  0.05  surrlangevingrad_compare/
python surrogate_pt_classifier_langevingrad_.py 6 50000 0.5 0.05 surrlangevingrad_compare/ 
python surrogate_pt_classifier_langevingrad_.py 7 50000  0.5  0.05  surrlangevingrad_compare/
python surrogate_pt_classifier_langevingrad_.py 8 50000 0.5 0.05 surrlangevingrad_compare/ 