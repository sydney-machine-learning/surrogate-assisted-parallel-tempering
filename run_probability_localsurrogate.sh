
#!/bin/sh 
echo Running all 	 
 
  



python surrogate_pt_classifier_rw_localsurrogate.py 3 50000   0.5  0.05  res_sprob_local/ 
python surrogate_pt_classifier_rw_localsurrogate.py 4 50000   0.5  0.05 res_sprob_local/
python surrogate_pt_classifier_rw_localsurrogate.py 5 50000   0.5  0.1  res_sprob_local/
python surrogate_pt_classifier_rw_localsurrogate.py 6 30000 0.5 0.1 res_sprob_local/ 
python surrogate_pt_classifier_rw_localsurrogate.py 7 30000  0.5  0.1  res_sprob_local/
python surrogate_pt_classifier_rw_localsurrogate.py 8 30000 0.5 0.05 res_sprob_local/ 

 