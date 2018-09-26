
#!/bin/sh 
echo Running all 	 
 
  
  
python surrogate_pt_sinterval.py 6 50000 0.5 0.05  
python surrogate_pt_sinterval.py 7 50000 0.5 0.05
python surrogate_pt_sinterval.py 8 50000 0.5 0.05
python surrogate_pt_sinterval.py 5 50000 0.5 0.05  


python surrogate_pt_sinterval.py 6 50000 0.5 0.1   
python surrogate_pt_sinterval.py 7 50000 0.5 0.1
python surrogate_pt_sinterval.py 8 50000 0.5 0.1
python surrogate_pt_sinterval.py 5 50000 0.5 0.1  


python surrogate_pt_sinterval.py 6 50000 0.5 0.2   
python surrogate_pt_sinterval.py 7 50000 0.5 0.2
python surrogate_pt_sinterval.py 8 50000 0.5 0.2
python surrogate_pt_sinterval.py 5 50000 0.5 0.2 

 