
#!/bin/sh 
echo Running all 	 

python surrogate_pt_sprob.py 6 50000 0.1  
python surrogate_pt_sprob.py 7 50000 0.1
python surrogate_pt_sprob.py 8 50000 0.1 
python surrogate_pt_sprob.py 5 50000 0.1 
 
python surrogate_pt_sprob.py 6 50000 0.25  
python surrogate_pt_sprob.py 7 50000 0.25
python surrogate_pt_sprob.py 8 50000 0.25 
python surrogate_pt_sprob.py 5 50000 0.25 
  
python surrogate_pt_sprob.py 6 50000 0.5  
python surrogate_pt_sprob.py 7 50000 0.5
python surrogate_pt_sprob.py 8 50000 0.5 
python surrogate_pt_sprob.py 5 50000 0.5   


python surrogate_pt_sprob.py 6 50000 0.75  
python surrogate_pt_sprob.py 7 50000 0.75
python surrogate_pt_sprob.py 8 50000 0.75 
python surrogate_pt_sprob.py 5 50000 0.75 


python surrogate_pt_sprob.py 6 50000 0  
python surrogate_pt_sprob.py 7 50000 0 
python surrogate_pt_sprob.py 8 50000 0 
python surrogate_pt_sprob.py 5 50000 0  