
#!/bin/sh 
echo Running all 	 
 
 

 
python surrogate_rw_common_interval.py 5 10000   0.5  0.05  num_replicas_rw/ 10  
  
python surrogate_rw_common_interval.py 6 10000   0.5  0.05  num_replicas_rw/ 10

 
python surrogate_rw_common_interval.py 7 10000   0.5  0.05  num_replicas_rw/ 10  

python surrogate_rw_common_interval.py 8 10000   0.5  0.05  num_replicas_rw/ 10
 