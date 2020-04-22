
#!/bin/sh 
echo Running all 	 
 
  
max_samples=50000
replicas=10
surrogate_prob=0.5
surrogate_interval=0.02
 
python surrogate_revamp.py 4 $max_samples  $surrogate_prob  $surrogate_interval revamp_rw/ $replicas 
 
python surrogate_revamp.py 5 $max_samples  $surrogate_prob  $surrogate_interval revamp_rw/ $replicas   
 
python surrogate_revamp.py 6 $max_samples  $surrogate_prob  $surrogate_interval  revamp_rw/ $replicas  
 
python surrogate_revamp.py 7 $max_samples  $surrogate_prob   $surrogate_interval  revamp_rw/ $replicas   

python surrogate_revamp.py 8 $max_samples  $surrogate_prob  $surrogate_interval  revamp_rw/ $replicas 