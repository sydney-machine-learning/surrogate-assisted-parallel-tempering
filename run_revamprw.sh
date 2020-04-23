
#!/bin/sh 
echo Running all 	 
 
  
max_samples=50000
replicas=10
#surrogate_prob=0.5
#surrogate_interval=0.05
 
#python surrogate_revamp.py 4 $max_samples  $surrogate_prob  $surrogate_interval revamp_rwexp/ $replicas    



for problem in  4 5 6 7 8
 	do
	for surrogate_interval in  0.05 #0.05 #0.02 0.05 0.10 0.1
  		do
		for surrogate_prob  in  0.6 # 0.2  0.4 0.6 0.8   #0.2 0.4 0.6 0.8 
			do   


			python _revamp.py $problem $max_samples  $surrogate_prob  $surrogate_interval revamp_rwexp/ $replicas   
  
  
	done  
 
  
		

		

		
	done
done
