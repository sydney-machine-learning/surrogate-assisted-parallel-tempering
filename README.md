# surrogate-assisted-parallel-tempering
Surrogate-Assisted Parallel Tempering for Bayesian Neural Learning

### code

We have two major versions 

1. surrogate parallel tempering using random-walk proposal distribution: [surrogate_pt_classifier_rw_common_interval.py] to run with [run_probability_commoninterval.sh]
2. surrogate parallel tempering using Langevin gradient proposal distribution: [surrogate_pt_classifier_langevingrad.py] to run with [run_langevin.sh]

paper online: [Surrogate-assisted parallel tempering for Bayesian neural learning](https://arxiv.org/abs/1811.08687)



### Prerequisites

The framework is built using: 

* [Paralel tempering neural networks with paralel processing ] (https://github.com/sydney-machine-learning/parallel-tempering-neural-net)
* [Neurocomputing paper by R. Chandra et. al] (https://www.sciencedirect.com/science/article/pii/S0925231219308069) with  [Arxiv open access](https://arxiv.org/abs/1811.04343)

### Installing


you   need to install  Tensorflow and scikitlearn for surrogate training. 

## Running the tests

Data for the experiments are given: [data](https://github.com/sydney-machine-learning/surrogate-assisted-parallel-tempering/tree/master/DATA)

###  Experiments 

Example results and sh script for running experiment is given here: [sample experiment results](https://github.com/sydney-machine-learning/surrogate-assisted-parallel-tempering/tree/master/experiments)

 
 
 

## Versioning
 
* TBA

## Authors
 
* R. Chandra, K Jain, A. Kapoor [Surrogate-assisted parallel tempering for Bayesian neural learning](https://arxiv.org/abs/1811.08687)

## License

* This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* R. Dietmar Muller and Danial Azam, University of Sydney
