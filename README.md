## Archived/Obsolete:
This repository has not been under active development for many years and was a part of my work at the department of engineering cybernetics at NTNU in 2019. The repository has therefore been archived. 

# CASCLIK
[CasADi](https://web.casadi.org/)-based closed-loop inverse kinematics.

## Introduction
This is a library for rapid prototyping of constraint-based, task-priority closed-loop inverse kinematics. In essence this means you feed it an arbitrary expression with robot variables, inputs, virtual variables, and the controllers try to give you exponential convergence to the desired expression. It supports controller such as the singularity-robust multiple task-priority set-based approach [1], and reactive QP approach (eTaSL/eTC) [2]. It is a prototyping library meant for comparing overall strategies, and although capable of running quite fast, is not the best solution for real-time or finalized industrial applications.  It conceptually borrows the architecture from the eTaSL software, and we suggest that users use that for more industrial and real-time sensitive implementations.


[1]: Moe et al., [Set-Based Tasks within the Singularity-Robust Multiple Task-Priority Inverse Kinematics Framework: General Formulation, Stability Analysis, and Experimental Results](https://www.frontiersin.org/articles/10.3389/frobt.2016.00016/full), 2016

[2]: Aertbeliën and De Schutter, [eTaSL/eTC: A constraint-based task specification language and robot controller using expression graphs](https://ieeexplore.ieee.org/document/6942760), 2014


## Article
[CASCLIK: CasADi-Based Closed-Loop Inverse Kinematics](https://arxiv.org/abs/1901.06713)


## Requirements
This software requires CasADi and by default it uses the "shell compiler" method of compiling the solvers and functions. Make sure that CasADi works with all the example code, and is capable of compiling code and functions.

## Installation instructions
1. Install pip
2. run `pip install --user .` in this folder.
3. Try importing casclik somewhere

## Examples
Check the examples folder. It currently only contains [jupyter](https://jupyter.org/) notebook examples. 

## Others
### urdf2casadi
Python module for automatically generating [CasADi](https://web.casadi.org/) functions of forward kinematics, either as transformation matrices or as dual quaternions.
It also supports Denavit-Hartenberg parameters. Link: [urdf2casadi](https://github.com/mahaarbo/urdf2casadi).

### casclik\_examples
This is a [ROS](http://www.ros.org/) metapackage with example integration of CASCLIK into [ROS](http://www.ros.org/). The `casclik_basics` package contains a robot interface for sending joint position commands to the robot. But check out `casclik_tests` first. Link: [casclik\_examples](https://github.com/mahaarbo/casclik_examples).
