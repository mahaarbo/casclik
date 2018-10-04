# CASCLIK
CasADi based closed-loop inverse kinematics.

## Introduction
This is a library for running some standard closed-loop inverse kinematics approaches such as the Set-Based task control approach [1], and reactive QP control [2].
It is a prototyping library meant for comparing overall strategies, and although capable of running quite fast, is not the best solution for real-time or finalized industrial applications. 

It conceptually borrows the architecture from the eTaSL software, and we suggest that users use that for more industrial and real-time sensitive implementations.


[1]: Signe article
[2]: eTaSL article

## Requirements
This software requires CasADi and by default it uses the "shell compiler" method of compiling the solvers and functions. Make sure that CasADi works with all the example code, and is capable of compiling code and functions.

## Installation instructions
1. Install pip
2. run `pip install --user .` in this folder.
3. Try importing casclik somewhere
