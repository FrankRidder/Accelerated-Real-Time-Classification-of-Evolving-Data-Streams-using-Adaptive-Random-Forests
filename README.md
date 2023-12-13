# Accelerated Real-Time Classification of Evolving Data Streams using Adaptive Random Forests​

Implementation for the paper with same title as repo (link will be added when available)
​
## Prerequisites
Vitis HLS - 2022.2 

## Data Preparation
Data generation and preperation can be found in https://github.com/ingako/gsarf

## Running the Tests
After adding the files in the source folder as source and the top_lvl function as top function.  
Add the testbench tb_arf.cpp and the datafolder as testbench files.  
Select the U55C or a simular FPGA as target for your solution in Vitis HLS.

You should now be able to run C simulation and synthesis.
