## Parallel Programming Final Project
This project implements a **GPU-accelerated** 1D heat equation solver using finite element method.

### Environment and Compiling
I run the code on the AMD-provided platform for this course. Use the following command to compile:
```
$ make seq // compile the sequential version
$ make para // compile the parallel version
$ make clean // remove executables
```

### Data input format and output format
The first 6 lines are ```n```, ```step```, ```Record_every_n_steps```, ```step_size```, ```width```, and ```kappa```, respectively. Then, followed by ```n``` numbers denoting the temperature at ```i / n * width``` (```i. The data can be simulated as long as it satisfies the above format.

The program will output the temperature at ```i / n * width``` every ```Record_every_n_steps``` moves.

### Run experiments
After compiling the code, use the following command to obtain the result mentioned in the report.
```
// temporarily directories for prof.py
$ mkdir seqout
$ mkdir paraout
$ python prof.py
```
This program will compare the result of the sequential version and the parallel version, and print the time of both programs.
