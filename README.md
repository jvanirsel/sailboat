# SAILboat
Personal repository for SAIL GEMINI and general computational work.

## Installing GEMINI on ERAU's VEGA with Bash

### Load modules
In `~/.bash_profile` append the following:
```sh
module load gcc/8.5.0-gcc-8.5.0-cokvw3c
module load openmpi/5.0.2-gcc-8.5.0-diludms
module load netlib-lapack/3.11.0-gcc-8.5.0-hlxv33x
module load cmake/3.27.9-gcc-8.5.0-5bfhfkk
```
These use the older GCC version (8.5.0) since this is the one that includes the lapack module. Next, run
```sh
source ~/.bash_profile
```

### Clone and build GEMINI
Clone the latest gemini3d repository and build it in parallel:
```sh
git clone https://github.com/gemini3d/gemini3d.git
cd gemini3d
cmake -B build
cmake --build build --parallel
```
Ensure you read
```sh
[100%] Built target gemini.bin
```

### Test GEMINI installation (**NOT YET WORKING**)
**Do not run ctest on VEGA login node!** First download the required tests,
```sh
ctest --test-dir build --preset download
```
and ensure you read
```sh
100% tests passed, 0 tests failed out of 8
```
Now, run an interactive PBS job,
```sh
qsub -I -l walltime=1:00:00
```
and navigate back to the `gemini3d` directory. Then run the tests,
```sh
ctest --test-dir build
```
and ensure you read
```sh
100% tests passed, 0 tests failed out of 73
```

## Installing PyGEMINI on ERAU's VEGA with Bash

### Load modules
In `~/.bash_profile` append the following:
```sh
module load python/3.11.7-gcc-8.5.0-wfpoppf
```
and source it,
```sh
source ~/.bash_profile
```

### Create GEMINI python environment
PyGEMINI cannot (and should not) be installed on the root VEGA python. Create a python environment in a local `.venvs` location:
```sh
mkdir ~/.venvs
python -m venv ~/.venvs/gemini
```
Next, in `~/.bash_profile`, append:
```sh
source ~/.venvs/gemini/bin/activate
```
and then run
```sh
source ~/.bash_profile
```

### Install PyGEMINI
Simply run
```sh
pip install gemini3d
```
