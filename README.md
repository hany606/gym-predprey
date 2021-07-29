# gym-predprey
This repository contains the code for gym environment for predator and prey agents based on [snolfi/longdpole](https://github.com/snolfi/longdpole) and [snolfi/evorobotpy2](https://github.com/snolfi/evorobotpy2)


## Installation

```
git clone https://github.com/hany606/gym-predprey
cd gym-predprey
cd predpreylib
python3 setupErPredPrey.py build_ext --inplace  
<!-- cp ErPredPrey*.so ../ -->
<!-- cp *.sample ../ -->
<!-- cp ErPredprey.ini ../ -->
cd ..    
pip3 install -e .
```

For now copy *.sample files to the place of the running scripts
```
cp *.sample <path>
```