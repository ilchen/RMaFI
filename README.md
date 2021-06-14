# RMaFI
Solutions for exercises from John C. Hull's [Risk Management and Financial Institutions](http://www-2.rotman.utoronto.ca/~hull/riskman/index.html)
using Python and [Pandas](https://pandas.pydata.org).

So far I tackled Chapters 13 _Market Risk VaR: The Historical Simulation Approach_, 14 _Market Risk VaR: The Model Building Approach_,
and 18 _Fundamental Review of the Trading Book_.

## Requirements
For the majority of problems the following will suffice:
```commandline
python3 -m pip install -r requirements.txt
```
For a few problems relying on the Extreme Value Theory in Chapter 13 _The Historical Simulation Approach_, where we need
to solve optimization problems, TensorFlow 2.x will be needed:
```commandline
python3 -m pip install -r requirements_extra.txt
```
## How to install using virtualenv
If you want to avoid mixing up packages in your global python installation, you can prepare the environment as follows:
```commandline
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_extra.txt
python3 -m pip install jupyter
```

## Data for Chapters 13, 14, and 18
The spreadsheet I used for exercises in these chapters can be downloaded from [this page](http://www-2.rotman.utoronto.ca/~hull/VaRExample/index.html).
I used `VaRExampleRMFI3eHistoricalSimulation.xls` for all these Chapters. Please remove the first row before feeding it to the code.

Code in `ch14/` replicates the functionality in [VaRExampleRMFI3eModelBuilding.xls](http://www-2.rotman.utoronto.ca/~hull/VaRExample/VaRExampleRMFI4eModelBuilding.xls). 

## How to run
* **Chapter 13**: Market Risk VaR: The Historical Simulation Approach
   ```commandline
   python3 ch13/hist_simulation.py ./ext/VaRExampleRMFI3eHistoricalSimulation.xls
   ```
   Or, much better, using Jupiter. For historical simulation exercises that don't utilize the _Extreme Value Theory_:
   ```commandline
   jupyter notebook ch13.ipynb
   ```
   For exercises that make use of the _Extreme Value Theory_:
   ```commandline
   jupyter notebook ch13_evt.ipynb
   ```

* **Chapter 14**: Market Risk VaR: The Model Building Approach
   ```commandline
   python3 ch14/model_build.py ./ext/VaRExampleRMFI3eHistoricalSimulation.xls
   ```
   Or, much better, using Jupiter.
   ```commandline
   jupyter notebook ch14.ipynb
   ```
* **Chapter 18**: Fundamental Review of the Trading Book:
   ```commandline
   python3 ch18/frtb.py ./ext/VaRExampleRMFI3eHistoricalSimulation.xls
   ```