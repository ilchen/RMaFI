# RMaFI
Solutions for exercises from John C. Hull's Risk Management and Financial Institutions. So far I tackled Chapters 14 _Market Risk VaR: The Historical Simulation Approach_ and 15 _Market Risk VaR: The Model Building Approach_.

## Requirements
```
Python            3.x
scipy             1.0.1
numpy             1.14.2
pandas            0.22.0
tensorflow        1.8.0   <-- for optimization problems in Exercises 14.8, 14.9, 14.10, and 14.11.
```

## Data for Chapters 14 and 15
The spreadsheet I used for exercices in these chapters can be downloaded from [this page](http://www-2.rotman.utoronto.ca/~hull/VaRExample/index.html). I used `VaRExampleRMFI3eHistoricalSimulation.xls` for both Chapters. Please remove the first row before feeding it to the code.
Code in `ch15/` replictes the functionality in `VaRExampleRMFI3eModelBuilding.xls`.
