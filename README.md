# Manipulation Playground

A set of python scripts which set up manipulation environments for the [Coppelia-Sim](https://www.coppeliarobotics.com/) simulator. Current scenarios are:

#### `dualRamTable.py`

Dual arms mounted on desk with assorted cubes.

 <img src='pictures/dualArm.png' height=200px style='display: block'>

#### `cupPour.py` 

Table of cups filled with liquid. Another table with bowl for transfer.

<img src='pictures/cupPour.png' height=200px style='display: block'>

#### `gearInsert.py`

A gear with hole in it, and a a peg to insert it on.

 <img src='pictures/gearInsert.png' height=200px style='display: block'>

## Setup / Running

First, make sure you have the latest version of [Coppellia-Sim](https://www.coppeliarobotics.com/) installed.

Next, install [PyRep](https://github.com/stepjam/PyRep), either by [following the instructions on their repository](https://github.com/stepjam/PyRep), or with:

```
pip install pyrep
```

Then, spin up the enironment of your choice using the relevant python script. E.g:

```
python dualArmTable.py
```

## Project Aims / Further Information

[Rough Draft Roadmap Document](https://www.overleaf.com/read/xvrdrpzdckfw)