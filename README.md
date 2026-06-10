# AME_mujoco_sim2sim

MuJoCo sim2sim deployment for AME Locomotion.

## Usage

First, clone the AME repository:

```bash
git clone git@github.com:unitreerobotics/AME_Locomotion.git
```

Then clone this repository into it:

```bash
cd AME_Locomotion
git clone git@github.com:Rorschach7771/AME_mujoco_sim2sim.git
```

## Requirements

```bash
pip install mujoco pygame pyaml
```

## Directory Tree

Your directory tree should look like this:

```
AME_Locomotion/
├── AME_mujoco_sim2sim/
│   ├── deploy_mujoco.py
│   ├── g1_29dof.yaml
│   └── lock_pos.py
├── logs/
├── pretrained/
├── rsl_rl/
├── scripts/
├── source/
├── unitree_model/
├── run_play.sh
├── run_train.sh
└── README.md
```

## Run

```bash
cd AME_mujoco_sim2sim
python deploy_mujoco.py
```

A joystick is required to control the robot. If you don't have a joystick, you need to modify the code to use other control methods.

## Acknowledgements

This project is built upon [AME (Adaptive Motion Engine)](https://sites.google.com/leggedrobotics.com/ame-2), a remarkable framework for legged robot locomotion. AME demonstrates an elegant and highly effective approach to robust locomotion control, achieving impressive agility and adaptability across diverse terrains. Huge thanks to the AME team for their outstanding work and open-source contribution!

- Project page: https://sites.google.com/leggedrobotics.com/ame-2
- Source code: https://github.com/SII-FUSC/AME_Locomotion
