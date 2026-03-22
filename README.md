# Skills_from_transition
## Backgrounds
The focus of the repo is the temporal abstraction of action sequences or motion trajectories.

Temporal abstraction involves breaking down a time-series signal into distinct task-level segments, a process also known as "skill discovery." For example, when controlling a robot to perform a water-fetching task, the overall behavior consists of high-level actions or "skills"—such as reach, grasp, pour, and place—each of which is composed of lower-level actions like joint position, velocity, and force commands. Given a full sequence of these low-level actions that successfully completes the entire task, the goal is to develop an end-to-end model that automatically segments a sequence into constituent task subsequences, which is the essence of temporal abstraction.

This is commonly achieved using a self-supervised approach: the original low-level action sequence is fed into a model, where an encoder transforms it into a sequence of skills. A decoder reconstructs the original low-level sequence from this skill representation. By minimizing reconstruction error while enforcing appropriate constraints, the model learns to form a meaningful intermediate representation of the underlying skills.

## Datasets
The model is currently traind on minimalist simulation data, i.e. the expert trajectories of **nine rooms** task.

FUrther extention includes the benchamarks from the **embodied intelligence** studies.

## Usage

Train and evaluate the model using the following commands:

```bash
python train.py   # training
python test.py    # evaluation
