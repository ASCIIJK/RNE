# RNE
Public code for RNE, which is a new SOTA for class incremental learning!

# Install
Use the commend "conda env create -f environment.yml" to build the basic environment.
If the automatic installation fails, please configure the environment manually according to the "requirements.txt" file. We do not test the installation process, any problems could be proposed in the comments section.

# How to use
We construct our project refering the PyCil library ( https://github.com/G-U-N/PyCIL ).If you wants to reproduce all the results, use the commend:

<p align="center"><strong>sh dist_train.sh</strong></p>
  
Results will be outputs in corresponding logs, which locate in "logs/'model_name'/'settings'/". And the check_point named "taskx.pkl" of last epoch at each task will also be saved in the same path.

if you just want to test the results on just one setting, you can use the commend:

<p align="center"><strong>python main.py --config=./exps/[MODEL NAME].json</strong></p>

# Updates
**2024.5.20** We build the project. And we update the code to test on CIFAR-100 for three incremental setting. (include "RNE" and "RNE-compress")

**2024.5.21** We update the code to test on ImageNet-100 and Food-101 for three setting (however, we do not test the result, any problems could be proposed in the comments section).
