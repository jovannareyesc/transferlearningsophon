# transferlearningsophon

## Transfer Learning Project

This README page aims to be an introduction for the ongoing transfer learning project with applications to particle physics. The project in it of itself is aimed at advancing and observing the machine learning applications to the world of particle physics, and specifically, to the task of jet-tagging. As described in the original repository for Sophon, _"...the model Sophon (Sophon (Signature-Oriented Pre-training for Heavy-resonance ObservatioN) is a method proposed for developing foundation AI models tailored for future usage in LHC experimental analyses..." _ More specifically, the Sophon is a deep learning framework developed with the goal of better classifiying jets—AKA, collimated sprays of particles produced in high-energy collisions at places like the LHC (Large Hadron Collider)—using both particle-level and jet-level features.

The bigger and more universal goal, however, is to explore representation learning in jet physics, focusing on how neuralnetowrk embeddings capture physical information across different datasets and simulation domains. By doing all of this, we are aiming for the following, overarching goal: Evaluate transfer learning potential across deep learning models and jet types (Sophon vs. ParT & Higgs, top, QCD, etc.)

This README file focuses on one core task: running inference with Sophon on a subset of the JetClass dataset (which can be accessed through https://zenodo.org/records/6619768 -> "JetClass_Pythia_val_5M.tar" -> Download & Extract) and extracting embeddings for visualization and classification. It is written to simplify and better explain the whole process.

### Steps to follow:
1. Set up a new Python venv for the project
2. Download and unzip the data from the .tar file and save to an accessible folder for the project (https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1)
3. Create a new .py file in the /sophon folder to run the model (model located in: example_ParticleTransforme_sophon.py file).
4. Run the inference script (reads ROOT files + writes a CSV file with the embeddings)
5. Explore embeddings (simple plotting)

## Requirements
- Python 3.10+
- PyTorch
- uproot, numpy, tqdm, awkward

## Install for the new venv:
'''
from repo_root/
conda create -n sophon python=3.10 -y
conda activate sophon
# Install PyTorch (pick the right command for your CUDA)
# See https://pytorch.org/get-started/locally/ for your platform; example (CPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Core deps
pip install uproot numpy tqdm
'''

# Data
Once you have downlaoded the subset of the JetClass dataset, place the .root files in data/JetClass/val_5M. The example config file as well as the inference scrip, expect around 5 of the validation files to successfully run inference on them: 
'''
HToBB_120.root, HToBB_121.root,...,HToBB_124.root
'''


   
