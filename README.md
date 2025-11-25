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
```sh
from repo_root/
conda create -n sophon python=3.10 -y
conda activate sophon
# Install PyTorch (pick the right command for your CUDA)
# See https://pytorch.org/get-started/locally/ for your platform; example (CPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Core deps
pip install uproot numpy tqdm
```

## Data
Once you have downlaoded the subset of the JetClass dataset, place the .root files in data/JetClass/val_5M. The example config file as well as the inference scrip, expect around 5 of the validation files to successfully run inference on them: 
'''
HToBB_120.root, HToBB_121.root,...,HToBB_124.root

## Inference script

Access the inference_jetclass.py file in this repository and download. The inference script:
1. Reads particle-flow and scalar features from each event in the .root files
2. Pads up to the 128 mparticles and skips events/logits events
3. Calls the function getm_model from example_ParticleTransformer_sophon.py
4. Writes CSV files with: file name, event index, truth label, some of the main kinematic features, and a 128-D vector embedding
5. From a local terminal, run:
```sh
python inference_with_embedding.py
```
_It will take a couple minutes per root file_

### Steps to follow in order to successfully run data through Sophon

The steps that are required to be followed in order to run a comparison between QCD (in this case, background noise) and any of the other jet classes available in the val_5M dataset through Sophon are as follows:
1. Select which jet class to compare to QCD.
- For example, let's take the HToCC files.
One will notice that in the val_5M dataset, there are a total of five files related to the HToCC class; all labeled as "HToCC_120.root", "HToCC_121.root",...,"HToCC_124.root". When feeding these to Sophon — or any other jet class, really — we must include all five files in our code as such:
```sh
root_files = ["HToCC_120.root", "HToCC_121.root","HToCC_122.root","HToCC_123.root","HToCC_124.root"]
```
2. Make sure you pf_keys is the correct dimension.
- If the model receives any other dimensions when it comes to what we are feeding it, it will not run.
Make sure pf_keys is composed of particle_keys + scalar_keys in order to run the data through the model successfully. Consequently, particle_keys and scalar_keys must each contain all labels that are in the inference script above. They must look like this:
```sh
particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]
scalar_keys = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
    'label_Tbqq', 'label_Tbl', 'jet_pt', 'jet_eta', 'jet_phi',
    'jet_energy', 'jet_nparticles', 'jet_sdmass', 'jet_tau1',
    'jet_tau2', 'jet_tau3', 'jet_tau4', 'aux_genpart_eta',
    'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt',
    'aux_truth_match'
]
```
3. If you want a record of the inference, make sure to save it to a csv file.
- In order to successfully save the data being run through Sophon, the cleanest and most accessible way to do it is to save the inference output into a csv file under an appropriate name (e.g.: inference_sophon.csv)
To do this, we must first "import csv" at the very top of our inference script.
-Then, before running the inference loop we create a new variable under an appropriate name as such:
```sh
OUTPUT_CSV = "inference_sophon.csv"
```
-For this to actually create and write into the CSV file, we must open the file using python's csv.writer before the event loop, and define the header row we want. For example:
```sh
with open(OUTPUT_CSV, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # write header (following format is what we have been using and what works best so far)
    header = ["file", "event_index", "truth_label", "label_name"] + \
             [f"emb_{j}" for j in range(128)]
    writer.writerow(header)

    # now begin your inference loop 
    for i in range(total_events):
        # compute embedding, truth label, etc.
        row = [file_name, i, truth_label, label_name] + list(embedding)
        writer.writerow(row)
```
In summary, the main points are:
1. Open the file using with open(...) before the loop.
2. Create the writer object.
3. Write in the header.
4. Write exactly one CSV row per event inside the inference loop.
