import os
import sys
import csv
import math
import torch
import uproot
import numpy as np
from tqdm import tqdm
from math import cos, sin, sinh

# Ensure the current directory is in sys.path to import local modules
sys.path.append(".")
# Import the get_model function from the local networks module
from networks.example_ParticleTransformer_sophon import get_model

# Define the list of particle and scalar feature keys to read from the ROOT files
particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]

# Define scalar features that have a single value per jet
scalar_keys = [
    'label_QCD','label_Hbb','label_Hcc','label_Hgg',
    'label_H4q','label_Hqql','label_Zqq','label_Wqq',
    'label_Tbqq','label_Tbl','jet_pt','jet_eta','jet_phi',
    'jet_energy','jet_nparticles','jet_sdmass','jet_tau1',
    'jet_tau2','jet_tau3','jet_tau4','aux_genpart_eta',
    'aux_genpart_phi','aux_genpart_pid','aux_genpart_pt',
    'aux_truth_match'
]

pf_keys = particle_keys + scalar_keys

# Define the input data location 
# You can change this path to the location of your file
root_dir = "../data/JetClass/val_5M"
# Defines the list of ROOT files to process
root_files = ["HToCC_120.root", "HToCC_121.root", 
              "HToCC_122.root", "HToCC_123.root", "HToCC_124.root"]

# Container to hold configuration settings for get_model function
class DummyDataConfig:
    # map of input feature indices
    input_dicts = {"pf_features": list(range(37))}
    # Name of the model input
    input_names = ["pf_points"]
    # Expected tensor shape for the model input
    input_shapes = {"pf_points": (128, 37)}
    # Names of output labels
    label_names = ["label"]
    # Number of output classes
    num_classes = 10


data_config = DummyDataConfig()

# Check for GPU availability and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Create the model
model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)
# Put the model in evaluation mode and move it to the appropriate device
model.eval().to(device)

# Define the path for the output CSV file
output_csv_path = "HToCC_inference_with_embedding.csv"

# Open the CSV file in write mode ('w')
with open(output_csv_path, mode="w", newline="") as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Define the base header for the CSV file (general jet info and labels)
    base_header = (["file", "event_index"] +
                   ["truth_label", "label_name",
                    "jet_sdmass", "jet_mass", "jet_pt", "jet_eta", "jet_phi"])
    # Create column headers for the 128-dimensional output embedding vector
    emb_header = [f"emb_{j}" for j in range(128)]
    # Write the complete header row to the CSV file
    writer.writerow(base_header + emb_header)

    # Iterate over each ROOT file in the list
    for file_name in root_files:
        print(f"\nRunning inference on: {file_name}")
        # Construct the full file path
        file_path = os.path.join(root_dir, file_name)
        # Open the ROOT file using uproot
        with uproot.open(file_path) as f:
            # Access the 'tree' object (contains the data)
            tree = f["tree"]
            # Read the required feature arrays from the tree into a NumPy dictionary
            arrays = tree.arrays(pf_keys, library="np")
        # Define the maximum number of particles the model can handle (128)
        max_part = 128
        # Get the total number of events (jets) in the file
        total_events = len(arrays["part_px"])

        # Loop through each event in the file, showing a progress bar
        for i in tqdm(range(total_events), desc=f"{file_name}"):
            try:
                # Get the actual number of constituent particles in the current jet
                n_part = arrays["part_px"][i].shape[0]
                # Skip the event if it has more particles than the model's limit
                if n_part > max_part:
                    continue

                # build input tensor
                # Extract particle features for the current event
                particle_feats = [arrays[k][i] for k in particle_keys]
                # Extract scalar features and tile them to have the same length as the particle features (n_part)
                scalar_feats = [np.full(n_part, arrays[k][i]) for k in scalar_keys]
                # Combine particle and scalar features
                all_feats = particle_feats + scalar_feats
                # Stack all features into a 2D array (n_part, n_features)
                pf_features = np.stack(all_feats, axis=1)

                # Create a zero-padded array for the model input (max_part, n_features)
                padded = np.zeros((max_part, pf_features.shape[1]), dtype=np.float32)
                # Fill the array with the actual feature data
                padded[:n_part, :] = pf_features
                
                # Convert the NumPy array to a PyTorch tensor, add a batch dimension (unsqueeze(0)), and move it to the device
                jet_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                # Extract Lorentz vectors (px, py, pz, E) and permute dimensions (batch, feature, particle)
                lorentz_vectors = jet_tensor[:, :, 0:4].transpose(1, 2)
                # Extract non-kinematic features and permute dimensions (batch, feature, particle)
                features = jet_tensor[:, :, 4:].transpose(1, 2)
                # Create a mask tensor: 1 for valid particles, 0 for padded (checks if the sum of all features for a particle is non-zero)
                mask = (jet_tensor.sum(dim=2) != 0).unsqueeze(1)
                # Placeholder for particle coordinates if needed (not used here)
                points = None

                # Disable gradient calculation during inference to save memory and speed up computation
                with torch.no_grad():
                    # Run the inference: get the logits (raw outputs) and the embedding vector
                    logits, embedding = model(points, features, lorentz_vectors, mask)
                    # Remove the batch dimension, move the tensor to CPU, and convert it to a NumPy array
                    embedding = embedding.squeeze(0).cpu().numpy()

                # truth labels
                # Extract the one-hot encoded truth labels for the current jet
                label_array = np.array([arrays[k][i] for k in [
                    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
                    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
                    'label_Tbqq', 'label_Tbl'
                ]])
                # Determine the true class index (0-9) by finding the index of the maximum value (1)
                truth_label = int(np.argmax(label_array))
                # List of label names
                label_names = ["QCD","Hbb","Hcc","Hgg","H4q","Hqql","Zqq","Wqq","Tbqq","Tbl"]
                # Get the corresponding name for the true class
                label_name = label_names[truth_label]

                # softdrop + ungroomed mass
                # Get the SoftDrop mass value
                jet_sdmass = float(arrays["jet_sdmass"][i])
                # Get the jet's kinematic properties
                pt  = float(arrays["jet_pt"][i])
                eta = float(arrays["jet_eta"][i])
                phi = float(arrays["jet_phi"][i])
                E   = float(arrays["jet_energy"][i])

                # Convert jet's transverse momentum (pt), pseudorapidity (eta), and azimuthal angle (phi) to Cartesian momentum components (px, py, pz)
                px = pt * cos(phi)
                py = pt * sin(phi)
                # The relationship between pz, pt, and eta is: pz = pt * sinh(eta)
                pz = pt * sinh(eta)
                # Calculate the square of the three-momentum (p^2 = px^2 + py^2 + pz^2)
                p2 = px*px + py*py + pz*pz
                # Calculate the square of the invariant mass (m^2 = E^2 - p^2). Ensure it's non-negative.
                m2 = max(E*E - p2, 0.0)
                # Calculate the invariant mass (ungroomed jet mass)
                jet_mass = float(np.sqrt(m2))

                # Create the data row for the CSV file: file info, labels, jet properties, and the embedding
                row = [file_name, i, truth_label, label_name,
                       jet_sdmass, jet_mass, pt, eta, phi] + list(embedding)
                writer.writerow(row)

            # Handle any exceptions (errors) that occur during the processing of a specific event
            except Exception as e:
                print(f"Error in event {i}: {e}")
                # Continue to the next event if an error occurs
                continue

# Print confirmation message after all files have been processed
print(f"Saved CSV data to {output_csv_path}")