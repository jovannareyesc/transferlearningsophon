import os
import sys
import torch
import uproot
import numpy as np
from tqdm import tqdm
import csv
from math import cos, sin, sinh

sys.path.append(".")
from networks.example_ParticleTransformer_sophon import get_model

# particle and scalar feature keys
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

pf_keys = particle_keys + scalar_keys

root_dir = "../data/JetClass/val_5M"
root_files = ["HToCC_120.root", "HToCC_121.root", 
              "HToCC_122.root", "HToCC_123.root", "HToCC_124.root"]

# dummy config for model
class DummyDataConfig:
    input_dicts = {"pf_features": list(range(37))}
    input_names = ["pf_points"]
    input_shapes = {"pf_points": (128, 37)}
    label_names = ["label"]
    num_classes = 10

data_config = DummyDataConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = get_model(data_config, num_classes=data_config.num_classes, export_embed=True)
model.eval().to(device)

output_csv_path = "HToCC_inference_with_embedding.csv"

with open(output_csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # deleted probs columns, only keep embedding
    base_header = (["file", "event_index"] +
                   ["truth_label", "label_name",
                    "jet_sdmass", "jet_mass", "jet_pt", "jet_eta", "jet_phi"])
    emb_header = [f"emb_{j}" for j in range(128)]
    writer.writerow(base_header + emb_header)

    for file_name in root_files:
        print(f"\nRunning inference on: {file_name}")
        file_path = os.path.join(root_dir, file_name)
        with uproot.open(file_path) as f:
            tree = f["tree"]
            arrays = tree.arrays(pf_keys, library="np")
        max_part = 128
        total_events = len(arrays["part_px"])

        for i in tqdm(range(total_events), desc=f"{file_name}"):
            try:
                n_part = arrays["part_px"][i].shape[0]
                if n_part > max_part:
                    continue

                # build input tensor
                particle_feats = [arrays[k][i] for k in particle_keys]
                scalar_feats = [np.full(n_part, arrays[k][i]) for k in scalar_keys]
                all_feats = particle_feats + scalar_feats
                pf_features = np.stack(all_feats, axis=1)

                padded = np.zeros((max_part, pf_features.shape[1]), dtype=np.float32)
                padded[:n_part, :] = pf_features

                jet_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                lorentz_vectors = jet_tensor[:, :, 0:4].transpose(1, 2)
                features = jet_tensor[:, :, 4:].transpose(1, 2)
                mask = (jet_tensor.sum(dim=2) != 0).unsqueeze(1)
                points = None

                with torch.no_grad():
                    logits, embedding = model(points, features, lorentz_vectors, mask)
                    embedding = embedding.squeeze(0).cpu().numpy()

                # truth labels
                label_array = np.array([arrays[k][i] for k in [
                    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
                    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
                    'label_Tbqq', 'label_Tbl'
                ]])
                truth_label = int(np.argmax(label_array))
                label_names = ["QCD","Hbb","Hcc","Hgg","H4q","Hqql","Zqq","Wqq","Tbqq","Tbl"]
                label_name = label_names[truth_label]

                # softdrop + ungroomed mass
                jet_sdmass = float(arrays["jet_sdmass"][i])
                pt  = float(arrays["jet_pt"][i])
                eta = float(arrays["jet_eta"][i])
                phi = float(arrays["jet_phi"][i])
                E   = float(arrays["jet_energy"][i])

                px = pt * cos(phi)
                py = pt * sin(phi)
                pz = pt * sinh(eta)
                p2 = px*px + py*py + pz*pz
                m2 = max(E*E - p2, 0.0)
                jet_mass = float(np.sqrt(m2))

                row = [file_name, i, truth_label, label_name,
                       jet_sdmass, jet_mass, pt, eta, phi] + list(embedding)
                writer.writerow(row)

            except Exception as e:
                print(f"Error in event {i}: {e}")
                continue

print(f"Saved CSV data to {output_csv_path}")