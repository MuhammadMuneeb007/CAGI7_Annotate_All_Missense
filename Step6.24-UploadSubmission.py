import synapseclient
from synapseclient import Project, Folder, File
import synapseclient
import synapseutils
from synapseclient import File
import os
# ------------------------------------------------
# Initialize Synapse connection with personal token
# ------------------------------------------------
syn = synapseclient.Synapse()
syn.login(authToken="eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc1Njc4NzE3MiwiaWF0IjoxNzU2Nzg3MTcyLCJqdGkiOiIyNTM3NiIsInN1YiI6IjM1NTAzNDUifQ.XYAeq_J8VY0kqf7P5OI8GCnYfN566ZJkg998oAWXBs_Mj58ottorfD7wtG2w3TMXXnhigsVY5A27vM5Xq00sU4kDPLhdUE_Yg_ukz3S9urMUdTfl9YU7g1VQkzT2__ky_2XysrvO5bsgjJFjU2QY3GGDiL0LqMJx7_7SphrPp7N1PS5T41XSBZjppKMhynZJ_GP6G-fuChy-RkJVZS_UleoQXpXOY6sJCsjDfkhCbCoEj91kMTlFeOlU-NxVV1mHTwdRew7B065m8L9HWW7aX8CruxFMKFKAbxM5hWE7t8ndlfhg3z0mejc0RFSx4H9qQ6xHRiyrTZ5zo_iGf7qkEA")   # replace with your token

# ------------------------------------------------
# 1. List all projects you have access to
# ------------------------------------------------
# ------------------------------------------------
# 1. List all projects you have access to
# ------------------------------------------------
PROJECT_ID = "syn69957325"       # CAGI7_Annotate_all_missense_UQ-BioSig
FOLDER_NAME = "UQ_BioSig_model_Final"
FILE_TO_UPLOAD = "Final_Submission1/UQ_BioSig_model_Final.tsv"  # <-- change this to your actual file name

 
# ----------------------------
# FIND OR CREATE FOLDER
# ----------------------------
project = syn.get(PROJECT_ID)
print(f"📂 Project: {project.name} ({PROJECT_ID})")

# check if folder exists
folders = list(syn.getChildren(PROJECT_ID, includeTypes=['folder']))
folder_id = None
for f in folders:
    if f['name'] == FOLDER_NAME:
        folder_id = f['id']
        break

# if not found, create folder
if folder_id is None:
    folder = synapseclient.Folder(name=FOLDER_NAME, parent=project)
    folder = syn.store(folder)
    folder_id = folder.id
    print(f"🆕 Created folder: {FOLDER_NAME} ({folder_id})")
else:
    print(f"✅ Found folder: {FOLDER_NAME} ({folder_id})")

# ----------------------------
# UPLOAD FILE
# ----------------------------
if not os.path.exists(FILE_TO_UPLOAD):
    raise FileNotFoundError(f"❌ File not found: {FILE_TO_UPLOAD}")

entity = synapseclient.File(FILE_TO_UPLOAD, parent=folder_id)
entity = syn.store(entity)

print(f"🎉 Uploaded file: {FILE_TO_UPLOAD} to {FOLDER_NAME} ({entity.id})")