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
# Never commit credentials. Configure SYNAPSE_AUTH_TOKEN in the environment.
token = os.environ.get("SYNAPSE_AUTH_TOKEN")
if not token:
    raise RuntimeError("Set SYNAPSE_AUTH_TOKEN in the environment before uploading.")
syn.login(authToken=token)   # replace with your token

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