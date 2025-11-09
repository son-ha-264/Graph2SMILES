mamba create -y -n graph2smiles_old2 python=3.8 tqdm
mamba activate graph2smiles_old2
mamba install -y rdkit -c conda-forge

# pip dependencies
pip3 install torch==2.1 torchvision torchtext --index-url https://download.pytorch.org/whl/cu121
pip3 install gdown numpy==1.22 OpenNMT-py==1.2.0  networkx==2.5 selfies==1.0.3