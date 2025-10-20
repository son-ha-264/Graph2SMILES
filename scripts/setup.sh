mamba create -y -n graph2smiles python=3.9 tqdm
mamba activate graph2smiles
#mamba install -y pytorch=2.1 torchvision torchtext -c pytorch
mamba install -y rdkit -c conda-forge

# pip dependencies
pip3 install torch==2.1 torchvision torchtext --index-url https://download.pytorch.org/whl/cu121
pip3 install gdown OpenNMT-py networkx selfies