import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem 

# conda activate my-rdkit-env  
suppl = Chem.SDMolSupplier('../datainfo/aa_raw_data/NCI60/ComboCompoundSet.sdf')
count = 0
for mol in suppl:
    count += 1
    if mol is None:
        continue
    else:
        print('---------- ' + str(count) + ' ----------')
        print(Chem.MolToSmiles(mol))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        fp_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, fp_arr)
        print(fp_arr)


# m2 = Chem.MolFromSmiles('C1CCC1')
# print(Chem.MolToMolBlock(m2)) 