
# DAttProt: A Double Attention Model for Enzyme Protein Classification Based on Transformer Encoders and Multi-Scale Convolutions

Code for our paper "DAttProt: A Double Attention Model for Enzyme Protein Classification Based on Transformer Encoders and Multi-Scale Convolutions"

## Requirements

The code has been tested running under Python 3.6.4 and 3.7.4, with the following packages and their dependencies installed:

| Python | 3.6.4 | 3.7.4 | Comment |
|--|--|--|--|
| numpy | 1.19.4 | 1.16.5 | |
| torch | 1.7.1 | 1.3.1 | We write a Transformer encoder module for torch < 1.3, it is however not compatible with ```nn.TransformerEncoder``` |
| pandas | 1.1.5 | 0.25.1 | |
| lxml | 4.2.5 | 4.4.1 | |
| tqdm | 4.51.0 | 4.36.1 | This package is not obligatory, you can remove ```from tqdm import tqdm``` and simply change ```for element in tqdm(iterable)``` to ```for element in iterable``` |


## Datasets

Dataset files applied in pre-training and fine-tuning are all large files and not included in this repository. The following dataset files are required:

- `/dataset/SwissProt/uniprot_sprot.xml`  is the Swiss-Prot dataset for pre-training, can be downloaded from [the UniProt database](https://www.uniprot.org/downloads).
- `/dataset/DEEPre/DEEPre_data.pkl` and `\dataset\ECPred\ECPred_data.pkl` are dataset files of the DEEPre and ECPred database. Our data representation and segmentation are same with that in "[UDSMProt: universal deep sequence models for protein classification](https://doi.org/10.1093/bioinformatics/btaa003)" by Strodthoff et al., 2020. Thus these dataset files can be directly pulled from [the Github repository of UDSMProt](https://github.com/nstrodt/UDSMProt/tree/master/git_data).
- `/utils/motif_utils/dataset/uniprot_sprot.dat` is the Swiss-Prot dataset for the model interpretability experiments, can also be downloaded from  [the UniProt database](https://www.uniprot.org/downloads).

## Usage

### Dataset Pre-processing
Generate `.npy` dataset files to `/dataset/SwissProt and DEEPre and ECPred/`
```bash
git clone https://github.com/zhanglabNKU/DAttProt.git
cd DAttProt/utils/dataset_utils/
python SwissProt.py
python DEEPre.py
python ECPred.py
```

### Pre-training
If you want to modify hyper-parameters, modify codes at the beginning of `pretrain.py`, please refer to the annotation of each parameter.
```bash
# continue from bash commands above
cd ../../
python pretrain.py
```

### Fine-tuning
If you want to modify hyper-parameters, modify codes at the beginning of `deepre.py` and `ecpred.py`, please refer to the annotation of each parameter.
```bash
# continue from bash commands above
python deepre.py
python ecpred.py
```

Here is an example of the `mainclasses` variable setting in `DEEPre.py`:

- Level 0:
  
  `mainclasses = ''` for enzymes and non-enzyme proteins
- Level 1:
  
  `mainclasses = '1'` for main class prediction of enzymes
- Level 2:
  
  `mainclasses = '1.1'` for Oxidoreductases with EC number 1.x.x.x
  
  `mainclasses = '1.2'` for Transferases with EC number 2.x.x.x
  
  `mainclasses = '1.3'` for Hydrolases with EC number 3.x.x.x
  
  `mainclasses = '1.4'` for Lyases with EC number 4.x.x.x
  
  `mainclasses = '1.5'` for Isomerases with EC number 5.x.x.x
  
  `mainclasses = '1.6'` for Ligases with EC number 6.x.x.x
  
### Interpretability Analysis
`/utils/motif_utils/generate_motif_test_data.py` selects sequences from DEEPre and ECPred datasets whose motif features are annotated in the Swiss-Prot database and generates `/utils/motif_utils/dataset/motif.json and motif.pkl`.
```bash
# continue from bash commands above
cp saved_models/{model_name}.pkl utils/motif_utils/saved_models/{model_name}.pkl
cd utils/motif_utils/
python generate_motif_test_data.py
python double_attention.py
```
