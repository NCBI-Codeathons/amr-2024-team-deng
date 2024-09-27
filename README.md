# Team Deng

List of participants and affiliations:

- Yixiang Deng (Team Leader, University of Delaware)
- Soham Shirolkar, University of South Florida
- Steven Weaver, Temple University
- Garima Rani, Friedrich Schiller University Jena
- Sharvari Narendra, University of Virginia
- Wengang Zhang, National Cancer Institute, NIH
- Shu Cheng, Tennessee Department of Health Laboratory Services
  
## Project Goals

This project aims to develop a deep-learning model to predict antibiotic resistance in bacteria using genomic and structural information of bacterial receptors and antibiotics. The model will utilize graph neural networks to identify AMR biomarkers and predict the antibiotic resistence of new compounds.

Keywords: Graph neural networks, AMR detection, Antibiotics, Genomic data, Structural data, Drug resistance, Machine learning

## Approach
![Workflow chart](https://github.com/NCBI-Codeathons/amr-2024-team-deng/blob/main/workflow_chart.png)
### Data collection 

- Using the VAMPr(VAriant Mapping and Prediction of antibiotic resistance) database, we acquired a dataset consisting of information about bacteria-antibiotic relationships.
- Retrieve data from AST (Antimicrobial Susceptibility Testing) and MicroBig-EE databases with a focus on E coli.
- Downlaod antibiotics' structure data from PubChem

### Data processing 
- Integrate and preprocess data from AST and VAMPr databases.
- Prepare molecular structure data for Chemprop input.
- Format antibiotic resistance data as labels for supervised learning.

### Model construction
Chemprop Workflow for Antibiotic Resistance Prediction

- Utilize Chemprop's built-in graph neural network architecture for molecular property prediction.
- Represent antibiotics as molecular graphs using SMILES notation.
- Incorporate bacterial genomic features as additional input to the model.

### Model training and evaluation

- Train the Chemprop model on the processed AST and VAMPr data.
- Use cross-validation to assess model performance and prevent overfitting.
- Evaluate the model by predicting resistance for known antibiotics and potentially new compounds.

## Getting Started
### Prerequisites
- Python 3.7+
- Chemprop
- RDKit
- PyTorch
- PyTorch Geometric
- pandas
- Google Cloud BigQuery
- scikit-learn

### Installation 
```pip install torch torch-geometric pandas google-cloud-bigquery chemprop rdkit scikit-learn```

### Usage

## Results

## References 
Kim, J., Greenberg, D. E., Pifer, R., Jiang, S., Xiao, G., Shelburne, S. A., Koh, A., Xie, Y., & Zhan, X. (2020). VAMPr: VAriant Mapping and Prediction of antibiotic resistance via explainable features and machine learning. PLoS computational biology, 16(1), e1007511. https://doi.org/10.1371/journal.pcbi.1007511 
Heid, Esther, et al. "Chemprop: a machine learning package for chemical property prediction." Journal of Chemical Information and Modeling 64.1 (2023): 9-17.

## Future Work
## NCBI Codeathon Disclaimer

This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)
