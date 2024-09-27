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
![resistance-heatmap](https://github.com/NCBI-Codeathons/amr-2024-team-deng/blob/main/resistance-heatmap.png)
### Data processing 
- Integrate and preprocess data from AST and VAMPr databases.
- Prepare molecular structure data for Chemprop input.
- Format antibiotic resistance data as labels for supervised learning.
- Calculated a resistance_rate variable based on the proportion of resistance cases for each antibiotic.
- Normalized the resistance rate using Wilson score interval for use in Chemprop.

### Model construction
Chemprop Workflow for Antibiotic Resistance Prediction

- Utilize Chemprop's built-in graph neural network architecture for molecular property prediction.
- Represent antibiotics as molecular graphs using SMILES notation.
- Tuned hyperparameters to improve modeling, including:
  - Increasing the default epoch
  - Adjusting batch size
  - Modifying model depth

### Model training and evaluation

- Train the Chemprop model on the processed AST and VAMPr data.
- Use cross-validation to assess model performance and prevent overfitting.
- Evaluate the model by predicting resistance for known antibiotics and potentially new compounds.



## Results
Our Chemprop-based model for predicting antibiotic resistance demonstrated promising results in several key areas:
 - The initial model using VAMPr data showed limitations due to the small dataset size.
 - The switch to AST data for E. coli provided more robust results.
 - Hyperparameter tuning demonstrated positive impacts on model performance.



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
```jupyter notebook notebooks/model.ipynb &```

## References 
- Kim, J., Greenberg, D. E., Pifer, R., Jiang, S., Xiao, G., Shelburne, S. A., Koh, A., Xie, Y., & Zhan, X. (2020). VAMPr: VAriant Mapping and Prediction of antibiotic resistance via explainable features and machine learning. PLoS computational biology, 16(1), e1007511. https://doi.org/10.1371/journal.pcbi.1007511 
- Heid, Esther, et al. "Chemprop: a machine learning package for chemical property prediction." Journal of Chemical Information and Modeling 64.1 (2023): 9-17.

## Future Work
- Expand the model to predict resistance for a wider range of bacteria and antibiotics.
- Incorporate additional molecular descriptors and bacterial genomic features to improve prediction accuracy.
- Develop a web interface for easy access to the AMR prediction tool.

## NCBI Codeathon Disclaimer

This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)
