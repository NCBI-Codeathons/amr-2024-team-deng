# Team Deng

List of participants and affiliations:

- Yixiang Deng (Team Leader)
- Soham Shirolkar, University of South Florida
- Steven Weaver, Temple University
- Garima Rani, Friedrich Schiller University Jena
- Sharvari Narendra, University of Virginia
- Wengang Zhang, National Cancer Institute, NIH
  
## Project Goals

This project aims to develop a deep-learning model to predict antibiotic resistance in bacteria using genomic and structural information of bacterial receptors and antibiotics. The model will utilize graph neural networks to identify AMR biomarkers and contribute to the development of effective treatment strategies.

Keywords: Graph neural networks, AMR biomarkers, Bacterial receptors, Antibiotics, Genomic data, Structural data, Drug resistance, Machine learning

## Approach

Data collection - Using the VAMPr(VAriant Mapping and Prediction of antibiotic resistance))bioinformatics tool, we acquired a dataset consisting of information about bacteria-antibiotic relationships. 

Model construction - 
GNN Workflow for Antibiotic Resistance Prediction

- Define Nodes:
  - Treat bacteria and antibiotics as nodes in the graph.
  - Node embeddings - Using genetic information or latent representations from genomic data.

- Create Adjacency Matrix:
  - Use binary data (resistance vs non-resistance) from the VAMPr data to build the adjacency matrix.

- Graph construction:
  - Nodes: Embedded bacteria and antibiotics.
  - Edges: Represent resistance or susceptibility between bacteria and antibiotics.

- Link Prediction:
  - Perform link prediction to predict whether an unobserved bacterium (e.g., B1) is resistant to antibiotics (A1 to An).
  - The model will learn to predict edges, where a link between a bacterium and antibiotic represents resistance.


Evaluate the model by predicting whether new bacteria exhibit resistance to specific antibiotics, based on the learned graph structure.

## Results

## References 
Kim, J., Greenberg, D. E., Pifer, R., Jiang, S., Xiao, G., Shelburne, S. A., Koh, A., Xie, Y., & Zhan, X. (2020). VAMPr: VAriant Mapping and Prediction of antibiotic resistance via explainable features and machine learning. PLoS computational biology, 16(1), e1007511. https://doi.org/10.1371/journal.pcbi.1007511 


## Future Work

## NCBI Codeathon Disclaimer

This software was created as part of an NCBI codeathon, a hackathon-style event focused on rapid innovation. While we encourage you to explore and adapt this code, please be aware that NCBI does not provide ongoing support for it.

For general questions about NCBI software and tools, please visit: [NCBI Contact Page](https://www.ncbi.nlm.nih.gov/home/about/contact/)
