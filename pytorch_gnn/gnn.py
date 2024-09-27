from google.cloud import bigquery
import pandas as pd
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# Initialize the BigQuery client
client = bigquery.Client()

query = """
    SELECT id, biosample_acc, taxgroup_name, scientific_name, epi_type, isolation_source, geo_loc_name,
           target_acc, antibiotic, phenotype, measurement_sign, mic, mic_secondary,
           disk_diffusion, disk_diffusion_secondary, standard, reagent, platform, vendor, host,
           collection_date, creation_date, bioproject_acc, checksum
    FROM `ncbi-2024-amr-codeathon.pdbrowser.ast`
    LIMIT 20000
"""

# Execute the query and load the data into a DataFrame
df = client.query(query).to_dataframe()

# Create graph nodes for bacteria (using 'id') and antibiotics
bacteria_samples = df['id'].unique()
antibiotics = df['antibiotic'].unique()

# Encode bacteria and antibiotics as nodes
bacteria_node_map = {b: i for i, b in enumerate(bacteria_samples)}
antibiotic_node_map = {a: len(bacteria_node_map) + i for i, a in enumerate(antibiotics)}

# Build edges based on resistance/susceptibility from the 'phenotype' column
edges = []
edge_labels = []

for _, row in df.iterrows():
    b_node = bacteria_node_map[row['id']]
    a_node = antibiotic_node_map[row['antibiotic']]
    
    # Create an edge between bacteria and antibiotic
    edges.append((b_node, a_node))
    
    # Edge label: 1 for resistance, 0 for susceptibility (using 'phenotype' column)
    label = 1 if row['phenotype'].lower() == 'resistant' else 0
    edge_labels.append(label)

# Convert edges and labels to torch tensors
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_labels = torch.tensor(edge_labels, dtype=torch.float)

# Create node features (you can use one-hot encoding or embeddings for bacteria/antibiotics)
num_nodes = len(bacteria_node_map) + len(antibiotic_node_map)
node_features = torch.eye(num_nodes, dtype=torch.float)

# Create PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index, y=edge_labels)
# Optional: Print some details
print(f'Number of nodes: {num_nodes}')
print(f'Number of edges: {len(edges)}')

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        # Define the layers for the GNN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Third Graph Convolution Layer (for output)
        x = self.conv3(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(in_channels=data.num_node_features, hidden_channels=16, out_channels=1).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for link prediction

def train():
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Get the source and target nodes for each edge
    source_nodes = data.edge_index[0]
    target_nodes = data.edge_index[1]

    # Average the model's output for both source and target nodes to create an edge prediction
    edge_predictions = (out[source_nodes] + out[target_nodes]) / 2

    # Compute the loss based on edge predictions and edge labels
    loss = criterion(edge_predictions.view(-1), data.y)
    loss.backward()
    
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        # Generate the output by passing node features through the model
        out = model(data.x, data.edge_index)  # x (node features) and edge_index

        # Get predictions for source and target nodes
        source_nodes = data.edge_index[0]
        target_nodes = data.edge_index[1]
        
        # Average the model's output for both source and target nodes
        edge_predictions = (out[source_nodes] + out[target_nodes]) / 2

        # Apply a threshold to convert predictions to binary (0 or 1)
        pred = (edge_predictions > 0.5).float()
        print(pred)

        # Ensure pred and data.y have the same size
        pred = pred.view(-1)
        true_labels = data.y.view(-1)

        # Compare the predicted labels with the true labels
        correct = pred.eq(true_labels).sum().item()

        # Compute accuracy
        accuracy = correct / true_labels.size(0)

        return accuracy

accuracy = test()
print(f'Test Accuracy: {accuracy:.4f}')
