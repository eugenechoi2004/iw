import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from networks.nets import EncoderMLP  # Assuming you have this module
from envs.tree import NaryTreeEnvironment
import matplotlib.pyplot as plt  # Added for plotting

# Parameters
gamma = 0.3
num_trajectories = 50000
dataset_size = 10000
batch_size = 128
num_epochs = 200
learning_rate = 0.001

# Initialize environment and generate trajectories
env = NaryTreeEnvironment(depth=4, branching_factor=2)
trajectories = env.get_trajectories(num_trajectories)

# Precompute episode lengths for efficiency
episode_lengths = np.array([len(traj) for traj in trajectories])

# Function to generate points
def generate_points_vectorized(num_points):
    min_L = 2 
    max_L = 8  
    traj_indices = np.random.randint(0, num_trajectories, size=num_points)
    traj_lengths = episode_lengths[traj_indices]

    # Generate random path lengths L between min_L and max_L (inclusive)
    L = np.random.randint(min_L, max_L + 1, size=num_points)

    # Ensure that L <= traj_lengths
    valid_mask = (L <= traj_lengths)
    while not valid_mask.all():
        invalid_indices = np.where(~valid_mask)[0]

        # Resample traj_indices and L for invalid indices
        traj_indices[invalid_indices] = np.random.randint(0, num_trajectories, size=len(invalid_indices))
        traj_lengths[invalid_indices] = episode_lengths[traj_indices[invalid_indices]]
        L[invalid_indices] = np.random.randint(min_L, max_L + 1, size=len(invalid_indices))

        valid_mask = (L <= traj_lengths)

    # Randomly select t0 ensuring that t0 + L <= traj_length
    t0 = np.random.randint(0, traj_lengths - L + 1)

    # Calculate t2 (goal index)
    t2 = t0 + L - 1

    # Calculate t1 (waypoint index)
    t1 = np.zeros_like(t0)
    odd_mask = (L % 2 == 1)
    even_mask = ~odd_mask

    # For odd lengths, waypoint is the middle node
    t1[odd_mask] = t0[odd_mask] + L[odd_mask] // 2

    # For even lengths, waypoint is the middle node closest to start
    t1[even_mask] = t0[even_mask] + (L[even_mask] // 2) - 1

    # Extract the nodes from trajectories
    starts = np.array([trajectories[traj_idx][t0_idx]
                       for traj_idx, t0_idx in zip(traj_indices, t0)])
    waypoints = np.array([trajectories[traj_idx][t1_idx]
                          for traj_idx, t1_idx in zip(traj_indices, t1)])
    goals = np.array([trajectories[traj_idx][t2_idx]
                      for traj_idx, t2_idx in zip(traj_indices, t2)])

    return starts, waypoints, goals


def evaluate(env, encoder):
    """
    Evaluates the environment and encoder by generating a random start and goal pair,
    encoding them, and finding the most similar node based on dot product similarity.

    Args:
        env: An instance of NaryTreeEnvironment.
        encoder: The encoder model.

    Returns:
        start: The randomly selected start node.
        goal: The randomly selected goal node.
        predicted_waypoint: The node with the highest similarity score.
        similarity_scores: A list of similarity scores for each node.
    """
    # Generate random start and goal nodes
    start, goal = np.random.randint(0, env.num_nodes, size=2)

    # Encode the start and goal pair and normalize
    start_goal_encoding = encoder(torch.tensor([start]).long(), torch.tensor([goal]).long())
    start_goal_encoding = start_goal_encoding / torch.norm(start_goal_encoding, dim=-1, keepdim=True)

    # Initialize variables to track the most similar node
    max_similarity = -float('inf')
    predicted_waypoint = None
    similarity_scores = []

    # Iterate through all nodes in the environment
    for node in range(env.num_nodes):
        # Encode the node as a waypoint and normalize
        waypoint_encoding = encoder(torch.tensor([node]).long(), torch.tensor([node]).long())
        waypoint_encoding = waypoint_encoding / torch.norm(waypoint_encoding, dim=-1, keepdim=True)

        # Compute similarity (dot product)
        similarity = torch.matmul(start_goal_encoding, waypoint_encoding.T).item()
        similarity_scores.append(similarity)

        # Update the most similar node
        if similarity > max_similarity:
            max_similarity = similarity
            predicted_waypoint = node

    # Print the final results
    print(f"Start node: {start}")
    print(f"Goal node: {goal}")
    print(f"Predicted waypoint: {predicted_waypoint}")
    print(f"Maximum similarity score: {max_similarity}")

    return start, goal, predicted_waypoint, similarity_scores

def evaluate_model(env, encoder, path_lengths=[3, 4, 5, 6], num_samples=100):
    """
    Evaluates the model by generating random start and goal pairs with specific path lengths,
    uses the model to predict waypoints, and computes accuracy.

    For even path lengths, if the predicted waypoint matches any of the middle two nodes,
    it is considered correct.

    Args:
        env: The NaryTreeEnvironment instance.
        encoder: The trained encoder model.
        path_lengths: List of path lengths to evaluate.
        num_samples: Number of samples per path length category.

    Returns:
        A dictionary with path length as key and accuracy as value.
    """
    results = {}
    for L in path_lengths:
        correct_predictions = 0
        total_predictions = 0
        samples_found = 0

        print(f"\nEvaluating path length {L}:")

        while samples_found < num_samples:
            # Randomly select start and goal nodes
            start = np.random.randint(0, env.num_nodes)
            goal = np.random.randint(0, env.num_nodes)

            # Get the path between start and goal
            node_path = env.get_node_path(start, goal)

            # Check if the path length is L
            if len(node_path) == L:
                # Use the model to predict the waypoint
                # Encode start and goal and normalize
                start_goal_encoding = encoder(torch.tensor([start]).long(), torch.tensor([goal]).long())
                start_goal_encoding = start_goal_encoding / torch.norm(start_goal_encoding, dim=-1, keepdim=True)

                # Vectorized encoding of all nodes as waypoints
                all_nodes = torch.arange(env.num_nodes).long()
                waypoint_encodings = encoder(all_nodes, all_nodes)
                waypoint_encodings = waypoint_encodings / torch.norm(waypoint_encodings, dim=-1, keepdim=True)

                # Compute similarities
                similarities = torch.matmul(start_goal_encoding, waypoint_encodings.T).squeeze(0).detach().numpy()

                # Find the node with the highest similarity
                predicted_waypoint = np.argmax(similarities)

                # Get the middle node(s)
                if L % 2 == 1:
                    # Odd length, single middle node
                    middle_index = L // 2
                    correct_waypoints = [node_path[middle_index]]
                else:
                    # Even length, two middle nodes
                    middle_indices = [L // 2 - 1, L // 2]
                    correct_waypoints = [node_path[i] for i in middle_indices]

                # Check if the predicted waypoint is among the correct middle node(s)
                if predicted_waypoint in correct_waypoints:
                    correct_predictions += 1

                total_predictions += 1
                samples_found += 1

        accuracy = correct_predictions / total_predictions
        results[L] = accuracy
        print(f"Accuracy for path length {L}: {accuracy * 100:.2f}% (based on {total_predictions} samples)")

    return results

# Generate dataset
starts, waypoints, goals = generate_points_vectorized(dataset_size)
dataset = np.stack([starts, waypoints, goals], axis=1)

# Custom Dataset Class
class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.long)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

# Create Dataset and DataLoader
trajectory_dataset = TrajectoryDataset(dataset)
data_loader = DataLoader(trajectory_dataset, batch_size=batch_size, shuffle=True)

# Encoder Network
num_cat = env.num_nodes
embedding_dim = 32
hidden_dims = [128, 128, 128, 128, 128, 128]
output_dim = 32
encoder = EncoderMLP(num_cat=num_cat, embedding_dim=embedding_dim, hidden_dims=hidden_dims, output_dim=output_dim)

# Optimizer and Loss
optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Initialize accuracy tracking
path_lengths = [3, 4, 5, 6]
accuracy_history = {L: [] for L in path_lengths}

# Training Loop
for epoch in range(num_epochs):
    encoder.train()
    epoch_loss = 0.0

    for batch in data_loader:
        start, waypoint, goal = batch[:, 0], batch[:, 1], batch[:, 2]  # Unpack batch

        # Encode start-goal and normalize
        sg_encoded = encoder(start.long(), goal.long())
        sg_encoded = sg_encoded / torch.norm(sg_encoded, dim=-1, keepdim=True)  # Normalize

        # Encode waypoint-waypoint and normalize
        ww_encoded = encoder(waypoint.long(), waypoint.long())
        ww_encoded = ww_encoded / torch.norm(ww_encoded, dim=-1, keepdim=True)  # Normalize

        # Compute similarity matrix (dot product)
        output = sg_encoded @ ww_encoded.T  # Matrix multiplication

        # Target: indices of correct classes (diagonal)
        target = torch.arange(output.size(0)).long().to(output.device)

        # Compute loss
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Evaluate model after each epoch
    start, goal, predicted_waypoint, similarity_scores = evaluate(env, encoder)
    results = evaluate_model(env, encoder, path_lengths=path_lengths, num_samples=100)

    # Record accuracies
    for L in path_lengths:
        accuracy_history[L].append(results[L])

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.4f}")

# Plotting the accuracies over epochs
plt.figure(figsize=(10, 6))
for L in path_lengths:
    plt.plot(range(1, num_epochs+1), accuracy_history[L], label=f'Path Length {L}')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs for Different Path Lengths')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
