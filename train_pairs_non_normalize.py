import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from networks.nets import EncoderMLP  # Assuming you have this module
from envs.tree import NaryTreeEnvironment
import matplotlib.pyplot as plt  # For plotting

# Parameters
gamma = 0.3
num_trajectories = 5000
dataset_size = 10000
batch_size = 32  
num_epochs = 500
learning_rate = 0.001

# Initialize environment and generate trajectories
env = NaryTreeEnvironment(depth=4, branching_factor=2)
trajectories = env.get_trajectories(num_trajectories)

# Precompute episode lengths for efficiency
episode_lengths = np.array([len(traj) for traj in trajectories])

# Function to generate points
def generate_points_vectorized(num_points):
    traj_indices = np.random.randint(0, num_trajectories, size=num_points)
    traj_lengths = episode_lengths[traj_indices]
    t0 = np.random.randint(0, traj_lengths)
    
    d1, d2 = np.random.geometric(p=gamma, size=(2, num_points))
    d1 = d2

    valid_mask = (t0 + d1 + d2 < traj_lengths)
    while not valid_mask.all():
        invalid_indices = np.where(~valid_mask)[0]
        traj_indices[invalid_indices] = np.random.randint(0, num_trajectories, size=len(invalid_indices))
        traj_lengths[invalid_indices] = episode_lengths[traj_indices[invalid_indices]]
        t0[invalid_indices] = np.random.randint(0, traj_lengths[invalid_indices])
        d1[invalid_indices], d2[invalid_indices] = np.random.geometric(p=gamma, size=(2, len(invalid_indices)))
        valid_mask = (t0 + d1 + d2 < traj_lengths)

    t1 = t0 + d1
    t2 = t1 + d2

    starts = np.array([trajectories[traj_idx][t0_idx] for traj_idx, t0_idx in zip(traj_indices, t0)])
    waypoints = np.array([trajectories[traj_idx][t1_idx] for traj_idx, t1_idx in zip(traj_indices, t1)])
    goals = np.array([trajectories[traj_idx][t2_idx] for traj_idx, t2_idx in zip(traj_indices, t2)])

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

    # Encode the start and goal pair
    start_goal_encoding = encoder(torch.tensor([start]).long(), torch.tensor([goal]).long())
    #start_goal_encoding = start_goal_encoding / torch.norm(start_goal_encoding, dim=-1, keepdim=True)  # Normalization removed

    # Initialize variables to track the most similar node
    max_similarity = -float('inf')
    predicted_waypoint = None
    similarity_scores = []

    # Iterate through all nodes in the environment
    for node in range(env.num_nodes):
        # Encode the node as a waypoint
        waypoint_encoding = encoder(torch.tensor([node]).long(), torch.tensor([node]).long())
        #waypoint_encoding = waypoint_encoding / torch.norm(waypoint_encoding, dim=-1, keepdim=True)  # Normalization removed

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
                # Encode start and goal
                start_goal_encoding = encoder(torch.tensor([start]).long(), torch.tensor([goal]).long())
                #start_goal_encoding = start_goal_encoding / torch.norm(start_goal_encoding, dim=-1, keepdim=True)  # Normalization removed

                # Vectorized encoding of all nodes as waypoints
                all_nodes = torch.arange(env.num_nodes).long()
                waypoint_encodings = encoder(all_nodes, all_nodes)
                #waypoint_encodings = waypoint_encodings / torch.norm(waypoint_encodings, dim=-1, keepdim=True)  # Normalization removed

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
embedding_dim = 16
hidden_dims = [128, 128, 128]
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

        # Encode start-goal
        sg_encoded = encoder(start.long(), goal.long())
        #sg_encoded = sg_encoded / torch.norm(sg_encoded, dim=-1, keepdim=True)  # Normalization removed

        # Encode waypoint-waypoint
        ww_encoded = encoder(waypoint.long(), waypoint.long())
        #ww_encoded = ww_encoded / torch.norm(ww_encoded, dim=-1, keepdim=True)  # Normalization removed

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
