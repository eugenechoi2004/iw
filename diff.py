import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from networks.nets import EncoderHyperbolicMLP  # Assuming you have this module
from envs.tree import NaryTreeEnvironment
import matplotlib.pyplot as plt  # For plotting
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianAdam
import os  # For directory operations


# Parameters
gamma = 0.3
num_trajectories = 50_000
dataset_size = 10_000
batch_size = 128
num_epochs = 200
learning_rate = 0.001

# Initialize environment and generate trajectories
env = NaryTreeEnvironment(depth=4, branching_factor=2)
trajectories = env.get_trajectories(num_trajectories)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    encoding them, and finding the most similar node based on hyperbolic distance.
    """
    # Generate random start and goal nodes
    start, goal = np.random.randint(0, env.num_nodes, size=2)
    
    # Encode the start and goal pair
    start_goal_encoding = encoder(torch.tensor([start]).long(), torch.tensor([goal]).long())
    
    # Ensure start_goal_encoding has shape [1, embedding_dim]
    if start_goal_encoding.dim() == 1:
        start_goal_encoding = start_goal_encoding.unsqueeze(0)
    
    # Encode all nodes as waypoints
    all_nodes = torch.arange(env.num_nodes).long()
    waypoint_encodings = encoder(all_nodes, all_nodes)
    
    # Compute hyperbolic distances between start_goal_encoding and all waypoint_encodings
    distances = manifold.dist(start_goal_encoding, waypoint_encodings)
    # Convert distances to similarities
    similarities = -distances.detach().numpy()
    
    # Find the node with the highest similarity (smallest distance)
    predicted_waypoint = np.argmax(similarities)
    max_similarity = similarities[predicted_waypoint]
    
    # Print the final results
    print(f"Start node: {start}")
    print(f"Goal node: {goal}")
    print(f"Predicted waypoint: {predicted_waypoint}")
    print(f"Maximum similarity score: {max_similarity}")
    
    return start, goal, predicted_waypoint, similarities

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
                # Encode start and goal
                start_tensor = torch.tensor([start]).long()
                goal_tensor = torch.tensor([goal]).long()
                start_goal_encoding = encoder(start_tensor, goal_tensor)

                # Ensure start_goal_encoding has shape [1, embedding_dim]
                if start_goal_encoding.dim() == 1:
                    start_goal_encoding = start_goal_encoding.unsqueeze(0)

                # Encode all nodes as waypoints
                all_nodes = torch.arange(env.num_nodes).long()
                waypoint_encodings = encoder(all_nodes, all_nodes)

                # Compute hyperbolic distances
                distances = manifold.dist(start_goal_encoding, waypoint_encodings)

                # Convert distances to similarities
                similarities = -distances.detach().numpy()

                # Find the node with the highest similarity
                predicted_waypoint = np.argmin(distances.detach().numpy())

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

def plot_state_pair_embeddings(encoder, env, manifold, device, epoch=0, save_dir="plots"):
    """
    Plot embeddings of *all* valid state pairs (s, g) on the Poincaré Ball,
    except those with path length == 2. Color each pair by its path length L.
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import torch

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1) Plot phi(s, s) for all s
    num_states = env.num_nodes
    states = torch.arange(num_states).long().to(device)

    with torch.no_grad():
        phi_ss = encoder(states, states)  # shape: [num_states, embedding_dim]

    # If phi_ss is a ManifoldTensor, unwrap it
    if hasattr(phi_ss, 'tensor'):
        phi_ss = phi_ss.tensor

    phi_ss_np = phi_ss.cpu().numpy()
    ax.scatter(phi_ss_np[:, 0], phi_ss_np[:, 1],
               color='blue', label=r'$\phi(s, s)$', alpha=0.7)

    # 2) Enumerate all valid pairs (s != g), skip L=2
    all_pairs = []
    for s_node in range(num_states):
        for g_node in range(num_states):
            if s_node == g_node:
                continue

            # Get path and length
            node_path = env.get_node_path(s_node, g_node)
            L = len(node_path)
            # Skip pairs with length == 2
            if L == 2:
                continue

            all_pairs.append((s_node, g_node, L))

    # If you worry about memory usage or time, you could randomly sample from all_pairs:
    # all_pairs = random.sample(all_pairs, k=1000)  # for example

    # 3) Convert to tensors for batch encoding
    s_list = [p[0] for p in all_pairs]
    g_list = [p[1] for p in all_pairs]
    L_list = [p[2] for p in all_pairs]

    s_tensors = torch.tensor(s_list).long().to(device)
    g_tensors = torch.tensor(g_list).long().to(device)

    # 4) Encode all (s, g) pairs in one big batch
    with torch.no_grad():
        phi_sg = encoder(s_tensors, g_tensors)

    if hasattr(phi_sg, 'tensor'):
        phi_sg = phi_sg.tensor

    phi_sg_np = phi_sg.cpu().numpy()

    # 5) Color map for path lengths
    color_map = {
        3: 'red',
        4: 'orange',
        5: 'green',
        6: 'purple',
        7: 'brown',
        8: 'pink',
        9: 'gray',
        10: 'cyan',
        # etc. add more as needed
    }

    # Keep track of which lengths we have *actually* plotted, to build a legend
    plotted_lengths = set()

    # 6) Plot each (s, g) embedding, skipping L=2, coloring by L
    for i, (s_node, g_node, L) in enumerate(all_pairs):
        # (x, y) coordinate
        x, y = phi_sg_np[i, 0], phi_sg_np[i, 1]

        # Get color from map (default black if we haven't assigned one)
        color = color_map.get(L, 'black')

        sc = ax.scatter(x, y, color=color, marker='x', s=30, alpha=0.7)

        # Add legend label for the first occurrence of each L
        if L not in plotted_lengths:
            sc.set_label(f'L={L}')
            plotted_lengths.add(L)

    # 7) Draw the Poincaré disk boundary
    c_val = manifold.c.value.data.item()
    radius = 1 / np.sqrt(c_val)
    circle = plt.Circle((0, 0), radius, color='black', fill=False, linewidth=2)
    ax.add_artist(circle)

    ax.set_xlim(-1.1 * radius, 1.1 * radius)
    ax.set_ylim(-1.1 * radius, 1.1 * radius)

    ax.set_title("All (s, g) Pairs (Except L=2) on the Poincaré Ball")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()

    # 8) Save the plot
    save_path = os.path.join(save_dir, f"all_pairs_excl_L2_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved state pair embeddings (excluding L=2) for epoch {epoch+1} at {save_path}")

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
# Encoder Network
num_cat = env.num_nodes
embedding_dim = 16
euc_hidden_dims = [64, 64, 32]
hyp_hidden_dims = [64, 64, 32]
output_dim = 2

from hypll.manifolds.poincare_ball import Curvature, PoincareBall

curvature_value = 1.0
curvature = Curvature(value=curvature_value, requires_grad=True)
manifold = PoincareBall(c=curvature)

encoder = EncoderHyperbolicMLP(
    cat_features=num_cat,
    embedding_dims=embedding_dim,
    euc_hidden_dims=euc_hidden_dims,
    hyp_hidden_dims=hyp_hidden_dims,
    output_dim=output_dim,
    manifold=manifold
)

# Optimizer and Loss
optimizer = RiemannianAdam(encoder.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Initialize accuracy tracking
path_lengths = [3, 4, 5, 6]
accuracy_history = {L: [] for L in path_lengths}

# Training Loop
for epoch in range(num_epochs):
    encoder.train()
    epoch_loss = 0.0

    for batch in data_loader:
        # batch shape: [batch_size, 3], columns = [start, middle, goal]
        start = batch[:, 0]  # shape: [batch_size]
        mid   = batch[:, 1]  # shape: [batch_size]
        goal  = batch[:, 2]  # shape: [batch_size]
        batch_size = batch.size(0)

        # Randomly select which of the three states [start, mid, goal] to use as the "waypoint."
        random_idx = torch.randint(
            low=0, 
            high=3, 
            size=(batch_size,), 
            device=batch.device
        )

        # Create a single tensor with all possible states for easy indexing
        # shape: [batch_size, 3]
        all_states = torch.stack([start, mid, goal], dim=1)

        # waypoint[i] = all_states[i, random_idx[i]]
        waypoint = all_states[torch.arange(batch_size), random_idx]

        # ------------------------------------------------
        # Encode (start, goal)
        # ------------------------------------------------
        sg_encoded = encoder(start.long(), goal.long())  
        # shape: [batch_size, embedding_dim]

        # ------------------------------------------------
        # Encode (waypoint, waypoint)
        # ------------------------------------------------
        ww_encoded = encoder(waypoint.long(), waypoint.long())  
        # shape: [batch_size, embedding_dim]

        # ------------------------------------------------
        # Compute pairwise distances for classification
        # ------------------------------------------------
        sg_expanded = sg_encoded.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        ww_expanded = ww_encoded.unsqueeze(0)  # [1, batch_size, embedding_dim]
        distances = manifold.dist(sg_expanded, ww_expanded)  # [batch_size, batch_size]

        # Negate distances -> logits for cross entropy
        logits = -distances

        # Diagonal = correct class
        target = torch.arange(batch_size, device=logits.device)

        # Compute loss
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # ------------------------------------------
    # Evaluate after each epoch
    # (same logic as your original code)
    # ------------------------------------------
    start_state, goal_state, predicted_waypoint, similarity_scores = evaluate(env, encoder)
    results = evaluate_model(env, encoder, path_lengths=path_lengths, num_samples=100)
    plot_state_pair_embeddings(encoder, env, manifold, device, epoch=epoch, save_dir="plots")

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
