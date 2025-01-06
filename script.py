# setup_project.py
import os

# Directory structure
directories = [
    "project",
    "project/networks",
    "project/envs",
    "project/utils",
    "project/plots"
]

for d in directories:
    os.makedirs(d, exist_ok=True)

########################################
# config.py
########################################
config_code = r'''import torch

class Config:
    # Environment & Data Generation
    depth = 4
    branching_factor = 2
    num_trajectories = 50_000
    dataset_size = 10_000
    min_L = 2
    max_L = 8

    # Training Parameters
    batch_size = 128
    num_epochs = 200
    learning_rate = 0.001

    # Model Hyperparameters
    embedding_dim = 16
    euc_hidden_dims = [64, 64, 32]
    hyp_hidden_dims = [64, 64, 32]
    output_dim = 2
    curvature_value = 1.0

    # RL gamma (unused currently, but kept for reference)
    gamma = 0.3

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Evaluation
    path_lengths = [3, 4, 5, 6]
    num_evaluation_samples = 100
'''

########################################
# envs/tree.py
########################################
tree_code = r'''import numpy as np
import graphviz
import matplotlib.pyplot as plt
import io

class NaryTreeEnvironment:
    """
    N-ary Tree environment for RL with categorical actions and improved display
    """

    def __init__(self, depth, branching_factor, start=0):
        self.depth = depth
        self.branching_factor = branching_factor
        # Calculate number of nodes in a full n-ary tree of given depth
        # depth levels: 0..depth-1
        self.num_nodes = sum(branching_factor**i for i in range(depth))
        self.agent_position = start
        self.action_map = {
            0: "up",
            **{i + 1: f"child_{i}" for i in range(branching_factor)},
            branching_factor + 1: "stay",
        }
        self.state_dim = self.num_nodes
        self.action_dim = self.branching_factor + 2

    def move_agent(self, action):
        direction = self.action_map[action]
        current_node = self.agent_position

        if direction == "up":
            new_position = (
                (current_node - 1) // self.branching_factor if current_node > 0 else 0
            )
        elif direction.startswith("child_"):
            child_index = int(direction.split("_")[1])
            new_position = self.branching_factor * current_node + child_index + 1
        elif direction == "stay":
            new_position = current_node
        else:
            raise ValueError(
                f"Invalid action. Use 0 (up), 1-{self.branching_factor} (children), or {self.branching_factor+1} (stay)."
            )

        if self.is_valid_move(new_position):
            self.agent_position = new_position
            return True
        return False

    def is_valid_move(self, position):
        return 0 <= position < self.num_nodes

    def get_state(self):
        return self.agent_position

    def is_leaf(self):
        # This leaf-check might not be accurate for all trees.
        # Adjust as necessary. For now, just assume nodes at depth = self.depth-1 are leaves.
        # The last level nodes range from sum of all nodes until depth-1.
        # But since we only track total nodes, let's do a heuristic:
        # Actually, let's assume leaves are those that can't produce children:
        start_leaf_level = sum(self.branching_factor**i for i in range(self.depth - 1))
        return self.agent_position >= start_leaf_level

    def reset(self):
        self.agent_position = 0
        return self.get_state()

    def get_possible_actions(self):
        actions = [self.branching_factor + 1]  # 'stay' is always possible
        if self.agent_position > 0:
            actions.append(0)  # 'up' is possible if not at root
        if not self.is_leaf():
            actions.extend(
                range(1, self.branching_factor + 1)
            )  # child moves are possible if not a leaf
        return actions

    def get_node_path(self, start_node, end_node):
        """
        Returns the path of nodes from start_node to end_node by tracing both up to root.
        """
        if start_node == end_node:
            return [start_node]

        path_to_root_start = []
        node = start_node
        while node >= 0:
            path_to_root_start.append(node)
            if node == 0:
                break
            node = (node - 1) // self.branching_factor

        path_to_root_end = []
        node = end_node
        while node >= 0:
            path_to_root_end.append(node)
            if node == 0:
                break
            node = (node - 1) // self.branching_factor

        # Find common ancestor
        i = 1
        while i <= len(path_to_root_start) and i <= len(path_to_root_end) and path_to_root_start[-i] == path_to_root_end[-i]:
            i += 1
        i -= 1

        common_ancestor = path_to_root_start[-i]

        # Construct path: start->...->common_ancestor->...->end
        start_to_ancestor = path_to_root_start[:-i] if i > 0 else path_to_root_start
        end_to_ancestor = path_to_root_end[:-i] if i > 0 else path_to_root_end
        full_path = start_to_ancestor + end_to_ancestor[::-1]
        return full_path

    def get_action_path(self, start_node, end_node):
        """
        Returns the sequence of actions to move from start_node to end_node.
        """
        node_path = self.get_node_path(start_node, end_node)
        actions = []

        for i in range(len(node_path) - 1):
            current_node = node_path[i]
            next_node = node_path[i + 1]

            if next_node < current_node:
                actions.append(0)  # Move up
            elif next_node == current_node:
                actions.append(self.branching_factor + 1)  # Stay
            else:
                action = next_node - self.branching_factor * current_node
                actions.append(action)  # Move to child

        actions.append(self.branching_factor + 1) # Stay at end
        return list(zip(node_path, actions))

    def valid_indices(self):
        return range(self.num_nodes)

    def display(self, highlight_path=None, ax=None, start=None, end=None):
        dot = graphviz.Graph()
        dot.attr(rankdir="TB")

        def add_nodes_edges(node, depth=0):
            if node >= self.num_nodes or depth >= self.depth:
                return

            if start is not None and node == start:
                dot.node(str(node), str(node), style="filled", color="red")
            elif end is not None and node == end:
                dot.node(str(node), str(node), style="filled", color="green")
            elif highlight_path and node in highlight_path:
                dot.node(str(node), str(node), style="filled", color="lightblue")
            else:
                dot.node(str(node), str(node))

            # Add edges to children
            if depth < self.depth - 1:
                for i in range(self.branching_factor):
                    child = self.branching_factor * node + i + 1
                    if child < self.num_nodes:
                        if highlight_path and node in highlight_path and child in highlight_path:
                            dot.edge(str(node), str(child), color="red", penwidth="2")
                        else:
                            dot.edge(str(node), str(child))
                        add_nodes_edges(child, depth + 1)

        add_nodes_edges(0)

        if ax is None:
            # In a notebook, can display(dot)
            pass
        else:
            png_data = dot.pipe(format='png')
            ax.imshow(plt.imread(io.BytesIO(png_data)))
            ax.axis('off')

    def get_trajectories(self, num_trajectories):
        trajectories = []
        for _ in range(num_trajectories):
            start, end = np.random.randint(0, self.num_nodes, size=2)
            trajectories.append(self.get_node_path(start, end))
        return trajectories
'''

########################################
# networks/encoder.py
########################################
encoder_code = r'''import torch
import torch.nn as nn
import torch.nn.functional as F
import hypll.nn as hnn
from hypll.tensors import TangentTensor

class EncoderMLP(nn.Module):
    def __init__(self, num_cat, embedding_dim, hidden_dims, output_dim):
        super(EncoderMLP, self).__init__()
        self.embedding = nn.Embedding(num_cat, embedding_dim)
        input_dim = embedding_dim * 2
        model = []
        for dim in hidden_dims:
            model.append(nn.Linear(input_dim, dim))
            model.append(nn.ReLU())
            input_dim = dim
        model.append(nn.Linear(input_dim, output_dim))
        self.encoder = nn.Sequential(*model)
    
    def forward(self, first, second):
        first_embed = self.embedding(first)
        second_embed = self.embedding(second)
        concat_embed = torch.cat((first_embed, second_embed), dim=1)
        return self.encoder(concat_embed)

class EncoderHyperbolicMLP(nn.Module):
    def __init__(self, cat_features, embedding_dims, euc_hidden_dims, hyp_hidden_dims, output_dim, manifold):
        super(EncoderHyperbolicMLP, self).__init__()
        
        self.manifold = manifold

        # First part: Euclidean MLP
        self.euc_mlp = EncoderMLP(cat_features, embedding_dims, euc_hidden_dims, hyp_hidden_dims[0])
        
        # Hyperbolic layers
        hyp_layers = []
        for i in range(1, len(hyp_hidden_dims)):
            hyp_layers.append(hnn.HLinear(hyp_hidden_dims[i-1], hyp_hidden_dims[i], manifold=manifold))
            hyp_layers.append(hnn.HReLU(manifold=manifold))
        
        hyp_layers.append(hnn.HLinear(hyp_hidden_dims[-1], output_dim, manifold=manifold))
        
        self.hyp_mlp = nn.Sequential(*hyp_layers)
    
    def forward(self, first, second):
        euc_output = self.euc_mlp(first, second)
        hyp_input = self.manifold_map(euc_output, self.manifold)
        output = self.hyp_mlp(hyp_input)
        return output
    
    def manifold_map(self, x, manifold):
        tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
        return manifold.expmap(tangents)
'''

########################################
# utils/data_utils.py
########################################
data_utils_code = r'''import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.long)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

def generate_points_vectorized(num_points, trajectories, episode_lengths, min_L, max_L):
    num_trajectories = len(trajectories)
    traj_indices = np.random.randint(0, num_trajectories, size=num_points)
    traj_lengths = episode_lengths[traj_indices]

    L = np.random.randint(min_L, max_L + 1, size=num_points)
    valid_mask = (L <= traj_lengths)
    while not valid_mask.all():
        invalid_indices = np.where(~valid_mask)[0]
        traj_indices[invalid_indices] = np.random.randint(0, num_trajectories, size=len(invalid_indices))
        traj_lengths[invalid_indices] = episode_lengths[traj_indices[invalid_indices]]
        L[invalid_indices] = np.random.randint(min_L, max_L + 1, size=len(invalid_indices))
        valid_mask = (L <= traj_lengths)

    t0 = np.random.randint(0, traj_lengths - L + 1)
    t2 = t0 + L - 1

    t1 = np.zeros_like(t0)
    odd_mask = (L % 2 == 1)
    even_mask = ~odd_mask
    t1[odd_mask] = t0[odd_mask] + L[odd_mask] // 2
    t1[even_mask] = t0[even_mask] + (L[even_mask] // 2) - 1

    starts = np.array([trajectories[traj_idx][t0_idx] for traj_idx, t0_idx in zip(traj_indices, t0)])
    waypoints = np.array([trajectories[traj_idx][t1_idx] for traj_idx, t1_idx in zip(traj_indices, t1)])
    goals = np.array([trajectories[traj_idx][t2_idx] for traj_idx, t2_idx in zip(traj_indices, t2)])

    return starts, waypoints, goals

def create_data_loader(starts, waypoints, goals, batch_size, shuffle=True):
    dataset = np.stack([starts, waypoints, goals], axis=1)
    trajectory_dataset = TrajectoryDataset(dataset)
    data_loader = DataLoader(trajectory_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
'''

########################################
# utils/evaluation.py
########################################
evaluation_code = r'''import torch
import numpy as np

def evaluate(env, encoder, manifold, device):
    start, goal = np.random.randint(0, env.num_nodes, size=2)
    start_goal_encoding = encoder(torch.tensor([start]).long().to(device), torch.tensor([goal]).long().to(device))

    if start_goal_encoding.dim() == 1:
        start_goal_encoding = start_goal_encoding.unsqueeze(0)

    all_nodes = torch.arange(env.num_nodes).long().to(device)
    waypoint_encodings = encoder(all_nodes, all_nodes)
    distances = manifold.dist(start_goal_encoding, waypoint_encodings)
    similarities = -distances.detach().cpu().numpy()

    predicted_waypoint = np.argmax(similarities)
    max_similarity = similarities[predicted_waypoint]

    print(f"Start node: {start}")
    print(f"Goal node: {goal}")
    print(f"Predicted waypoint: {predicted_waypoint}")
    print(f"Maximum similarity score: {max_similarity}")

    return start, goal, predicted_waypoint, similarities

def evaluate_model(env, encoder, manifold, device, path_lengths=[3,4,5,6], num_samples=100):
    results = {}
    for L in path_lengths:
        correct_predictions = 0
        total_predictions = 0
        samples_found = 0

        print(f"\\nEvaluating path length {L}:")

        while samples_found < num_samples:
            start = np.random.randint(0, env.num_nodes)
            goal = np.random.randint(0, env.num_nodes)
            node_path = env.get_node_path(start, goal)

            if len(node_path) == L:
                start_tensor = torch.tensor([start]).long().to(device)
                goal_tensor = torch.tensor([goal]).long().to(device)
                start_goal_encoding = encoder(start_tensor, goal_tensor)

                if start_goal_encoding.dim() == 1:
                    start_goal_encoding = start_goal_encoding.unsqueeze(0)

                all_nodes = torch.arange(env.num_nodes).long().to(device)
                waypoint_encodings = encoder(all_nodes, all_nodes)
                distances = manifold.dist(start_goal_encoding, waypoint_encodings)
                predicted_waypoint = np.argmin(distances.detach().cpu().numpy())

                if L % 2 == 1:
                    middle_index = L // 2
                    correct_waypoints = [node_path[middle_index]]
                else:
                    middle_indices = [L // 2 - 1, L // 2]
                    correct_waypoints = [node_path[i] for i in middle_indices]

                if predicted_waypoint in correct_waypoints:
                    correct_predictions += 1

                total_predictions += 1
                samples_found += 1

        accuracy = correct_predictions / total_predictions
        print(f"Accuracy for path length {L}: {accuracy * 100:.2f}% (based on {total_predictions} samples)")
        results[L] = accuracy

    return results
'''

########################################
# utils/plotting.py
########################################
plotting_code = r'''import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_state_pair_embeddings(encoder, env, manifold, device, epoch=0, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    num_states = env.num_nodes
    states = torch.arange(num_states).long().to(device)

    with torch.no_grad():
        phi_ss = encoder(states, states)

    if hasattr(phi_ss, 'tensor'):
        phi_ss = phi_ss.tensor

    phi_ss_np = phi_ss.cpu().numpy()
    ax.scatter(phi_ss_np[:, 0], phi_ss_np[:, 1], color='blue', label=r'$\\phi(s, s)$', alpha=0.7)

    specific_pairs = [(3, 5), (7, 14), (9, 11), (8, 11)]
    s_list = [s for s,g in specific_pairs]
    g_list = [g for s,g in specific_pairs]

    s_tensors = torch.tensor(s_list).long().to(device)
    g_tensors = torch.tensor(g_list).long().to(device)

    with torch.no_grad():
        phi_sg = encoder(s_tensors, g_tensors)

    if hasattr(phi_sg, 'tensor'):
        phi_sg = phi_sg.tensor

    phi_sg_np = phi_sg.cpu().numpy()

    for i in range(len(s_list)):
        x, y = phi_sg_np[i, 0], phi_sg_np[i, 1]
        s_node, g_node = s_list[i], g_list[i]
        ax.scatter(x, y, color='red', marker='x', s=100)
        ax.text(x, y, f'({s_node},{g_node})', fontsize=9, color='red', ha='right', va='bottom')

    c = manifold.c.value.data.item()
    radius = 1 / np.sqrt(c)
    circle = plt.Circle((0, 0), radius, color='black', fill=False, linewidth=2)
    ax.add_artist(circle)

    ax.set_xlim(-1.1 * radius, 1.1 * radius)
    ax.set_ylim(-1.1 * radius, 1.1 * radius)

    ax.set_title("Embeddings of State Pairs on the Poincaré Ball")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    save_path = os.path.join(save_dir, f"state_pair_embeddings_epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved state pair embeddings plot for epoch {epoch+1} at {save_path}")
'''

########################################
# utils/training.py
########################################
training_code = r'''import torch

def train_encoder(encoder, manifold, optimizer, criterion, data_loader, num_epochs, evaluate_fn, evaluate_model_fn, plot_fn, env, config):
    accuracy_history = {L: [] for L in config.path_lengths}

    for epoch in range(num_epochs):
        encoder.train()
        epoch_loss = 0.0

        for batch in data_loader:
            start, waypoint, goal = batch[:, 0], batch[:, 1], batch[:, 2]

            start = start.long().to(config.device)
            waypoint = waypoint.long().to(config.device)
            goal = goal.long().to(config.device)

            sg_encoded = encoder(start, goal)
            ww_encoded = encoder(waypoint, waypoint)

            sg_expanded = sg_encoded.unsqueeze(1)
            ww_expanded = ww_encoded.unsqueeze(0)

            distances = manifold.dist(sg_expanded, ww_expanded)
            logits = -distances

            target = torch.arange(logits.size(0)).long().to(config.device)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation
        encoder.eval()
        start, goal, predicted_waypoint, similarity_scores = evaluate_fn(env, encoder, manifold, config.device)
        results = evaluate_model_fn(env, encoder, manifold, config.device, path_lengths=config.path_lengths, num_samples=config.num_evaluation_samples)
        plot_fn(encoder, env, manifold, config.device, epoch=epoch, save_dir="plots")

        for L in config.path_lengths:
            accuracy_history[L].append(results[L])

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.4f}")

    return accuracy_history
'''

########################################
# main.py
########################################
main_code = r'''import torch
from config import Config
from envs.tree import NaryTreeEnvironment
from networks.encoder import EncoderHyperbolicMLP
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianAdam
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.data_utils import generate_points_vectorized, create_data_loader
from utils.training import train_encoder
from utils.evaluation import evaluate, evaluate_model
from utils.plotting import plot_state_pair_embeddings

def main():
    config = Config()

    # Create environment and trajectories
    env = NaryTreeEnvironment(depth=config.depth, branching_factor=config.branching_factor)
    trajectories = env.get_trajectories(config.num_trajectories)
    episode_lengths = [len(traj) for traj in trajectories]

    # Create dataset
    starts, waypoints, goals = generate_points_vectorized(config.dataset_size, trajectories, episode_lengths, config.min_L, config.max_L)
    data_loader = create_data_loader(starts, waypoints, goals, config.batch_size, shuffle=True)

    # Setup manifold
    curvature = Curvature(value=config.curvature_value, requires_grad=True)
    manifold = PoincareBall(c=curvature)

    # Encoder
    encoder = EncoderHyperbolicMLP(
        cat_features=env.num_nodes,
        embedding_dims=config.embedding_dim,
        euc_hidden_dims=config.euc_hidden_dims,
        hyp_hidden_dims=config.hyp_hidden_dims,
        output_dim=config.output_dim,
        manifold=manifold
    ).to(config.device)

    # Optimizer & Loss
    optimizer = RiemannianAdam(encoder.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train
    accuracy_history = train_encoder(
        encoder=encoder,
        manifold=manifold,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=data_loader,
        num_epochs=config.num_epochs,
        evaluate_fn=evaluate,
        evaluate_model_fn=evaluate_model,
        plot_fn=plot_state_pair_embeddings,
        env=env,
        config=config
    )

    # Plot accuracies
    plt.figure(figsize=(10, 6))
    for L in config.path_lengths:
        plt.plot(range(1, config.num_epochs+1), accuracy_history[L], label=f'Path Length {L}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs for Different Path Lengths')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_over_epochs.png")
    plt.show()

if __name__ == "__main__":
    main()
'''

# Write code to files
files = {
    "project/config.py": config_code,
    "project/envs/tree.py": tree_code,
    "project/networks/encoder.py": encoder_code,
    "project/utils/data_utils.py": data_utils_code,
    "project/utils/evaluation.py": evaluation_code,
    "project/utils/plotting.py": plotting_code,
    "project/utils/training.py": training_code,
    "project/main.py": main_code
}

for filepath, code in files.items():
    with open(filepath, "w") as f:
        f.write(code)

print("Project structure created successfully. You can now run 'python project/main.py' to start training (assuming all dependencies are met).")
