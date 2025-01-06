import torch
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
