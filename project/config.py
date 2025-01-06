import torch

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
