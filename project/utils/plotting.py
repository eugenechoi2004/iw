import os
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
    ax.scatter(phi_ss_np[:, 0], phi_ss_np[:, 1], 
            color='blue', label='$\\phi(s, s)$', alpha=0.7)


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
