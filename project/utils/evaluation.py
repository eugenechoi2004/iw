import torch
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
