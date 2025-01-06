import torch

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
        # plot_fn(encoder, env, manifold, config.device, epoch=epoch, save_dir="plots")

        for L in config.path_lengths:
            accuracy_history[L].append(results[L])

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.4f}")

    return accuracy_history
