import numpy as np
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
