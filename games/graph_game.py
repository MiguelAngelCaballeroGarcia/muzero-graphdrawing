# graph_game.py
import datetime
import pathlib
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import random

from .abstract_game import AbstractGame
from .GraphDrawing import graph_generation, graph_drawing


class MuZeroConfig:
    def __init__(self, grid_size: int = 6, sparsity: float = 0.6, seed: int = 0):
        # ---- game-specific params (editable) ----
        self.grid_size = grid_size
        self.sparsity = sparsity  # passed to generator

        self.seed = seed

        # Derived observation shape: channels = 1 + n*n, shape (C, n, n)
        C = 1 + self.grid_size * self.grid_size
        self.observation_shape = (C, self.grid_size, self.grid_size)

        # Action space: fixed discrete set: k nodes (== grid_size) × (n*n target cells)
        # action_id = node_index * (n*n) + (flatpos - 1)
        self.action_space = list(range(self.grid_size * (self.grid_size * self.grid_size)))

        # Single-player style environment: MuZero plays alone (no opponent)
        self.players = list(range(1))
        self.stacked_observations = 0

        # Evaluate / self-play settings (tunable)
        self.muzero_player = 0
        self.opponent = None

        # Self-play / training
        # Number of parallel self-play workers (adjust per node CPUs)
        self.num_workers = 1

        # Self-play actors typically do environment + MCTS; keep them on CPU by default.
        # Reserve the GPU for training when using a single GPU node.
        self.selfplay_on_gpu = False

        # Let the runtime determine the available GPUs (use torch.cuda.device_count() and Slurm allocation).
        # Keep None so muzero.py can detect the allocated GPUs automatically.
        self.max_num_gpus = None

        self.max_moves = 500  # you can lower this (e.g., n*n * 4)
        self.num_simulations = 50
        self.discount = 1
        self.temperature_threshold = None

        # Root prior noise and UCB settings (defaults)
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network (set to FC/resnet as you like)
        self.network = "resnet"
        self.support_size = 10

        self.encoding_size = 32

        # <-- ADD THIS MISSING ATTRIBUTE -->
        self.blocks = 3  # Use num_resnet_blocks here for compatibility
        
        self.num_channels = 64    # Number of convolutional channels (filters)
        self.num_resnet_blocks = 3 # Number of residual blocks (depth)
        self.downsample = False    # Set to True if input size > output size (not needed here)

        self.channels = self.num_channels

        # <--- ADD THESE RESNET HEAD LAYERS --->
        self.reduced_channels_reward = 16  # Channels for the reward prediction head
        self.reduced_channels_value = 16   # Channels for the value prediction head
        self.reduced_channels_policy = 16  # Channels for the policy prediction head
        
        # Define the FC layers within the ResNet heads
        self.resnet_fc_reward_layers = [32] 
        self.resnet_fc_value_layers = [32]  
        self.resnet_fc_policy_layers = [32] 
        # <--- END MISSING RESNET HEAD LAYERS --->
        
        # ADD ALL MISSING FC LAYER DEFINITIONS HERE:
        self.fc_representation_layers = [] # Added
        self.fc_dynamics_layers = [64]    # Added
        self.fc_reward_layers = [64]      # Already added (from previous fix)
        self.fc_value_layers = []         # Added
        self.fc_policy_layers = []        # Added

        # Training hyperparams (examples)
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 100000
        
        # Increase batch size for GPU usage (change down for small tests).
        # If you have limited GPU memory, reduce this accordingly.
        self.batch_size = 32

        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25

        # Default to using GPU for training on the cluster; you can override at runtime.
        self.train_on_gpu = True


        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = 0.003
        self.lr_decay_rate = 1
        self.lr_decay_steps = 10000

        # Replay buffer etc.
        self.replay_buffer_size = 3000
        self.num_unroll_steps = 20
        self.td_steps = 20
        self.PER = True
        self.PER_alpha = 0.5
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Play/training ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = None

    def visit_softmax_temperature_fn(self, trained_steps):
        return 1


class Game(AbstractGame):
    """MuZero wrapper to expose the GraphEnv to MuZero."""

    def __init__(self, seed: Optional[int] = None):
        cfg = MuZeroConfig()
        self.env = GraphEnv(n=cfg.grid_size, s=cfg.sparsity, max_moves=cfg.max_moves, seed=seed)

    def step(self, action: int):
        obs, reward, done = self.env.step(action)
        # wrapper scaling (MuZero examples sometimes scale reward)
        return obs, reward*10, done

    def to_play(self):
        # Single-player (always player 0)
        return 0

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def expert_agent(self):
        # optional: provide a simple heuristic agent
        return self.env.expert_action()

    def action_to_string(self, action_number: int):
        node_idx = action_number // (self.env.n * self.env.n)
        flat = (action_number % (self.env.n * self.env.n)) + 1
        return f"Move node_index={node_idx} to flatpos={flat}"


# -------------------------
# Environment implementation
# -------------------------
class GraphEnv:
    """
    Environment that wraps GraphDrawing and exposes a fixed discrete action space.

    Action encoding:
        action_id = node_index * (n*n) + (flatpos - 1)
    where:
        - node_index is index in sorted(self.graph.node_ids) (0 .. k-1)
        - flatpos in 1..(n*n) is the target cell (row,col) -> flat = row*n + col + 1
    """

    def __init__(self, n: int = 4, s: float = 0.3, max_moves: int = 200, seed: Optional[int] = None):
        self.n = n
        self.s = s
        self.max_moves = max_moves
        self._rng = random.Random(seed)

        # number of movable nodes (by your design this equals n)
        self.k = n

        # placeholders
        self.graph: Optional[graph_drawing.GraphDrawing] = None
        self.steps = 0
        self._build_new_game()

    # ----- helpers for encoding/decoding actions -----
    def _node_list(self) -> List[int]:
        """Return a stable ordered list of node IDs used to encode actions."""
        # sorted order ensures fixed mapping
        return sorted(self.graph.node_ids)

    def _action_decode(self, action: int) -> Tuple[int, Tuple[int, int]]:
        """Decode action id -> (node_id, (i,j))."""
        cells = self.n * self.n
        node_idx = action // cells
        flat = (action % cells) + 1
        node_list = self._node_list()
        if node_idx < 0 or node_idx >= len(node_list):
            raise ValueError("action node_idx out of range")
        node_id = node_list[node_idx]
        i = (flat - 1) // self.n
        j = (flat - 1) % self.n
        return node_id, (i, j)

    def _action_encode(self, node_idx: int, flatpos: int) -> int:
        return node_idx * (self.n * self.n) + (flatpos - 1)         # N*n*n (N = number of nodes, n*n = size of the grid)

    # ----- environment lifecycle -----
    def _build_new_game(self):
        """Construct a fresh graph drawing from generator."""
        n, pos_map, edges = graph_generation.generate_connected_random_graph(self.n, self.s)
        # generator returns n, pos_map, edges where n == self.n
        self.graph = graph_drawing.GraphDrawing(n=n, pos=pos_map, edges=edges, build_tensor=True)
        self.steps = 0

    def reset(self):
        self._build_new_game()
        # MuZero expects the raw observation array
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        # GraphDrawing.tensor shape (C, n, n), convert to float32
        return self.graph.tensor.astype(np.float32)

    # ----- legal actions -----
    def legal_actions(self) -> List[int]:
        """
        Build list of valid action ids:
         - node must exist in node list
         - target cell must be empty (not occupied by other node)
         - optionally disallow moving a node to the same cell (we treat that as illegal)
        """
        actions = []
        node_list = self._node_list()
        occupied = set(self.graph.pos.values())
        cells = self.n * self.n

        for node_idx, node_id in enumerate(node_list):
            for flat in range(1, cells + 1):
                i = (flat - 1) // self.n
                j = (flat - 1) % self.n
                if (i, j) in occupied:
                    # If it's the same node's current position, skip (no-op)
                    if self.graph.pos[node_id] == (i, j):
                        continue
                    # otherwise it's occupied by another node → illegal
                    continue
                # empty cell → legal
                actions.append(self._action_encode(node_idx, flat))
        return actions

    # ----- step / reward -----
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Apply action if legal. Returns (observation, reward, done).
        Reward = previous_crossings - new_crossings (positive if we decrease crossings).
        If no change, reward = 0.001 (small positive).
        Illegal actions produce a small negative reward and do not change state.
        """
        legal = set(self.legal_actions())
        if action not in legal:
            # illegal action: small penalty
            return self.get_observation(), -10.0, False

        node_id, new_pos = self._action_decode(action)

        prev_cross = self.graph.crossings
        # do the move (GraphDrawing.move_node_to updates tensor and crossings incrementally)
        self.graph.move_node_to(node_id, new_pos)
        new_cross = self.graph.crossings

        # reward rule
        diff = prev_cross - new_cross
        if diff == 0:
            reward = -0.01
        else:
            reward = float(diff)

        self.steps += 1
        done = (self.steps >= self.max_moves) or (self.graph.crossings == 0)

        # --- TERMINAL BONUS FIX ---
        if self.graph.crossings == 0 and done:
            # Add a large positive reward for solving the game
            reward += 1000.0
        
        return self.get_observation(), reward, done

    # ----- optional helpers -----
    def expert_action(self) -> int:
        """Return a simple heuristic: choose first legal action that reduces crossings, else random legal."""
        legal = self.legal_actions()
        random.shuffle(legal)
        for a in legal:
            node_id, new_pos = self._action_decode(a)
            # simulate move on a cloned graph
            gcopy = self.graph.clone()
            gcopy.move_node_to(node_id, new_pos)
            if gcopy.crossings < self.graph.crossings:
                return a
        # no reducing move found → random legal
        if legal:
            return legal[0]
        # fallback
        return 0

    def render(self):
        self.graph.plot(show_ids=True, title=f"Steps: {self.steps}, Crossings: {self.graph.crossings}")

