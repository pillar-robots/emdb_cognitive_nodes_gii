import rclpy
import pandas as pd
import numpy as np
from collections import deque
from copy import deepcopy

from core.cognitive_node import CognitiveNode
from cognitive_nodes.episode import Episode, Action, episode_msg_to_obj

from cognitive_node_interfaces.msg import Episode as EpisodeMsg




class EpisodicBuffer:
    """
    Class that creates a buffer of episodes to be used as a STM and learn models. 
    It supports a main buffer for training and a secondary buffer for testing.
    """    
    def __init__(self, node:CognitiveNode, main_size, secondary_size, train_split=1.0, inputs=[], outputs=[], random_seed=0, **params) -> None:
        """Initialize an EpisodicBuffer.

        Creates bounded buffers for episodes and configures label extraction and RNG.

        :param node: Owning cognitive node used for logging and ROS2 integration.
        :type node: CognitiveNode
        :param main_size: Maximum number of episodes stored in the main (training) buffer.
        :type main_size: int
        :param secondary_size: Maximum number of episodes stored in the secondary (testing) buffer.
        :type secondary_size: int
        :param train_split: Fraction in [0, 1] of incoming episodes routed to the main buffer; the remainder goes to secondary.
        :type train_split: float
        :param inputs: Episode fields considered inputs/features (e.g., ['old_perception', 'action']).
        :type inputs: list[str]
        :param outputs: Episode fields considered outputs/targets (e.g., ['perception']).
        :type outputs: list[str]
        :param random_seed: Seed for the internal Numpy RNG used for shuffling and sampling.
        :type random_seed: int
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict

        :return: None
        :rtype: None

        Notes:
        - Buffers are implemented as bounded deques.
        - Labels are derived from observed episodes based on inputs/outputs.
        """        
        self.node=node
        self.train_split=train_split # Percentage of samples used for training
        self.inputs=inputs #Fields of the episode that are considered inputs (Used for prediction)
        self.outputs=outputs #Fields of the episode that are considered outputs (Predicted), or a post calculated value (e.g. Value)
        self.input_labels=[]
        self.output_labels=[]
        self.is_input=[]
        self.main_buffer=deque(maxlen=main_size) # Main buffer, used for training
        self.secondary_buffer=deque(maxlen=secondary_size) # Secondary buffer, used for testing
        self.main_dataframe_inputs=None # DataFrame for the main buffer
        self.main_dataframe_outputs=None # DataFrame for the main buffer
        self.secondary_dataframe_inputs=None # DataFrame for the secondary buffer
        self.secondary_dataframe_outputs=None # DataFrame for the secondary buffer
        self.new_sample_count_main=0 # Counter for new samples in the main buffer
        self.new_sample_count_secondary=0 # Counter for new samples in the secondary buffer
        self.rng = np.random.default_rng(random_seed) # Configuration of the random number generator

    def configure_labels(self, episode: Episode):
        """
        Creates the label list.

        :param episode: Episode object.
        :type episode: cognitive_node_interfaces.msg.Episode
        """
        self.node.get_logger().info(f"Configuring labels for episodic buffer: {episode}")
        self.input_labels.clear()
        self.output_labels.clear()
        self._extract_labels(self.inputs, episode, self.input_labels)
        self._extract_labels(self.outputs, episode, self.output_labels)
        self.node.get_logger().info(f"Configuration finished - Input labels: {self.input_labels}, Output labels: {self.output_labels}")

    def update_labels(self, episode: Episode):
        """
        Updates the label list based on the given episode.

        :param episode: Episode object.
        :type episode: cognitive_node_interfaces.msg.Episode
        """
        self.node.get_logger().debug(f"Updating labels for episodic buffer: {episode}")
        new_input_labels = []
        new_output_labels = []
        self._extract_labels(self.inputs, episode, new_input_labels)
        self._extract_labels(self.outputs, episode, new_output_labels)
        # Add new input labels and log warnings
        new_inputs = set(new_input_labels) - set(self.input_labels)
        if new_inputs:
            self.input_labels.extend(new_inputs)
            self.node.get_logger().warning(f"New input labels {new_inputs} added to episodic buffer.")

        # Add new output labels and log warnings
        new_outputs = set(new_output_labels) - set(self.output_labels)
        if new_outputs:
            self.output_labels.extend(new_outputs)
            self.node.get_logger().warning(f"New output labels {new_outputs} added to episodic buffer.")


    def add_episode(self, episode: Episode, reward=0.0):
        """Add an episode to the buffer.

        Validates content, updates labels, and routes the episode to the main or
        secondary buffer based on train_split.

        :param episode: Episode instance to add.
        :type episode: Episode
        :param reward: Optional reward value (unused in EpisodicBuffer; used by TraceBuffer overrides).
        :type reward: float

        :return: None
        :rtype: None

        Notes:
        - Empty episodes (w.r.t configured inputs/outputs) are skipped with a warning.
        - Labels are configured lazily on the first episode and updated when new dimensions appear.
        """        
        if self.empty_episode(episode, self.inputs, self.outputs):
            self.node.get_logger().warning("The episode is empty, not adding to buffer.")
        else:
            if (not self.input_labels and self.inputs) or (not self.output_labels and self.outputs):
                self.configure_labels(episode)
            else:
                self.update_labels(episode)
            if self.rng.uniform() < self.train_split:
                # Add to main buffer
                self.main_buffer.append(deepcopy(episode))
                self.new_sample_count_main += 1
            else:
                # Add to secondary buffer
                self.secondary_buffer.append(deepcopy(episode))
                self.new_sample_count_secondary += 1
        
    def remove_episode(self, index=None, remove_from_main=True):
        """Remove an episode from the buffer.

        If index is provided, removes the episode at that position; otherwise,
        removes the oldest (leftmost) episode from the selected buffer.

        :param index: Position to remove. If None, removes the oldest entry.
        :type index: int | None
        :param remove_from_main: True to remove from main_buffer; False for secondary_buffer.
        :type remove_from_main: bool

        :return: None
        :rtype: None

        """        
        if remove_from_main:
            if index is not None:
                self.main_buffer.remove(self.main_buffer[index]) 
            else:
                self.main_buffer.popleft()
        else:
            if index is not None:
                self.secondary_buffer.remove(self.secondary_buffer[index])
            else:
                self.secondary_buffer.popleft()

    def clear(self):
        """
        Clears the episodic buffer.
        """
        self.main_buffer.clear()
        self.secondary_buffer.clear()
        self.main_dataframe = None
        self.secondary_dataframe = None
        self.new_sample_count_main = 0
        self.new_sample_count_secondary = 0
    
    def create_dataframes(self):
        """
        Creates pandas DataFrames from the main and secondary buffers.
        """
        if self.input_labels:
            if len(self.main_buffer) > 0:
                self.main_dataframe_inputs = self.buffer_to_dataframe(self.main_buffer, self.input_labels)
            if len(self.secondary_buffer) > 0:
                self.secondary_dataframe_inputs = self.buffer_to_dataframe(self.secondary_buffer, self.input_labels)
        if self.output_labels:
            if len(self.main_buffer) > 0:
                self.main_dataframe_outputs = self.buffer_to_dataframe(self.main_buffer, self.output_labels)
            if len(self.secondary_buffer) > 0:
                self.secondary_dataframe_outputs = self.buffer_to_dataframe(self.secondary_buffer, self.output_labels)

    def is_compatible(self, episode: Episode):
        """
        Checks if the episode is compatible with the current buffer configuration.

        :param episode: Episode object to check compatibility.
        :type episode: cognitive_node_interfaces.msg.Episode
        :return: True if compatible, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError("is_compatible method is not implemented yet.")
        

    #### GETTERS / SETTERS ####

    def get_input_labels(self):
        """
        Returns the input labels of the episodic buffer.

        :return: List of input labels.
        :rtype: list
        """
        return self.input_labels
    
    def get_output_labels(self):
        """
        Returns the output labels of the episodic buffer.

        :return: List of output labels.
        :rtype: list
        """
        return self.output_labels

    def get_sample(self, index, main=True):
        """
        Method to obtain a sample from the buffer.

        :param index: Index of the sample to obtain.
        :type index: int
        :param main: Whether to get the sample from the main buffer or secondary buffer.
        :type main: bool
        :return: The requested sample.
        :rtype: list
        """
        if main:
            return self.main_buffer[index]
        else:
            return self.secondary_buffer[index]

    def get_dataset(self, shuffle=False, n_samples=None):
        """
        Returns the dataset as numpy arrays.

        :param shuffle: Option to shuffle the dataset , defaults to False
        :type shuffle: bool, optional
        :param n_samples: Option to limit the number of samples returned, defaults to None
        :type n_samples: int, optional
        :return: Tuple containing training inputs, training outputs, test inputs, and test outputs as numpy arrays.
        :rtype: tuple
        """
        x_train, y_train = self._get_samples_from_buffer(self.main_buffer, shuffle=shuffle, n_samples=n_samples)
        x_test, y_test = self._get_samples_from_buffer(self.secondary_buffer, shuffle=shuffle, n_samples=n_samples)
        return x_train, y_train, x_test, y_test


    def get_train_samples(self, shuffle=False, n_samples=None):
        """Returns the training samples as lists of input and output dicts.

        :param shuffle: Option to shuffle the samples, defaults to False
        :type shuffle: bool, optional
        :param n_samples: Option to limit the number of samples returned, defaults to None
        :type n_samples: int, optional
        :return: Tuple containing training inputs and training outputs as numpy arrays.
        :rtype: tuple
        """
        return self._get_samples_from_buffer(self.main_buffer, shuffle=shuffle, n_samples=n_samples)

    def get_test_samples(self, shuffle=False, n_samples=None):
        """Returns the test samples as lists of input and output dicts.

        :param shuffle: Option to shuffle the samples, defaults to False
        :type shuffle: bool, optional
        :param n_samples: Option to limit the number of samples returned, defaults to None
        :type n_samples: int, optional
        :return: Tuple containing test inputs and test outputs as numpy arrays.
        :rtype: tuple
        """
        return self._get_samples_from_buffer(self.secondary_buffer, shuffle=shuffle, n_samples=n_samples)
    
    def get_dataframes(self):
        """
        Returns the DataFrames of the main and secondary buffers.

        :return: Tuple with the main and secondary DataFrames.
        :rtype: tuple
        """
        self.create_dataframes()
        return self.main_dataframe_inputs, self.main_dataframe_outputs, self.secondary_dataframe_inputs, self.secondary_dataframe_outputs
    
    def reset_new_sample_count(self, main=True, secondary=True):
        """
        Resets the new sample count for the main and/or secondary buffers.

        :param main: Whether to reset the main buffer count, defaults to True.
        :type main: bool
        :param secondary: Whether to reset the secondary buffer count, defaults to True.
        :type secondary: bool
        """
        if main:
            self.new_sample_count_main = 0
        if secondary:
            self.new_sample_count_secondary = 0

    @property
    def main_max_size(self):
        """
        Returns the max size of the main buffer.

        :return: The maximum size of the main buffer.
        :rtype: int
        """        
        return self.main_buffer.maxlen if self.main_buffer.maxlen is not None else float('inf')

    @property
    def secondary_max_size(self):
        """
        Returns the max size of the secondary buffer.

        :return: The maximum size of the secondary buffer.
        :rtype: int
        """        
        return self.secondary_buffer.maxlen if self.secondary_buffer.maxlen is not None else float('inf')
    
    @property
    def main_size(self):
        """
        Returns the current size of the main buffer.

        :return: The current size of the main buffer.
        :rtype: int
        """
        return len(self.main_buffer)

    @property
    def secondary_size(self):
        """
        Returns the current size of the secondary buffer.

        :return: The current size of the secondary buffer.
        :rtype: int
        """        
        return len(self.secondary_buffer)
    
    # HELPER METHODS

    @staticmethod
    def episode_to_flat_dict(episode: Episode, labels):
        """
        Converts an episode to a dict representation matching the labels.

        :param episode: The episode to convert.
        :type episode: Episode
        :param labels: The labels to match in the dict representation.
        :type labels: list
        :return: A dict representation of the episode matching the labels.
        :rtype: dict
        """    
        vector = {}
        dimensions = [label.split(':') for label in labels]
        for label, instance in zip(labels, dimensions):
            if instance[0] == "action":
                if instance[1] == 'policy':
                    value = episode.action.policy_id
                else:
                    value = episode.action.actuation.get(instance[1], [{}])[0].get(instance[2], 0.0)
            elif instance[0] == "reward_list":
                value = episode.reward_list.get(instance[1], 0.0)
            elif instance[0] == "parent_policy":
                value = episode.parent_policy
            else:
                value = getattr(episode, instance[0]).get(instance[1], [{}])[0].get(instance[2], np.nan)
            vector[label] = value
        return vector
    
    @staticmethod
    def empty_episode(episode: Episode, inputs: list, outputs: list):
        """
        Checks if an episode is empty.

        :param episode: Episode object.
        :type episode: Episode
        :param inputs: Input labels to check.
        :type inputs: list
        :param outputs: Output labels to check.
        :type outputs: list
        :return: True if the episode is empty, False otherwise.
        :rtype: bool
        """
        empty = True
        instances = inputs + outputs
        for instance in instances:
            if instance == "action":
                if episode.action.actuation or episode.action.policy_id:
                    empty = False
            else:
                if getattr(episode, instance):
                    empty = False
        return empty

    @staticmethod
    def episode_to_vector(episode: Episode, labels):
        """
        Converts an episode to a vector representation ordered according to the given labels.

        :param episode: Episode object to convert.
        :type episode: Episode
        :param labels: Labels to match in the vector representation.
        :type labels: list
        :return: Vector representation of the episode.
        :rtype: np.ndarray
        """        

        flat_dict = EpisodicBuffer.episode_to_flat_dict(episode, labels)
        vector = np.zeros(len(labels))
        for i, label in enumerate(labels):
            vector[i] = flat_dict[label]
        return vector
    
    @staticmethod
    def vector_to_episode(vector, labels):
        """
        Converts a vector representation to an episode object. 

        :param vector: Vector representation of the episode.
        :type vector: np.ndarray
        :param labels: Labels to match in the vector representation.
        :type labels: list
        :raises ValueError: If the length of the vector does not match the number of labels.
        :return: Episode object created from the vector.
        :rtype: Episode
        """        
        episode = Episode()
        if len(labels) != len(vector):
            raise ValueError("The length of the vector does not match the number of labels.")
        for i, label in enumerate(labels):
            instance = label.split(':')
            if instance[0] == "action":
                if instance[1] == 'policy':
                    episode.action.policy_id = vector[i]
                else:
                    if not episode.action.actuation.get(instance[1], None):
                        episode.action.actuation[instance[1]] = [{}]
                    episode.action.actuation[instance[1]][0][instance[2]] = vector[i]
            elif instance[0] == "reward_list":
                episode.reward_list[instance[1]] = vector[i]
            else:
                if not getattr(episode, instance[0]).get(instance[1], None):
                    getattr(episode, instance[0])[instance[1]] = [{}]
                getattr(episode, instance[0])[instance[1]][0][instance[2]] = vector[i]
        return episode

    @staticmethod
    def buffer_to_dict_list(buffer, labels):
        """
        Converts a buffer of episodes to a list of dicts using the given labels.
        """
        return [EpisodicBuffer.episode_to_flat_dict(ep, labels) for ep in buffer]

    @staticmethod
    def buffer_to_dataframe(buffer, labels):
        """
        Converts a buffer of episodes to a pandas DataFrame using the given labels.
        
        :param buffer: Buffer of episodes to convert.
        :type buffer: deque
        :param labels: Labels to use for the DataFrame columns.
        :type labels: list
        :return: DataFrame containing the episodes in the buffer.
        :rtype: pd.DataFrame
        """
        return pd.DataFrame(EpisodicBuffer.buffer_to_dict_list(buffer, labels), columns=labels)
    
    @staticmethod
    def buffer_to_matrix(buffer, labels):
        """
        Converts a buffer of episodes to a numpy matrix using the given labels.
        
        :param buffer: Buffer of episodes to convert.
        :type buffer: deque
        :param labels: Labels to use for the matrix columns.
        :type labels: list
        :return: Numpy matrix containing the episodes in the buffer.
        :rtype: np.ndarray
        """
        return np.array([EpisodicBuffer.episode_to_vector(ep, labels) for ep in buffer])
    
    @staticmethod
    def matrix_to_buffer(matrix, labels):
        """
        Converts a numpy matrix to a buffer of episodes using the given labels.
        
        :param matrix: Numpy matrix to convert.
        :type matrix: np.ndarray
        :param labels: Labels to use for the episodes.
        :type labels: list
        :return: Buffer of episodes created from the matrix.
        :rtype: list
        """
        buffer = [EpisodicBuffer.vector_to_episode(row, labels) for row in matrix]
        return buffer

    def _get_samples_from_buffer(self, buffer, shuffle=False, n_samples=None):
        """
        Internal helper to get (inputs, outputs) numpy arrays from a buffer.

        :param buffer: Episode buffer to extract samples from.
        :type buffer: deque
        :param shuffle: Whether to shuffle the samples, defaults to False
        :type shuffle: bool, optional
        :param n_samples: Number of samples to extract, defaults to None
        :type n_samples: int, optional
        :return: Tuple of (inputs, outputs) numpy arrays
        :rtype: tuple
        """
        inputs = self.buffer_to_matrix(buffer, self.input_labels)
        if self.output_labels:
            outputs = self.buffer_to_matrix(buffer, self.output_labels)
        else:
            outputs = np.empty((inputs.shape[0], 0)) if inputs.size else np.empty((0,0))

        if shuffle:
            inputs, outputs = self._shuffle_dataset(inputs, outputs)

        # If requested, sample up to n_samples without replacement using the object's RNG
        if n_samples is not None:
            total = inputs.shape[0]
            if total > n_samples:
                idx = self.rng.choice(total, size=n_samples, replace=False)
                inputs = inputs[idx]
                if outputs.size:
                    outputs = outputs[idx]
        return inputs, outputs

    def _shuffle_dataset(self, inputs, outputs):
        """
        Helper method to shuffle the dataset represented by numpy arrays.

        :param inputs: Inputs array.
        :type inputs: np.ndarray
        :param outputs: Outputs array.
        :type outputs: np.ndarray
        :return: Tuple of shuffled (inputs, outputs) arrays.
        :rtype: tuple
        """        
        idx = self.rng.permutation(inputs.shape[0])
        inputs = inputs[idx]
        if len(outputs) > 0:
            outputs = outputs[idx]
        return inputs, outputs

    @staticmethod
    def _extract_labels(io_list, episode, label_list):
        """
        Helper method to obtain the label list from a given episode.

        :param io_list: List that specifies the fields to extract labels from.
        :type io_list: list
        :param episode: Episode object to extract labels from.
        :type episode: Episode
        :param label_list: List to append the extracted labels to.
        :type label_list: list
        """        
        for io in io_list:
            io_obj = getattr(episode, io)
            if isinstance(io_obj, Action):
                for group, dims_list in io_obj.actuation.items():
                    dims = dims_list[0]
                    for dim in dims:
                        label_list.append(f"{io}:{group}:{dim}")
            elif isinstance(io_obj, str):
                label_list.append(io)
            
            else:
                for group, dims_list in io_obj.items():
                    if isinstance(dims_list, list):
                        dims = dims_list[0]
                        for dim in dims:
                            label_list.append(f"{io}:{group}:{dim}")
                    else:
                        label_list.append(f"{io}:{group}")

class TraceBuffer(EpisodicBuffer):
    """
    Trace Buffer class, a specialized version of the Episodic Buffer that stores traces of episodes.
    """
    def __init__(self, node, main_size, secondary_size=0, max_traces=10, min_traces=1, max_antitraces=5, train_split=1.0, inputs=[], outputs=[], evaluation_method='linear', reward_factor=1.0, random_seed=0, **params):
        """Initialize a TraceBuffer for storing episode traces with utility values.

        Creates buffers for successful traces and antitraces, evaluating episode utilities
        based on final rewards using configurable evaluation methods.

        :param node: Owning cognitive node used for logging and ROS2 integration.
        :type node: CognitiveNode
        :param main_size: Maximum number of episodes stored in a single trace before completion.
        :type main_size: int
        :param secondary_size: Maximum number of episodes in secondary buffer (unused in TraceBuffer), defaults to 0
        :type secondary_size: int, optional
        :param max_traces: Maximum number of successful traces to retain, defaults to 10
        :type max_traces: int, optional
        :param min_traces: Minimum number of traces required before allowing antitraces to be added, defaults to 1
        :type min_traces: int, optional
        :param max_antitraces: Maximum number of failed traces (antitraces) to retain, defaults to 5
        :type max_antitraces: int, optional
        :param train_split: Fraction of episodes for training (inherited but unused in TraceBuffer), defaults to 1.0
        :type train_split: float, optional
        :param inputs: Episode fields considered inputs/features (e.g., ['old_perception', 'action']), defaults to []
        :type inputs: list, optional
        :param outputs: Episode fields considered outputs/targets (e.g., ['perception']), defaults to []
        :type outputs: list, optional
        :param evaluation_method: Method for computing utility values ('linear', 'exponential', 'goal_only'), defaults to 'linear'
        :type evaluation_method: str, optional
        :param reward_factor: Multiplicative factor applied to the final reward in utility calculation, defaults to 1.0
        :type reward_factor: float, optional
        :param random_seed: Seed for the internal Numpy RNG used for shuffling and sampling, defaults to 0
        :type random_seed: int, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict


        Notes:
        - Traces are stored as lists of (episode, utility) tuples.
        - Utility values are computed when a trace completes based on the evaluation method.
        - Antitraces represent failed sequences and have zero utility values.
        """
        super().__init__(node, main_size, secondary_size, train_split, inputs, outputs, random_seed, **params)
        self.traces_buffer = deque(maxlen=max_traces)
        self.antitraces_buffer = deque(maxlen=max_antitraces)
        self.new_traces = 0
        self.min_utility_fraction = 0.01
        self.min_traces = float(min_traces)
        self.evaluation_method = evaluation_method
        self.reward_factor = reward_factor

    def add_episode(self, episode, reward=0.0):
        """Add an episode to the trace buffer and complete the trace if reward is positive.

        Appends the episode to the main buffer. If a positive reward is provided,
        evaluates utilities for all episodes in the current trace, stores the trace,
        and clears the buffer to begin a new trace.

        :param episode: Episode instance to add to the current trace.
        :type episode: Episode
        :param reward: Reward value for the trace; positive values trigger trace completion, defaults to 0.0
        :type reward: float, optional
        :raises ValueError: If episode is not of type Episode.

        :return: None
        :rtype: None

        Notes:
        - Only episodes of type Episode are accepted.
        - Labels are configured on the first episode if not already set.
        - Positive rewards trigger utility evaluation and trace storage.
        - The buffer is cleared after storing a successful trace.
        """
        if type(episode) is not Episode:
            raise ValueError("The episode must be of type Episode.")
        if (not self.input_labels and self.inputs) or (not self.output_labels and self.outputs):
            self.configure_labels(episode)
        self.main_buffer.append(deepcopy(episode))
        self.new_sample_count_main += 1

        #Add corresponding trace
        if reward > 0: # If the reward is positive, consider it a successful trace
            utility_trace = self.evaluate_trace(reward)
            self.traces_buffer.append(list(zip(self.main_buffer, utility_trace)))
            self.new_traces += 1
            self.node.get_logger().info(f"Adding trace with {self.main_size} episodes. New traces: {self.new_traces}")
            self.clear()

    def add_antitrace(self):
        """Add an antitrace to the buffer if enough traces exist.
        """        
        if self.n_traces >= self.min_traces: # If the buffer is full, and there are enough traces, add an antitrace
            self.node.get_logger().info("Adding antitrace")
            self.antitraces_buffer.append(list(zip(self.main_buffer, np.zeros(self.main_max_size))))
            self.clear()

    def evaluate_trace(self, reward):
        """Compute utility values for each episode in the current trace.

        Uses the configured evaluation method to assign utility values to all episodes
        in the main buffer based on the final reward. The evaluation method determines
        how utility is distributed across the episode sequence.

        :param reward: Final reward value for the trace used to compute utilities.
        :type reward: float
        :return: List of utility values, one per episode in the trace.
        :rtype: list[float]
        """
        n = len(self.main_buffer)
        if n == 0:
            return []
        min_val = reward * self.min_utility_fraction
        values = getattr(self, f"eval_{self.evaluation_method}", self.eval_default)(reward, min_val, self.reward_factor, n, self.main_max_size)
        return values

    def get_flattened_traces(self, n_samples=None):
        """Flatten stored traces into a single buffer with corresponding utilities.

        Optionally samples a subset of traces, then flattens all (episode, utility) pairs
        from the selected traces into a single sequential buffer.

        :param n_samples: Maximum number of traces to sample; if None, uses all traces, defaults to None
        :type n_samples: int | None, optional
        :return: Tuple of (episode buffer, utilities array) containing flattened trace data.
        :rtype: tuple[list, np.ndarray]
        """
        if n_samples is not None:
            if len(self.traces_buffer) > n_samples:
                idx = self.rng.choice(len(self.traces_buffer), size=n_samples, replace=False)
                selected_traces = [self.traces_buffer[i] for i in idx]
            else:
                selected_traces = list(self.traces_buffer)
        else:
            selected_traces = list(self.traces_buffer)
        flattened_traces = [item for trace in selected_traces for item in trace]
        buffer, utilities = zip(*flattened_traces) if flattened_traces else ([], [])
        return buffer, np.array(utilities)

    def get_dataset(self, shuffle=True, n_samples=None):
        """Return training dataset from flattened traces as numpy arrays.

        Flattens stored traces into episode states and utilities, optionally shuffling
        the resulting dataset.

        :param shuffle: Whether to shuffle the dataset, defaults to True
        :type shuffle: bool, optional
        :param n_samples: Maximum number of traces to include; if None, uses all traces, defaults to None
        :type n_samples: int | None, optional
        :return: Tuple of (states, utilities) numpy arrays for training.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        buffer, utilities = self.get_flattened_traces(n_samples)
        states, _ = self._get_samples_from_buffer(buffer, shuffle=False)
        x_train, y_train = self._shuffle_dataset(states, utilities) if shuffle else (states, utilities)
        return x_train, y_train
    
    def reset_new_sample_count(self, main=True, secondary=True):
        """
        Resets the new sample count for the trace buffer.

        :param main: Unused in this class, defaults to True.
        :type main: bool
        :param secondary: Unused in this class, defaults to True.
        :type secondary: bool
        """
        self.new_traces = 0

    @staticmethod
    def eval_default(reward, min_val, reward_factor, n, full_length):
        """Default evaluation method placeholder.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value derived from the reward (e.g., reward * min_utility_fraction).
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (main buffer capacity).
        :type full_length: int
        :raises NotImplementedError: Always, as this method is a placeholder.
        """
        raise NotImplementedError(f"Evaluation method requested is not implemented.")

    @staticmethod
    def eval_linear(reward, min_val, reward_factor, n, full_length):
        """Linear evaluation: interpolate utilities from a start value to the final reward.

        Computes a start value consistent with the full-length trace and linearly
        interpolates utilities across episodes, ending at reward * reward_factor.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value derived from the reward.
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (main buffer capacity).
        :type full_length: int
        :return: List of length n with linearly increasing utility values.
        :rtype: list[float]
        """
        if n == 1:
            return [reward]
        
        # Compute the start value at the equivalent position in a full-length trace
        start_val = min_val + (reward - min_val) * (full_length - n) / (full_length - 1)
        
        # Linear interpolation from start_val to reward over n steps
        values = []
        for i in range(n-1):
            value = start_val + (reward - start_val) * i / (n - 1)
            values.append(value)
        values.append(reward*reward_factor)
        return values

    @staticmethod
    def eval_exponential(reward, min_val, reward_factor, n, full_length):
        """Exponential evaluation: grow utilities exponentially towards the final reward.

        Uses k = ln(reward/min_val) / (full_length - 1) and evaluates utilities at
        positions [full_length - n, ..., full_length - 2], appending reward * reward_factor
        as the final value.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value derived from the reward; adjusted if non-positive.
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (main buffer capacity).
        :type full_length: int
        :return: List of length n with exponentially increasing utility values.
        :rtype: list[float]
        """
        if n == 1:
            return [reward*reward_factor]
        
        # Ensure min_val is positive for logarithm calculation
        if min_val <= 0:
            min_val = 0.001
        
        # Calculate the exponential growth rate based on full sequence
        k = np.log(reward / min_val) / (full_length - 1)
        
        # Generate exponential values for the last n positions of the full sequence
        values = []
        start_position = full_length - n
        for i in range(n-1):
            position = start_position + i
            value = min_val * np.exp(k * position)
            values.append(value)
        values.append(reward*reward_factor)
        return values
    
    @staticmethod
    def eval_goal_only(reward, min_val, reward_factor, n, full_length):
        """Goal-only evaluation: zero utility except at the final episode.

        Assigns 0.0 to all episodes except the last, which receives reward * reward_factor.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value (unused in this method).
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (unused in this method).
        :type full_length: int
        :return: List of length n with zero utilities except the final element.
        :rtype: list[float]
        """
        values = [0.0] * (n - 1) + [reward*reward_factor]
        return values

    @property
    def max_traces(self):
        """Maximum capacity of the traces buffer.

        :return: Max number of traces retained.
        :rtype: int
        """
        return self.traces_buffer.maxlen

    @property
    def max_antitraces(self):
        """Maximum capacity of the antitraces buffer.

        :return: Max number of antitraces retained.
        :rtype: int
        """
        return self.antitraces_buffer.maxlen
    
    @property
    def n_traces(self):
        """Current number of stored traces.

        :return: Count of traces in the buffer.
        :rtype: int
        """
        return len(self.traces_buffer)

    @property
    def n_antitraces(self):
        """Current number of stored antitraces.

        :return: Count of antitraces in the buffer.
        :rtype: int
        """
        return len(self.antitraces_buffer)


class TestEpisodicBuffer(CognitiveNode):
    """
    Test Episodic Buffer class, this class is a test implementation of the Episodic Buffer.
    It is used to test the functionality of the Episodic Buffer.
    """
    def __init__(self, name='test_episodic_buffer', **params):
        """
        Constructor of the Test Episodic Buffer class.

        :param name: The name of the Test Episodic Buffer instance.
        :type name: str
        """
        super().__init__(name, **params)
        self.episodic_buffer = EpisodicBuffer(self, main_size=10, secondary_size=5, train_split=0.8, inputs=['old_perception', 'action'], outputs=['perception'], random_seed=42)
        self.episode_subscription = self.create_subscription(
            EpisodeMsg,
            '/main_loop/episodes',
            self.episode_callback,
            1
        )

    def episode_callback(self, msg: EpisodeMsg):
        """
        Callback for the episode subscription. It receives an episode message and adds it to the episodic buffer.

        :param msg: The episode message received.
        :type msg: cognitive_node_interfaces.msg.Episode
        """
        episode = episode_msg_to_obj(msg)
        self.episodic_buffer.add_episode(episode)
        self.get_logger().info(f"Episode added to buffer: \n {episode} \n New main samples: {self.episodic_buffer.new_sample_count_main}, New secondary samples: {self.episodic_buffer.new_sample_count_secondary}")

        self.get_logger().info(f"MAIN BUFFER CONTENTS: ")
        for i, episode in enumerate(self.episodic_buffer.main_buffer):
            self.get_logger().info(f" - Episode {i}:\n {episode}")

        self.get_logger().info(f"SECONDARY BUFFER CONTENTS: ")
        for i, episode in enumerate(self.episodic_buffer.secondary_buffer):
            self.get_logger().info(f" - Episode {i}:\n {episode}")
        
        if self.episodic_buffer.new_sample_count_main >= 10:
            x_train, y_train, x_test, y_test = self.episodic_buffer.get_dataset()
            self.get_logger().info(f"Features Train: \n {x_train}")
            self.get_logger().info(f"Targets Train: \n {y_train}")
            self.get_logger().info(f"Features Test: \n {x_test}")
            self.get_logger().info(f"Targets Test: \n {y_test}")

def test_episodic_buffer(args=None):
    rclpy.init(args=args)

    generic_model = TestEpisodicBuffer()

    rclpy.spin(generic_model)

    generic_model.destroy_node()
    rclpy.shutdown()
