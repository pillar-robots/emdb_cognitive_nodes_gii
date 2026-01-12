import rclpy
import gc
from rclpy.impl.rcutils_logger import RcutilsLogger

import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.losses import Loss
from keras._tf_keras.keras.callbacks import TensorBoard
from keras import layers, metrics, losses, Sequential

from core.cognitive_node import CognitiveNode
from core.utils import class_from_classname, msg_to_dict
from cognitive_node_interfaces.srv import SetActivation, Predict, GetSuccessRate, IsCompatible
from cognitive_nodes.episodic_buffer import EpisodicBuffer
from cognitive_nodes.episode import Episode, Action, episode_msg_to_obj, episode_msg_list_to_obj_list, episode_obj_list_to_msg_list

class DeliberativeModel(CognitiveNode):
    """
    Deliberative Model class, this class is a generic model that can be used to implement different types of deliberative models. 
    """
    def __init__(self, name='model', class_name = 'cognitive_nodes.deliberative_model.DeliberativeModel', node_type="deliberative_model", prediction_srv_type=None, **params):
        """
        Constructor of the Deliberative Model class.

        Initializes a Deliberative instance with the given name.

        :param name: The name of the Deliberative Model instance.
        :type name: str
        :param class_name: The name of the DeliberativeModel class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "deliberative_model".
        :type node_type: str
        """
        super().__init__(name, class_name, **params)

        self.episodic_buffer=None
        self.learner=None
        self.confidence_evaluator=None

        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation,
            node_type+ "/" + str(name) + '/set_activation',
            self.set_activation_callback,
            callback_group=self.cbgroup_server
        )

        # N: Predict Service
        if prediction_srv_type is not None:
            prediction_srv_type = class_from_classname(prediction_srv_type)
        else:
            raise ValueError("prediction_srv_type must be provided and be a valid class name.")
        self.predict_service = self.create_service(
            prediction_srv_type,
            node_type+ "/" + str(name) + '/predict',
            self.predict_callback,
            callback_group=self.cbgroup_server
        )

        # N: Get Success Rate Service
        self.get_success_rate_service = self.create_service(
            GetSuccessRate,
            node_type+ "/" + str(name) + '/get_success_rate',
            self.get_success_rate_callback,
            callback_group=self.cbgroup_server
        )

        # N: Is Compatible Service
        self.is_compatible_service = self.create_service(
            IsCompatible,
            node_type+ "/" + str(name) + '/is_compatible',
            self.is_compatible_callback,
            callback_group=self.cbgroup_server
        )

        #TODO: Set activation from main_loop
        #self.activation.activation = 1.0

    def set_activation_callback(self, request, response):
        """
        Some processes can modify the activation of a Model.

        :param request: The request that contains the new activation value.
        :type request: cognitive_node_interfaces.srv.SetActivation.Request
        :param response: The response indicating if the activation was set.
        :type response: cognitive_node_interfaces.srv.SetActivation.Response
        :return: The response indicating if the activation was set.
        :rtype: cognitive_node_interfaces.srv.SetActivation.Response
        """
        activation = request.activation
        self.get_logger().info('Setting activation ' + str(activation) + '...')
        self.activation.activation = activation
        self.activation.timestamp = self.get_clock().now().to_msg()
        response.set = True
        return response
    
    def predict_callback(self, request, response):
        """
        Get predicted perception values for the last perceptions not newer than a given
        timestamp and for a given policy.

        :param request: The request that contains the timestamp and the policy.
        :type request: cognitive_node_interfaces.srv.Predict.Request
        :param response: The response that included the obtained perception.
        :type response: cognitive_node_interfaces.srv.Predict.Response
        :return: The response that included the obtained perception.
        :rtype: cognitive_node_interfaces.srv.Predict.Response
        """
        self.get_logger().info('Predicting ...') 
        input_episodes = episode_msg_list_to_obj_list(request.input_episodes)
        output_episodes = self.predict(input_episodes)
        response.output_episodes = episode_obj_list_to_msg_list(output_episodes)
        self.get_logger().info(f"Prediction made... ")
        return response
    
    def get_success_rate_callback(self, request, response): # TODO: implement
        """
        Get a prediction success rate based on a historic of previous predictions.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.GetSuccessRate.Request
        :param response: The response that contains the predicted success rate.
        :type response: cognitive_node_interfaces.srv.GetSuccessRate.Response
        :return: The response that contains the predicted success rate.
        :rtype: cognitive_node_interfaces.srv.GetSuccessRate.Response
        """
        self.get_logger().info('Getting success rate..')
        raise NotImplementedError
        response.success_rate = 0.5
        return response
    
    def is_compatible_callback(self, request, response): # TODO: implement
        """
        Check if the Model is compatible with the current available perceptions.

        :param request: The request that contains the current available perceptions.
        :type request: cognitive_node_interfaces.srv.IsCompatible.Request
        :param response: The response indicating if the Model is compatible or not.
        :type response: cognitive_node_interfaces.srv.IsCompatible.Response
        :return: The response indicating if the Model is compatible or not.
        :rtype: cognitive_node_interfaces.srv.IsCompatible.Response
        """
        self.get_logger().info('Checking if compatible..')
        raise NotImplementedError
        response.compatible = True
        return response

    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the activation value of the Model.

        :param perception: Perception does not influence the activation.
        :type perception: dict
        :param activation_list: List of activation values from other sources, defaults to None.
        :type activation_list: list
        :return: The activation of the instance and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation

    def predict(self, input_episodes: list[Episode]) -> list:
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        predictions = self.learner.call(input_data)
        if predictions is None:
            predicted_episodes = input_episodes  # If the model is not configured, return the input episodes
        else:
            self.get_logger().info(f"Predictions: {predictions}")
            self.get_logger().info(f"Output labels: {self.episodic_buffer.output_labels}")
            predicted_episodes = self.episodic_buffer.matrix_to_buffer(predictions, self.episodic_buffer.output_labels)
        self.get_logger().info(f"Prediction made: {predicted_episodes}")
        return predicted_episodes
    
    
    
class Learner:
    """
    Class that wraps around a learning model (Linear Classifier, ANN, SVM...)
    """    
    def __init__(self, node:CognitiveNode, buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the Learner class.

        :param buffer: Episodic buffer to use.
        :type buffer: generic_model.EpisodicBuffer
        """        
        self.node = node
        self.model=None
        self.buffer=buffer
        self.configured=False

    def train(self):
        """
        Placeholder method for training the model.

        :raises NotImplementedError: Not implemented yet.
        """        
        raise NotImplementedError
    
    def call(self, x):
        """
        Placeholder method for predicting an outcome.

        :param perception: Perception dictionary.
        :type perception: dict
        :param action: Candidate action dictionary.
        :type action: dict
        :raises NotImplementedError: Not implemented yet.
        """        
        raise NotImplementedError
    

class ANNLearner(Learner):
    def __init__(self, node, buffer, batch_size=32, epochs=50, output_activation='sigmoid', hidden_activation='relu', hidden_layers=[128], learning_rate=0.001, model_file=None, tensorboard=False, tensorboard_log_dir=None, **params):
        super().__init__(node, buffer, **params)
        tf.config.set_visible_devices([], 'GPU') # TODO: Handle GPU usage properly
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.configured = False
        self.model_file = model_file
        self.tensorboard = tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir
        self._run_counter = 0
        if self.model_file is not None:
            self.configure_model(0,0) # Load model from file

    def configure_model(self, input_length, output_length):
        """
        Configure the ANN model with the given input shape, output shape, and labels.

        :param input_shape: The shape of the input data.
        :type input_shape: int
        :param output_shape: The shape of the output data.
        :type output_shape: int
        """
        if self.model_file is None:
            self.model = Sequential()
            
            ## TODO: USE THE LABELS TO SEPARATE THE INPUTS INTO REGUAR INPUTS AND THE POLICY ID INPUT, THEN CONCATNATE ##
            # TODO: THIS MIGHT REQUIRE TO USE THE FUNCTIONAL API INSTEAD OF SEQUENTIAL
            # --- Inputs ---
            # object_input = layers.Input(shape=(), dtype=tf.int32, name="object_id")
            # numeric_input = layers.Input(shape=(num_numeric_features,), dtype=tf.float32, name="numeric_features")

            # --- Embedding Layer ---
            #embedding_layer = layers.Embedding(input_dim=num_objects, output_dim=embedding_dim)
            #embedded_object = embedding_layer(object_input)  # shape: (batch_size, embedding_dim)

            ## TODO: USE THE LABELS TO SEPARATE THE INPUTS INTO REGUAR INPUTS AND THE POLICY ID INPUT, THEN CONCATNATE ##

            self.model.add(layers.Input(shape=(input_length,)))
            for units in self.hidden_layers:
                self.model.add(layers.Dense(units,
                                        activation=self.hidden_activation,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-5)))
                self.model.add(layers.BatchNormalization())   # stabilizes training
                self.model.add(layers.Dropout(0.1))           # mild dropout
            self.model.add(layers.Dense(output_length, activation=self.output_activation))
            #self.model.compile(optimizer=self.optimizer, loss=AsymmetricMSE(underestimation_penalty=3.0), metrics=['mae'])
            self.model.compile(optimizer=self.optimizer, loss="mse", metrics=['mae'])
            self.input_length = input_length
            self.output_length = output_length
            self.configured = True

        else:
            self.node.get_logger().info(f"Loading model from {self.model_file}")
            self.model = tf.keras.models.load_model(self.model_file, custom_objects={"AsymmetricMSE": AsymmetricMSE})
            self.input_length = self.model.input_shape[1]
            self.output_length = self.model.output_shape[1]
            self.configured = True               
    
    def reset_model_state(self):
        """Reset only the optimizer state, keeping model weights unchanged."""
        if self.configured:
            weights = self.model.get_weights()
            # Create a fresh optimizer instance
            fresh_optimizer = Adam(learning_rate=self.learning_rate)
            del self.model
            gc.collect()
            self.configure_model(self.input_length, self.output_length)
            # Recompile with fresh optimizer (weights unchanged)
            self.model.compile(optimizer=fresh_optimizer, loss="mse", metrics=['mae'])
            self.model.set_weights(weights)

    def train(self, x_train, y_train, epochs=None, batch_size=None, validation_split=0.0, x_val=None, y_val=None, verbose=1, reset_optimizer=True):
        # Ensure x_train and y_train are at least 2D
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        # Handle validation data if provided
        validation_data = None
        if x_val is not None and y_val is not None:
            # Ensure x_val and y_val are at least 2D
            if len(x_val.shape) == 1:
                x_val = x_val.reshape(-1, 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (x_val, y_val)
            # If validation_data is provided, set validation_split to 0.0
            validation_split = 0.0

        
        if not epochs:
            epochs = self.epochs
        if not batch_size:
            batch_size = self.batch_size
        if not self.configured:
            self.configure_model(x_train.shape[1], y_train.shape[1])
        callbacks = []

        if reset_optimizer:
            self.reset_model_state()

        # Learning rate scheduling
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=epochs // 10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=epochs // 5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        if self.tensorboard:
            if self.tensorboard_log_dir is None:
                self.tensorboard_log_dir = f'logs/fit/{self.node.name}'
            run_dict = self.tensorboard_log_dir + f'/run_{self._run_counter}'
            self._run_counter += 1
            tensorboard_callback = TensorBoard(log_dir=run_dict, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks
        )

    def call(self, x):
        if not self.configured:
            return None
        return self.model.predict(x)
    
    def evaluate(self, x_test, y_test):
        if not self.configured:
            return None
        return self.model.evaluate(x_test, y_test, verbose=0)[1]
    
    def get_weights(self):
        """
        Get the current model weights.
        
        :return: List of weight arrays or None if model not configured.
        :rtype: list or None
        """
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot get weights.")
            return None
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """
        Set the model weights.
        
        :param weights: List of weight arrays to set.
        :type weights: list
        :return: True if weights were set successfully, False otherwise.
        :rtype: bool
        """
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot set weights.")
            return False
        
        try:
            self.model.set_weights(weights)
            self.node.get_logger().info("Weights set successfully.")
            return True
        except Exception as e:
            self.node.get_logger().error(f"Failed to set weights: {e}")
            return False

class AsymmetricMSE(Loss):
    def __init__(self, underestimation_penalty=1.0, overestimation_penalty=1.0, name="asymmetric_mse"):
        """
        underestimation_penalty: float, multiplier applied when y_pred < y_true
        overestimation_penalty: float, multiplier applied when y_pred > y_true
        """
        super().__init__(name=name)
        self.underestimation_penalty = underestimation_penalty
        self.overestimation_penalty = overestimation_penalty

    def call(self, y_true, y_pred):
        error = y_pred - y_true
        weight = tf.where(error < 0, self.underestimation_penalty, self.overestimation_penalty)
        return tf.reduce_mean(weight * tf.square(error))

    def get_config(self):
        config = super().get_config()
        config.update({
            "underestimation_penalty": self.underestimation_penalty
        })
        return config



################################################
####   PYTORCH VERSION - WORK IN PROGRESS   ####
################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os


class ANNLearner_torch(Learner):
    def __init__(self, node, buffer, batch_size=32, epochs=50, output_activation='sigmoid', 
                 hidden_activation='relu', hidden_layers=[128], learning_rate=0.001, loss_function=nn.MSELoss, val_function=nn.L1Loss, 
                 model_file=None, tensorboard=False, tensorboard_log_dir=None, 
                 device='cpu', **params):
        super().__init__(node, buffer, **params)
        
        # PyTorch specific setup
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        self.batch_size = batch_size
        self.epochs = epochs
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.configured = False
        self.model_file = model_file
        self.tensorboard = tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir
        self._run_counter = 0
        
        # Model and optimizer will be initialized later
        self.model = None
        self.optimizer = None
        self.criterion = loss_function()
        self.val_criterion = val_function()
        
        if self.model_file is not None:
            self.load_model()

    def configure_model(self, input_length, output_length):
        """Configure the ANN model with PyTorch."""
        self.model = ANNModel(
            input_size=input_length,
            output_size=output_length,
            hidden_layers=self.hidden_layers,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.input_length = input_length
        self.output_length = output_length
        self.configured = True
        
        self.node.get_logger().info(f"Model configured with input: {input_length}, output: {output_length}")

    def load_model(self):
        """Load model from file."""
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file, map_location=self.device)
            
            # Extract model configuration from checkpoint
            self.input_length = checkpoint['input_length']
            self.output_length = checkpoint['output_length']
            self.hidden_layers = checkpoint['hidden_layers']
            self.hidden_activation = checkpoint['hidden_activation']
            self.output_activation = checkpoint['output_activation']
            self.learning_rate = checkpoint['learning_rate']
            
            # Configure model architecture
            self.configure_model(self.input_length, self.output_length)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.node.get_logger().info(f"Model loaded from {self.model_file}")
        else:
            self.node.get_logger().warning(f"Model file {self.model_file} not found")

    def save_model(self, filepath=None):
        """Save model to file."""
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot save.")
            return False
            
        save_path = filepath or self.model_file
        if save_path is None:
            self.node.get_logger().warning("No save path provided")
            return False
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_length': self.input_length,
            'output_length': self.output_length,
            'hidden_layers': self.hidden_layers,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate
        }
        
        torch.save(checkpoint, save_path)
        self.node.get_logger().info(f"Model saved to {save_path}")
        return True

    def reset_model_state(self):
        """Reset optimizer state while keeping model weights."""
        if self.configured:
            # Save current weights
            weights = self.model.state_dict().copy()
            
            # Reinitialize optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Restore weights
            self.model.load_state_dict(weights)

    def train(self, x_train, y_train, epochs=None, batch_size=None, validation_split=0.0, 
              x_val=None, y_val=None, verbose=1, reset_optimizer=True):
        """Train the model using PyTorch."""
        
        # Convert numpy arrays to torch tensors
        if isinstance(x_train, np.ndarray):
            x_train = torch.FloatTensor(x_train)
        if isinstance(y_train, np.ndarray):
            y_train = torch.FloatTensor(y_train)
            
        # Ensure proper dimensions
        if len(x_train.shape) == 1:
            x_train = x_train.unsqueeze(1)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
            
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        
        # Configure model if not done
        if not self.configured:
            self.configure_model(x_train.shape[1], y_train.shape[1])
            
        if reset_optimizer:
            self.reset_model_state()
            
        # Prepare validation data
        val_loader = None
        if x_val is not None and y_val is not None:
            if isinstance(x_val, np.ndarray):
                x_val = torch.FloatTensor(x_val)
            if isinstance(y_val, np.ndarray):
                y_val = torch.FloatTensor(y_val)
                
            if len(x_val.shape) == 1:
                x_val = x_val.unsqueeze(1)
            if len(y_val.shape) == 1:
                y_val = y_val.unsqueeze(1)
                
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        elif validation_split > 0:
            # Split training data for validation
            split_idx = int(len(x_train) * (1 - validation_split))
            x_val = x_train[split_idx:]
            y_val = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]
            
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Create data loader for training
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, 
            patience=epochs//10, min_lr=1e-7,
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = epochs // 5
        best_epoch = 0
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
            avg_train_loss = train_loss / num_batches
            
            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = self.val_criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                        
                val_loss = val_loss / val_batches
                self.model.train()
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    self.best_weights = self.model.state_dict().copy()
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        self.node.get_logger().info(f"Early stopping at epoch {epoch+1}. Restoring weights from epoch {best_epoch+1} with val loss {best_val_loss:.4f}.")
                    # Restore best weights
                    self.model.load_state_dict(self.best_weights)
                    break
                    
            if verbose > 0:
                if val_loss is not None:
                    self.node.get_logger().info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    self.node.get_logger().info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")

    def call(self, x):
        """Make predictions with the model."""
        if not self.configured:
            return None
            
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
            
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
            
        x = x.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x)
            
        return predictions.cpu().numpy()

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return MAE."""
        if not self.configured:
            return None
            
        # Convert to tensors
        if isinstance(x_test, np.ndarray):
            x_test = torch.FloatTensor(x_test)
        if isinstance(y_test, np.ndarray):
            y_test = torch.FloatTensor(y_test)
            
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(1)
        if len(y_test.shape) == 1:
            y_test = y_test.unsqueeze(1)
            
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_test)
            mae = torch.mean(torch.abs(predictions - y_test))
            
        return mae.cpu().item()

    def get_weights(self):
        """Get the current model weights."""
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot get weights.")
            return None
        return self.model.state_dict()

    def set_weights(self, weights):
        """Set the model weights."""
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot set weights.")
            return False
            
        try:
            self.model.load_state_dict(weights)
            self.node.get_logger().info("Weights set successfully.")
            return True
        except Exception as e:
            self.node.get_logger().error(f"Failed to set weights: {e}")
            return False


class ANNModel(nn.Module):
    """PyTorch neural network model."""
    
    def __init__(self, input_size, output_size, hidden_layers=[128], 
                 hidden_activation='relu', output_activation='sigmoid'):
        super(ANNModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.LayerNorm(hidden_size))
            self.layers.append(self._get_activation(hidden_activation))
            self.layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
            
        # Output layer
        self.layers.append(nn.Linear(prev_size, output_size))
        
        # Output activation
        if output_activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            self.layers.append(nn.Tanh())
        elif output_activation == 'relu':
            self.layers.append(nn.ReLU())
        # No activation for linear output
        
    def _get_activation(self, activation_name):
        """Get activation function by name."""
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()  # Default
            
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x


class AsymmetricMSELoss(nn.Module):
    """PyTorch version of AsymmetricMSE loss."""
    
    def __init__(self, underestimation_penalty=1.0, overestimation_penalty=1.0):
        super(AsymmetricMSELoss, self).__init__()
        self.underestimation_penalty = underestimation_penalty
        self.overestimation_penalty = overestimation_penalty
        
    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        weight = torch.where(error < 0, self.underestimation_penalty, self.overestimation_penalty)
        return torch.mean(weight * torch.square(error))

class Evaluator:
    """
    Class that evaluates the success rate of a model based on its predictions.
    """    
    def __init__(self, node:CognitiveNode, learner:Learner,  buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the Learner class.

        :param node: Cognitive node that uses this model.
        :type node: CognitiveNode
        :param learner: Learner instance to use for predictions.
        :type learner: Learner
        :param buffer: Episodic buffer to use.
        :type buffer: generic_model.EpisodicBuffer
        """
        self.node = node
        self.learner = learner
        self.buffer = buffer
        self.learner = learner

    def evaluate(self):
        """
        Placeholder method for evaluating the model's success rate.

        :raises NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError


