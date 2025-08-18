#!/usr/bin/env python3
"""
Brain-inspired network topology builder for neuromorphic computing.

This module implements realistic brain-like network architectures including:
- Distance-dependent connectivity with spatial layouts
- Excitatory/inhibitory balance (80/20 ratio)
- Modular network structures
- Small-world network properties
- Biologically realistic connection patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math

from .network import NeuromorphicNetwork, NetworkLayer


class NeuronType(Enum):
    """Types of neurons in brain-inspired networks."""
    EXCITATORY_PYRAMIDAL = "excitatory_pyramidal"
    EXCITATORY_SPINY_STELLATE = "excitatory_spiny_stellate"
    INHIBITORY_BASKET = "inhibitory_basket"
    INHIBITORY_CHANDELIER = "inhibitory_chandelier"
    INHIBITORY_MARTINOTTI = "inhibitory_martinotti"


@dataclass
class SpatialPosition:
    """3D spatial position for neurons."""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'SpatialPosition') -> float:
        """Calculate Euclidean distance to another position."""
        return math.sqrt(
            (self.x - other.x)**2 + 
            (self.y - other.y)**2 + 
            (self.z - other.z)**2
        )


@dataclass
class NetworkModule:
    """A module in a modular network architecture."""
    name: str
    center_position: SpatialPosition
    radius: float
    neuron_ids: List[int]
    excitatory_fraction: float = 0.8


class SpatialNetworkLayout:
    """
    Spatial layout system for positioning neurons in 2D/3D space.
    
    Supports various layout patterns:
    - Grid layout (regular spacing)
    - Random layout (uniform random positions)
    - Clustered layout (modules with local clustering)
    - Cortical column layout (layered structure)
    """
    
    def __init__(self, dimensions: int = 2, bounds: Tuple[float, float, float] = (100.0, 100.0, 10.0)):
        """
        Initialize spatial layout system.
        
        Args:
            dimensions: Number of spatial dimensions (2 or 3)
            bounds: (width, height, depth) of the spatial area
        """
        self.dimensions = dimensions
        self.bounds = bounds
        self.neuron_positions: Dict[int, SpatialPosition] = {}
        self.modules: Dict[str, NetworkModule] = {}
        
    def create_grid_layout(self, num_neurons: int, spacing: float = 1.0) -> Dict[int, SpatialPosition]:
        """
        Create a regular grid layout of neurons.
        
        Args:
            num_neurons: Number of neurons to position
            spacing: Distance between adjacent neurons
            
        Returns:
            Dictionary mapping neuron IDs to positions
        """
        positions = {}
        
        if self.dimensions == 2:
            # 2D grid
            grid_size = int(math.ceil(math.sqrt(num_neurons)))
            for i in range(num_neurons):
                row = i // grid_size
                col = i % grid_size
                positions[i] = SpatialPosition(
                    x=col * spacing,
                    y=row * spacing,
                    z=0.0
                )
        else:
            # 3D grid
            grid_size = int(math.ceil(num_neurons**(1/3)))
            for i in range(num_neurons):
                layer = i // (grid_size * grid_size)
                remainder = i % (grid_size * grid_size)
                row = remainder // grid_size
                col = remainder % grid_size
                positions[i] = SpatialPosition(
                    x=col * spacing,
                    y=row * spacing,
                    z=layer * spacing
                )
                
        self.neuron_positions.update(positions)
        return positions
        
    def create_random_layout(self, num_neurons: int) -> Dict[int, SpatialPosition]:
        """
        Create a random uniform layout of neurons.
        
        Args:
            num_neurons: Number of neurons to position
            
        Returns:
            Dictionary mapping neuron IDs to positions
        """
        positions = {}
        
        for i in range(num_neurons):
            x = np.random.uniform(0, self.bounds[0])
            y = np.random.uniform(0, self.bounds[1])
            z = np.random.uniform(0, self.bounds[2]) if self.dimensions == 3 else 0.0
            
            positions[i] = SpatialPosition(x=x, y=y, z=z)
            
        self.neuron_positions.update(positions)
        return positions
        
    def create_clustered_layout(
        self, 
        num_neurons: int, 
        num_clusters: int = 4,
        cluster_radius: float = 20.0,
        intra_cluster_std: float = 5.0
    ) -> Dict[int, SpatialPosition]:
        """
        Create a clustered layout with multiple modules.
        
        Args:
            num_neurons: Number of neurons to position
            num_clusters: Number of clusters/modules
            cluster_radius: Radius of each cluster
            intra_cluster_std: Standard deviation of positions within clusters
            
        Returns:
            Dictionary mapping neuron IDs to positions
        """
        positions = {}
        neurons_per_cluster = num_neurons // num_clusters
        
        # Create cluster centers
        cluster_centers = []
        for i in range(num_clusters):
            angle = 2 * math.pi * i / num_clusters
            center_x = self.bounds[0] / 2 + cluster_radius * math.cos(angle)
            center_y = self.bounds[1] / 2 + cluster_radius * math.sin(angle)
            center_z = self.bounds[2] / 2 if self.dimensions == 3 else 0.0
            
            cluster_centers.append(SpatialPosition(center_x, center_y, center_z))
            
        # Position neurons around cluster centers
        neuron_id = 0
        for cluster_idx, center in enumerate(cluster_centers):
            cluster_neurons = neurons_per_cluster
            if cluster_idx == num_clusters - 1:
                # Last cluster gets remaining neurons
                cluster_neurons = num_neurons - neuron_id
                
            cluster_neuron_ids = []
            for _ in range(cluster_neurons):
                # Gaussian distribution around cluster center
                x = np.random.normal(center.x, intra_cluster_std)
                y = np.random.normal(center.y, intra_cluster_std)
                z = np.random.normal(center.z, intra_cluster_std) if self.dimensions == 3 else 0.0
                
                # Keep within bounds
                x = np.clip(x, 0, self.bounds[0])
                y = np.clip(y, 0, self.bounds[1])
                z = np.clip(z, 0, self.bounds[2])
                
                positions[neuron_id] = SpatialPosition(x=x, y=y, z=z)
                cluster_neuron_ids.append(neuron_id)
                neuron_id += 1
                
            # Create module
            module = NetworkModule(
                name=f"module_{cluster_idx}",
                center_position=center,
                radius=cluster_radius,
                neuron_ids=cluster_neuron_ids
            )
            self.modules[module.name] = module
            
        self.neuron_positions.update(positions)
        return positions
        
    def create_cortical_column_layout(
        self,
        num_neurons: int,
        num_layers: int = 6,
        column_radius: float = 15.0,
        layer_thickness: float = 5.0
    ) -> Dict[int, SpatialPosition]:
        """
        Create a cortical column layout with layered structure.
        
        Args:
            num_neurons: Number of neurons to position
            num_layers: Number of cortical layers
            column_radius: Radius of the cortical column
            layer_thickness: Thickness of each layer
            
        Returns:
            Dictionary mapping neuron IDs to positions
        """
        positions = {}
        neurons_per_layer = num_neurons // num_layers
        
        neuron_id = 0
        for layer_idx in range(num_layers):
            layer_neurons = neurons_per_layer
            if layer_idx == num_layers - 1:
                # Last layer gets remaining neurons
                layer_neurons = num_neurons - neuron_id
                
            layer_z = layer_idx * layer_thickness
            
            for _ in range(layer_neurons):
                # Random position within circular column
                angle = np.random.uniform(0, 2 * math.pi)
                radius = np.random.uniform(0, column_radius)
                
                x = self.bounds[0] / 2 + radius * math.cos(angle)
                y = self.bounds[1] / 2 + radius * math.sin(angle)
                z = layer_z
                
                positions[neuron_id] = SpatialPosition(x=x, y=y, z=z)
                neuron_id += 1
                
        self.neuron_positions.update(positions)
        return positions
        
    def get_distance_matrix(self, neuron_ids: List[int]) -> np.ndarray:
        """
        Calculate distance matrix between specified neurons.
        
        Args:
            neuron_ids: List of neuron IDs
            
        Returns:
            Distance matrix (symmetric)
        """
        n = len(neuron_ids)
        distance_matrix = np.zeros((n, n))
        
        for i, id1 in enumerate(neuron_ids):
            for j, id2 in enumerate(neuron_ids):
                if i != j and id1 in self.neuron_positions and id2 in self.neuron_positions:
                    distance_matrix[i, j] = self.neuron_positions[id1].distance_to(
                        self.neuron_positions[id2]
                    )
                    
        return distance_matrix


class DistanceDependentConnectivity:
    """
    Distance-dependent connectivity builder implementing realistic connection patterns.
    
    Features:
    - Exponential decay connection probability with distance
    - Configurable spatial scales for different connection types
    - Support for different neuron types with specific connection rules
    """
    
    def __init__(self, spatial_layout: SpatialNetworkLayout):
        """
        Initialize distance-dependent connectivity builder.
        
        Args:
            spatial_layout: Spatial layout of neurons
        """
        self.spatial_layout = spatial_layout
        
        # Default connection parameters
        self.connection_params = {
            # Excitatory to excitatory
            'E_to_E': {
                'base_probability': 0.1,
                'spatial_scale': 50.0,  # μm
                'max_distance': 200.0,
                'weight_mean': 1.0,
                'weight_std': 0.2
            },
            # Excitatory to inhibitory
            'E_to_I': {
                'base_probability': 0.2,
                'spatial_scale': 30.0,
                'max_distance': 100.0,
                'weight_mean': 1.5,
                'weight_std': 0.3
            },
            # Inhibitory to excitatory
            'I_to_E': {
                'base_probability': 0.3,
                'spatial_scale': 40.0,
                'max_distance': 150.0,
                'weight_mean': -2.0,  # Inhibitory
                'weight_std': 0.4
            },
            # Inhibitory to inhibitory
            'I_to_I': {
                'base_probability': 0.15,
                'spatial_scale': 25.0,
                'max_distance': 80.0,
                'weight_mean': -1.0,  # Inhibitory
                'weight_std': 0.2
            }
        }
        
    def set_connection_parameters(self, connection_type: str, **params):
        """
        Set connection parameters for a specific connection type.
        
        Args:
            connection_type: Type of connection ('E_to_E', 'E_to_I', 'I_to_E', 'I_to_I')
            **params: Connection parameters to update
        """
        if connection_type in self.connection_params:
            self.connection_params[connection_type].update(params)
        else:
            self.connection_params[connection_type] = params
            
    def compute_connection_probability(
        self, 
        pre_neuron_id: int, 
        post_neuron_id: int,
        connection_type: str
    ) -> float:
        """
        Compute connection probability based on distance and connection type.
        
        Args:
            pre_neuron_id: Presynaptic neuron ID
            post_neuron_id: Postsynaptic neuron ID
            connection_type: Type of connection
            
        Returns:
            Connection probability (0-1)
        """
        if (pre_neuron_id not in self.spatial_layout.neuron_positions or 
            post_neuron_id not in self.spatial_layout.neuron_positions):
            return 0.0
            
        # Get distance between neurons
        distance = self.spatial_layout.neuron_positions[pre_neuron_id].distance_to(
            self.spatial_layout.neuron_positions[post_neuron_id]
        )
        
        # Get connection parameters
        params = self.connection_params.get(connection_type, self.connection_params['E_to_E'])
        
        # Check maximum distance
        if distance > params['max_distance']:
            return 0.0
            
        # Exponential decay with distance
        base_prob = params['base_probability']
        spatial_scale = params['spatial_scale']
        
        probability = base_prob * np.exp(-distance / spatial_scale)
        
        return min(probability, 1.0)
        
    def generate_connection_matrix(
        self,
        pre_neuron_ids: List[int],
        post_neuron_ids: List[int],
        pre_neuron_types: List[NeuronType],
        post_neuron_types: List[NeuronType]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate connection matrix and weight matrix based on distance and neuron types.
        
        Args:
            pre_neuron_ids: List of presynaptic neuron IDs
            post_neuron_ids: List of postsynaptic neuron IDs
            pre_neuron_types: Types of presynaptic neurons
            post_neuron_types: Types of postsynaptic neurons
            
        Returns:
            Tuple of (connection_matrix, weight_matrix)
        """
        n_pre = len(pre_neuron_ids)
        n_post = len(post_neuron_ids)
        
        connection_matrix = np.zeros((n_post, n_pre), dtype=bool)
        weight_matrix = np.zeros((n_post, n_pre), dtype=float)
        
        for i, pre_id in enumerate(pre_neuron_ids):
            for j, post_id in enumerate(post_neuron_ids):
                if pre_id == post_id:
                    continue  # No self-connections
                    
                # Determine connection type
                pre_type = pre_neuron_types[i]
                post_type = post_neuron_types[j]
                
                if self._is_excitatory(pre_type):
                    if self._is_excitatory(post_type):
                        conn_type = 'E_to_E'
                    else:
                        conn_type = 'E_to_I'
                else:
                    if self._is_excitatory(post_type):
                        conn_type = 'I_to_E'
                    else:
                        conn_type = 'I_to_I'
                        
                # Compute connection probability
                prob = self.compute_connection_probability(pre_id, post_id, conn_type)
                
                # Stochastic connection
                if np.random.random() < prob:
                    connection_matrix[j, i] = True
                    
                    # Generate weight
                    params = self.connection_params[conn_type]
                    weight = np.random.normal(params['weight_mean'], params['weight_std'])
                    
                    # Ensure inhibitory weights are negative
                    if not self._is_excitatory(pre_type):
                        weight = -abs(weight)
                    else:
                        weight = abs(weight)
                        
                    weight_matrix[j, i] = weight
                    
        return connection_matrix, weight_matrix
        
    def _is_excitatory(self, neuron_type: NeuronType) -> bool:
        """Check if neuron type is excitatory."""
        return neuron_type in [NeuronType.EXCITATORY_PYRAMIDAL, NeuronType.EXCITATORY_SPINY_STELLATE]


class ExcitatoryInhibitoryBalance:
    """
    System for implementing realistic excitatory/inhibitory balance in neural networks.
    
    Features:
    - 80% excitatory, 20% inhibitory neuron populations
    - Realistic connection probabilities for E→E, E→I, I→E, I→I
    - Different inhibitory neuron types (basket, chandelier, Martinotti cells)
    - Proper synaptic strength ratios
    """
    
    def __init__(self):
        """Initialize E/I balance system."""
        # Standard cortical E/I ratios
        self.excitatory_fraction = 0.8
        self.inhibitory_fraction = 0.2
        
        # Realistic connection probabilities (based on cortical data)
        self.connection_probabilities = {
            'E_to_E': 0.05,   # Excitatory to excitatory (sparse)
            'E_to_I': 0.15,   # Excitatory to inhibitory (higher)
            'I_to_E': 0.20,   # Inhibitory to excitatory (strong)
            'I_to_I': 0.10    # Inhibitory to inhibitory (moderate)
        }
        
        # Synaptic strength ratios (relative to E→E = 1.0)
        self.synaptic_strengths = {
            'E_to_E': 1.0,    # Baseline excitatory strength
            'E_to_I': 1.2,    # Slightly stronger to drive inhibition
            'I_to_E': -3.0,   # Strong inhibition (negative)
            'I_to_I': -1.5    # Moderate inhibitory-inhibitory
        }
        
        # Inhibitory neuron type distributions
        self.inhibitory_types = {
            'basket': 0.60,      # Basket cells (perisomatic inhibition)
            'chandelier': 0.25,  # Chandelier cells (axon initial segment)
            'martinotti': 0.15   # Martinotti cells (dendritic inhibition)
        }
        
    def calculate_population_sizes(self, total_neurons: int) -> Dict[str, int]:
        """
        Calculate population sizes maintaining E/I balance.
        
        Args:
            total_neurons: Total number of neurons
            
        Returns:
            Dictionary with population sizes
        """
        n_excitatory = int(total_neurons * self.excitatory_fraction)
        n_inhibitory = total_neurons - n_excitatory
        
        # Distribute inhibitory neurons by type
        n_basket = int(n_inhibitory * self.inhibitory_types['basket'])
        n_chandelier = int(n_inhibitory * self.inhibitory_types['chandelier'])
        n_martinotti = n_inhibitory - n_basket - n_chandelier
        
        return {
            'excitatory': n_excitatory,
            'inhibitory_total': n_inhibitory,
            'basket': n_basket,
            'chandelier': n_chandelier,
            'martinotti': n_martinotti
        }
        
    def get_connection_parameters(self, pre_type: str, post_type: str) -> Dict[str, float]:
        """
        Get connection parameters for specific pre/post neuron types.
        
        Args:
            pre_type: Presynaptic neuron type ('excitatory' or 'inhibitory')
            post_type: Postsynaptic neuron type ('excitatory' or 'inhibitory')
            
        Returns:
            Connection parameters (probability, strength, etc.)
        """
        # Determine connection type
        if pre_type == 'excitatory' and post_type == 'excitatory':
            conn_type = 'E_to_E'
        elif pre_type == 'excitatory' and post_type == 'inhibitory':
            conn_type = 'E_to_I'
        elif pre_type == 'inhibitory' and post_type == 'excitatory':
            conn_type = 'I_to_E'
        else:  # inhibitory to inhibitory
            conn_type = 'I_to_I'
            
        return {
            'connection_probability': self.connection_probabilities[conn_type],
            'synaptic_strength': self.synaptic_strengths[conn_type],
            'connection_type': conn_type
        }
        
    def validate_ei_balance(self, network_stats: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate that network maintains proper E/I balance.
        
        Args:
            network_stats: Network statistics
            
        Returns:
            Balance metrics
        """
        total_neurons = network_stats.get('total_neurons', 0)
        if total_neurons == 0:
            return {'error': 'No neurons found'}
            
        # Calculate actual E/I ratios
        excitatory_count = 0
        inhibitory_count = 0
        
        for layer_name, layer_info in network_stats.get('layers', {}).items():
            if 'excitatory' in layer_name.lower():
                excitatory_count += layer_info.get('size', 0)
            elif ('inhibitory' in layer_name.lower() or 
                  'basket' in layer_name.lower() or 
                  'chandelier' in layer_name.lower() or 
                  'martinotti' in layer_name.lower() or
                  'cells' in layer_name.lower()):
                inhibitory_count += layer_info.get('size', 0)
                
        actual_e_fraction = excitatory_count / total_neurons if total_neurons > 0 else 0
        actual_i_fraction = inhibitory_count / total_neurons if total_neurons > 0 else 0
        
        # Calculate balance metrics
        e_balance_error = abs(actual_e_fraction - self.excitatory_fraction)
        i_balance_error = abs(actual_i_fraction - self.inhibitory_fraction)
        
        return {
            'actual_excitatory_fraction': actual_e_fraction,
            'actual_inhibitory_fraction': actual_i_fraction,
            'target_excitatory_fraction': self.excitatory_fraction,
            'target_inhibitory_fraction': self.inhibitory_fraction,
            'excitatory_balance_error': e_balance_error,
            'inhibitory_balance_error': i_balance_error,
            'balance_quality': 1.0 - (e_balance_error + i_balance_error),  # 1.0 = perfect
            'is_balanced': (e_balance_error < 0.05 and i_balance_error < 0.05)
        }


class BrainInspiredNetworkBuilder:
    """
    Builder for creating brain-inspired network architectures.
    
    Combines spatial layouts, distance-dependent connectivity, and realistic
    neuron type distributions to create biologically plausible networks.
    """
    
    def __init__(self):
        """Initialize brain-inspired network builder."""
        self.spatial_layout = None
        self.connectivity_builder = None
        self.network = None
        self.ei_balance = ExcitatoryInhibitoryBalance()
        
    def create_cortical_network(
        self,
        total_neurons: int = 1000,
        excitatory_fraction: Optional[float] = None,
        spatial_bounds: Tuple[float, float, float] = (200.0, 200.0, 50.0),
        layout_type: str = "clustered",
        use_detailed_inhibitory_types: bool = False,
        **layout_params
    ) -> NeuromorphicNetwork:
        """
        Create a cortical-like network with realistic E/I balance.
        
        Args:
            total_neurons: Total number of neurons
            excitatory_fraction: Fraction of excitatory neurons (uses default 0.8 if None)
            spatial_bounds: (width, height, depth) of spatial area
            layout_type: Type of spatial layout ('grid', 'random', 'clustered', 'cortical')
            use_detailed_inhibitory_types: Whether to create separate layers for inhibitory types
            **layout_params: Additional parameters for spatial layout
            
        Returns:
            Configured neuromorphic network with proper E/I balance
        """
        # Use default E/I balance if not specified
        if excitatory_fraction is not None:
            self.ei_balance.excitatory_fraction = excitatory_fraction
            self.ei_balance.inhibitory_fraction = 1.0 - excitatory_fraction
            
        # Calculate population sizes with proper E/I balance
        population_sizes = self.ei_balance.calculate_population_sizes(total_neurons)
        
        # Create spatial layout
        self.spatial_layout = SpatialNetworkLayout(dimensions=3, bounds=spatial_bounds)
        
        # Position neurons based on layout type
        if layout_type == "grid":
            positions = self.spatial_layout.create_grid_layout(total_neurons, **layout_params)
        elif layout_type == "random":
            positions = self.spatial_layout.create_random_layout(total_neurons)
        elif layout_type == "clustered":
            positions = self.spatial_layout.create_clustered_layout(total_neurons, **layout_params)
        elif layout_type == "cortical":
            positions = self.spatial_layout.create_cortical_column_layout(total_neurons, **layout_params)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
            
        # Create connectivity builder with E/I-aware parameters
        self.connectivity_builder = DistanceDependentConnectivity(self.spatial_layout)
        self._configure_ei_connectivity_parameters()
        
        # Create network
        self.network = NeuromorphicNetwork()
        
        # Create layers based on E/I balance
        self._create_ei_balanced_layers(population_sizes, use_detailed_inhibitory_types)
        
        # Create connections with proper E/I balance
        self._create_ei_balanced_connections(population_sizes, use_detailed_inhibitory_types)
        
        return self.network
        
    def _configure_ei_connectivity_parameters(self):
        """Configure connectivity parameters for proper E/I balance."""
        # Update connection parameters based on E/I balance requirements
        ei_params = {
            'E_to_E': {
                'base_probability': self.ei_balance.connection_probabilities['E_to_E'],
                'spatial_scale': 80.0,  # μm
                'max_distance': 300.0,
                'weight_mean': abs(self.ei_balance.synaptic_strengths['E_to_E']),
                'weight_std': 0.2
            },
            'E_to_I': {
                'base_probability': self.ei_balance.connection_probabilities['E_to_I'],
                'spatial_scale': 60.0,  # Shorter range for E→I
                'max_distance': 200.0,
                'weight_mean': abs(self.ei_balance.synaptic_strengths['E_to_I']),
                'weight_std': 0.3
            },
            'I_to_E': {
                'base_probability': self.ei_balance.connection_probabilities['I_to_E'],
                'spatial_scale': 100.0,  # Longer range for inhibition
                'max_distance': 400.0,
                'weight_mean': abs(self.ei_balance.synaptic_strengths['I_to_E']),
                'weight_std': 0.5
            },
            'I_to_I': {
                'base_probability': self.ei_balance.connection_probabilities['I_to_I'],
                'spatial_scale': 50.0,  # Short range for I→I
                'max_distance': 150.0,
                'weight_mean': abs(self.ei_balance.synaptic_strengths['I_to_I']),
                'weight_std': 0.3
            }
        }
        
        # Apply parameters to connectivity builder
        for conn_type, params in ei_params.items():
            self.connectivity_builder.set_connection_parameters(conn_type, **params)
            
    def _create_ei_balanced_layers(self, population_sizes: Dict[str, int], detailed_inhibitory: bool):
        """Create network layers with proper E/I balance."""
        # Add excitatory layer
        self.network.add_layer(
            "excitatory", 
            population_sizes['excitatory'], 
            neuron_type="adex"
        )
        
        if detailed_inhibitory:
            # Create separate layers for different inhibitory types
            if population_sizes['basket'] > 0:
                self.network.add_layer(
                    "basket_cells", 
                    population_sizes['basket'], 
                    neuron_type="lif"
                )
                
            if population_sizes['chandelier'] > 0:
                self.network.add_layer(
                    "chandelier_cells", 
                    population_sizes['chandelier'], 
                    neuron_type="lif"
                )
                
            if population_sizes['martinotti'] > 0:
                self.network.add_layer(
                    "martinotti_cells", 
                    population_sizes['martinotti'], 
                    neuron_type="lif"
                )
        else:
            # Single inhibitory layer
            self.network.add_layer(
                "inhibitory", 
                population_sizes['inhibitory_total'], 
                neuron_type="lif"
            )
            
    def _create_ei_balanced_connections(self, population_sizes: Dict[str, int], detailed_inhibitory: bool):
        """Create connections with proper E/I balance."""
        if detailed_inhibitory:
            # Detailed connections with specific inhibitory types
            inhibitory_layers = []
            if population_sizes['basket'] > 0:
                inhibitory_layers.append('basket_cells')
            if population_sizes['chandelier'] > 0:
                inhibitory_layers.append('chandelier_cells')
            if population_sizes['martinotti'] > 0:
                inhibitory_layers.append('martinotti_cells')
                
            # E → E connections
            self._add_ei_connection("excitatory", "excitatory", "E_to_E")
            
            # E → I connections (to all inhibitory types)
            for inh_layer in inhibitory_layers:
                self._add_ei_connection("excitatory", inh_layer, "E_to_I")
                
            # I → E connections (from all inhibitory types)
            for inh_layer in inhibitory_layers:
                self._add_ei_connection(inh_layer, "excitatory", "I_to_E")
                
            # I → I connections (between inhibitory types)
            for i, inh_layer1 in enumerate(inhibitory_layers):
                for j, inh_layer2 in enumerate(inhibitory_layers):
                    if i != j:  # Different inhibitory types
                        self._add_ei_connection(inh_layer1, inh_layer2, "I_to_I")
        else:
            # Simple E/I connections
            self._add_ei_connection("excitatory", "excitatory", "E_to_E")
            self._add_ei_connection("excitatory", "inhibitory", "E_to_I")
            self._add_ei_connection("inhibitory", "excitatory", "I_to_E")
            self._add_ei_connection("inhibitory", "inhibitory", "I_to_I")
            
    def _add_ei_connection(self, pre_layer: str, post_layer: str, connection_type: str):
        """Add connection with E/I balance parameters."""
        params = self.ei_balance.get_connection_parameters(
            'excitatory' if 'excitatory' in pre_layer else 'inhibitory',
            'excitatory' if 'excitatory' in post_layer else 'inhibitory'
        )
        
        # Determine synapse type based on connection
        if connection_type == 'I_to_E' or connection_type == 'I_to_I':
            synapse_type = "stdp"  # Could use inhibitory-specific synapse type
        else:
            synapse_type = "stdp"
            
        self.network.connect_layers(
            pre_layer, 
            post_layer,
            synapse_type=synapse_type,
            connection_probability=params['connection_probability'],
            weight=abs(params['synaptic_strength'])  # Network handles sign
        )
        
    def _create_distance_dependent_connections(
        self,
        excitatory_ids: List[int],
        inhibitory_ids: List[int], 
        neuron_types: List[NeuronType]
    ):
        """Create connections based on distance and neuron types."""
        
        # E -> E connections
        exc_types = [neuron_types[i] for i in excitatory_ids]
        conn_matrix, weight_matrix = self.connectivity_builder.generate_connection_matrix(
            excitatory_ids, excitatory_ids, exc_types, exc_types
        )
        self._add_connection_from_matrix("excitatory", "excitatory", conn_matrix, weight_matrix)
        
        # E -> I connections
        inh_types = [neuron_types[i] for i in inhibitory_ids]
        conn_matrix, weight_matrix = self.connectivity_builder.generate_connection_matrix(
            excitatory_ids, inhibitory_ids, exc_types, inh_types
        )
        self._add_connection_from_matrix("excitatory", "inhibitory", conn_matrix, weight_matrix)
        
        # I -> E connections
        conn_matrix, weight_matrix = self.connectivity_builder.generate_connection_matrix(
            inhibitory_ids, excitatory_ids, inh_types, exc_types
        )
        self._add_connection_from_matrix("inhibitory", "excitatory", conn_matrix, weight_matrix)
        
        # I -> I connections
        conn_matrix, weight_matrix = self.connectivity_builder.generate_connection_matrix(
            inhibitory_ids, inhibitory_ids, inh_types, inh_types
        )
        self._add_connection_from_matrix("inhibitory", "inhibitory", conn_matrix, weight_matrix)
        
    def _add_connection_from_matrix(
        self, 
        pre_layer: str, 
        post_layer: str, 
        conn_matrix: np.ndarray, 
        weight_matrix: np.ndarray
    ):
        """Add connection to network from connectivity matrix."""
        # Calculate effective connection probability
        total_possible = conn_matrix.size
        actual_connections = np.sum(conn_matrix)
        effective_prob = actual_connections / total_possible if total_possible > 0 else 0.0
        
        # Add connection to network
        self.network.connect_layers(
            pre_layer, 
            post_layer,
            synapse_type="stdp",
            connection_probability=effective_prob,
            weight=np.mean(weight_matrix[weight_matrix != 0]) if np.any(weight_matrix != 0) else 1.0
        )
        
    def _is_excitatory_type(self, neuron_type: NeuronType) -> bool:
        """Check if neuron type is excitatory."""
        return neuron_type in [NeuronType.EXCITATORY_PYRAMIDAL, NeuronType.EXCITATORY_SPINY_STELLATE]
        
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the created network."""
        if self.network is None:
            return {}
            
        # Basic network statistics
        network_info = self.network.get_network_info()
        
        stats = {
            'total_neurons': network_info['total_neurons'],
            'total_connections': network_info['total_synapses'],
            'layers': network_info['layers'],
            'connections': network_info['connections']
        }
        
        # Add spatial information if available
        if self.spatial_layout is not None:
            stats.update({
                'spatial_bounds': self.spatial_layout.bounds,
                'modules': len(self.spatial_layout.modules),
                'spatial_dimensions': self.spatial_layout.dimensions
            })
            
        # Add E/I balance validation
        ei_balance_metrics = self.ei_balance.validate_ei_balance(stats)
        stats['ei_balance'] = ei_balance_metrics
        
        # Connection type statistics
        connection_stats = {}
        for (pre, post), conn in self.network.connections.items():
            if conn.synapse_population:
                conn_name = f'{pre}_to_{post}'
                connection_stats[conn_name] = {
                    'num_synapses': len(conn.synapse_population.synapses),
                    'connection_probability': conn.connection_probability,
                    'synapse_type': conn.synapse_type
                }
                
        stats['connection_details'] = connection_stats
        
        return stats
        
    def validate_network_balance(self) -> Dict[str, Any]:
        """
        Validate that the network maintains proper E/I balance and connectivity.
        
        Returns:
            Validation results with recommendations
        """
        if self.network is None:
            return {'error': 'No network created'}
            
        stats = self.get_network_statistics()
        validation = {
            'ei_balance': stats.get('ei_balance', {}),
            'recommendations': [],
            'warnings': [],
            'overall_quality': 'unknown'
        }
        
        # Check E/I balance
        ei_metrics = stats.get('ei_balance', {})
        if ei_metrics.get('is_balanced', False):
            validation['recommendations'].append("E/I balance is within acceptable range")
        else:
            e_error = ei_metrics.get('excitatory_balance_error', 0)
            i_error = ei_metrics.get('inhibitory_balance_error', 0)
            if e_error > 0.1 or i_error > 0.1:
                validation['warnings'].append(f"E/I balance significantly off target (E error: {e_error:.3f}, I error: {i_error:.3f})")
            else:
                validation['warnings'].append("E/I balance slightly off target but acceptable")
                
        # Check connectivity
        total_connections = stats.get('total_connections', 0)
        total_neurons = stats.get('total_neurons', 1)
        connection_density = total_connections / (total_neurons ** 2) if total_neurons > 0 else 0
        
        if 0.01 <= connection_density <= 0.1:  # Realistic cortical density
            validation['recommendations'].append(f"Connection density ({connection_density:.4f}) is realistic")
        elif connection_density < 0.01:
            validation['warnings'].append(f"Connection density ({connection_density:.4f}) may be too sparse")
        else:
            validation['warnings'].append(f"Connection density ({connection_density:.4f}) may be too dense")
            
        # Overall quality assessment
        balance_quality = ei_metrics.get('balance_quality', 0)
        if balance_quality > 0.9 and 0.01 <= connection_density <= 0.1:
            validation['overall_quality'] = 'excellent'
        elif balance_quality > 0.8 and 0.005 <= connection_density <= 0.15:
            validation['overall_quality'] = 'good'
        elif balance_quality > 0.6:
            validation['overall_quality'] = 'acceptable'
        else:
            validation['overall_quality'] = 'poor'
            
        return validation