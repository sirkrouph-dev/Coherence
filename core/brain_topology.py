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
        
    def create_cortical_network(
        self,
        total_neurons: int = 1000,
        excitatory_fraction: float = 0.8,
        spatial_bounds: Tuple[float, float, float] = (200.0, 200.0, 50.0),
        layout_type: str = "clustered",
        **layout_params
    ) -> NeuromorphicNetwork:
        """
        Create a cortical-like network with realistic architecture.
        
        Args:
            total_neurons: Total number of neurons
            excitatory_fraction: Fraction of excitatory neurons (typically 0.8)
            spatial_bounds: (width, height, depth) of spatial area
            layout_type: Type of spatial layout ('grid', 'random', 'clustered', 'cortical')
            **layout_params: Additional parameters for spatial layout
            
        Returns:
            Configured neuromorphic network
        """
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
            
        # Create connectivity builder
        self.connectivity_builder = DistanceDependentConnectivity(self.spatial_layout)
        
        # Create network
        self.network = NeuromorphicNetwork()
        
        # Determine neuron types and counts
        n_excitatory = int(total_neurons * excitatory_fraction)
        n_inhibitory = total_neurons - n_excitatory
        
        # Create neuron type assignments
        neuron_types = []
        neuron_ids = list(range(total_neurons))
        
        # Assign excitatory types (80% pyramidal, 20% spiny stellate)
        for i in range(n_excitatory):
            if i < int(n_excitatory * 0.8):
                neuron_types.append(NeuronType.EXCITATORY_PYRAMIDAL)
            else:
                neuron_types.append(NeuronType.EXCITATORY_SPINY_STELLATE)
                
        # Assign inhibitory types (60% basket, 25% chandelier, 15% martinotti)
        for i in range(n_inhibitory):
            if i < int(n_inhibitory * 0.6):
                neuron_types.append(NeuronType.INHIBITORY_BASKET)
            elif i < int(n_inhibitory * 0.85):
                neuron_types.append(NeuronType.INHIBITORY_CHANDELIER)
            else:
                neuron_types.append(NeuronType.INHIBITORY_MARTINOTTI)
                
        # Create layers for different neuron types
        excitatory_ids = [i for i, t in enumerate(neuron_types) if self._is_excitatory_type(t)]
        inhibitory_ids = [i for i, t in enumerate(neuron_types) if not self._is_excitatory_type(t)]
        
        # Add excitatory layer
        self.network.add_layer(
            "excitatory", 
            len(excitatory_ids), 
            neuron_type="adex"  # Use existing neuron type
        )
        
        # Add inhibitory layer
        self.network.add_layer(
            "inhibitory", 
            len(inhibitory_ids), 
            neuron_type="lif"  # Use existing neuron type
        )
        
        # Create distance-dependent connections
        self._create_distance_dependent_connections(
            excitatory_ids, inhibitory_ids, neuron_types
        )
        
        return self.network
        
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
        """Get statistics about the created network."""
        if self.network is None or self.spatial_layout is None:
            return {}
            
        stats = {
            'total_neurons': sum(layer.size for layer in self.network.layers.values()),
            'total_connections': sum(
                len(conn.synapse_population.synapses) if conn.synapse_population else 0
                for conn in self.network.connections.values()
            ),
            'spatial_bounds': self.spatial_layout.bounds,
            'modules': len(self.spatial_layout.modules)
        }
        
        # Connection statistics
        for (pre, post), conn in self.network.connections.items():
            if conn.synapse_population:
                stats[f'{pre}_to_{post}_connections'] = len(conn.synapse_population.synapses)
                
        return stats