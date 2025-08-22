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
        
        # Inhibitory neuron properties
        self.inhibitory_properties = {
            'basket': {
                'target_location': 'soma',  # Perisomatic inhibition
                'time_constant': 10.0,      # Fast spiking
                'threshold': -50.0,         # Low threshold
                'adaptation': 0.0,          # Minimal adaptation
                'connection_range': 50.0    # Local connections
            },
            'chandelier': {
                'target_location': 'axon_initial_segment',  # AIS inhibition
                'time_constant': 8.0,       # Very fast
                'threshold': -48.0,         # Very low threshold
                'adaptation': 0.0,          # No adaptation
                'connection_range': 30.0    # Very local
            },
            'martinotti': {
                'target_location': 'dendrites',  # Dendritic inhibition
                'time_constant': 20.0,      # Slower dynamics
                'threshold': -55.0,         # Higher threshold
                'adaptation': 0.2,          # Some adaptation
                'connection_range': 100.0   # Longer range connections
            }
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
        
    def get_inhibitory_neuron_parameters(self, inhibitory_type: str) -> Dict[str, Any]:
        """
        Get parameters for specific inhibitory neuron types.
        
        Args:
            inhibitory_type: Type of inhibitory neuron ('basket', 'chandelier', 'martinotti')
            
        Returns:
            Neuron parameters for the specified type
        """
        if inhibitory_type not in self.inhibitory_properties:
            raise ValueError(f"Unknown inhibitory type: {inhibitory_type}")
            
        base_params = self.inhibitory_properties[inhibitory_type].copy()
        
        # Add common inhibitory neuron parameters (mapped to LIF parameter names)
        base_params.update({
            'neuron_type': 'lif',  # Use LIF for inhibitory neurons
            'tau_m': base_params['time_constant'],  # Map time_constant to tau_m
            'v_rest': -70.0,       # Resting potential
            'v_thresh': base_params['threshold'],   # Map threshold to v_thresh
            'v_reset': -70.0,      # Reset potential
            'refractory_period': 2.0
        })
        
        # Remove the original parameter names that don't match LIF constructor
        base_params.pop('time_constant', None)
        base_params.pop('threshold', None)
        
        return base_params
        
    def create_inhibitory_populations(
        self, 
        total_inhibitory: int,
        spatial_layout: Optional['SpatialNetworkLayout'] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Create populations of different inhibitory neuron types.
        
        Args:
            total_inhibitory: Total number of inhibitory neurons
            spatial_layout: Optional spatial layout for positioning
            
        Returns:
            Dictionary with inhibitory populations
        """
        populations = {}
        
        # Calculate population sizes
        pop_sizes = {
            'basket': int(total_inhibitory * self.inhibitory_types['basket']),
            'chandelier': int(total_inhibitory * self.inhibitory_types['chandelier']),
            'martinotti': total_inhibitory - int(total_inhibitory * self.inhibitory_types['basket']) - int(total_inhibitory * self.inhibitory_types['chandelier'])
        }
        
        neuron_id = 0
        for inh_type, size in pop_sizes.items():
            if size > 0:
                # Get neuron parameters for this type
                params = self.get_inhibitory_neuron_parameters(inh_type)
                
                # Create neuron IDs
                neuron_ids = list(range(neuron_id, neuron_id + size))
                neuron_id += size
                
                populations[inh_type] = {
                    'size': size,
                    'neuron_ids': neuron_ids,
                    'parameters': params,
                    'type': inh_type
                }
                
        return populations
        
    def calculate_ei_activity_balance(
        self, 
        excitatory_activity: float, 
        inhibitory_activity: float,
        target_balance: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate E/I activity balance and suggest adjustments.
        
        Args:
            excitatory_activity: Average excitatory firing rate
            inhibitory_activity: Average inhibitory firing rate
            target_balance: Target E/I activity ratio
            
        Returns:
            Balance analysis and adjustment suggestions
        """
        if inhibitory_activity == 0:
            ei_ratio = float('inf') if excitatory_activity > 0 else 0
        else:
            ei_ratio = excitatory_activity / inhibitory_activity
            
        balance_error = abs(ei_ratio - target_balance)
        
        # Suggest adjustments
        if ei_ratio > target_balance * 1.2:  # Too much excitation
            suggestion = 'increase_inhibition'
            adjustment_factor = ei_ratio / target_balance
        elif ei_ratio < target_balance * 0.8:  # Too much inhibition
            suggestion = 'increase_excitation'
            adjustment_factor = target_balance / ei_ratio
        else:
            suggestion = 'balanced'
            adjustment_factor = 1.0
            
        return {
            'ei_ratio': ei_ratio,
            'target_ratio': target_balance,
            'balance_error': balance_error,
            'balance_quality': max(0, 1.0 - balance_error / target_balance),
            'suggestion': suggestion,
            'adjustment_factor': adjustment_factor,
            'is_balanced': balance_error < target_balance * 0.2
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


class ModularNetworkArchitecture:
    """
    System for creating modular network architectures with hierarchical organization.
    
    Features:
    - Dense intra-module connectivity (high clustering)
    - Sparse inter-module connections (long-range)
    - Hierarchical module organization
    - Small-world network properties
    """
    
    def __init__(self):
        """Initialize modular network architecture system."""
        # Modular connectivity parameters
        self.intra_module_probability = 0.3  # High connectivity within modules
        self.inter_module_probability = 0.02  # Sparse connectivity between modules
        self.hub_probability = 0.1  # Probability of hub connections
        
        # Small-world parameters
        self.rewiring_probability = 0.1  # For small-world rewiring
        self.clustering_target = 0.6  # Target clustering coefficient
        self.path_length_target = 3.0  # Target average path length
        
        # Hierarchical parameters
        self.hierarchy_levels = 3  # Number of hierarchical levels
        self.modules_per_level = [4, 2, 1]  # Modules at each level
        
        # Storage for created modules and connections
        self.modules = {}
        self.inter_module_connections = {}
        self.hierarchy_levels_map = {}
        
    def create_modular_layout(
        self, 
        total_neurons: int, 
        num_modules: int = 4,
        module_separation: float = 100.0,
        module_radius: float = 30.0
    ) -> Tuple[Dict[int, SpatialPosition], Dict[str, NetworkModule]]:
        """
        Create spatial layout with modular organization.
        
        Args:
            total_neurons: Total number of neurons
            num_modules: Number of modules to create
            module_separation: Distance between module centers
            module_radius: Radius of each module
            
        Returns:
            Tuple of (neuron_positions, modules)
        """
        neurons_per_module = total_neurons // num_modules
        positions = {}
        modules = {}
        
        # Create module centers in a grid or circular arrangement
        if num_modules <= 4:
            # Circular arrangement for small numbers
            angles = np.linspace(0, 2*np.pi, num_modules, endpoint=False)
            centers = [
                SpatialPosition(
                    x=module_separation * np.cos(angle),
                    y=module_separation * np.sin(angle),
                    z=0.0
                )
                for angle in angles
            ]
        else:
            # Grid arrangement for larger numbers
            grid_size = int(np.ceil(np.sqrt(num_modules)))
            centers = []
            for i in range(num_modules):
                row = i // grid_size
                col = i % grid_size
                centers.append(SpatialPosition(
                    x=col * module_separation,
                    y=row * module_separation,
                    z=0.0
                ))
        
        # Distribute neurons within modules
        neuron_id = 0
        for module_idx, center in enumerate(centers):
            if module_idx == num_modules - 1:
                # Last module gets remaining neurons
                module_neurons = total_neurons - neuron_id
            else:
                module_neurons = neurons_per_module
                
            module_neuron_ids = []
            
            # Position neurons within module using Gaussian distribution
            for _ in range(module_neurons):
                # Gaussian distribution around module center
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.exponential(module_radius / 3)  # Exponential for realistic density
                radius = min(radius, module_radius)  # Cap at module radius
                
                x = center.x + radius * np.cos(angle)
                y = center.y + radius * np.sin(angle)
                z = center.z + np.random.normal(0, 5.0)  # Small z variation
                
                positions[neuron_id] = SpatialPosition(x=x, y=y, z=z)
                module_neuron_ids.append(neuron_id)
                neuron_id += 1
                
            # Create module
            module = NetworkModule(
                name=f"module_{module_idx}",
                center_position=center,
                radius=module_radius,
                neuron_ids=module_neuron_ids,
                excitatory_fraction=0.8
            )
            modules[module.name] = module
            
        return positions, modules
        
    def compute_modular_connectivity(
        self,
        modules: Dict[str, NetworkModule],
        neuron_positions: Dict[int, SpatialPosition]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute modular connectivity matrix with intra/inter-module structure.
        
        Args:
            modules: Dictionary of network modules
            neuron_positions: Neuron position mapping
            
        Returns:
            Dictionary mapping (pre_id, post_id) to connection strength
        """
        connections = {}
        
        # Create neuron-to-module mapping
        neuron_to_module = {}
        for module_name, module in modules.items():
            for neuron_id in module.neuron_ids:
                neuron_to_module[neuron_id] = module_name
                
        # Generate connections
        all_neuron_ids = list(neuron_positions.keys())
        
        for pre_id in all_neuron_ids:
            for post_id in all_neuron_ids:
                if pre_id == post_id:
                    continue
                    
                pre_module = neuron_to_module.get(pre_id)
                post_module = neuron_to_module.get(post_id)
                
                if pre_module == post_module:
                    # Intra-module connection
                    base_prob = self.intra_module_probability
                else:
                    # Inter-module connection
                    base_prob = self.inter_module_probability
                    
                # Distance-dependent modulation
                distance = neuron_positions[pre_id].distance_to(neuron_positions[post_id])
                distance_factor = np.exp(-distance / 50.0)  # 50μm spatial scale
                
                # Final connection probability
                connection_prob = base_prob * distance_factor
                
                # Stochastic connection
                if np.random.random() < connection_prob:
                    # Weight based on connection type
                    if pre_module == post_module:
                        weight = np.random.normal(1.0, 0.2)  # Strong intra-module
                    else:
                        weight = np.random.normal(0.5, 0.1)  # Weaker inter-module
                        
                    connections[(pre_id, post_id)] = max(0.1, weight)
                    
        return connections
        
    def add_small_world_rewiring(
        self,
        connections: Dict[Tuple[int, int], float],
        rewiring_prob: Optional[float] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Add small-world rewiring to create shortcuts between distant modules.
        
        Args:
            connections: Existing connection dictionary
            rewiring_prob: Probability of rewiring each connection
            
        Returns:
            Updated connections with small-world properties
        """
        if rewiring_prob is None:
            rewiring_prob = self.rewiring_probability
            
        rewired_connections = connections.copy()
        all_neuron_ids = list(set([pre for pre, post in connections.keys()] + 
                                 [post for pre, post in connections.keys()]))
        
        # Rewire some connections to create shortcuts
        connections_to_rewire = list(connections.keys())
        np.random.shuffle(connections_to_rewire)
        
        num_to_rewire = int(len(connections_to_rewire) * rewiring_prob)
        
        for i in range(num_to_rewire):
            old_connection = connections_to_rewire[i]
            pre_id, old_post_id = old_connection
            old_weight = connections[old_connection]
            
            # Remove old connection
            if old_connection in rewired_connections:
                del rewired_connections[old_connection]
                
            # Create new random connection (shortcut)
            possible_targets = [nid for nid in all_neuron_ids if nid != pre_id]
            if possible_targets:
                new_post_id = np.random.choice(possible_targets)
                
                # Add new connection with similar weight
                new_connection = (pre_id, new_post_id)
                if new_connection not in rewired_connections:
                    rewired_connections[new_connection] = old_weight
                
        return rewired_connections
        
    def analyze_network_properties(
        self,
        connections: Dict[Tuple[int, int], float],
        modules: Dict[str, NetworkModule]
    ) -> Dict[str, float]:
        """
        Analyze small-world and modular properties of the network.
        
        Args:
            connections: Network connections
            modules: Network modules
            
        Returns:
            Dictionary with network analysis metrics
        """
        # Build adjacency list
        adjacency = {}
        all_neurons = set()
        
        for (pre, post), weight in connections.items():
            all_neurons.add(pre)
            all_neurons.add(post)
            if pre not in adjacency:
                adjacency[pre] = []
            adjacency[pre].append(post)
            
        # Ensure all neurons are in adjacency list
        for neuron in all_neurons:
            if neuron not in adjacency:
                adjacency[neuron] = []
                
        # Calculate clustering coefficient
        clustering_coefficients = []
        for neuron in all_neurons:
            neighbors = adjacency[neuron]
            if len(neighbors) < 2:
                clustering_coefficients.append(0.0)
                continue
                
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if neighbor2 in adjacency.get(neighbor1, []):
                        triangles += 1
                        
            clustering = triangles / possible_triangles if possible_triangles > 0 else 0.0
            clustering_coefficients.append(clustering)
            
        # Handle empty case
        avg_clustering = np.mean(clustering_coefficients) if clustering_coefficients else 0.0
        if np.isnan(avg_clustering):
            avg_clustering = 0.0
        
        # Calculate average path length (simplified BFS)
        path_lengths = []
        sample_size = min(100, len(all_neurons))  # Sample for efficiency
        sampled_neurons = np.random.choice(list(all_neurons), sample_size, replace=False)
        
        for start_neuron in sampled_neurons:
            # BFS to find shortest paths
            visited = {start_neuron}
            queue = [(start_neuron, 0)]
            distances = []
            
            while queue and len(distances) < 50:  # Limit for efficiency
                current, distance = queue.pop(0)
                
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, distance + 1))
                        distances.append(distance + 1)
                        
            if distances:
                path_lengths.extend(distances)
                
        avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
        if np.isnan(avg_path_length):
            avg_path_length = 0.0
        
        # Calculate modularity
        intra_module_connections = 0
        inter_module_connections = 0
        
        neuron_to_module = {}
        for module_name, module in modules.items():
            for neuron_id in module.neuron_ids:
                neuron_to_module[neuron_id] = module_name
                
        for (pre, post) in connections.keys():
            if neuron_to_module.get(pre) == neuron_to_module.get(post):
                intra_module_connections += 1
            else:
                inter_module_connections += 1
                
        total_connections = intra_module_connections + inter_module_connections
        modularity = intra_module_connections / total_connections if total_connections > 0 else 0.0
        
        # Small-world index
        small_world_index = avg_clustering / avg_path_length if avg_path_length > 0 else 0.0
        
        return {
            'clustering_coefficient': avg_clustering,
            'average_path_length': avg_path_length,
            'modularity': modularity,
            'small_world_index': small_world_index,
            'total_connections': total_connections,
            'intra_module_connections': intra_module_connections,
            'inter_module_connections': inter_module_connections
        }
        
    def create_hierarchical_modules(
        self, 
        total_neurons: int,
        hierarchy_levels: int = 3,
        modules_per_level: List[int] = None,
        module_size_range: Tuple[int, int] = (50, 200)
    ) -> Dict[str, NetworkModule]:
        """
        Create hierarchical module structure.
        
        Args:
            total_neurons: Total number of neurons to distribute
            hierarchy_levels: Number of hierarchical levels
            modules_per_level: Number of modules at each level (if None, auto-calculate)
            module_size_range: (min, max) neurons per module
            
        Returns:
            Dictionary of created modules
        """
        if modules_per_level is None:
            # Auto-calculate modules per level (more at lower levels)
            modules_per_level = []
            for level in range(hierarchy_levels):
                # More modules at lower levels (sensory), fewer at higher levels (cognitive)
                num_modules = max(1, int(8 / (level + 1)))
                modules_per_level.append(num_modules)
                
        # Distribute neurons across hierarchy
        total_modules = sum(modules_per_level)
        neurons_per_module = total_neurons // total_modules
        
        modules = {}
        neuron_id = 0
        
        for level in range(hierarchy_levels):
            level_modules = modules_per_level[level]
            
            for module_idx in range(level_modules):
                module_name = f"L{level}_M{module_idx}"
                
                # Calculate module size (vary within range)
                if module_idx == level_modules - 1:  # Last module gets remaining
                    remaining_neurons = total_neurons - neuron_id
                    module_size = min(remaining_neurons, 
                                    np.random.randint(module_size_range[0], module_size_range[1] + 1))
                else:
                    module_size = min(neurons_per_module,
                                    np.random.randint(module_size_range[0], module_size_range[1] + 1))
                    
                # Create module neuron IDs
                module_neuron_ids = list(range(neuron_id, neuron_id + module_size))
                neuron_id += module_size
                
                # Create module with spatial position
                # Higher levels are more centrally located
                center_x = 50 + (level - hierarchy_levels/2) * 20
                center_y = 50 + (module_idx - level_modules/2) * 30
                center_z = level * 15  # Vertical hierarchy
                
                module = NetworkModule(
                    name=module_name,
                    center_position=SpatialPosition(center_x, center_y, center_z),
                    radius=20.0 + level * 5,  # Larger modules at higher levels
                    neuron_ids=module_neuron_ids
                )
                
                modules[module_name] = module
                self.hierarchy_levels_map[module_name] = level
                
        self.modules = modules
        return modules
        
    def create_small_world_connections(
        self,
        modules: Dict[str, NetworkModule],
        spatial_layout: SpatialNetworkLayout
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """
        Create small-world connections between modules.
        
        Args:
            modules: Dictionary of network modules
            spatial_layout: Spatial layout for distance calculations
            
        Returns:
            Dictionary of inter-module connections with parameters
        """
        connections = {}
        module_names = list(modules.keys())
        
        # 1. Create local clustering (nearby modules strongly connected)
        for i, module1_name in enumerate(module_names):
            module1 = modules[module1_name]
            
            for j, module2_name in enumerate(module_names[i+1:], i+1):
                module2 = modules[module2_name]
                
                # Calculate distance between module centers
                distance = module1.center_position.distance_to(module2.center_position)
                
                # Local clustering: high probability for nearby modules
                if distance < 50.0:  # Local neighborhood
                    connection_prob = 0.8 * np.exp(-distance / 30.0)
                else:
                    connection_prob = 0.05 * np.exp(-distance / 100.0)
                    
                # Hierarchical bias: same level modules connect more
                level1 = self.hierarchy_levels_map[module1_name]
                level2 = self.hierarchy_levels_map[module2_name]
                
                if level1 == level2:
                    connection_prob *= 2.0  # Same level bonus
                elif abs(level1 - level2) == 1:
                    connection_prob *= 1.5  # Adjacent level bonus
                    
                # Stochastic connection
                if np.random.random() < connection_prob:
                    connections[(module1_name, module2_name)] = {
                        'connection_probability': connection_prob,
                        'distance': distance,
                        'level_difference': abs(level1 - level2),
                        'connection_type': 'local' if distance < 50.0 else 'long_range'
                    }
                    
        # 2. Add long-range connections for small-world property
        num_long_range = int(len(module_names) * self.rewiring_probability)
        
        for _ in range(num_long_range):
            # Random long-range connection
            module1_name = np.random.choice(module_names)
            module2_name = np.random.choice(module_names)
            
            if module1_name != module2_name:
                connection_key = tuple(sorted([module1_name, module2_name]))
                
                if connection_key not in connections:
                    module1 = modules[module1_name]
                    module2 = modules[module2_name]
                    distance = module1.center_position.distance_to(module2.center_position)
                    
                    level1 = self.hierarchy_levels_map[module1_name]
                    level2 = self.hierarchy_levels_map[module2_name]
                    
                    connections[connection_key] = {
                        'connection_probability': 0.1,  # Weak long-range
                        'distance': distance,
                        'level_difference': abs(level1 - level2),
                        'connection_type': 'long_range'
                    }
                    
        self.inter_module_connections = connections
        return connections
        
    def calculate_network_properties(self, connections: Dict) -> Dict[str, float]:
        """
        Calculate small-world network properties.
        
        Args:
            connections: Dictionary of connections
            
        Returns:
            Network property metrics
        """
        module_names = list(self.modules.keys())
        n_modules = len(module_names)
        
        if n_modules < 2:
            return {'clustering_coefficient': 0.0, 'average_path_length': 0.0}
            
        # Build adjacency matrix
        adjacency = np.zeros((n_modules, n_modules))
        name_to_idx = {name: i for i, name in enumerate(module_names)}
        
        for (mod1, mod2), conn_info in connections.items():
            if mod1 in name_to_idx and mod2 in name_to_idx:
                i, j = name_to_idx[mod1], name_to_idx[mod2]
                adjacency[i, j] = adjacency[j, i] = 1
                
        # Calculate clustering coefficient
        clustering_coeffs = []
        for i in range(n_modules):
            neighbors = np.where(adjacency[i] == 1)[0]
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs.append(0.0)
            else:
                # Count triangles
                triangles = 0
                for j in range(len(neighbors)):
                    for l in range(j+1, len(neighbors)):
                        if adjacency[neighbors[j], neighbors[l]] == 1:
                            triangles += 1
                            
                clustering_coeff = 2 * triangles / (k * (k - 1))
                clustering_coeffs.append(clustering_coeff)
                
        avg_clustering = np.mean(clustering_coeffs)
        
        # Calculate average path length (simplified BFS)
        total_path_length = 0
        path_count = 0
        
        for start in range(n_modules):
            # BFS from start node
            visited = set([start])
            queue = [(start, 0)]
            
            while queue:
                node, dist = queue.pop(0)
                
                for neighbor in range(n_modules):
                    if adjacency[node, neighbor] == 1 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
                        total_path_length += dist + 1
                        path_count += 1
                        
        avg_path_length = total_path_length / path_count if path_count > 0 else 0.0
        
        # Small-world coefficient
        # Compare to random network with same degree
        avg_degree = np.sum(adjacency) / n_modules
        random_clustering = avg_degree / n_modules if n_modules > 0 else 0.0
        random_path_length = np.log(n_modules) / np.log(avg_degree) if avg_degree > 1 else 1.0
        
        small_world_coeff = (avg_clustering / random_clustering) / (avg_path_length / random_path_length) if random_clustering > 0 and random_path_length > 0 else 0.0
        
        return {
            'clustering_coefficient': avg_clustering,
            'average_path_length': avg_path_length,
            'small_world_coefficient': small_world_coeff,
            'number_of_connections': len(connections),
            'connection_density': len(connections) / (n_modules * (n_modules - 1) / 2) if n_modules > 1 else 0.0
        }
        
    def get_module_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the modular architecture.
        
        Returns:
            Dictionary with module statistics
        """
        if not self.modules:
            return {}
            
        # Module size statistics
        module_sizes = [len(module.neuron_ids) for module in self.modules.values()]
        
        # Hierarchy statistics
        levels = list(self.hierarchy_levels_map.values())
        modules_per_level = {}
        for level in set(levels):
            modules_per_level[f'level_{level}'] = levels.count(level)
            
        # Connection statistics
        connection_types = {}
        for conn_info in self.inter_module_connections.values():
            conn_type = conn_info['connection_type']
            connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
            
        return {
            'total_modules': len(self.modules),
            'total_neurons': sum(module_sizes),
            'average_module_size': np.mean(module_sizes),
            'module_size_std': np.std(module_sizes),
            'hierarchy_levels': len(set(levels)),
            'modules_per_level': modules_per_level,
            'total_inter_module_connections': len(self.inter_module_connections),
            'connection_types': connection_types
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
        self.modular_architecture = ModularNetworkArchitecture()
        
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
        
        # Create connections with proper E/I balance (infer detail from created layers)
        self._create_ei_balanced_connections(population_sizes)
        
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
            
    def _create_ei_balanced_connections(self, population_sizes: Dict[str, int], detailed_inhibitory: Optional[bool] = None) -> None:
        """Create connections with proper E/I balance.

        If detailed_inhibitory is None, infer from presence of specific inhibitory layers.
        """
        if detailed_inhibitory is None:
            detailed_inhibitory = any(
                name in self.network.layers
                for name in ("basket_cells", "chandelier_cells", "martinotti_cells")
            )

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
        
    def create_modular_cortical_network(
        self,
        total_neurons: int = 1000,
        hierarchy_levels: int = 3,
        modules_per_level: Optional[List[int]] = None,
        excitatory_fraction: Optional[float] = None,
        spatial_bounds: Tuple[float, float, float] = (300.0, 300.0, 100.0),
        create_small_world: bool = True,
        **kwargs
    ) -> NeuromorphicNetwork:
        """
        Create a modular cortical network with hierarchical organization and small-world properties.
        
        Args:
            total_neurons: Total number of neurons
            hierarchy_levels: Number of hierarchical levels
            modules_per_level: Number of modules at each level
            excitatory_fraction: Fraction of excitatory neurons
            spatial_bounds: (width, height, depth) of spatial area
            create_small_world: Whether to add small-world connections
            **kwargs: Additional parameters
            
        Returns:
            Modular neuromorphic network
        """
        # Use default E/I balance if not specified
        if excitatory_fraction is not None:
            self.ei_balance.excitatory_fraction = excitatory_fraction
            self.ei_balance.inhibitory_fraction = 1.0 - excitatory_fraction
            
        # Create hierarchical modules
        modules = self.modular_architecture.create_hierarchical_modules(
            total_neurons, hierarchy_levels, modules_per_level
        )
        
        # Create spatial layout for the entire network
        self.spatial_layout = SpatialNetworkLayout(dimensions=3, bounds=spatial_bounds)
        
        # Position neurons based on their module assignments
        positions = {}
        for module in modules.values():
            # Create clustered layout for each module
            module_positions = self._create_module_spatial_layout(
                module, self.spatial_layout
            )
            positions.update(module_positions)
            
        # Update spatial layout with all positions
        self.spatial_layout.neuron_positions.update(positions)
        
        # Create connectivity builder
        self.connectivity_builder = DistanceDependentConnectivity(self.spatial_layout)
        self._configure_ei_connectivity_parameters()
        
        # Create network
        self.network = NeuromorphicNetwork()
        
        # Create layers for each module
        self._create_modular_layers(modules)
        
        # Create intra-module connections (standard E/I)
        self._create_intra_module_connections(modules)
        
        # Create inter-module connections
        if create_small_world:
            inter_module_connections = self.modular_architecture.create_small_world_connections(
                modules, self.spatial_layout
            )
            self._create_inter_module_connections(inter_module_connections, modules)
            
        return self.network
        
    def _create_module_spatial_layout(
        self, 
        module: NetworkModule, 
        spatial_layout: SpatialNetworkLayout
    ) -> Dict[int, SpatialPosition]:
        """
        Create spatial layout for neurons within a module.
        
        Args:
            module: Network module
            spatial_layout: Global spatial layout
            
        Returns:
            Dictionary mapping neuron IDs to positions
        """
        positions = {}
        center = module.center_position
        radius = module.radius
        
        for i, neuron_id in enumerate(module.neuron_ids):
            # Random position within module radius
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(0, radius)
            
            x = center.x + distance * np.cos(angle)
            y = center.y + distance * np.sin(angle)
            z = center.z + np.random.normal(0, radius * 0.1)  # Small z variation
            
            # Keep within global bounds
            x = np.clip(x, 0, spatial_layout.bounds[0])
            y = np.clip(y, 0, spatial_layout.bounds[1])
            z = np.clip(z, 0, spatial_layout.bounds[2])
            
            positions[neuron_id] = SpatialPosition(x, y, z)
            
        return positions
        
    def _create_modular_layers(self, modules: Dict[str, NetworkModule]):
        """Create network layers for each module with detailed E/I populations."""
        for module_name, module in modules.items():
            module_size = len(module.neuron_ids)
            
            # Calculate detailed E/I populations for this module
            module_populations = self.ei_balance.calculate_population_sizes(module_size)
            
            # Create excitatory layer for this module
            if module_populations['excitatory'] > 0:
                self.network.add_layer(
                    f"{module_name}_excitatory",
                    module_populations['excitatory'],
                    neuron_type="adex"
                )
                
            # Create specific inhibitory neuron type layers
            inhibitory_populations = self.ei_balance.create_inhibitory_populations(
                module_populations['inhibitory_total'], self.spatial_layout
            )
            
            for inh_type, pop_info in inhibitory_populations.items():
                if pop_info['size'] > 0:
                    layer_name = f"{module_name}_{inh_type}_cells"
                    
                    # Use parameters specific to inhibitory type
                    params = pop_info['parameters']
                    
                    # Extract neuron parameters (only include LIF-compatible parameters)
                    lif_compatible_params = ['tau_m', 'v_rest', 'v_thresh', 'v_reset', 'refractory_period']
                    neuron_params = {k: v for k, v in params.items() 
                                   if k in lif_compatible_params}
                    
                    self.network.add_layer(
                        layer_name,
                        pop_info['size'],
                        neuron_type=params['neuron_type'],
                        **neuron_params
                    )
                    
    def _create_intra_module_connections(self, modules: Dict[str, NetworkModule]):
        """Create connections within each module."""
        for module_name, module in modules.items():
            exc_layer = f"{module_name}_excitatory"
            
            # Get all inhibitory layers for this module
            inhibitory_types = ['basket_cells', 'chandelier_cells', 'martinotti_cells']
            inh_layers = {}
            for inh_type in inhibitory_types:
                layer_name = f"{module_name}_{inh_type}"
                if layer_name in self.network.layers:
                    inh_layers[inh_type] = layer_name
            
            # Intra-module connections (dense)
            if exc_layer in self.network.layers:
                # E→E connections
                self._add_ei_connection(exc_layer, exc_layer, "E_to_E")
                
                # E→I connections (to all inhibitory types)
                for inh_type, inh_layer in inh_layers.items():
                    self._add_ei_connection(exc_layer, inh_layer, "E_to_I")
                    
                # I→E connections (from all inhibitory types)
                for inh_type, inh_layer in inh_layers.items():
                    self._add_ei_connection(inh_layer, exc_layer, "I_to_E")
                    
                # I→I connections (between inhibitory types)
                inh_layer_names = list(inh_layers.values())
                for i, inh_layer1 in enumerate(inh_layer_names):
                    for j, inh_layer2 in enumerate(inh_layer_names):
                        if i != j:  # Different inhibitory types
                            self._add_ei_connection(inh_layer1, inh_layer2, "I_to_I")
                            
    def _create_inter_module_connections(
        self, 
        inter_connections: Dict[Tuple[str, str], Dict[str, float]], 
        modules: Dict[str, NetworkModule]
    ):
        """Create connections between modules."""
        for (module1_name, module2_name), conn_info in inter_connections.items():
            # Get layer names
            exc1_layer = f"{module1_name}_excitatory"
            exc2_layer = f"{module2_name}_excitatory"
            
            # Check if layers exist
            if (exc1_layer in self.network.layers and 
                exc2_layer in self.network.layers):
                
                # Inter-module connections are typically excitatory
                connection_prob = conn_info['connection_probability'] * 0.1  # Weaker than intra-module
                
                # Bidirectional excitatory connections
                self.network.connect_layers(
                    exc1_layer,
                    exc2_layer,
                    synapse_type="stdp",
                    connection_probability=connection_prob,
                    weight=0.5  # Weaker inter-module weights
                )
                
                self.network.connect_layers(
                    exc2_layer,
                    exc1_layer,
                    synapse_type="stdp",
                    connection_probability=connection_prob,
                    weight=0.5
                )
                
    def validate_small_world_properties(self) -> Dict[str, Any]:
        """
        Validate that the network exhibits small-world properties.
        
        Returns:
            Dictionary with small-world validation results
        """
        if not hasattr(self, 'modular_architecture') or not self.modular_architecture.modules:
            return {'error': 'No modular architecture found'}
            
        # Calculate network properties
        if self.modular_architecture.inter_module_connections:
            properties = self.modular_architecture.calculate_network_properties(
                self.modular_architecture.inter_module_connections
            )
        else:
            return {'error': 'No inter-module connections found'}
            
        # Small-world criteria
        clustering = properties.get('clustering_coefficient', 0)
        path_length = properties.get('average_path_length', 0)
        small_world_coeff = properties.get('small_world_coefficient', 0)
        
        # Validation criteria
        has_high_clustering = clustering > 0.2  # Higher than random
        has_short_paths = path_length < 6.0     # Reasonable path length
        has_small_world = small_world_coeff > 1.0  # Small-world index > 1
        
        # Overall assessment
        small_world_quality = 0.0
        if has_high_clustering:
            small_world_quality += 0.4
        if has_short_paths:
            small_world_quality += 0.3
        if has_small_world:
            small_world_quality += 0.3
            
        return {
            'clustering_coefficient': clustering,
            'average_path_length': path_length,
            'small_world_coefficient': small_world_coeff,
            'has_high_clustering': has_high_clustering,
            'has_short_paths': has_short_paths,
            'has_small_world_properties': has_small_world,
            'small_world_quality': small_world_quality,
            'validation_passed': small_world_quality >= 0.7,
            'network_type': self._classify_network_topology(clustering, path_length, small_world_coeff)
        }
        
    def calculate_clustering_coefficient(self, adjacency_matrix: np.ndarray) -> float:
        """
        Calculate clustering coefficient for a network.
        
        Args:
            adjacency_matrix: Binary adjacency matrix
            
        Returns:
            Average clustering coefficient
        """
        n_nodes = adjacency_matrix.shape[0]
        clustering_coeffs = []
        
        for i in range(n_nodes):
            neighbors = np.where(adjacency_matrix[i] == 1)[0]
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs.append(0.0)
                continue
                
            # Count triangles involving node i
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if adjacency_matrix[neighbors[j], neighbors[l]] == 1:
                        triangles += 1
                        
            # Clustering coefficient for node i
            possible_triangles = k * (k - 1) // 2
            clustering = triangles / possible_triangles if possible_triangles > 0 else 0.0
            clustering_coeffs.append(clustering)
            
        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
        
    def calculate_shortest_path_lengths(self, adjacency_matrix: np.ndarray) -> float:
        """
        Calculate average shortest path length using optimal algorithm.
        
        Automatically selects between BFS (small networks) and advanced algorithms (large networks).
        
        Args:
            adjacency_matrix: Binary adjacency matrix
            
        Returns:
            Average shortest path length
        """
        n_nodes = adjacency_matrix.shape[0]
        
        # For large networks, we should use the new Tsinghua algorithm
        if n_nodes > 1000:
            # TODO: Implement Tsinghua algorithm for large sparse graphs
            # For now, use optimized BFS with sampling
            return self._calculate_paths_large_network(adjacency_matrix)
        else:
            # Use standard BFS for small networks
            return self._calculate_paths_bfs(adjacency_matrix)
            
    def _calculate_paths_bfs(self, adjacency_matrix: np.ndarray) -> float:
        """Standard BFS implementation for small networks."""
        n_nodes = adjacency_matrix.shape[0]
        all_path_lengths = []
        
        for start in range(n_nodes):
            # BFS from start node
            distances = [-1] * n_nodes
            distances[start] = 0
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                current_dist = distances[current]
                
                # Check all neighbors
                for neighbor in range(n_nodes):
                    if adjacency_matrix[current, neighbor] == 1 and distances[neighbor] == -1:
                        distances[neighbor] = current_dist + 1
                        queue.append(neighbor)
                        
            # Collect path lengths (excluding unreachable nodes)
            for dist in distances:
                if dist > 0:  # Exclude self (distance 0) and unreachable (-1)
                    all_path_lengths.append(dist)
                    
        return np.mean(all_path_lengths) if all_path_lengths else 0.0
        
    def _calculate_paths_large_network(self, adjacency_matrix: np.ndarray) -> float:
        """
        Optimized path calculation for large networks using degree-based landmarks.

        Strategy:
        - Select top-k high-degree nodes as landmarks (good graph coverage)
        - Optionally add a few random nodes to cover low-degree regions
        - Run BFS from each landmark, aggregate finite shortest path lengths
        - Return the mean of collected distances as the estimated average path length

        This provides a strong approximation in O(k * (V + E)) time.
        """
        n_nodes = adjacency_matrix.shape[0]
        if n_nodes == 0:
            return 0.0

        # Compute degrees efficiently
        degrees = np.asarray(adjacency_matrix.sum(axis=1)).ravel() if hasattr(adjacency_matrix, 'sum') else np.sum(adjacency_matrix, axis=1)

        # Choose number of landmarks: sqrt scaling with upper/lower bounds
        k_core = max(10, int(np.sqrt(n_nodes)))
        k_core = min(k_core, 200)  # cap for performance

        # Select top-degree nodes
        top_indices = np.argpartition(-degrees, kth=min(k_core-1, n_nodes-1))[:k_core]

        # Add a few random nodes to improve coverage in sparse/disconnected graphs
        k_rand = min(20, max(5, n_nodes // 200))
        random_pool = np.setdiff1d(np.arange(n_nodes), top_indices, assume_unique=False)
        if random_pool.size > 0 and k_rand > 0:
            rand_indices = np.random.choice(random_pool, size=min(k_rand, random_pool.size), replace=False)
            landmarks = np.unique(np.concatenate([top_indices, rand_indices]))
        else:
            landmarks = np.unique(top_indices)

        # BFS utility (array/list based for speed, avoid Python sets where possible)
        def bfs_distances(start: int) -> np.ndarray:
            distances = np.full(n_nodes, -1, dtype=np.int32)
            distances[start] = 0
            queue = [start]
            head = 0
            # Convert row access once for speed
            while head < len(queue):
                current = queue[head]
                head += 1
                # Neighbors where adjacency[current, j] == 1
                row = adjacency_matrix[current]
                if hasattr(row, 'nonzero'):
                    neighbors = row.nonzero()[0]
                else:
                    neighbors = np.where(row == 1)[0]
                for neighbor in neighbors:
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            return distances

        all_path_lengths: list[int] = []
        for start in landmarks:
            dists = bfs_distances(int(start))
            # Exclude 0 (self) and -1 (unreachable)
            valid = dists[(dists > 0)]
            if valid.size:
                all_path_lengths.extend(valid.tolist())

        return float(np.mean(all_path_lengths)) if all_path_lengths else 0.0
        
    def implement_tsinghua_shortest_path(self, adjacency_matrix: np.ndarray) -> float:
        """
        Placeholder for implementing the new Tsinghua University shortest path algorithm.
        
        This algorithm achieves O(E × (log V)^0.67) complexity for sparse graphs,
        potentially much faster than Dijkstra's O(E + V log V) for large sparse networks.
        
        Key techniques:
        - Combines Dijkstra and Bellman-Ford approaches
        - Uses frontier shrinking with pivot nodes
        - Employs divide-and-conquer graph partitioning
        - Avoids sorting bottleneck
        
        Args:
            adjacency_matrix: Binary adjacency matrix
            
        Returns:
            Average shortest path length
            
        TODO: Implement when algorithm details become available
        """
        # For now, fall back to optimized BFS
        return self._calculate_paths_large_network(adjacency_matrix)
        
    def calculate_small_world_index(
        self, 
        clustering: float, 
        path_length: float, 
        n_nodes: int, 
        avg_degree: float
    ) -> float:
        """
        Calculate small-world index (sigma).
        
        Args:
            clustering: Actual clustering coefficient
            path_length: Actual average path length
            n_nodes: Number of nodes
            avg_degree: Average degree
            
        Returns:
            Small-world index
        """
        # Expected values for random network
        random_clustering = avg_degree / n_nodes if n_nodes > 0 else 0.0
        random_path_length = np.log(n_nodes) / np.log(avg_degree) if avg_degree > 1 else 1.0
        
        # Small-world index
        if random_clustering > 0 and random_path_length > 0:
            clustering_ratio = clustering / random_clustering
            path_ratio = path_length / random_path_length
            return clustering_ratio / path_ratio if path_ratio > 0 else 0.0
        else:
            return 0.0
            
    def get_modular_network_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics including modular properties.
        
        Returns:
            Statistics with modular network analysis
        """
        stats = self.get_network_statistics()
        
        # Add modular architecture statistics
        if self.modular_architecture.modules:
            module_stats = self.modular_architecture.get_module_statistics()
            stats['modular_architecture'] = module_stats
            
            # Calculate small-world properties
            if self.modular_architecture.inter_module_connections:
                network_props = self.modular_architecture.calculate_network_properties(
                    self.modular_architecture.inter_module_connections
                )
                stats['small_world_properties'] = network_props
                
                # Assess small-world quality
                clustering = network_props.get('clustering_coefficient', 0)
                path_length = network_props.get('average_path_length', 0)
                small_world_coeff = network_props.get('small_world_coefficient', 0)
                
                stats['network_topology_assessment'] = {
                    'has_small_world_properties': small_world_coeff > 1.0,
                    'clustering_quality': 'high' if clustering > 0.2 else 'medium' if clustering > 0.1 else 'low',
                    'path_length_efficiency': 'good' if path_length < 4.0 else 'moderate' if path_length < 6.0 else 'poor',
                    'topology_type': self._classify_network_topology(clustering, path_length, small_world_coeff)
                }
                
        return stats
        
    def _classify_network_topology(
        self, 
        clustering: float, 
        path_length: float, 
        small_world_coeff: float
    ) -> str:
        """Classify the network topology type."""
        if small_world_coeff > 1.0 and clustering > 0.2:
            return 'small_world'
        elif clustering > 0.4:
            return 'clustered'
        elif path_length < 3.0:
            return 'random'
        elif clustering < 0.1 and path_length > 5.0:
            return 'sparse'
        else:
            return 'intermediate'
        
    def create_large_scale_brain_network(
        self,
        total_neurons: int = 100_000,
        hierarchy_levels: int = 5,
        modules_per_level: Optional[List[int]] = None,
        spatial_bounds: Tuple[float, float, float] = (1000.0, 1000.0, 200.0),
        create_small_world: bool = True,
        **kwargs
    ) -> NeuromorphicNetwork:
        """
        Create a large-scale brain network suitable for realistic simulations.
        
        This method creates networks with 10K-1M+ neurons, using optimized algorithms
        for large-scale network analysis including the new shortest path algorithms.
        
        Args:
            total_neurons: Total number of neurons (10K-1M+)
            hierarchy_levels: Number of hierarchical levels (3-7)
            modules_per_level: Modules at each level (auto-calculated if None)
            spatial_bounds: Large spatial area for realistic layouts
            create_small_world: Whether to add small-world connections
            **kwargs: Additional parameters
            
        Returns:
            Large-scale neuromorphic network
        """
        print(f"Creating large-scale brain network with {total_neurons:,} neurons...")
        
        if total_neurons < 10_000:
            print("Warning: Network size < 10K neurons. Consider using create_modular_cortical_network() instead.")
            
        # Auto-calculate modules for large networks
        if modules_per_level is None:
            modules_per_level = self._auto_modules_per_level(total_neurons, hierarchy_levels)
                
        print(f"Hierarchy: {hierarchy_levels} levels with {modules_per_level} modules per level")
        
        # Create hierarchical modules
        modules = self.modular_architecture.create_hierarchical_modules(
            total_neurons, hierarchy_levels, modules_per_level
        )
        
        n_modules = len(modules)
        print(f"Created {n_modules} modules (path analysis complexity: O({n_modules**2:,}))")
        
        # Use optimized spatial layout for large networks
        self.spatial_layout = SpatialNetworkLayout(dimensions=3, bounds=spatial_bounds)
        
        # Efficient positioning for large networks
        print("Positioning neurons in spatial layout...")
        positions = self._create_efficient_spatial_layout(modules, spatial_bounds)
        self.spatial_layout.neuron_positions.update(positions)
        
        # Create connectivity builder
        self.connectivity_builder = DistanceDependentConnectivity(self.spatial_layout)
        self._configure_ei_connectivity_parameters()
        
        # Create network
        self.network = NeuromorphicNetwork()
        
        # Create layers efficiently
        print("Creating network layers...")
        self._create_modular_layers(modules)
        
        # Create connections
        print("Creating intra-module connections...")
        self._create_intra_module_connections(modules)
        
        if create_small_world:
            print("Creating small-world inter-module connections...")
            inter_module_connections = self.modular_architecture.create_small_world_connections(
                modules, self.spatial_layout
            )
            self._create_inter_module_connections(inter_module_connections, modules)
            
        total_created = sum(layer.size for layer in self.network.layers.values())
        total_connections = len(self.network.connections)
        
        print(f"✓ Large-scale network created:")
        print(f"  - Neurons: {total_created:,}")
        print(f"  - Connections: {total_connections:,}")
        print(f"  - Modules: {n_modules}")
        print(f"  - Layers: {len(self.network.layers)}")
        
        return self.network

    def _auto_modules_per_level(self, total_neurons: int, hierarchy_levels: int) -> List[int]:
        """
        Determine modules per level based on total neurons targeting reasonable module sizes.

        Heuristics:
        - Aim for ~500–2000 neurons per lowest-level module
        - Increase module count for larger networks using a geometric progression
        - Ensure at least 1 module per level and non-increasing modules with higher levels
        """
        # Target module size range
        min_module = 500
        max_module = 2000
        # Desired number of leaf modules
        target_leaf_modules = max(1, int(total_neurons / max_module))
        # Cap to avoid explosion
        target_leaf_modules = min(target_leaf_modules, max(10, int(np.sqrt(total_neurons / 100))))

        # Distribute across hierarchy with geometric decrease
        levels = max(1, hierarchy_levels)
        modules = []
        remaining = target_leaf_modules
        for level in range(levels):
            # Higher levels have fewer modules
            factor = max(1, int(target_leaf_modules / (2 ** (levels - level - 1))))
            modules_at_level = max(1, min(remaining, factor))
            modules.append(modules_at_level)
            remaining = max(1, modules_at_level // 2)

        # Normalize to be non-increasing with level index
        for i in range(1, len(modules)):
            modules[i] = min(modules[i], modules[i-1])

        return modules
        
    def _create_efficient_spatial_layout(
        self, 
        modules: Dict[str, NetworkModule], 
        spatial_bounds: Tuple[float, float, float]
    ) -> Dict[int, SpatialPosition]:
        """
        Create efficient spatial layout for large networks.
        
        Uses optimized positioning to avoid O(N²) operations.
        """
        positions = {}
        
        for module in modules.values():
            # Use vectorized operations for efficiency
            n_neurons = len(module.neuron_ids)
            center = module.center_position
            radius = module.radius
            
            # Generate random positions in batch
            angles = np.random.uniform(0, 2 * np.pi, n_neurons)
            distances = np.random.uniform(0, radius, n_neurons)
            z_offsets = np.random.normal(0, radius * 0.1, n_neurons)
            
            # Vectorized position calculation
            x_coords = center.x + distances * np.cos(angles)
            y_coords = center.y + distances * np.sin(angles)
            z_coords = center.z + z_offsets
            
            # Clip to bounds
            x_coords = np.clip(x_coords, 0, spatial_bounds[0])
            y_coords = np.clip(y_coords, 0, spatial_bounds[1])
            z_coords = np.clip(z_coords, 0, spatial_bounds[2])
            
            # Create positions efficiently
            for i, neuron_id in enumerate(module.neuron_ids):
                positions[neuron_id] = SpatialPosition(
                    x_coords[i], y_coords[i], z_coords[i]
                )
                
        return positions
        
    def create_ei_balanced_network(
        self,
        total_neurons: int = 1000,
        spatial_bounds: Tuple[float, float, float] = (200.0, 200.0, 50.0),
        excitatory_fraction: Optional[float] = None,
        **kwargs
    ) -> NeuromorphicNetwork:
        """
        Create a network with proper E/I balance and specific inhibitory neuron types.
        
        Args:
            total_neurons: Total number of neurons
            spatial_bounds: (width, height, depth) of spatial area
            excitatory_fraction: Fraction of excitatory neurons (default: 0.8)
            **kwargs: Additional parameters
            
        Returns:
            E/I balanced neuromorphic network
        """
        # Set E/I balance if specified
        if excitatory_fraction is not None:
            self.ei_balance.excitatory_fraction = excitatory_fraction
            self.ei_balance.inhibitory_fraction = 1.0 - excitatory_fraction
            
        # Calculate population sizes
        population_sizes = self.ei_balance.calculate_population_sizes(total_neurons)
        
        # Create spatial layout
        self.spatial_layout = SpatialNetworkLayout(dimensions=3, bounds=spatial_bounds)
        
        # Position neurons randomly in space
        positions = {}
        for neuron_id in range(total_neurons):
            x = np.random.uniform(0, spatial_bounds[0])
            y = np.random.uniform(0, spatial_bounds[1])
            z = np.random.uniform(0, spatial_bounds[2])
            positions[neuron_id] = SpatialPosition(x, y, z)
            
        self.spatial_layout.neuron_positions = positions
        
        # Create connectivity builder
        self.connectivity_builder = DistanceDependentConnectivity(self.spatial_layout)
        self._configure_ei_connectivity_parameters()
        
        # Create network
        self.network = NeuromorphicNetwork()
        
        # Create excitatory layer
        if population_sizes['excitatory'] > 0:
            self.network.add_layer(
                "excitatory",
                population_sizes['excitatory'],
                neuron_type="adex"
            )
            
        # Create specific inhibitory layers
        inhibitory_populations = self.ei_balance.create_inhibitory_populations(
            population_sizes['inhibitory_total'], self.spatial_layout
        )
        
        for inh_type, pop_info in inhibitory_populations.items():
            if pop_info['size'] > 0:
                layer_name = f"{inh_type}_cells"
                params = pop_info['parameters']
                
                # Extract neuron parameters (only include LIF-compatible parameters)
                lif_compatible_params = ['tau_m', 'v_rest', 'v_thresh', 'v_reset', 'refractory_period']
                neuron_params = {k: v for k, v in params.items() 
                               if k in lif_compatible_params}
                
                self.network.add_layer(
                    layer_name,
                    pop_info['size'],
                    neuron_type=params['neuron_type'],
                    **neuron_params
                )
                
        # Create E/I connections
        self._create_ei_balanced_connections(inhibitory_populations)
        
        return self.network
        
    def _create_ei_balanced_connections(self, inhibitory_populations: Dict[str, Dict[str, Any]]):
        """Create E/I balanced connections with specific inhibitory types."""
        exc_layer = "excitatory"
        
        if exc_layer in self.network.layers:
            # E→E connections
            self._add_ei_connection(exc_layer, exc_layer, "E_to_E")
            
            # E→I and I→E connections for each inhibitory type
            for inh_type, pop_info in inhibitory_populations.items():
                inh_layer = f"{inh_type}_cells"
                
                if inh_layer in self.network.layers:
                    # E→I connections
                    self._add_ei_connection(exc_layer, inh_layer, "E_to_I")
                    
                    # I→E connections
                    self._add_ei_connection(inh_layer, exc_layer, "I_to_E")
                    
            # I→I connections between different inhibitory types
            inh_layer_names = [f"{inh_type}_cells" for inh_type in inhibitory_populations.keys()]
            for i, inh_layer1 in enumerate(inh_layer_names):
                for j, inh_layer2 in enumerate(inh_layer_names):
                    if i != j and inh_layer1 in self.network.layers and inh_layer2 in self.network.layers:
                        self._add_ei_connection(inh_layer1, inh_layer2, "I_to_I")
        
    def _create_modular_ei_layers(self, modules: Dict[str, NetworkModule], population_sizes: Dict[str, int]):
        """Create E/I balanced layers for each module with specific inhibitory types."""
        total_neurons = sum(len(module.neuron_ids) for module in modules.values())
        
        for module_name, module in modules.items():
            module_size = len(module.neuron_ids)
            
            # Calculate detailed E/I populations for this module
            module_populations = self.ei_balance.calculate_population_sizes(module_size)
            
            # Create excitatory layer for this module
            if module_populations['excitatory'] > 0:
                self.network.add_layer(
                    f"{module_name}_excitatory",
                    module_populations['excitatory'],
                    neuron_type="adex"
                )
                
            # Create specific inhibitory neuron type layers
            inhibitory_populations = self.ei_balance.create_inhibitory_populations(
                module_populations['inhibitory_total'], self.spatial_layout
            )
            
            for inh_type, pop_info in inhibitory_populations.items():
                if pop_info['size'] > 0:
                    layer_name = f"{module_name}_{inh_type}_cells"
                    
                    # Use parameters specific to inhibitory type
                    params = pop_info['parameters']
                    
                    # Extract neuron parameters (only include LIF-compatible parameters)
                    lif_compatible_params = ['tau_m', 'v_rest', 'v_thresh', 'v_reset', 'refractory_period']
                    neuron_params = {k: v for k, v in params.items() 
                                   if k in lif_compatible_params}
                    
                    self.network.add_layer(
                        layer_name,
                        pop_info['size'],
                        neuron_type=params['neuron_type'],
                        **neuron_params
                    )
                
    def _create_modular_connections(self, modules: Dict[str, NetworkModule], enable_small_world: bool):
        """Create modular connections with intra/inter-module structure."""
        # Get modular connectivity
        modular_connections = self.modular_architecture.compute_modular_connectivity(
            modules, self.spatial_layout.neuron_positions
        )
        
        # Add small-world rewiring if enabled
        if enable_small_world:
            modular_connections = self.modular_architecture.add_small_world_rewiring(
                modular_connections
            )
            
        # Create connections between module layers with specific inhibitory types
        module_names = list(modules.keys())
        inhibitory_types = ['basket_cells', 'chandelier_cells', 'martinotti_cells']
        
        for module_name in module_names:
            exc_layer = f"{module_name}_excitatory"
            
            # Get all inhibitory layers for this module
            inh_layers = {}
            for inh_type in inhibitory_types:
                layer_name = f"{module_name}_{inh_type}"
                if layer_name in self.network.layers:
                    inh_layers[inh_type] = layer_name
            
            # Intra-module connections (dense)
            if exc_layer in self.network.layers:
                # E→E connections
                self._add_ei_connection(exc_layer, exc_layer, "E_to_E")
                
                # E→I connections (to all inhibitory types)
                for inh_type, inh_layer in inh_layers.items():
                    self._add_ei_connection(exc_layer, inh_layer, "E_to_I")
                    
                # I→E connections (from all inhibitory types)
                for inh_type, inh_layer in inh_layers.items():
                    self._add_ei_connection(inh_layer, exc_layer, "I_to_E")
                    
                # I→I connections (between inhibitory types)
                inh_layer_names = list(inh_layers.values())
                for i, inh_layer1 in enumerate(inh_layer_names):
                    for j, inh_layer2 in enumerate(inh_layer_names):
                        if i != j:  # Different inhibitory types
                            self._add_ei_connection(inh_layer1, inh_layer2, "I_to_I")
                            
        # Inter-module connections (sparse, mainly excitatory)
        for i, module1_name in enumerate(module_names):
            for j, module2_name in enumerate(module_names[i+1:], i+1):
                exc1_layer = f"{module1_name}_excitatory"
                exc2_layer = f"{module2_name}_excitatory"
                
                if exc1_layer in self.network.layers and exc2_layer in self.network.layers:
                    # Sparse bidirectional excitatory connections between modules
                    inter_module_prob = self.modular_architecture.inter_module_probability
                    
                    self.network.connect_layers(
                        exc1_layer, exc2_layer,
                        synapse_type="stdp",
                        connection_probability=inter_module_prob,
                        weight=0.3  # Weaker inter-module connections
                    )
                    
                    self.network.connect_layers(
                        exc2_layer, exc1_layer,
                        synapse_type="stdp", 
                        connection_probability=inter_module_prob,
                        weight=0.3
                    )
                
            # Intra-module E→I
            if exc_layer in self.network.layers and inh_layer in self.network.layers:
                self._add_modular_connection(
                    exc_layer, inh_layer, "E_to_I", 
                    self.modular_architecture.intra_module_probability * 1.5
                )
                
            # Intra-module I→E
            if inh_layer in self.network.layers and exc_layer in self.network.layers:
                self._add_modular_connection(
                    inh_layer, exc_layer, "I_to_E", 
                    self.modular_architecture.intra_module_probability * 1.2
                )
                
            # Intra-module I→I
            if inh_layer in self.network.layers:
                self._add_modular_connection(
                    inh_layer, inh_layer, "I_to_I", 
                    self.modular_architecture.intra_module_probability * 0.8
                )
                    
    def _add_modular_connection(self, pre_layer: str, post_layer: str, connection_type: str, probability: float):
        """Add modular connection with appropriate parameters."""
        params = self.ei_balance.get_connection_parameters(
            'excitatory' if 'excitatory' in pre_layer else 'inhibitory',
            'excitatory' if 'excitatory' in post_layer else 'inhibitory'
        )
        
        # Use modular probability instead of default
        effective_probability = min(probability, 0.5)  # Cap at 50%
        
        self.network.connect_layers(
            pre_layer,
            post_layer,
            synapse_type="stdp",
            connection_probability=effective_probability,
            weight=abs(params['synaptic_strength'])
        )
        
    def analyze_network_modularity(self) -> Dict[str, Any]:
        """
        Analyze the modularity and small-world properties of the created network.
        
        Returns:
            Dictionary with network analysis results
        """
        if self.network is None or not hasattr(self.spatial_layout, 'modules'):
            return {'error': 'No modular network created'}
            
        # Extract connections from network
        connections = {}
        neuron_offset = 0
        layer_neuron_mapping = {}
        
        # Create mapping from layer neurons to global neuron IDs
        for layer_name, layer in self.network.layers.items():
            layer_neuron_mapping[layer_name] = list(range(neuron_offset, neuron_offset + layer.size))
            neuron_offset += layer.size
            
        # Extract connections
        for (pre_layer, post_layer), connection in self.network.connections.items():
            if connection.synapse_population:
                pre_neurons = layer_neuron_mapping[pre_layer]
                post_neurons = layer_neuron_mapping[post_layer]
                
                for (pre_idx, post_idx), synapse in connection.synapse_population.synapses.items():
                    if pre_idx < len(pre_neurons) and post_idx < len(post_neurons):
                        global_pre = pre_neurons[pre_idx]
                        global_post = post_neurons[post_idx]
                        connections[(global_pre, global_post)] = synapse.weight
                        
        # Analyze network properties
        analysis = self.modular_architecture.analyze_network_properties(
            connections, self.spatial_layout.modules
        )
        
        # Add additional modular metrics
        analysis.update({
            'num_modules': len(self.spatial_layout.modules),
            'neurons_per_module': [len(module.neuron_ids) for module in self.spatial_layout.modules.values()],
            'module_names': list(self.spatial_layout.modules.keys())
        })
        
        return analysis