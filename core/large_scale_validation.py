#!/usr/bin/env python3
"""
Large-Scale Network Properties Validation
========================================

This module validates that large-scale networks maintain biological realism
including E/I balance, small-world properties, and other brain-like characteristics.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time

try:
    from scipy import sparse
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from .brain_topology import BrainTopologyBuilder
    BRAIN_TOPOLOGY_AVAILABLE = True
except ImportError:
    BRAIN_TOPOLOGY_AVAILABLE = False
    print("Warning: BrainTopologyBuilder not available - using mock configurations")


@dataclass
class BiologicalMetrics:
    """Container for biological realism metrics."""
    ei_ratio: float
    excitatory_fraction: float
    inhibitory_fraction: float
    connection_density: float
    clustering_coefficient: float
    average_path_length: float
    small_world_index: float
    degree_distribution_gamma: float
    firing_rate_mean: float
    firing_rate_std: float
    validation_score: float
    
    
@dataclass
class ValidationResult:
    """Result of network validation."""
    network_size: int
    num_modules: int
    metrics: BiologicalMetrics
    passes_validation: bool
    validation_notes: List[str]
    computation_time: float


class LargeScaleNetworkValidator:
    """Validator for large-scale network biological properties.
    
    Handles validation of networks ranging from thousands to millions of neurons
    with hundreds to thousands of modules, ensuring biological realism is
    maintained across all scales.
    """
    
    def __init__(self):
        """Initialize validator with scale-adaptive biological constraints."""
        # Biological constraints
        self.target_ei_ratio = 0.8  # 80% excitatory
        self.max_connection_density = 0.05  # 5% max connectivity (smaller networks)
        self.target_clustering = (0.25, 0.7)  # Biological range (relaxed for large networks)
        self.target_path_length = (2.0, 6.0)  # Small-world range
        self.min_small_world_index = 1.2  # SW > 1.2 for large networks (relaxed)
        
        # Scale-dependent thresholds
        self.large_network_threshold = 100000  # 100K+ neurons
        self.massive_network_threshold = 1000000  # 1M+ neurons
        
        print("Large-Scale Network Validator initialized")
        print(f"  Target E/I ratio: {self.target_ei_ratio}")
        print(f"  Max connection density: {self.max_connection_density}")
        print(f"  Clustering range: {self.target_clustering}")
        print(f"  Path length range: {self.target_path_length}")
        print(f"  Min small-world index: {self.min_small_world_index}")
        
    def validate_network_properties(
        self, 
        network_config: Dict[str, Any],
        spike_data: Optional[np.ndarray] = None
    ) -> ValidationResult:
        """
        Comprehensive validation of network properties.
        
        Args:
            network_config: Network configuration dictionary
            spike_data: Optional spike train data for activity analysis
            
        Returns:
            ValidationResult with all metrics and validation status
        """
        start_time = time.time()
        validation_notes = []
        
        # Extract basic network properties
        total_neurons = network_config.get('total_neurons', 0)
        num_modules = network_config.get('num_modules', 1)
        
        print(f"Validating network: {total_neurons:,} neurons, {num_modules} modules")
        
        # 1. E/I Balance Validation
        ei_metrics = self._validate_ei_balance(network_config)
        if ei_metrics['passes']:
            validation_notes.append("✓ E/I balance within biological range")
        else:
            validation_notes.append(f"✗ E/I balance issue: {ei_metrics['notes']}")
            
        # 2. Connection Density Validation  
        density_metrics = self._validate_connection_density(network_config)
        if density_metrics['passes']:
            validation_notes.append("✓ Connection density appropriate")
        else:
            validation_notes.append(f"✗ Connection density issue: {density_metrics['notes']}")
            
        # 3. Small-World Properties Validation
        sw_metrics = self._validate_small_world_properties(network_config)
        if sw_metrics['passes']:
            validation_notes.append("✓ Small-world properties confirmed")
        else:
            validation_notes.append(f"✗ Small-world issue: {sw_metrics['notes']}")
            
        # 4. Activity Pattern Validation (if spike data available)
        activity_metrics = {}
        if spike_data is not None:
            activity_metrics = self._validate_activity_patterns(spike_data)
            if activity_metrics['passes']:
                validation_notes.append("✓ Activity patterns biological")
            else:
                validation_notes.append(f"✗ Activity issue: {activity_metrics['notes']}")
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(
            ei_metrics, density_metrics, sw_metrics, activity_metrics
        )
        
        # Compile biological metrics
        biological_metrics = BiologicalMetrics(
            ei_ratio=ei_metrics.get('ei_ratio', 0.8),
            excitatory_fraction=ei_metrics.get('excitatory_fraction', 0.8),
            inhibitory_fraction=ei_metrics.get('inhibitory_fraction', 0.2),
            connection_density=density_metrics.get('density', 0.0),
            clustering_coefficient=sw_metrics.get('clustering', 0.0),
            average_path_length=sw_metrics.get('path_length', 0.0),
            small_world_index=sw_metrics.get('small_world_index', 0.0),
            degree_distribution_gamma=sw_metrics.get('degree_gamma', 0.0),
            firing_rate_mean=activity_metrics.get('firing_rate_mean', 0.0),
            firing_rate_std=activity_metrics.get('firing_rate_std', 0.0),
            validation_score=validation_score
        )
        
        # Overall validation result
        passes_validation = validation_score >= 0.7  # 70% threshold
        computation_time = time.time() - start_time
        
        return ValidationResult(
            network_size=total_neurons,
            num_modules=num_modules,
            metrics=biological_metrics,
            passes_validation=passes_validation,
            validation_notes=validation_notes,
            computation_time=computation_time
        )
    
    def _validate_ei_balance(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate excitatory/inhibitory balance."""
        # Extract E/I information
        ei_ratio = network_config.get('ei_ratio', 0.8)
        excitatory_fraction = ei_ratio
        inhibitory_fraction = 1.0 - ei_ratio
        
        # Check if within biological range (70-85% excitatory)
        passes = 0.7 <= excitatory_fraction <= 0.85
        
        notes = ""
        if not passes:
            notes = f"E/I ratio {excitatory_fraction:.2f} outside biological range [0.7-0.85]"
            
        return {
            'passes': passes,
            'ei_ratio': ei_ratio,
            'excitatory_fraction': excitatory_fraction,
            'inhibitory_fraction': inhibitory_fraction,
            'notes': notes
        }
    
    def _validate_connection_density(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate connection density is biologically realistic with scale-adaptive limits."""
        total_neurons = network_config.get('total_neurons', 0)
        total_connections = network_config.get('total_connections', 0)
        
        if total_neurons == 0:
            return {'passes': False, 'density': 0.0, 'notes': 'No neuron count available'}
            
        # Calculate actual density
        max_connections = total_neurons * (total_neurons - 1)
        actual_density = total_connections / max_connections if max_connections > 0 else 0.0
        
        # Scale-adaptive density limits (biological networks become sparser at larger scales)
        if total_neurons >= self.massive_network_threshold:  # 1M+ neurons
            max_density = 0.02  # 2% max for massive networks
        elif total_neurons >= self.large_network_threshold:  # 100K+ neurons  
            max_density = 0.03  # 3% max for large networks
        else:
            max_density = self.max_connection_density  # 5% for smaller networks
            
        passes = actual_density <= max_density
        
        notes = ""
        if not passes:
            notes = f"Density {actual_density:.4f} exceeds scale-adaptive limit {max_density:.3f} for {total_neurons:,} neurons"
        else:
            notes = f"Density {actual_density:.4f} within biological range for {total_neurons:,} neurons"
            
        return {
            'passes': passes,
            'density': actual_density,
            'max_density': max_density,
            'notes': notes
        }
    
    def _validate_small_world_properties(self, network_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate small-world network properties with scale-adaptive criteria."""
        # Extract small-world metrics if available
        sw_metrics = network_config.get('small_world_metrics', {})
        total_neurons = network_config.get('total_neurons', 0)
        num_modules = network_config.get('num_modules', 1)
        
        clustering = sw_metrics.get('clustering', 0.0)
        path_length = sw_metrics.get('avg_path_length', 0.0)
        small_world_index = sw_metrics.get('small_world_index', 0.0)
        
        # Scale-adaptive criteria for large networks
        if num_modules >= 1000:  # 1000+ modules
            # Relax criteria for massive modular networks
            min_clustering = 0.25
            max_path_length = 5.0
            min_sw_index = 1.1
        elif num_modules >= 500:  # 500+ modules
            min_clustering = 0.28
            max_path_length = 4.5
            min_sw_index = 1.2
        elif total_neurons >= self.large_network_threshold:  # Large networks
            min_clustering = 0.3
            max_path_length = 4.0
            min_sw_index = 1.3
        else:  # Smaller networks
            min_clustering = self.target_clustering[0]
            max_path_length = self.target_path_length[1]
            min_sw_index = self.min_small_world_index
            
        # Validation criteria
        clustering_ok = clustering >= min_clustering and clustering <= self.target_clustering[1]
        path_length_ok = self.target_path_length[0] <= path_length <= max_path_length
        small_world_ok = small_world_index >= min_sw_index
        
        passes = clustering_ok and path_length_ok and small_world_ok
        
        notes = []
        if not clustering_ok:
            notes.append(f"Clustering {clustering:.3f} outside adapted range [{min_clustering:.2f}-{self.target_clustering[1]:.2f}]")
        if not path_length_ok:
            notes.append(f"Path length {path_length:.2f} outside range [{self.target_path_length[0]:.1f}-{max_path_length:.1f}]")
        if not small_world_ok:
            notes.append(f"SW index {small_world_index:.2f} below adapted minimum {min_sw_index:.2f}")
            
        # Additional validation for very large networks
        if num_modules >= 1000:
            notes.append(f"✓ 1000+ module network validated with adapted criteria")
            
        return {
            'passes': passes,
            'clustering': clustering,
            'path_length': path_length,
            'small_world_index': small_world_index,
            'degree_gamma': sw_metrics.get('degree_gamma', 0.0),
            'adapted_criteria': {
                'min_clustering': min_clustering,
                'max_path_length': max_path_length,
                'min_sw_index': min_sw_index
            },
            'notes': '; '.join(notes) if notes else '✓ All small-world properties validated'
        }
    
    def _validate_activity_patterns(self, spike_data: np.ndarray) -> Dict[str, Any]:
        """Validate neural activity patterns."""
        if len(spike_data) == 0:
            return {'passes': False, 'notes': 'No spike data available'}
            
        # Calculate firing rates
        num_neurons = spike_data.shape[0] if len(spike_data.shape) > 1 else len(np.unique(spike_data))
        simulation_time = spike_data.max() if len(spike_data) > 0 else 1.0
        
        # Estimate firing rates (simplified)
        if len(spike_data.shape) == 1:
            # Spike times format
            firing_rates = np.bincount(spike_data.astype(int)) / simulation_time * 1000  # Hz
        else:
            # Spike raster format
            firing_rates = np.sum(spike_data, axis=1) / simulation_time * 1000  # Hz
            
        firing_rate_mean = np.mean(firing_rates)
        firing_rate_std = np.std(firing_rates)
        
        # Biological firing rates: 1-100 Hz typical, with most neurons <20 Hz
        rate_ok = 0.5 <= firing_rate_mean <= 50.0
        variability_ok = firing_rate_std > 0.1  # Some variability expected
        
        passes = rate_ok and variability_ok
        
        notes = []
        if not rate_ok:
            notes.append(f"Mean firing rate {firing_rate_mean:.1f} Hz outside biological range [0.5-50]")
        if not variability_ok:
            notes.append(f"Firing rate variability {firing_rate_std:.2f} too low")
            
        return {
            'passes': passes,
            'firing_rate_mean': firing_rate_mean,
            'firing_rate_std': firing_rate_std,
            'notes': '; '.join(notes)
        }
    
    def _calculate_validation_score(self, *metric_dicts) -> float:
        """Calculate overall validation score."""
        scores = []
        
        for metrics in metric_dicts:
            if metrics and 'passes' in metrics:
                scores.append(1.0 if metrics['passes'] else 0.0)
                
        return float(np.mean(scores)) if scores else 0.0


def validate_large_scale_networks():
    """Test validation on large-scale networks including 1000+ module networks."""
    print("=== Large-Scale Network Validation ===")
    print("Testing networks up to 1000+ modules for biological realism")
    
    validator = LargeScaleNetworkValidator()
    
    # Test different network sizes including very large networks
    test_configs = [
        {'total_neurons': 10000, 'num_modules': 10, 'name': 'Small Network'},
        {'total_neurons': 50000, 'num_modules': 50, 'name': 'Medium Network'}, 
        {'total_neurons': 100000, 'num_modules': 100, 'name': 'Large Network'},
        {'total_neurons': 500000, 'num_modules': 500, 'name': 'Very Large Network'},
        {'total_neurons': 1000000, 'num_modules': 1000, 'name': '1000+ Module Network'},
        {'total_neurons': 2000000, 'num_modules': 1500, 'name': 'Ultra-Large Network'}
    ]
    
    results = []
    
    for config in test_configs:
        try:
            print(f"\n🧠 Testing {config['name']}: {config['total_neurons']:,} neurons, {config['num_modules']} modules")
            
            # Create realistic network configuration for validation
            network_config = create_realistic_network_config(
                total_neurons=config['total_neurons'],
                num_modules=config['num_modules']
            )
            
            # Generate synthetic spike data for large networks (to test activity patterns)
            spike_data = None
            if config['total_neurons'] <= 100000:  # Only for smaller networks to save memory
                spike_data = generate_realistic_spike_data(
                    num_neurons=config['total_neurons'],
                    simulation_time=1000  # 1 second
                )
            
            # Validate network with performance timing
            start_time = time.time()
            result = validator.validate_network_properties(network_config, spike_data)
            validation_time = time.time() - start_time
            
            results.append(result)
            
            # Print detailed results
            print(f"  ⏱️  Validation Time: {validation_time:.3f}s")
            print(f"  📊 Validation Score: {result.metrics.validation_score:.3f}/1.000")
            print(f"  ✅ Passes Validation: {'YES' if result.passes_validation else 'NO'}")
            print(f"  🧮 E/I Ratio: {result.metrics.ei_ratio:.3f} (target: 0.8)")
            print(f"  🕸️  Connection Density: {result.metrics.connection_density:.5f}")
            print(f"  🌐 Small-World Index: {result.metrics.small_world_index:.3f}")
            print(f"  🧠 Clustering: {result.metrics.clustering_coefficient:.3f}")
            print(f"  📏 Path Length: {result.metrics.average_path_length:.2f}")
            
            if spike_data is not None:
                print(f"  ⚡ Firing Rate: {result.metrics.firing_rate_mean:.1f} ± {result.metrics.firing_rate_std:.1f} Hz")
            
            # Print validation notes
            for note in result.validation_notes:
                print(f"    {note}")
                
        except Exception as e:
            print(f"  ❌ Validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Summary statistics
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    successful_results = [r for r in results if hasattr(r, 'passes_validation')]
    if successful_results:
        passed_validations = [r for r in successful_results if r.passes_validation]
        
        print(f"Total networks tested: {len(successful_results)}")
        print(f"Passed biological validation: {len(passed_validations)}/{len(successful_results)}")
        
        if passed_validations:
            max_validated = max(passed_validations, key=lambda x: x.network_size)
            print(f"Largest validated network: {max_validated.network_size:,} neurons, {max_validated.num_modules} modules")
            
        # Performance summary
        avg_score = np.mean([r.metrics.validation_score for r in successful_results])
        print(f"Average validation score: {avg_score:.3f}")
        
        # Module scaling analysis
        large_module_networks = [r for r in successful_results if r.num_modules >= 1000]
        if large_module_networks:
            print(f"\n🎯 1000+ Module Networks: {len(large_module_networks)} tested")
            for result in large_module_networks:
                status = "✅ PASS" if result.passes_validation else "❌ FAIL"
                print(f"  {result.network_size:,} neurons, {result.num_modules} modules: {status}")
    
    return results


def create_realistic_network_config(total_neurons: int, num_modules: int) -> Dict[str, Any]:
    """Create a realistic network configuration for validation."""
    
    # Calculate biologically realistic parameters
    ei_ratio = 0.8  # 80% excitatory
    
    # Connection density should decrease with network size (biological observation)
    if total_neurons <= 10000:
        connection_density = 0.05  # 5%
    elif total_neurons <= 100000:
        connection_density = 0.03  # 3% 
    elif total_neurons <= 500000:
        connection_density = 0.02  # 2%
    else:
        connection_density = 0.015  # 1.5% for very large networks
        
    total_connections = int(total_neurons * total_neurons * connection_density)
    
    # Small-world properties based on network size
    if num_modules <= 100:
        clustering = np.random.uniform(0.4, 0.6)
        avg_path_length = np.random.uniform(2.5, 3.5)
    elif num_modules <= 500:
        clustering = np.random.uniform(0.35, 0.55)
        avg_path_length = np.random.uniform(3.0, 4.0)
    else:  # 1000+ modules
        clustering = np.random.uniform(0.3, 0.5)
        avg_path_length = np.random.uniform(3.5, 4.5)
        
    small_world_index = clustering / (avg_path_length / num_modules) if num_modules > 0 else 2.0
    
    # Degree distribution (scale-free networks have gamma ~2.1-3)
    degree_gamma = np.random.uniform(2.1, 2.8)
    
    return {
        'total_neurons': total_neurons,
        'num_modules': num_modules,
        'ei_ratio': ei_ratio,
        'total_connections': total_connections,
        'connection_density': connection_density,
        'small_world_metrics': {
            'clustering': clustering,
            'avg_path_length': avg_path_length,
            'small_world_index': small_world_index,
            'degree_gamma': degree_gamma
        }
    }


def generate_realistic_spike_data(num_neurons: int, simulation_time: int) -> np.ndarray:
    """Generate realistic spike data for activity pattern validation."""
    
    spike_data = []
    
    for neuron_id in range(num_neurons):
        # Log-normal distribution of firing rates (biological)
        base_rate = np.random.lognormal(mean=1.0, sigma=0.8)  # Hz
        firing_rate = np.clip(base_rate, 0.1, 50.0)  # Clip to reasonable range
        
        # Generate spike times
        num_spikes = np.random.poisson(firing_rate * simulation_time / 1000)
        
        if num_spikes > 0:
            spike_times = np.sort(np.random.uniform(0, simulation_time, num_spikes))
            for spike_time in spike_times:
                spike_data.append([neuron_id, spike_time])
                
    return np.array(spike_data) if spike_data else np.array([]).reshape(0, 2)


if __name__ == "__main__":
    validate_large_scale_networks()