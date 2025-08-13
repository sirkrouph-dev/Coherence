"""
Benchmark Results Visualization
================================

This module provides visualization tools for analyzing benchmark results
and generating plots for reports.
"""

import os
import csv
import json
import glob
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def load_benchmark_results(results_dir: str = "benchmark_results") -> List[Dict]:
    """Load all benchmark results from CSV files."""
    csv_files = glob.glob(os.path.join(results_dir, "benchmark_results_*.csv"))
    
    all_results = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                row['network_size'] = int(row['network_size'])
                row['wall_clock_time_s'] = float(row['wall_clock_time_s'])
                row['memory_peak_mb'] = float(row['memory_peak_mb'])
                row['memory_average_mb'] = float(row['memory_average_mb'])
                row['total_spikes'] = int(row['total_spikes'])
                row['spike_rate_hz'] = float(row['spike_rate_hz'])
                row['spike_throughput'] = float(row['spike_throughput'])
                row['success'] = row['success'].lower() == 'true'
                all_results.append(row)
    
    return all_results


def create_performance_plots(results: List[Dict], output_dir: str = "benchmark_plots"):
    """Create comprehensive performance plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful results
    results = [r for r in results if r['success']]
    
    if not results:
        print("No successful benchmark results to plot")
        return
    
    # Group by network size and platform
    grouped = {}
    for r in results:
        size = r['network_size']
        platform = r['platform']
        if size not in grouped:
            grouped[size] = {}
        if platform not in grouped[size]:
            grouped[size][platform] = []
        grouped[size][platform].append(r)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define colors for platforms
    colors = {
        'cpu': '#1f77b4',
        'gpu_torch': '#ff7f0e',
        'gpu_cupy': '#2ca02c'
    }
    
    # Plot 1: Wall-clock time vs network size
    ax1 = fig.add_subplot(gs[0, 0])
    for platform in colors:
        sizes = []
        times = []
        for size in sorted(grouped.keys()):
            if platform in grouped[size]:
                sizes.append(size)
                avg_time = np.mean([r['wall_clock_time_s'] for r in grouped[size][platform]])
                times.append(avg_time)
        if sizes:
            ax1.plot(sizes, times, 'o-', label=platform, color=colors[platform], markersize=8)
    ax1.set_xlabel('Network Size (neurons)')
    ax1.set_ylabel('Wall-clock Time (seconds)')
    ax1.set_title('Simulation Time vs Network Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Memory usage vs network size
    ax2 = fig.add_subplot(gs[0, 1])
    for platform in colors:
        sizes = []
        memory = []
        for size in sorted(grouped.keys()):
            if platform in grouped[size]:
                sizes.append(size)
                avg_mem = np.mean([r['memory_peak_mb'] for r in grouped[size][platform]])
                memory.append(avg_mem)
        if sizes:
            ax2.plot(sizes, memory, 's-', label=platform, color=colors[platform], markersize=8)
    ax2.set_xlabel('Network Size (neurons)')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('Memory Usage vs Network Size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Spike throughput vs network size
    ax3 = fig.add_subplot(gs[0, 2])
    for platform in colors:
        sizes = []
        throughput = []
        for size in sorted(grouped.keys()):
            if platform in grouped[size]:
                sizes.append(size)
                avg_throughput = np.mean([r['spike_throughput'] for r in grouped[size][platform]])
                throughput.append(avg_throughput)
        if sizes:
            ax3.plot(sizes, throughput, '^-', label=platform, color=colors[platform], markersize=8)
    ax3.set_xlabel('Network Size (neurons)')
    ax3.set_ylabel('Spike Throughput (spikes/second)')
    ax3.set_title('Spike Processing Throughput')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Bar chart comparing platforms for each size
    ax4 = fig.add_subplot(gs[1, :])
    network_sizes = sorted(grouped.keys())
    platforms = list(colors.keys())
    x = np.arange(len(network_sizes))
    width = 0.25
    
    for i, platform in enumerate(platforms):
        times = []
        for size in network_sizes:
            if platform in grouped[size]:
                avg_time = np.mean([r['wall_clock_time_s'] for r in grouped[size][platform]])
                times.append(avg_time)
            else:
                times.append(0)
        ax4.bar(x + i * width, times, width, label=platform, color=colors[platform])
    
    ax4.set_xlabel('Network Size (neurons)')
    ax4.set_ylabel('Wall-clock Time (seconds)')
    ax4.set_title('Performance Comparison Across Platforms')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([f'{size:,}' for size in network_sizes])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Speedup factors (if GPU results available)
    ax5 = fig.add_subplot(gs[2, 0])
    gpu_platforms = ['gpu_torch', 'gpu_cupy']
    for gpu_platform in gpu_platforms:
        sizes = []
        speedups = []
        for size in sorted(grouped.keys()):
            if 'cpu' in grouped[size] and gpu_platform in grouped[size]:
                cpu_time = np.mean([r['wall_clock_time_s'] for r in grouped[size]['cpu']])
                gpu_time = np.mean([r['wall_clock_time_s'] for r in grouped[size][gpu_platform]])
                if gpu_time > 0:
                    sizes.append(size)
                    speedups.append(cpu_time / gpu_time)
        if sizes:
            ax5.plot(sizes, speedups, 'o-', label=f'{gpu_platform} vs CPU', 
                    markersize=8, linewidth=2)
    
    ax5.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Network Size (neurons)')
    ax5.set_ylabel('Speedup Factor')
    ax5.set_title('GPU Speedup vs CPU')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)
    if ax5.get_lines():  # Only add legend if there are lines
        ax5.legend()
    
    # Plot 6: Memory efficiency (spikes per MB)
    ax6 = fig.add_subplot(gs[2, 1])
    for platform in colors:
        sizes = []
        efficiency = []
        for size in sorted(grouped.keys()):
            if platform in grouped[size]:
                sizes.append(size)
                avg_spikes = np.mean([r['total_spikes'] for r in grouped[size][platform]])
                avg_mem = np.mean([r['memory_peak_mb'] for r in grouped[size][platform]])
                if avg_mem > 0:
                    efficiency.append(avg_spikes / avg_mem)
        if sizes:
            ax6.plot(sizes, efficiency, 'd-', label=platform, color=colors[platform], markersize=8)
    ax6.set_xlabel('Network Size (neurons)')
    ax6.set_ylabel('Spikes per MB')
    ax6.set_title('Memory Efficiency')
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Scaling efficiency
    ax7 = fig.add_subplot(gs[2, 2])
    base_size = min(grouped.keys())
    for platform in colors:
        sizes = []
        scaling = []
        if platform in grouped.get(base_size, {}):
            base_time = np.mean([r['wall_clock_time_s'] for r in grouped[base_size][platform]])
            for size in sorted(grouped.keys()):
                if platform in grouped[size]:
                    sizes.append(size)
                    current_time = np.mean([r['wall_clock_time_s'] for r in grouped[size][platform]])
                    expected_time = base_time * (size / base_size)
                    actual_scaling = current_time / base_time
                    ideal_scaling = size / base_size
                    efficiency = ideal_scaling / actual_scaling if actual_scaling > 0 else 0
                    scaling.append(efficiency * 100)  # Convert to percentage
        if sizes:
            ax7.plot(sizes, scaling, 'p-', label=platform, color=colors[platform], markersize=8)
    
    ax7.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal')
    ax7.set_xlabel('Network Size (neurons)')
    ax7.set_ylabel('Scaling Efficiency (%)')
    ax7.set_title('Scaling Efficiency (Linear = 100%)')
    ax7.set_xscale('log')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Add overall title
    fig.suptitle('Neuromorphic Network Performance Benchmarks', fontsize=16, fontweight='bold')
    
    # Save plot
    output_file = os.path.join(output_dir, 'benchmark_analysis.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Performance plots saved to: {output_file}")
    
    # Also save as PDF for reports
    pdf_file = os.path.join(output_dir, 'benchmark_analysis.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_file}")
    
    plt.close()


def create_summary_table(results: List[Dict], output_dir: str = "benchmark_plots"):
    """Create a summary table of benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful results
    results = [r for r in results if r['success']]
    
    if not results:
        print("No successful benchmark results to summarize")
        return
    
    # Group by network size and platform
    summary = []
    grouped = {}
    for r in results:
        key = (r['network_size'], r['platform'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    
    # Calculate statistics
    for (size, platform), runs in grouped.items():
        avg_time = np.mean([r['wall_clock_time_s'] for r in runs])
        std_time = np.std([r['wall_clock_time_s'] for r in runs])
        avg_memory = np.mean([r['memory_peak_mb'] for r in runs])
        avg_throughput = np.mean([r['spike_throughput'] for r in runs])
        
        summary.append({
            'Network Size': f'{size:,}',
            'Platform': platform,
            'Avg Time (s)': f'{avg_time:.2f} ± {std_time:.2f}',
            'Avg Memory (MB)': f'{avg_memory:.1f}',
            'Avg Throughput': f'{avg_throughput:.0f}',
            'Runs': len(runs)
        })
    
    # Sort by network size and platform
    summary.sort(key=lambda x: (int(x['Network Size'].replace(',', '')), x['Platform']))
    
    # Create matplotlib table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    columns = list(summary[0].keys())
    cell_text = [[row[col] for col in columns] for row in summary]
    
    # Create table
    table = ax.table(cellText=cell_text, colLabels=columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.title('Benchmark Results Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Save table
    output_file = os.path.join(output_dir, 'benchmark_summary_table.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Summary table saved to: {output_file}")
    
    plt.close()


def main():
    """Main entry point for visualization."""
    print("Loading benchmark results...")
    results = load_benchmark_results()
    
    if not results:
        print("No benchmark results found in 'benchmark_results' directory")
        return
    
    print(f"Found {len(results)} benchmark results")
    
    print("Creating performance plots...")
    create_performance_plots(results)
    
    print("Creating summary table...")
    create_summary_table(results)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
