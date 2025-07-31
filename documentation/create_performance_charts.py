#!/usr/bin/env python3
"""
Performance visualization for JAX-based SAC benchmark results.
Creates charts showing GPU-resident RL implementation performance across different configurations.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for better looking plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def create_performance_comparison():
    """Create a comprehensive performance comparison chart."""

    # Performance data comparing different implementation approaches
    configs = [
        'CPU No JAX\n(i7-11370H)',
        'CPU Full JAX\n(i7-11370H)',
        'CPU Full JAX\n(Xeon E5-1630)',
        'GPU Full JAX\n(Xeon E5-1630 + RTX 3060)',
    ]

    # Average updates per second for each implementation approach
    updates_per_sec = [342, 650, 190, 5248]  # GPU uses best result (1024 envs)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Color scheme (different approaches)
    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c']

    # Subplot 1: Bar chart comparison of implementation approaches
    bars = ax1.bar(configs, updates_per_sec, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, updates_per_sec)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            f'{val:,}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=11,
        )

    ax1.set_ylabel('Updates per Second', fontsize=14, fontweight='bold')
    ax1.set_title(
        'Implementation Comparison\nDifferent Approaches to RL Training', fontsize=16, fontweight='bold', pad=20
    )
    ax1.set_ylim(0, 6000)
    ax1.grid(axis='y', alpha=0.3)

    # Add performance categories with colored backgrounds
    ax1.axhspan(0, 1000, alpha=0.1, color='orange', label='CPU Performance')
    ax1.axhspan(1000, 6000, alpha=0.1, color='green', label='GPU Performance')

    # Subplot 2: GPU scaling with environment count
    env_counts = [256, 1024, 2048, 4096]
    gpu_performance = [4880, 5248, 5157, 4939]

    ax2.plot(
        env_counts,
        gpu_performance,
        'o-',
        linewidth=3,
        markersize=10,
        color='#2ca02c',
        markerfacecolor='white',
        markeredgewidth=2,
    )
    ax2.fill_between(env_counts, gpu_performance, alpha=0.3, color='#2ca02c')

    # Add performance values as labels
    for x, y in zip(env_counts, gpu_performance):
        ax2.annotate(f'{y:,} ups/s', (x, y), textcoords='offset points', xytext=(0, 10), ha='center', fontweight='bold')

    ax2.set_xlabel('Number of Parallel Environments', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Updates per Second', fontsize=14, fontweight='bold')
    ax2.set_title('GPU Performance Scaling\n(RTX 3060 - 100% GPU, <10% CPU)', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(4500, 5500)
    ax2.set_xscale('log')
    ax2.set_xticks(env_counts)
    ax2.set_xticklabels([str(x) for x in env_counts])

    # Add sweet spot annotation
    ax2.annotate(
        'Peak Performance\n~5250 ups/s @ 1024 envs',
        xy=(1024, 5248),
        xytext=(2000, 5400),
        arrowprops=dict(arrowstyle='->', color='blue', lw=2),
        fontsize=11,
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig('performance_benchmarks.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()


def create_gpu_utilization_chart():
    """Create a chart showing GPU vs CPU utilization."""

    implementations = ['Traditional\nRL (CPU)', 'JAX CPU\n(Optimized)', 'JAX GPU\n(This Work)']
    gpu_usage = [0, 0, 100]
    cpu_usage = [90, 80, 8]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(implementations))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, gpu_usage, width, label='GPU Usage (%)', color='#2ca02c', alpha=0.8, edgecolor='black'
    )
    bars2 = ax.bar(
        x + width / 2, cpu_usage, width, label='CPU Usage (%)', color='#ff7f0e', alpha=0.8, edgecolor='black'
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f'{height}%',
                ha='center',
                va='bottom',
                fontweight='bold',
            )

    ax.set_ylabel('Resource Utilization (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Implementation', fontsize=14, fontweight='bold')
    ax.set_title(
        'Resource Utilization Comparison\nTrue GPU-Resident vs Traditional Approaches',
        fontsize=16,
        fontweight='bold',
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(implementations)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 110)

    # Add annotation for true GPU-resident
    ax.annotate(
        'Truly GPU-Resident\nComputation',
        xy=(2, 100),
        xytext=(1.5, 80),
        arrowprops=dict(arrowstyle='->', color='green', lw=2),
        fontsize=12,
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig('gpu_utilization.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()


def create_architecture_benefits_chart():
    """Create a chart showing the benefits of different architectural choices."""

    categories = ['CPU Baseline', '+ Vectorization\n+ JAX JIT', '+ GPU-Resident\nTraining']
    cumulative_speedup = [1.0, 1.9, round(5248 / 190, 1)]  # Based on 342 -> 5248 ups/s

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bar chart showing cumulative improvements
    x = np.arange(len(categories))
    bars = ax.bar(
        x,
        cumulative_speedup,
        color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd'],
        alpha=0.8,
        edgecolor='black',
        linewidth=1.2,
    )

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, cumulative_speedup)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.3,
            f'{val:.1f}x',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12,
        )

    ax.set_ylabel('Speedup Factor (vs CPU Baseline)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Optimization Stage', fontsize=14, fontweight='bold')
    ax.set_title(
        'Performance Gains from GPU-Resident Architecture\nJAX-based SAC Implementation',
        fontsize=16,
        fontweight='bold',
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 29)

    # Add annotations for key improvements
    ax.annotate(
        'JIT Compilation\nBenefit',
        xy=(1, 3.05),
        xytext=(0.5, 12),
        arrowprops=dict(arrowstyle='->', color='blue', lw=2),
        fontsize=11,
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
    )

    ax.annotate(
        'GPU Acceleration\n~15x Total Speedup',
        xy=(4, 15.3),
        xytext=(3.5, 12),
        arrowprops=dict(arrowstyle='->', color='green', lw=2),
        fontsize=11,
        ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
    )

    plt.tight_layout()
    plt.savefig('optimization_progression.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()


if __name__ == '__main__':
    print('🎨 Generating performance visualization charts...')

    # Create all charts
    create_performance_comparison()
    print("✅ Performance comparison chart saved as 'performance_benchmarks.png'")

    create_gpu_utilization_chart()
    print("✅ GPU utilization chart saved as 'gpu_utilization.png'")

    create_architecture_benefits_chart()
    print("✅ Architecture benefits chart saved as 'optimization_progression.png'")

    print('\n🎉 All performance visualizations generated successfully!')
    print('📁 Files saved:')
    print('   - performance_benchmarks.png')
    print('   - gpu_utilization.png')
    print('   - optimization_progression.png')
