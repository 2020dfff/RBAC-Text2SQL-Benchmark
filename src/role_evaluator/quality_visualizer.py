"""
Role Quality Visualization Suite

Provides visualization utilities for role generation quality metrics:
- Deny rate distribution charts
- Role-schema semantic alignment plots
- Policy overlap heatmaps
- Comprehensive quality dashboard

Usage:
    from src.role_evaluator.quality_visualizer import RoleQualityVisualizer
    
    visualizer = RoleQualityVisualizer(evaluator)
    
    # Individual plots
    visualizer.plot_deny_rate_distribution()
    visualizer.plot_semantic_alignment()
    visualizer.plot_policy_overlap_heatmap()
    
    # Comprehensive dashboard
    visualizer.plot_quality_dashboard()
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib import colormaps
from collections import defaultdict
from typing import Dict, List, Optional, Any
import random

from .quality_metrics import (
    RoleQualityEvaluator, 
    QualityReport,
    DenyRateMetrics,
    SemanticMetrics,
    OverlapMetrics
)


class RoleQualityVisualizer:
    """Visualization suite for role generation quality metrics."""
    
    def __init__(
        self, 
        evaluator: Optional[RoleQualityEvaluator] = None,
        report: Optional[QualityReport] = None
    ):
        """
        Initialize visualizer with evaluator or pre-computed report.
        
        Args:
            evaluator: RoleQualityEvaluator instance
            report: Pre-computed QualityReport
        """
        self.evaluator = evaluator
        self.report = report or (evaluator.evaluate_all() if evaluator else None)
    
    def plot_deny_rate_distribution(
        self, 
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot deny rate distribution charts.
        
        Creates two subplots:
        - Left: Overall allow/deny distribution (entry-level)
        - Right: Per-database deny rate comparison
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if not self.report or not self.report.deny_rate_metrics:
            raise ValueError("No deny rate metrics available.")
        
        metrics = self.report.deny_rate_metrics
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Overall distribution
        ax1 = axes[0]
        total_allowed = sum(m.allowed_count for m in metrics.values())
        total_denied = sum(m.denied_count for m in metrics.values())
        
        bars = ax1.bar(
            ['Allowed', 'Denied'], 
            [total_allowed, total_denied],
            color=['#4CAF50', '#F44336']
        )
        ax1.set_ylabel('Number of Entries')
        ax1.set_title('Overall Allow/Deny Distribution (Entry-Level)')
        
        total = total_allowed + total_denied
        for bar, count in zip(bars, [total_allowed, total_denied]):
            ax1.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + total * 0.01,
                f'{count}\n({count/total*100:.1f}%)', 
                ha='center', va='bottom'
            )
        
        # Right: Per-database deny rate
        ax2 = axes[1]
        db_names = sorted(metrics.keys())
        deny_rates = [metrics[db].deny_rate for db in db_names]
        
        colors = ['#F44336' if metrics[db].issues else '#4CAF50' for db in db_names]
        bars = ax2.bar(db_names, deny_rates, color=colors)
        
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Min threshold (5%)')
        ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Max threshold (90%)')
        
        ax2.set_ylabel('Deny Rate (%)')
        ax2.set_title('Per-Database Deny Rate')
        ax2.set_xticklabels(db_names, rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_semantic_alignment(
        self,
        figsize: tuple = None,
        save_path: Optional[str] = None,
        random_seed: int = 13
    ) -> plt.Figure:
        """
        Plot role-schema semantic alignment visualization.
        
        Similar to role_semantic_analysis.py visualization:
        - X-axis: Semantic similarity score
        - Y-axis: Database
        - Circle size: Column coverage ratio
        - Color: Similarity intensity
        - Triangle: SystemManager roles
        
        Args:
            figsize: Figure size tuple (auto-calculated if None)
            save_path: Optional path to save the figure
            random_seed: Random seed for jitter consistency
            
        Returns:
            matplotlib Figure object
        """
        if not self.evaluator:
            raise ValueError("Evaluator required for semantic alignment plot.")
        
        random.seed(random_seed)
        
        # Get role records with computed similarities
        self.evaluator._compute_embeddings()
        role_specs = self.evaluator._prepare_role_records()
        
        # Build role records with similarities
        db_to_entries = defaultdict(list)
        similarities = []
        
        for spec in role_specs:
            schema_emb = self.evaluator._schema_emb_dict.get(spec["schema_text"])
            role_emb = self.evaluator._role_emb_dict.get(spec["role_text"])
            
            raw_sim = self.evaluator.cosine_similarity(role_emb, schema_emb)
            similarity = max(0.0, min(1.0, (raw_sim + 1.0) / 2.0))
            similarities.append(similarity)
            
            column_coverage = spec["accessible_columns"] / spec["total_columns"] if spec["total_columns"] > 0 else 0
            size = 90.0 + column_coverage * 220.0
            
            db_to_entries[spec["db_id"]].append({
                "similarity": similarity,
                "size": size,
                "is_system_manager": spec["is_system_manager"],
                "role": spec["role"],
                "column_coverage": column_coverage
            })
        
        # Setup plot
        db_ids = sorted(db_to_entries.keys())
        min_sim, max_sim = min(similarities), max(similarities)
        if max_sim - min_sim < 1e-6:
            max_sim = min_sim + 1e-6
        
        norm = PowerNorm(gamma=0.55, vmin=min_sim, vmax=max_sim)
        cmap = colormaps["viridis"]
        
        # Auto-size figure
        if figsize is None:
            figsize = (12, max(6, len(db_ids) * 0.5))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_ticks = []
        y_labels = []
        
        for row_idx, db_id in enumerate(db_ids):
            y = float(row_idx)
            y_ticks.append(y)
            y_labels.append(db_id)
            ax.axhline(y, color="#d0d0d0", linestyle="--", linewidth=0.9, zorder=0)
            
            sim_span = max_sim - min_sim if max_sim > min_sim else 1.0
            
            for entry in db_to_entries[db_id]:
                similarity = entry["similarity"]
                base_x = (similarity - min_sim) / sim_span
                jittered_x = base_x + random.uniform(-0.012, 0.012)
                
                if entry["is_system_manager"]:
                    ax.scatter(
                        jittered_x, y, s=200, c="#d62728", alpha=0.9,
                        edgecolors="#8c1c15", linewidths=0.5, marker="^", zorder=3
                    )
                else:
                    jittered_y = y + random.uniform(-0.16, 0.16)
                    ax.scatter(
                        jittered_x, jittered_y, s=entry["size"] * 2,
                        c=[similarity], cmap=cmap, norm=norm, alpha=0.82,
                        edgecolors="#333333", linewidths=0.45, marker="o"
                    )
        
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=12, weight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.6, len(db_ids) - 1 + 0.6)
        
        # X-axis with actual similarity values
        tick_positions = np.linspace(0.0, 1.0, 5)
        sim_span = max_sim - min_sim
        tick_labels = [f"{(min_sim + pos * sim_span):.2f}" for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=11)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.45)
        
        ax.set_xlabel("Semantic Similarity (Role ↔ Schema)", fontsize=14, weight='bold')
        ax.set_ylabel("Database", fontsize=14, weight='bold')
        ax.set_title("Role-Schema Semantic Alignment", fontsize=16, weight='bold')
        
        # Colorbar
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_map.set_array([])
        cbar = fig.colorbar(scalar_map, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Similarity", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_policy_overlap_bars(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot policy overlap as bar chart.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if not self.report or not self.report.overlap_metrics:
            raise ValueError("No overlap metrics available.")
        
        metrics = self.report.overlap_metrics
        
        fig, ax = plt.subplots(figsize=figsize)
        
        db_names = sorted(metrics.keys())
        avg_overlaps = [metrics[db].avg_overlap * 100 for db in db_names]
        max_overlaps = [metrics[db].max_overlap * 100 for db in db_names]
        
        x = np.arange(len(db_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, avg_overlaps, width, label='Avg Overlap', color='#2196F3')
        bars2 = ax.bar(x + width/2, max_overlaps, width, label='Max Overlap', color='#FF9800', alpha=0.7)
        
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Warning threshold (70%)')
        
        ax.set_ylabel('Policy Overlap (%)')
        ax.set_title('Role Policy Overlap by Database')
        ax.set_xticks(x)
        ax.set_xticklabels(db_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_quality_dashboard(
        self,
        figsize: tuple = (16, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive quality dashboard with all metrics.
        
        Creates a 2x2 grid:
        - Top-left: Deny rate distribution
        - Top-right: Per-database deny rates
        - Bottom-left: Semantic alignment summary
        - Bottom-right: Policy overlap
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if not self.report:
            raise ValueError("No report available.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Top-left: Overall allow/deny
        ax1 = axes[0, 0]
        if self.report.deny_rate_metrics:
            metrics = self.report.deny_rate_metrics
            total_allowed = sum(m.allowed_count for m in metrics.values())
            total_denied = sum(m.denied_count for m in metrics.values())
            
            bars = ax1.bar(['Allowed', 'Denied'], [total_allowed, total_denied],
                          color=['#4CAF50', '#F44336'])
            ax1.set_ylabel('Number of Entries')
            ax1.set_title('Overall Allow/Deny Distribution')
            
            total = total_allowed + total_denied
            for bar, count in zip(bars, [total_allowed, total_denied]):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{count}\n({count/total*100:.1f}%)', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No deny rate data', ha='center', va='center')
            ax1.set_title('Overall Allow/Deny Distribution')
        
        # Top-right: Per-database deny rates
        ax2 = axes[0, 1]
        if self.report.deny_rate_metrics:
            db_names = sorted(metrics.keys())
            deny_rates = [metrics[db].deny_rate for db in db_names]
            colors = ['#F44336' if metrics[db].issues else '#4CAF50' for db in db_names]
            
            x_pos = range(len(db_names))
            ax2.bar(x_pos, deny_rates, color=colors)
            ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7)
            ax2.axhline(y=90, color='red', linestyle='--', alpha=0.7)
            ax2.set_ylabel('Deny Rate (%)')
            ax2.set_title('Per-Database Deny Rate')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(db_names, rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'No deny rate data', ha='center', va='center')
            ax2.set_title('Per-Database Deny Rate')
        
        # Bottom-left: Semantic quality summary
        ax3 = axes[1, 0]
        if self.report.semantic_metrics:
            sem_metrics = self.report.semantic_metrics
            db_names = sorted(sem_metrics.keys())
            similarities = [sem_metrics[db].avg_similarity for db in db_names]
            colors = ['#F44336' if sem_metrics[db].issues else '#4CAF50' for db in db_names]
            
            ax3.barh(db_names, similarities, color=colors)
            ax3.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Min threshold')
            ax3.set_xlabel('Avg Semantic Similarity')
            ax3.set_title('Role-Schema Semantic Alignment')
            ax3.set_xlim(0, 1)
        else:
            ax3.text(0.5, 0.5, 'No semantic data', ha='center', va='center')
            ax3.set_title('Role-Schema Semantic Alignment')
        
        # Bottom-right: Policy overlap
        ax4 = axes[1, 1]
        if self.report.overlap_metrics:
            ovl_metrics = self.report.overlap_metrics
            db_names = sorted(ovl_metrics.keys())
            overlaps = [ovl_metrics[db].avg_overlap * 100 for db in db_names]
            colors = ['#F44336' if ovl_metrics[db].issues else '#4CAF50' for db in db_names]
            
            ax4.barh(db_names, overlaps, color=colors)
            ax4.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='Max threshold')
            ax4.set_xlabel('Avg Policy Overlap (%)')
            ax4.set_title('Role Policy Overlap')
            ax4.set_xlim(0, 100)
        else:
            ax4.text(0.5, 0.5, 'No overlap data', ha='center', va='center')
            ax4.set_title('Role Policy Overlap')
        
        # Overall title
        pass_rate = self.report.pass_rate
        status = "PASS" if pass_rate >= 0.8 else "WARN" if pass_rate >= 0.5 else "FAIL"
        fig.suptitle(
            f'[{status}] Role Generation Quality Dashboard (Pass Rate: {pass_rate:.1%})',
            fontsize=16, weight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def print_summary(self):
        """Print text summary of quality metrics."""
        if not self.report:
            print("No report available.")
            return
        
        from .quality_metrics import print_quality_report
        print_quality_report(self.report, verbose=True)
