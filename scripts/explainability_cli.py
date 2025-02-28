#!/usr/bin/env python3
"""
Explainability CLI

Command-line interface for explainability analysis including
feature importance, SHAP explanations, and what-if analysis.
"""

import argparse
import logging
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from airaware.explainability import (
    FeatureImportanceAnalyzer, FeatureImportanceConfig,
    SHAPExplainer, SHAPConfig,
    WhatIfAnalyzer, WhatIfConfig,
    plot_feature_importance, plot_shap_summary, plot_what_if_scenarios,
    create_explainability_report, evaluate_explainability_quality
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_feature_importance(args):
    """Analyze feature importance"""
    logger.info("🔧 Analyzing feature importance...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        numeric_cols = ['hour', 'day_of_week', 'month']
        for col in numeric_cols:
            if col not in train_data.columns:
                if col == 'hour':
                    train_data[col] = train_data['datetime_utc'].dt.hour
                    test_data[col] = test_data['datetime_utc'].dt.hour
                elif col == 'day_of_week':
                    train_data[col] = train_data['datetime_utc'].dt.dayofweek
                    test_data[col] = test_data['datetime_utc'].dt.dayofweek
                elif col == 'month':
                    train_data[col] = train_data['datetime_utc'].dt.month
                    test_data[col] = test_data['datetime_utc'].dt.month
    
    X_train = train_data[numeric_cols].values
    y_train = train_data[args.target_col].values
    X_test = test_data[numeric_cols].values
    y_test = test_data[args.target_col].values
    
    # Train model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Configure feature importance analyzer
    config = FeatureImportanceConfig(
        use_permutation_importance=args.use_permutation,
        use_tree_importance=args.use_tree,
        use_correlation_analysis=args.use_correlation,
        use_mutual_information=args.use_mutual_info,
        use_stability_analysis=args.use_stability,
        top_k_features=args.top_k
    )
    
    # Create and fit analyzer
    analyzer = FeatureImportanceAnalyzer(config)
    start_time = time.time()
    analyzer.fit(model, X_train, y_train, numeric_cols)
    analysis_time = time.time() - start_time
    
    # Get results
    importance_summary = analyzer.get_importance_summary()
    
    # Save results
    results = {
        'model_type': 'Feature Importance Analysis',
        'config': config.__dict__,
        'analysis_time': analysis_time,
        'importance_summary': importance_summary,
        'analyzer_info': analyzer.get_analyzer_info()
    }
    
    results_path = os.path.join(args.output_dir, 'feature_importance_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed results
    analyzer.save_results(os.path.join(args.output_dir, 'feature_importance_detailed.json'))
    
    # Create plots
    if args.create_plots:
        for method in analyzer.importance_results.keys():
            # Skip temporal importance as it has a different structure
            if method == 'temporal':
                continue
            plot_path = os.path.join(args.output_dir, f'feature_importance_{method}.png')
            plot_feature_importance(analyzer.importance_results, method, args.top_k, plot_path)
    
    logger.info(f"✅ Feature importance analysis completed!")
    logger.info(f"Analysis time: {analysis_time:.2f} seconds")
    logger.info(f"Methods used: {list(analyzer.importance_results.keys())}")
    logger.info(f"Results saved to: {results_path}")

def analyze_shap_explanations(args):
    """Analyze SHAP explanations"""
    logger.info("🔧 Analyzing SHAP explanations...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        numeric_cols = ['hour', 'day_of_week', 'month']
        for col in numeric_cols:
            if col not in train_data.columns:
                if col == 'hour':
                    train_data[col] = train_data['datetime_utc'].dt.hour
                    test_data[col] = test_data['datetime_utc'].dt.hour
                elif col == 'day_of_week':
                    train_data[col] = train_data['datetime_utc'].dt.dayofweek
                    test_data[col] = test_data['datetime_utc'].dt.dayofweek
                elif col == 'month':
                    train_data[col] = train_data['datetime_utc'].dt.month
                    test_data[col] = test_data['datetime_utc'].dt.month
    
    X_train = train_data[numeric_cols].values
    y_train = train_data[args.target_col].values
    X_test = test_data[numeric_cols].values
    y_test = test_data[args.target_col].values
    
    # Train model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Configure SHAP explainer
    config = SHAPConfig(
        explainer_type=args.explainer_type,
        max_samples=args.max_samples,
        explanation_type=args.explanation_type,
        top_k_features=args.top_k
    )
    
    # Create and fit explainer
    explainer = SHAPExplainer(config)
    start_time = time.time()
    explainer.fit(model, X_train, y_train, numeric_cols)
    analysis_time = time.time() - start_time
    
    # Get explanations
    explanation_summary = explainer.get_explanation_summary()
    
    # Explain specific instances
    instance_explanations = []
    for i in range(min(5, len(X_test))):
        explanation = explainer.explain_instance(X_test[i], i)
        instance_explanations.append(explanation)
    
    # Get global explanation
    global_explanation = explainer.explain_global()
    
    # Save results
    results = {
        'model_type': 'SHAP Explanation Analysis',
        'config': config.__dict__,
        'analysis_time': analysis_time,
        'explanation_summary': explanation_summary,
        'global_explanation': global_explanation,
        'instance_explanations': instance_explanations,
        'explainer_info': explainer.get_explainer_info()
    }
    
    results_path = os.path.join(args.output_dir, 'shap_explanations_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save explainer
    explainer.save_explainer(os.path.join(args.output_dir, 'shap_explainer.json'))
    
    # Create plots
    if args.create_plots and explainer.shap_values is not None:
        plot_path = os.path.join(args.output_dir, 'shap_summary.png')
        plot_shap_summary(explainer.shap_values, numeric_cols, X_test, plot_path)
    
    logger.info(f"✅ SHAP explanation analysis completed!")
    logger.info(f"Analysis time: {analysis_time:.2f} seconds")
    logger.info(f"Explainer type: {config.explainer_type}")
    logger.info(f"Results saved to: {results_path}")

def analyze_what_if_scenarios(args):
    """Analyze what-if scenarios"""
    logger.info("🔧 Analyzing what-if scenarios...")
    
    # Load data
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    data = pd.read_parquet(args.data_path)
    logger.info(f"Loaded data: {len(data)} records")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Split data
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['datetime_utc', args.target_col]]
    numeric_cols = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_data[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        numeric_cols = ['hour', 'day_of_week', 'month']
        for col in numeric_cols:
            if col not in train_data.columns:
                if col == 'hour':
                    train_data[col] = train_data['datetime_utc'].dt.hour
                    test_data[col] = test_data['datetime_utc'].dt.hour
                elif col == 'day_of_week':
                    train_data[col] = train_data['datetime_utc'].dt.dayofweek
                    test_data[col] = test_data['datetime_utc'].dt.dayofweek
                elif col == 'month':
                    train_data[col] = train_data['datetime_utc'].dt.month
                    test_data[col] = test_data['datetime_utc'].dt.month
    
    X_train = train_data[numeric_cols].values
    y_train = train_data[args.target_col].values
    X_test = test_data[numeric_cols].values
    y_test = test_data[args.target_col].values
    
    # Train model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Configure what-if analyzer
    config = WhatIfConfig(
        n_scenarios=args.n_scenarios,
        scenario_type=args.scenario_type,
        perturbation_method=args.perturbation_method,
        perturbation_range=args.perturbation_range,
        use_sensitivity_analysis=args.use_sensitivity,
        use_counterfactual=args.use_counterfactual
    )
    
    # Create and fit analyzer
    analyzer = WhatIfAnalyzer(config)
    start_time = time.time()
    analyzer.fit(model, X_train, y_train, numeric_cols)
    analysis_time = time.time() - start_time
    
    # Get results
    scenario_summary = analyzer.get_scenario_summary()
    sensitivity_summary = analyzer.get_sensitivity_summary()
    counterfactual_summary = analyzer.get_counterfactual_summary()
    
    # Save results
    results = {
        'model_type': 'What-If Scenario Analysis',
        'config': config.__dict__,
        'analysis_time': analysis_time,
        'scenario_summary': scenario_summary,
        'sensitivity_summary': sensitivity_summary,
        'counterfactual_summary': counterfactual_summary,
        'analyzer_info': analyzer.get_analyzer_info()
    }
    
    results_path = os.path.join(args.output_dir, 'what_if_scenarios_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed results
    analyzer.save_results(os.path.join(args.output_dir, 'what_if_scenarios_detailed.json'))
    
    # Create plots
    if args.create_plots:
        plot_path = os.path.join(args.output_dir, 'what_if_scenarios.png')
        plot_what_if_scenarios(scenario_summary, plot_path)
    
    logger.info(f"✅ What-if scenario analysis completed!")
    logger.info(f"Analysis time: {analysis_time:.2f} seconds")
    logger.info(f"Scenarios generated: {scenario_summary.get('n_scenarios', 0)}")
    logger.info(f"Results saved to: {results_path}")

def create_comprehensive_report(args):
    """Create comprehensive explainability report"""
    logger.info("🔧 Creating comprehensive explainability report...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load existing results
    explainability_results = {}
    
    # Load feature importance results
    fi_path = os.path.join(args.results_dir, 'feature_importance', 'feature_importance_detailed.json')
    if os.path.exists(fi_path):
        with open(fi_path, 'r') as f:
            explainability_results['feature_importance'] = json.load(f)
    
    # Load SHAP results
    shap_path = os.path.join(args.results_dir, 'shap_explanations', 'shap_explanations_results.json')
    if os.path.exists(shap_path):
        with open(shap_path, 'r') as f:
            explainability_results['shap_analysis'] = json.load(f)
    
    # Load what-if results
    whatif_path = os.path.join(args.results_dir, 'what_if_scenarios', 'what_if_scenarios_detailed.json')
    if os.path.exists(whatif_path):
        with open(whatif_path, 'r') as f:
            explainability_results['what_if_analysis'] = json.load(f)
    
    # Create comprehensive report
    report_path = os.path.join(args.output_dir, 'comprehensive_explainability_report.json')
    report = create_explainability_report(explainability_results, report_path)
    
    # Evaluate quality
    quality_metrics = evaluate_explainability_quality(explainability_results)
    
    # Add quality metrics to report
    report['quality_metrics'] = quality_metrics
    
    # Save updated report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"✅ Comprehensive explainability report created!")
    logger.info(f"Analysis methods: {report['summary']['n_analysis_methods']}")
    logger.info(f"Overall quality: {quality_metrics['overall_quality']:.3f}")
    logger.info(f"Report saved to: {report_path}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Explainability CLI for Air Quality Forecasting')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--data-path', type=str, required=True,
                              help='Path to the data file (parquet format)')
    common_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for results')
    common_parser.add_argument('--target-col', type=str, default='pm25',
                              help='Target column name')
    common_parser.add_argument('--top-k', type=int, default=20,
                              help='Number of top features to analyze')
    common_parser.add_argument('--create-plots', action='store_true',
                              help='Create visualization plots')
    
    # Feature importance
    fi_parser = subparsers.add_parser('feature-importance', parents=[common_parser],
                                    help='Analyze feature importance')
    fi_parser.add_argument('--use-permutation', action='store_true',
                          help='Use permutation importance')
    fi_parser.add_argument('--use-tree', action='store_true',
                          help='Use tree-based importance')
    fi_parser.add_argument('--use-correlation', action='store_true',
                          help='Use correlation analysis')
    fi_parser.add_argument('--use-mutual-info', action='store_true',
                          help='Use mutual information')
    fi_parser.add_argument('--use-stability', action='store_true',
                          help='Use stability analysis')
    
    # SHAP explanations
    shap_parser = subparsers.add_parser('shap', parents=[common_parser],
                                      help='Analyze SHAP explanations')
    shap_parser.add_argument('--explainer-type', type=str, default='tree',
                            choices=['tree', 'kernel', 'linear'],
                            help='SHAP explainer type')
    shap_parser.add_argument('--max-samples', type=int, default=100,
                            help='Maximum samples for SHAP analysis')
    shap_parser.add_argument('--explanation-type', type=str, default='both',
                            choices=['local', 'global', 'both'],
                            help='Type of explanation to generate')
    
    # What-if scenarios
    whatif_parser = subparsers.add_parser('what-if', parents=[common_parser],
                                        help='Analyze what-if scenarios')
    whatif_parser.add_argument('--n-scenarios', type=int, default=10,
                              help='Number of scenarios to generate')
    whatif_parser.add_argument('--scenario-type', type=str, default='systematic',
                              choices=['systematic', 'random', 'custom'],
                              help='Type of scenarios to generate')
    whatif_parser.add_argument('--perturbation-method', type=str, default='percentage',
                              choices=['percentage', 'absolute', 'standard_deviation'],
                              help='Method for feature perturbation')
    whatif_parser.add_argument('--perturbation-range', type=float, nargs=2, default=[-0.2, 0.2],
                              help='Range for perturbation')
    whatif_parser.add_argument('--use-sensitivity', action='store_true',
                              help='Perform sensitivity analysis')
    whatif_parser.add_argument('--use-counterfactual', action='store_true',
                              help='Perform counterfactual analysis')
    
    # Comprehensive report
    report_parser = subparsers.add_parser('report', help='Create comprehensive explainability report')
    report_parser.add_argument('--results-dir', type=str, required=True,
                              help='Directory containing explainability results')
    report_parser.add_argument('--output-dir', type=str, required=True,
                              help='Output directory for report')
    
    args = parser.parse_args()
    
    if args.command == 'feature-importance':
        analyze_feature_importance(args)
    elif args.command == 'shap':
        analyze_shap_explanations(args)
    elif args.command == 'what-if':
        analyze_what_if_scenarios(args)
    elif args.command == 'report':
        create_comprehensive_report(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
