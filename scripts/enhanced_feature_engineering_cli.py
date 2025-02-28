#!/usr/bin/env python3
"""Enhanced Feature Engineering CLI with advanced capabilities for AirAware PM₂.₅ nowcasting."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from airaware.features import (
    FeatureBuilder, FeatureConfig, FeatureValidator,
    AdvancedFeatureSelector, FeatureSelectionConfig,
    DomainFeatureGenerator, DomainFeatureConfig,
    FeatureInteractionDetector, InteractionConfig
)


def run_enhanced_pipeline_command(args):
    """Run enhanced feature engineering pipeline with all improvements."""
    console = Console()
    
    try:
        console.print(Panel.fit("🚀 Enhanced Feature Engineering Pipeline", style="bold blue"))
        
        # Initialize components
        feature_config = FeatureConfig()
        feature_config.save_intermediate_steps = args.save_intermediate
        feature_config.save_feature_importance = True
        
        domain_config = DomainFeatureConfig()
        if args.no_domain_features:
            domain_config.air_quality_indices = False
            domain_config.emission_source_proxies = False
            domain_config.dispersion_modeling = False
            domain_config.seasonal_patterns = False
            domain_config.urban_effects = False
        
        interaction_config = InteractionConfig()
        interaction_config.max_interactions = args.max_interactions
        interaction_config.domain_guided = args.domain_interactions
        
        selection_config = FeatureSelectionConfig()
        selection_config.max_features = args.max_features
        
        console.print(f"\n🔧 Enhanced Pipeline Configuration:")
        console.print(f"  • Domain Features: {'Enabled' if not args.no_domain_features else 'Disabled'}")
        console.print(f"  • Feature Interactions: {'Enabled' if args.interactions else 'Disabled'}")
        console.print(f"  • Advanced Selection: {'Enabled' if args.advanced_selection else 'Disabled'}")
        console.print(f"  • Max Interactions: {args.max_interactions}")
        console.print(f"  • Max Features: {args.max_features if args.max_features else 'No limit'}")
        
        # Step 1: Run base feature engineering
        console.print(f"\n📊 Step 1: Base Feature Engineering")
        
        feature_builder = FeatureBuilder(config=feature_config)
        base_result = feature_builder.build_features()
        
        if not base_result.success:
            console.print(f"[red]❌ Base feature engineering failed[/red]")
            return
        
        console.print(f"✅ Base features: {base_result.total_features} features")
        
        # Load the base features
        import pandas as pd
        features_df = pd.read_parquet(base_result.output_files["features"])
        
        # Step 2: Add domain-specific features
        if not args.no_domain_features:
            console.print(f"\n🏭 Step 2: Domain-Specific Features")
            
            domain_generator = DomainFeatureGenerator(domain_config)
            features_df = domain_generator.generate_features(features_df)
            
            domain_features = [col for col in features_df.columns 
                             if any(col.startswith(prefix) for prefix in 
                                   ['aqi_', 'emission_', 'dispersion_', 'seasonal_', 'urban_', 'topo_', 'temporal_', 'threshold_'])]
            
            console.print(f"✅ Domain features: +{len(domain_features)} features")
        
        # Step 3: Detect and create interactions
        if args.interactions:
            console.print(f"\n🔄 Step 3: Feature Interactions")
            
            interaction_detector = FeatureInteractionDetector(interaction_config)
            features_df, interaction_result = interaction_detector.detect_and_create_interactions(features_df)
            
            console.print(f"✅ Interaction features: +{interaction_result.interactions_created} features")
            
            # Save interaction report
            interaction_analysis = interaction_detector.analyze_interaction_importance(
                features_df, interaction_result.interaction_features
            )
            
            interaction_report = interaction_detector.generate_interaction_report(
                interaction_result, interaction_analysis
            )
            
            interaction_report_path = Path("data/processed/interaction_report.txt")
            with open(interaction_report_path, 'w') as f:
                f.write(interaction_report)
            
            console.print(f"📄 Interaction report saved: {interaction_report_path}")
        
        # Step 4: Advanced feature selection
        if args.advanced_selection:
            console.print(f"\n🎯 Step 4: Advanced Feature Selection")
            
            advanced_selector = AdvancedFeatureSelector(selection_config)
            selection_result = advanced_selector.select_features(features_df)
            
            # Apply selection
            if selection_result.selected_features:
                base_cols = ['datetime_utc', 'station_id', 'pm25']
                selected_cols = base_cols + selection_result.selected_features
                features_df = features_df[selected_cols]
                
                console.print(f"✅ Advanced selection: {len(selection_result.selected_features)} features selected")
                
                # Save selection report
                selection_report = advanced_selector.generate_selection_report(selection_result)
                
                selection_report_path = Path("data/processed/advanced_selection_report.txt")
                with open(selection_report_path, 'w') as f:
                    f.write(selection_report)
                
                console.print(f"📄 Selection report saved: {selection_report_path}")
        
        # Save enhanced features
        enhanced_features_path = Path("data/processed/enhanced_features.parquet")
        features_df.to_parquet(enhanced_features_path, index=False)
        
        # Final validation
        console.print(f"\n🔍 Step 5: Final Validation")
        
        validator = FeatureValidator()
        validation_metrics = validator.validate_features(features_df)
        
        # Display final results
        console.print(f"\n🎊 Enhanced Feature Engineering Complete!")
        
        results_table = Table(title="Enhanced Feature Engineering Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="white")
        
        total_features = len(features_df.columns) - 3  # Exclude datetime_utc, station_id, pm25
        
        results_table.add_row("Total Features", str(total_features))
        results_table.add_row("Records", f"{len(features_df):,}")
        results_table.add_row("Quality Score", f"{validation_metrics.data_quality_score:.1%}")
        results_table.add_row("Missing Values", f"{validation_metrics.missing_values_rate:.1%}")
        
        if args.interactions:
            results_table.add_row("Interactions Created", str(interaction_result.interactions_created))
        
        console.print(results_table)
        
        # Feature breakdown
        feature_breakdown = _analyze_enhanced_features(features_df)
        
        breakdown_table = Table(title="Enhanced Feature Breakdown")
        breakdown_table.add_column("Category", style="cyan")
        breakdown_table.add_column("Count", style="green")
        breakdown_table.add_column("Percentage", style="white")
        
        for category, count in feature_breakdown.items():
            if count > 0:
                pct = (count / total_features) * 100
                breakdown_table.add_row(category, str(count), f"{pct:.1f}%")
        
        console.print(breakdown_table)
        
        console.print(f"\n📁 Enhanced Features: {enhanced_features_path}")
        
    except Exception as e:
        console.print(f"[red]❌ Enhanced feature engineering failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_interaction_analysis_command(args):
    """Run standalone interaction analysis."""
    console = Console()
    
    try:
        console.print(Panel.fit("🔄 Feature Interaction Analysis", style="bold blue"))
        
        # Load features
        features_path = Path("data/processed/features.parquet")
        if not features_path.exists():
            console.print(f"[red]❌ Features file not found: {features_path}[/red]")
            sys.exit(1)
        
        import pandas as pd
        features_df = pd.read_parquet(features_path)
        
        console.print(f"✅ Loaded {len(features_df):,} records with {len(features_df.columns)} columns")
        
        # Configure interaction detector
        interaction_config = InteractionConfig()
        interaction_config.max_interactions = args.max_interactions
        interaction_config.domain_guided = args.domain_guided
        
        # Detect interactions
        interaction_detector = FeatureInteractionDetector(interaction_config)
        enhanced_df, interaction_result = interaction_detector.detect_and_create_interactions(features_df)
        
        # Analyze interactions
        interaction_analysis = interaction_detector.analyze_interaction_importance(
            enhanced_df, interaction_result.interaction_features
        )
        
        # Generate report
        interaction_report = interaction_detector.generate_interaction_report(
            interaction_result, interaction_analysis
        )
        
        console.print(interaction_report)
        
        # Save results
        if args.save_interactions:
            output_path = Path("data/processed/features_with_interactions.parquet")
            enhanced_df.to_parquet(output_path, index=False)
            console.print(f"\n💾 Enhanced features saved: {output_path}")
            
            report_path = Path("data/processed/interaction_analysis_report.txt")
            with open(report_path, 'w') as f:
                f.write(interaction_report)
            console.print(f"📄 Report saved: {report_path}")
        
    except Exception as e:
        console.print(f"[red]❌ Interaction analysis failed: {e}[/red]")
        sys.exit(1)


def run_domain_features_command(args):
    """Run standalone domain feature generation."""
    console = Console()
    
    try:
        console.print(Panel.fit("🏭 Domain-Specific Feature Generation", style="bold blue"))
        
        # Load features
        features_path = Path("data/processed/features.parquet")
        if not features_path.exists():
            console.print(f"[red]❌ Features file not found: {features_path}[/red]")
            sys.exit(1)
        
        import pandas as pd
        features_df = pd.read_parquet(features_path)
        
        console.print(f"✅ Loaded {len(features_df):,} records")
        
        # Configure domain generator
        domain_config = DomainFeatureConfig()
        if hasattr(args, 'air_quality_indices'):
            domain_config.air_quality_indices = args.air_quality_indices
        if hasattr(args, 'emission_sources'):
            domain_config.emission_source_proxies = args.emission_sources
        if hasattr(args, 'dispersion_modeling'):
            domain_config.dispersion_modeling = args.dispersion_modeling
        if hasattr(args, 'seasonal_patterns'):
            domain_config.seasonal_patterns = args.seasonal_patterns
        if hasattr(args, 'urban_effects'):
            domain_config.urban_effects = args.urban_effects
        
        # Generate domain features
        domain_generator = DomainFeatureGenerator(domain_config)
        enhanced_df = domain_generator.generate_features(features_df)
        
        # Analyze new features
        domain_features = [col for col in enhanced_df.columns 
                          if any(col.startswith(prefix) for prefix in 
                                ['aqi_', 'emission_', 'dispersion_', 'seasonal_', 'urban_', 'topo_', 'temporal_', 'threshold_'])]
        
        console.print(f"✅ Generated {len(domain_features)} domain-specific features")
        
        # Show feature categories
        categories = domain_generator.get_feature_categories()
        
        cat_table = Table(title="Domain Feature Categories")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="green")
        cat_table.add_column("Examples", style="white")
        
        for category, feature_list in categories.items():
            available_features = [f for f in feature_list if f in enhanced_df.columns]
            if available_features:
                examples = ", ".join(available_features[:3])
                if len(available_features) > 3:
                    examples += "..."
                cat_table.add_row(category.replace("_", " ").title(), 
                                str(len(available_features)), examples)
        
        console.print(cat_table)
        
        # Save results
        if args.save_domain:
            output_path = Path("data/processed/features_with_domain.parquet")
            enhanced_df.to_parquet(output_path, index=False)
            console.print(f"\n💾 Enhanced features saved: {output_path}")
        
    except Exception as e:
        console.print(f"[red]❌ Domain feature generation failed: {e}[/red]")
        sys.exit(1)


def _analyze_enhanced_features(df) -> dict:
    """Analyze enhanced feature breakdown."""
    
    feature_cols = [col for col in df.columns 
                   if col not in ['datetime_utc', 'station_id', 'pm25']]
    
    breakdown = {
        "Base Temporal": 0,
        "Base Meteorological": 0,
        "Domain AQI": 0,
        "Domain Emission": 0,
        "Domain Dispersion": 0,
        "Domain Seasonal": 0,
        "Domain Urban": 0,
        "Domain Threshold": 0,
        "Interactions": 0,
        "Other": 0
    }
    
    for col in feature_cols:
        if col.startswith('lag_') or col.startswith('rolling_') or col.startswith('calendar_') or col.startswith('cyclical_'):
            breakdown["Base Temporal"] += 1
        elif col.startswith('wind_') or col.startswith('temp_') or col.startswith('stability_') or col.startswith('comfort_'):
            breakdown["Base Meteorological"] += 1
        elif col.startswith('aqi_'):
            breakdown["Domain AQI"] += 1
        elif col.startswith('emission_'):
            breakdown["Domain Emission"] += 1
        elif col.startswith('dispersion_'):
            breakdown["Domain Dispersion"] += 1
        elif col.startswith('seasonal_'):
            breakdown["Domain Seasonal"] += 1
        elif col.startswith('urban_') or col.startswith('topo_'):
            breakdown["Domain Urban"] += 1
        elif col.startswith('threshold_'):
            breakdown["Domain Threshold"] += 1
        elif col.startswith('interact_'):
            breakdown["Interactions"] += 1
        else:
            breakdown["Other"] += 1
    
    return breakdown


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Feature Engineering Pipeline for AirAware")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Enhanced pipeline command
    enhanced_parser = subparsers.add_parser("enhanced", help="Run enhanced feature engineering pipeline")
    enhanced_parser.add_argument("--save-intermediate", action="store_true",
                               help="Save intermediate processing steps")
    enhanced_parser.add_argument("--no-domain-features", action="store_true",
                               help="Skip domain-specific features")
    enhanced_parser.add_argument("--interactions", action="store_true", default=True,
                               help="Enable feature interactions")
    enhanced_parser.add_argument("--advanced-selection", action="store_true",
                               help="Enable advanced feature selection")
    enhanced_parser.add_argument("--max-interactions", type=int, default=20,
                               help="Maximum number of interactions to create")
    enhanced_parser.add_argument("--max-features", type=int,
                               help="Maximum number of features to select")
    enhanced_parser.add_argument("--domain-interactions", action="store_true", default=True,
                               help="Use domain-guided interactions")
    
    # Interaction analysis command
    interaction_parser = subparsers.add_parser("interactions", help="Analyze feature interactions")
    interaction_parser.add_argument("--max-interactions", type=int, default=30,
                                  help="Maximum interactions to analyze")
    interaction_parser.add_argument("--domain-guided", action="store_true", default=True,
                                  help="Use domain knowledge")
    interaction_parser.add_argument("--save-interactions", action="store_true",
                                  help="Save features with interactions")
    
    # Domain features command
    domain_parser = subparsers.add_parser("domain", help="Generate domain-specific features")
    domain_parser.add_argument("--air-quality-indices", action="store_true", default=True,
                             help="Generate AQI features")
    domain_parser.add_argument("--emission-sources", action="store_true", default=True,
                             help="Generate emission source features")
    domain_parser.add_argument("--dispersion-modeling", action="store_true", default=True,
                             help="Generate dispersion features")
    domain_parser.add_argument("--seasonal-patterns", action="store_true", default=True,
                             help="Generate seasonal features")
    domain_parser.add_argument("--urban-effects", action="store_true", default=True,
                             help="Generate urban microclimate features")
    domain_parser.add_argument("--save-domain", action="store_true",
                             help="Save features with domain features")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate command handler
    if args.command == "enhanced":
        run_enhanced_pipeline_command(args)
    elif args.command == "interactions":
        run_interaction_analysis_command(args)
    elif args.command == "domain":
        run_domain_features_command(args)


if __name__ == "__main__":
    main()
