"""Local data discovery and manifest generation for AirAware project.

This module scans existing data files, safely unpacks data.zip if present,
and creates a comprehensive manifest for data availability assessment.
"""

import hashlib
import json
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)

# Dataset type detection patterns
DATASET_PATTERNS = {
    "openaq_targets": ["openaq", "targets", "pm25", "measurements"],
    "era5_hourly": ["era5", "hourly", "u10", "v10", "t2m", "blh"],
    "joined": ["joined", "combined", "merged"],
    "train_ready": ["train", "ready", "features", "processed"],
}

# Expected columns for schema validation
EXPECTED_SCHEMAS = {
    "openaq_targets": ["station_id", "ts_utc", "pm25"],
    "era5_hourly": ["ts_utc", "u10", "v10", "t2m", "blh"],
    "joined": ["station_id", "ts_utc", "pm25", "u10", "v10", "t2m", "blh"],
    "train_ready": ["station_id", "ts_utc", "pm25", "u10", "v10", "t2m", "blh"],
}


class DataScanner:
    """Scans local data directories and creates manifest."""

    def __init__(self, data_root: Path = Path("data")) -> None:
        """Initialize data scanner.
        
        Args:
            data_root: Root directory containing data subdirectories
        """
        self.data_root = data_root
        self.manifest_path = data_root / "artifacts" / "local_data_manifest.json"
        
        # Ensure artifacts directory exists
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def safe_unpack_zip(self, zip_path: Path) -> List[str]:
        """Safely unpack data.zip, skipping files that are newer locally.
        
        Args:
            zip_path: Path to data.zip file
            
        Returns:
            List of files that were extracted
        """
        extracted_files = []
        
        if not zip_path.exists():
            logger.info(f"No zip file found at {zip_path}")
            return extracted_files
            
        logger.info(f"Found {zip_path}, checking contents...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.is_dir():
                    continue
                    
                # Target path for extraction
                target_path = self.data_root / zip_info.filename
                
                # Skip if local file is newer
                if target_path.exists():
                    zip_mtime = datetime(*zip_info.date_time).timestamp()
                    local_mtime = target_path.stat().st_mtime
                    
                    if local_mtime > zip_mtime:
                        logger.debug(f"Skipping {zip_info.filename} (local is newer)")
                        continue
                
                # Extract file
                zip_ref.extract(zip_info, self.data_root)
                extracted_files.append(zip_info.filename)
                logger.info(f"Extracted: {zip_info.filename}")
        
        if extracted_files:
            console.print(f"[green]Extracted {len(extracted_files)} files from {zip_path}[/green]")
        else:
            console.print(f"[yellow]No files extracted from {zip_path} (all up to date)[/yellow]")
            
        return extracted_files

    def compute_file_hash(self, file_path: Path, quick_mode: bool = False) -> Optional[str]:
        """Compute SHA256 hash of file.
        
        Args:
            file_path: Path to file
            quick_mode: Skip hash computation for speed
            
        Returns:
            SHA256 hash or None if quick_mode or error
        """
        if quick_mode:
            return None
            
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not compute hash for {file_path}: {e}")
            return None

    def detect_dataset_type(self, file_path: Path) -> Set[str]:
        """Detect dataset type based on file path and name.
        
        Args:
            file_path: Path to file
            
        Returns:
            Set of detected dataset types
        """
        file_str = str(file_path).lower()
        detected_types = set()
        
        for dataset_type, patterns in DATASET_PATTERNS.items():
            if any(pattern in file_str for pattern in patterns):
                detected_types.add(dataset_type)
                
        return detected_types

    def validate_parquet_schema(self, file_path: Path, quick_mode: bool = False) -> Dict[str, Any]:
        """Validate parquet file schema against expected schemas.
        
        Args:
            file_path: Path to parquet file
            quick_mode: Skip detailed validation
            
        Returns:
            Validation results dictionary
        """
        if quick_mode or not file_path.suffix.lower() == '.parquet':
            return {"validated": False, "reason": "skipped"}
            
        try:
            # Read just the schema without loading data - use nrows=0 for efficiency
            import pyarrow.parquet as pq
            
            # Use pyarrow to read schema only (more efficient)
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema_arrow
            columns = [field.name for field in schema]
            
            # Get basic file stats
            num_rows = parquet_file.metadata.num_rows
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Check against expected schemas
            validation_results = {
                "validated": True, 
                "columns": columns,
                "num_columns": len(columns),
                "num_rows": num_rows,
                "size_mb": round(file_size_mb, 2),
                "matches": [],
                "schema_score": 0.0,
            }
            
            # Calculate schema match scores
            best_score = 0.0
            for schema_name, expected_cols in EXPECTED_SCHEMAS.items():
                matching_cols = sum(1 for col in expected_cols if col in columns)
                score = matching_cols / len(expected_cols) if expected_cols else 0.0
                
                if score >= 0.8:  # 80% column match threshold
                    validation_results["matches"].append({
                        "schema": schema_name,
                        "score": score,
                        "missing_cols": [col for col in expected_cols if col not in columns]
                    })
                
                best_score = max(best_score, score)
            
            validation_results["schema_score"] = best_score
            
            # Add data quality flags
            if num_rows == 0:
                validation_results["warnings"] = ["empty_file"]
            elif num_rows < 100:
                validation_results["warnings"] = ["small_dataset"]
                
            return validation_results
            
        except Exception as e:
            return {"validated": False, "reason": f"error: {e}"}

    def scan_file(self, file_path: Path, quick_mode: bool = False) -> Dict[str, Any]:
        """Scan single file and extract metadata.
        
        Args:
            file_path: Path to file
            quick_mode: Skip expensive operations
            
        Returns:
            File metadata dictionary
        """
        try:
            stat = file_path.stat()
            
            metadata = {
                "path": str(file_path.relative_to(self.data_root)),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "mtime_timestamp": stat.st_mtime,
                "extension": file_path.suffix.lower(),
                "dataset_types": list(self.detect_dataset_type(file_path)),
                "sha256": self.compute_file_hash(file_path, quick_mode),
                "schema_validation": self.validate_parquet_schema(file_path, quick_mode),
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")
            return {
                "path": str(file_path.relative_to(self.data_root)),
                "error": str(e),
                "mtime": None,
                "size_bytes": 0,
            }

    def scan_directory(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Scan all data directories and create manifest.
        
        Args:
            quick_mode: Skip expensive operations for faster discovery
            
        Returns:
            Complete manifest dictionary
        """
        console.print("[bold blue]Scanning local data directories...[/bold blue]")
        
        # Find all files to scan
        all_files = []
        for subdir in ["raw", "interim", "processed", "artifacts"]:
            subdir_path = self.data_root / subdir
            if subdir_path.exists():
                all_files.extend(subdir_path.rglob("*"))
        
        # Filter to actual files
        files_to_scan = [f for f in all_files if f.is_file()]
        
        if not files_to_scan:
            console.print("[yellow]No data files found[/yellow]")
            return {
                "scan_timestamp": datetime.now().isoformat(),
                "total_files": 0,
                "total_size_mb": 0,
                "files": [],
                "summary": {"by_type": {}, "by_directory": {}},
            }
        
        # Scan files with progress bar and enhanced mode messaging
        scanned_files = []
        
        if quick_mode:
            console.print(f"[yellow]⚡ Quick mode: scanning {len(files_to_scan)} files (skipping hashes & detailed validation)...[/yellow]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Quick scanning...", total=len(files_to_scan))
                
                for file_path in files_to_scan:
                    metadata = self.scan_file(file_path, quick_mode=True)
                    scanned_files.append(metadata)
                    progress.advance(task)
        else:
            console.print(f"[green]🔍 Full scan: processing {len(files_to_scan)} files (with hashes & schema validation)...[/green]")
            with ThreadPoolExecutor(max_workers=4) as executor:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Full analysis...", total=len(files_to_scan))
                    
                    # Submit all tasks
                    future_to_file = {
                        executor.submit(self.scan_file, f, False): f 
                        for f in files_to_scan
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_file):
                        metadata = future.result()
                        scanned_files.append(metadata)
                        progress.advance(task)
        
        # Create summary
        summary = self._create_summary(scanned_files)
        
        manifest = {
            "scan_timestamp": datetime.now().isoformat(),
            "quick_mode": quick_mode,
            "total_files": len(scanned_files),
            "total_size_mb": sum(f.get("size_mb", 0) for f in scanned_files),
            "files": scanned_files,
            "summary": summary,
        }
        
        return manifest

    def _create_summary(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics from scanned files."""
        summary = {
            "by_type": {},
            "by_directory": {},
            "coverage": {
                "openaq_stations": set(),
                "era5_dates": set(),
                "date_range": {"start": None, "end": None},
            }
        }
        
        for file_data in files:
            # By dataset type
            for dataset_type in file_data.get("dataset_types", []):
                if dataset_type not in summary["by_type"]:
                    summary["by_type"][dataset_type] = {"files": 0, "size_mb": 0}
                summary["by_type"][dataset_type]["files"] += 1
                summary["by_type"][dataset_type]["size_mb"] += file_data.get("size_mb", 0)
            
            # By directory
            path_parts = Path(file_data["path"]).parts
            if path_parts:
                directory = path_parts[0]
                if directory not in summary["by_directory"]:
                    summary["by_directory"][directory] = {"files": 0, "size_mb": 0}
                summary["by_directory"][directory]["files"] += 1
                summary["by_directory"][directory]["size_mb"] += file_data.get("size_mb", 0)
        
        # Convert sets to lists for JSON serialization
        summary["coverage"]["openaq_stations"] = list(summary["coverage"]["openaq_stations"])
        summary["coverage"]["era5_dates"] = list(summary["coverage"]["era5_dates"])
        
        return summary

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Save manifest to JSON file."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        console.print(f"[green]Manifest saved to {self.manifest_path}[/green]")

    def print_summary(self, manifest: Dict[str, Any]) -> None:
        """Print formatted summary of discovered data."""
        console.print("\n[bold green]📊 Data Discovery Summary[/bold green]")
        
        # Overall stats
        table = Table(title="Overall Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Files", str(manifest["total_files"]))
        table.add_row("Total Size", f"{manifest['total_size_mb']:.1f} MB")
        table.add_row("Scan Mode", "⚡ Quick" if manifest.get("quick_mode") else "🔍 Full")
        table.add_row("Last Scan", manifest["scan_timestamp"])
        
        # Add schema validation summary for full scans
        if not manifest.get("quick_mode"):
            validated_files = sum(1 for f in manifest["files"] 
                                if f.get("schema_validation", {}).get("validated"))
            table.add_row("Schema Validated", f"{validated_files} files")
        
        console.print(table)
        
        # By dataset type
        if manifest["summary"]["by_type"]:
            type_table = Table(title="By Dataset Type")
            type_table.add_column("Dataset Type", style="cyan")
            type_table.add_column("Files", style="white")
            type_table.add_column("Size (MB)", style="white")
            
            for dataset_type, stats in manifest["summary"]["by_type"].items():
                type_table.add_row(
                    dataset_type,
                    str(stats["files"]),
                    f"{stats['size_mb']:.1f}"
                )
            
            console.print(type_table)
        
        # By directory
        if manifest["summary"]["by_directory"]:
            dir_table = Table(title="By Directory")
            dir_table.add_column("Directory", style="cyan")
            dir_table.add_column("Files", style="white")
            dir_table.add_column("Size (MB)", style="white")
            
            for directory, stats in manifest["summary"]["by_directory"].items():
                dir_table.add_row(
                    directory,
                    str(stats["files"]),
                    f"{stats['size_mb']:.1f}"
                )
            
            console.print(dir_table)
        
        # Schema validation summary for full scans
        if not manifest.get("quick_mode") and manifest["files"]:
            self._print_schema_validation_summary(manifest)

    def _print_schema_validation_summary(self, manifest: Dict[str, Any]) -> None:
        """Print schema validation summary for full scans."""
        parquet_files = [f for f in manifest["files"] 
                        if f.get("extension") == ".parquet"]
        
        if not parquet_files:
            return
            
        schema_table = Table(title="Schema Validation Results")
        schema_table.add_column("File", style="cyan")
        schema_table.add_column("Rows", style="white") 
        schema_table.add_column("Cols", style="white")
        schema_table.add_column("Score", style="white")
        schema_table.add_column("Best Match", style="green")
        
        for file_data in parquet_files[:10]:  # Show top 10
            validation = file_data.get("schema_validation", {})
            if validation.get("validated"):
                matches = validation.get("matches", [])
                best_match = matches[0]["schema"] if matches else "Unknown"
                score = f"{validation.get('schema_score', 0):.1%}"
                
                schema_table.add_row(
                    Path(file_data["path"]).name,
                    f"{validation.get('num_rows', 0):,}",
                    str(validation.get('num_columns', 0)),
                    score,
                    best_match
                )
        
        if schema_table.rows:
            console.print(schema_table)
            
            if len(parquet_files) > 10:
                console.print(f"[dim]... and {len(parquet_files) - 10} more parquet files[/dim]")

    def run_discovery(self, quick_mode: bool = False, unpack_zip: bool = True) -> Dict[str, Any]:
        """Run complete data discovery process.
        
        Args:
            quick_mode: Skip expensive operations
            unpack_zip: Whether to unpack data.zip if found
            
        Returns:
            Complete manifest
        """
        console.print("[bold blue]AirAware Data Discovery[/bold blue]\n")
        
        # Step 1: Unpack zip if requested
        if unpack_zip:
            zip_path = self.data_root / "data.zip"
            self.safe_unpack_zip(zip_path)
        
        # Step 2: Scan directories
        manifest = self.scan_directory(quick_mode=quick_mode)
        
        # Step 3: Save manifest
        self.save_manifest(manifest)
        
        # Step 4: Print summary
        self.print_summary(manifest)
        
        return manifest


def main(
    data_root: str = typer.Argument("data", help="Root data directory"),
    quick: bool = typer.Option(False, "--quick", help="Skip hash computation for faster discovery"),
    no_unpack: bool = typer.Option(False, "--no-unpack", help="Skip unpacking data.zip"),
) -> None:
    """Discover and catalog local data files."""
    logging.basicConfig(level=logging.INFO)
    
    scanner = DataScanner(Path(data_root))
    scanner.run_discovery(quick_mode=quick, unpack_zip=not no_unpack)


if __name__ == "__main__":
    typer.run(main)
