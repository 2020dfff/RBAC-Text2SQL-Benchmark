"""
Bird Database Description Generator

This module processes Bird dataset database description files and generates
structured schema descriptions for role generation.

Each Bird database has a database_description/ folder containing CSV files
that describe each table with columns:
- original_column_name: Original column name in the database
- column_name: Processed column name
- column_description: Description of what the column contains
- data_format: Data type/format of the column
- value_description: Description of possible values or constraints
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ColumnInfo:
    """Data class to hold column information from Bird description files"""
    original_name: str
    column_name: str
    description: str
    data_format: str
    value_description: str


@dataclass
class TableInfo:
    """Data class to hold table information"""
    table_name: str
    columns: List[ColumnInfo]
    column_count: int


@dataclass
class DatabaseInfo:
    """Data class to hold complete database information"""
    db_name: str
    tables: List[TableInfo]
    total_tables: int
    total_columns: int


class BirdDatabaseDescriber:
    """
    Generate structured descriptions of Bird databases from CSV description files.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.bird_data_dir = project_root / 'data' / 'Bird'
        
    def read_database_descriptions(self, db_path: Path) -> Optional[DatabaseInfo]:
        """
        Read all CSV description files for a database and return structured information.
        
        Args:
            db_path: Path to the database directory
            
        Returns:
            DatabaseInfo object containing all database schema information
        """
        desc_dir = db_path / 'database_description'
        if not desc_dir.exists():
            logger.warning(f"No database_description directory found for {db_path.name}")
            return None
            
        csv_files = list(desc_dir.glob('*.csv'))
        if not csv_files:
            logger.warning(f"No CSV description files found in {desc_dir}")
            return None
            
        tables = []
        total_columns = 0
        
        for csv_file in sorted(csv_files):
            table_name = csv_file.stem  # Remove .csv extension
            
            try:
                # Try to read with UTF-8, fallback to other encodings if needed
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(csv_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_file, encoding='cp1252')
                
                # Validate expected columns
                expected_cols = ['original_column_name', 'column_name', 'column_description', 
                               'data_format', 'value_description']
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in {csv_file}: {missing_cols}")
                    continue
                
                # Parse column information
                columns = []
                for _, row in df.iterrows():
                    col_info = ColumnInfo(
                        original_name=str(row.get('original_column_name', '')).strip(),
                        column_name=str(row.get('column_name', '')).strip(),
                        description=str(row.get('column_description', '')).strip(),
                        data_format=str(row.get('data_format', '')).strip(),
                        value_description=str(row.get('value_description', '')).strip()
                    )
                    columns.append(col_info)
                
                table_info = TableInfo(
                    table_name=table_name,
                    columns=columns,
                    column_count=len(columns)
                )
                tables.append(table_info)
                total_columns += len(columns)
                
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {str(e)}")
                continue
        
        if not tables:
            logger.warning(f"No valid table descriptions found for {db_path.name}")
            return None
            
        return DatabaseInfo(
            db_name=db_path.name,
            tables=tables,
            total_tables=len(tables),
            total_columns=total_columns
        )
    
    def generate_schema_description(self, db_info: DatabaseInfo, format_type: str = "detailed") -> str:
        """
        Generate a structured schema description suitable for role generation prompts.
        
        Args:
            db_info: DatabaseInfo object containing database schema information
            format_type: Type of format - "detailed", "summary", or "compact"
            
        Returns:
            Formatted schema description string
        """
        if not db_info or not db_info.tables:
            return f"Database {db_info.db_name if db_info else 'unknown'} has no schema information available."
        
        if format_type == "detailed":
            return self._generate_detailed_description(db_info)
        elif format_type == "summary":
            return self._generate_summary_description(db_info)
        elif format_type == "compact":
            return self._generate_compact_description(db_info)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def _generate_detailed_description(self, db_info: DatabaseInfo) -> str:
        """Generate detailed schema description with full column information."""
        lines = [
            f"Database: {db_info.db_name}",
            f"Total Tables: {db_info.total_tables}",
            f"Total Columns: {db_info.total_columns}",
            "",
            "Table Schemas:"
        ]
        
        for table in db_info.tables:
            lines.append(f"\nTable: {table.table_name} ({table.column_count} columns)")
            lines.append("-" * (len(f"Table: {table.table_name} ({table.column_count} columns)")))
            
            for col in table.columns:
                # Format column information
                col_line = f"• {col.original_name}"
                if col.data_format and col.data_format != 'nan':
                    col_line += f" ({col.data_format})"
                
                if col.description and col.description != 'nan':
                    col_line += f": {col.description}"
                
                lines.append(col_line)
                
                # Add value description if available
                if col.value_description and col.value_description != 'nan' and len(col.value_description.strip()) > 0:
                    # Handle multi-line value descriptions
                    value_desc = col.value_description.strip()
                    if len(value_desc) > 100:
                        value_desc = value_desc[:100] + "..."
                    lines.append(f"  Values: {value_desc}")
        
        return "\n".join(lines)
    
    def _generate_summary_description(self, db_info: DatabaseInfo) -> str:
        """Generate summary schema description with key information only."""
        lines = [
            f"Database: {db_info.db_name}",
            f"Tables: {db_info.total_tables}, Columns: {db_info.total_columns}",
            ""
        ]
        
        for table in db_info.tables:
            lines.append(f"Table: {table.table_name} ({table.column_count} columns)")
            
            # Show key columns (first few with descriptions)
            key_columns = []
            for col in table.columns[:5]:  # Show first 5 columns
                col_desc = f"{col.original_name}"
                if col.data_format and col.data_format != 'nan':
                    col_desc += f" ({col.data_format})"
                key_columns.append(col_desc)
            
            lines.append(f"  Key columns: {', '.join(key_columns)}")
            if table.column_count > 5:
                lines.append(f"  ... and {table.column_count - 5} more columns")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_compact_description(self, db_info: DatabaseInfo) -> str:
        """Generate compact schema description for token-efficient prompts."""
        table_summaries = []
        for table in db_info.tables:
            # Get column names with data types
            col_specs = []
            for col in table.columns:
                if col.data_format and col.data_format != 'nan':
                    col_specs.append(f"{col.original_name}({col.data_format})")
                else:
                    col_specs.append(col.original_name)
            
            table_summaries.append(f"{table.table_name}[{', '.join(col_specs)}]")
        
        return f"Database: {db_info.db_name}\nTables: {'; '.join(table_summaries)}"
    
    def generate_role_prompt(self, db_info: DatabaseInfo, format_type: str = "summary") -> str:
        """
        Generate a complete prompt for role generation using the database schema.
        
        Args:
            db_info: DatabaseInfo object containing database schema information
            format_type: Format for schema description
            
        Returns:
            Complete prompt string ready for LLM role generation
        """
        schema_description = self.generate_schema_description(db_info, format_type)
        
        # Import prompt template from configs
        from configs.prompts import USER_PROMPT_TEMPLATE
        
        return USER_PROMPT_TEMPLATE.format(schema_content=schema_description)
    
    def get_database_paths(self, dataset_type: str = 'dev') -> List[Path]:
        """
        Get all database paths for the specified dataset type.
        
        Args:
            dataset_type: 'dev' or 'train'
            
        Returns:
            List of Path objects for each database directory
        """
        if dataset_type == 'dev':
            db_dir = self.bird_data_dir / 'dev_20240627' / 'dev_databases'
        elif dataset_type == 'train':
            db_dir = self.bird_data_dir / 'train' / 'train_databases'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        if not db_dir.exists():
            logger.warning(f"Database directory not found: {db_dir}")
            return []
        
        return [db for db in db_dir.iterdir() if db.is_dir() and not db.name.startswith('.')]
    
    def process_all_databases(self, dataset_type: str = 'dev', format_type: str = "summary") -> Dict[str, str]:
        """
        Process all databases in the specified dataset and generate descriptions.
        
        Args:
            dataset_type: 'dev' or 'train'
            format_type: Format for schema descriptions
            
        Returns:
            Dictionary mapping database names to their schema descriptions
        """
        db_paths = self.get_database_paths(dataset_type)
        results = {}
        
        logger.info(f"Processing {len(db_paths)} databases from {dataset_type} dataset...")
        
        for db_path in db_paths:
            try:
                db_info = self.read_database_descriptions(db_path)
                if db_info:
                    schema_desc = self.generate_schema_description(db_info, format_type)
                    results[db_info.db_name] = schema_desc
                    logger.info(f"✓ Processed {db_info.db_name}: {db_info.total_tables} tables, {db_info.total_columns} columns")
                else:
                    logger.warning(f"✗ Failed to process {db_path.name}")
                    
            except Exception as e:
                logger.error(f"✗ Error processing {db_path.name}: {str(e)}")
        
        logger.info(f"Successfully processed {len(results)}/{len(db_paths)} databases")
        return results


def main():
    """Test function to demonstrate usage"""
    import sys
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    project_root = Path('/home/feiy/Role-SQL-benchmark')
    describer = BirdDatabaseDescriber(project_root)
    
    # Test with first dev database
    db_paths = describer.get_database_paths('dev')
    if db_paths:
        test_db = db_paths[0]
        print(f"Testing with database: {test_db.name}")
        
        # Read database info
        db_info = describer.read_database_descriptions(test_db)
        if db_info:
            print(f"\nDatabase Info:")
            print(f"  Name: {db_info.db_name}")
            print(f"  Tables: {db_info.total_tables}")
            print(f"  Columns: {db_info.total_columns}")
            
            # Generate different format descriptions
            for fmt in ["compact", "summary", "detailed"]:
                print(f"\n{fmt.upper()} FORMAT:")
                print("-" * 50)
                desc = describer.generate_schema_description(db_info, fmt)
                print(desc[:500] + "..." if len(desc) > 500 else desc)


if __name__ == "__main__":
    main()
