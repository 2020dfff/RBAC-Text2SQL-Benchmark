"""
Generate role-based access control text2sql dataset for Bird dataset.

This module adapts the Spider RoleSQLGenerator for Bird dataset's structure,
which uses detailed schema files instead of tables.json.
"""

import json
import sys
import logging
import sqlparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

from .spider_role_sql_generator import RoleSQLGenerator

logger = logging.getLogger('bird_role_sql')


class BirdRoleSQLGenerator(RoleSQLGenerator):
    """Generate role-based SQL queries dataset for Bird dataset."""
    
    def __init__(self, project_root: Path, output_dir: str = None):
        """
        Initialize Bird generator with project root path and optional output directory.
        
        Args:
            project_root (Path): Path to project root directory
            output_dir (str, optional): Path to output directory for generated files
        """
        super().__init__(project_root, output_dir)
        
        # Override data source mapping for Bird dataset
        self.data_sources = {
            'dev': self.project_root / 'outputs' / 'bird_dev_data.json'
        }
        
        # Bird dataset doesn't use tables.json, we'll load from detailed schema files
        self.bird_data_dir = project_root / 'data' / 'Bird' / 'dev_20240627' / 'dev_databases'
        
    def load_bird_data(self, data_source: str = 'dev') -> List[Dict[str, str]]:
        """
        Load Bird dataset data.
        
        Args:
            data_source (str): Data source type (currently only 'dev' is supported)
            
        Returns:
            List[Dict[str, str]]: Bird dataset entries
        """
        if data_source not in self.data_sources:
            logger.error(f"Unsupported data source: {data_source}")
            return []
            
        data_file = self.data_sources[data_source]
        
        if not data_file.exists():
            logger.error(f"Bird data file not found: {data_file}")
            return []
            
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} examples from Bird {data_source} dataset")
            return data
            
        except Exception as e:
            logger.error(f"Error loading Bird {data_source} data: {str(e)}")
            return []
    
    def load_bird_table_info(self, db_id: str) -> Optional[Dict[str, Any]]:
        """
        Load table information for a specific Bird database from its detailed schema file.
        
        Args:
            db_id (str): Database ID
            
        Returns:
            Optional[Dict[str, Any]]: Table information or None if not found
        """
        db_path = self.bird_data_dir / db_id
        schema_file = db_path / f"{db_id}_schema_detailed.txt"
        
        if not schema_file.exists():
            logger.warning(f"Detailed schema file not found for {db_id}: {schema_file}")
            return None
            
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_content = f.read()
            
            # Extract the actual schema description (skip the header comments)
            schema_lines = schema_content.split('\n')
            schema_desc = []
            for line in schema_lines:
                if not line.startswith('#') and line.strip():
                    schema_desc.append(line)
            
            schema_text = '\n'.join(schema_desc)
            
            # For Bird, we'll extract table names from the schema text
            tables = self._extract_table_names_from_schema(schema_text)
            
            return {
                'db_id': db_id,
                'schema_text': schema_text,
                'table_names': list(tables)
            }
            
        except Exception as e:
            logger.error(f"Error loading table info for {db_id}: {str(e)}")
            return None
    
    def _extract_table_names_from_schema(self, schema_text: str) -> Set[str]:
        """
        Extract table names from Bird's detailed schema text.
        
        Args:
            schema_text (str): Detailed schema text
            
        Returns:
            Set[str]: Set of table names
        """
        tables = set()
        
        # Look for table definitions in the schema text
        lines = schema_text.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that indicate table names
            if line.startswith('Table:') or line.startswith('TABLE:'):
                # Extract table name after "Table:" or "TABLE:"
                parts = line.split(':', 1)
                if len(parts) > 1:
                    table_name = parts[1].strip()
                    if table_name:
                        tables.add(table_name)
            elif 'Database:' in line or 'database' in line.lower():
                # Skip database identifier lines
                continue
            elif line and not line.startswith('-') and not line.startswith('='):
                # Look for table names in other formats
                # This is a heuristic approach based on Bird's schema format
                import re
                # Match potential table names at the beginning of lines
                table_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*[:.]', line)
                if table_match:
                    potential_table = table_match.group(1)
                    # Filter out common non-table words
                    if potential_table.lower() not in ['columns', 'column', 'tables', 'primary', 'foreign', 'key', 'description']:
                        tables.add(potential_table)
        
        logger.debug(f"Extracted tables from schema: {tables}")
        return tables
    
    def generate_bird_database_instruction(self, db_id: str, table_info: Dict[str, Any]) -> str:
        """
        Generate database instruction for Bird dataset using detailed schema.
        
        Args:
            db_id (str): Database ID
            table_info (Dict[str, Any]): Table information from detailed schema
            
        Returns:
            str: Generated instruction string
        """
        if not table_info:
            return f"{db_id} database schema information."
            
        try:
            # Use the detailed schema text directly as it contains rich information
            schema_text = table_info.get('schema_text', '')
            
            if schema_text:
                # Format the schema text into a more instruction-like format
                instruction = f"Database {db_id} contains the following structure:\n\n{schema_text}"
                return instruction
            else:
                # Fallback to basic table listing
                tables = table_info.get('table_names', [])
                if tables:
                    return f"{db_id} contains tables: {', '.join(tables)}."
                else:
                    return f"{db_id} database schema information."
                    
        except Exception as e:
            logger.error(f"Error generating instruction for {db_id}: {str(e)}")
            return f"{db_id} database schema information."
    
    def extract_tables_from_bird_query(self, query: str, db_id: str) -> Set[str]:
        """
        Extract table names from Bird SQL query with database-specific context.
        
        Args:
            query (str): SQL query string
            db_id (str): Database ID for context
            
        Returns:
            Set[str]: Set of table names used in the query
        """
        # First try the standard table extraction
        tables = super().extract_tables_from_query(query)
        
        # If no tables found or we want to validate, we can cross-reference
        # with the database's known tables
        if not tables or len(tables) == 0:
            logger.warning(f"No tables extracted from Bird query for {db_id}: {query}")
            
            # Try to get the database's table list for validation
            table_info = self.load_bird_table_info(db_id)
            if table_info and 'table_names' in table_info:
                known_tables = set(t.lower() for t in table_info['table_names'])
                
                # Look for table names mentioned in the query text
                query_lower = query.lower()
                for table in table_info['table_names']:
                    if table.lower() in query_lower:
                        tables.add(table)
                        logger.debug(f"Found table '{table}' in query text for {db_id}")
        
        return tables
    
    def generate_bird_role_sql_dataset(self, role_file_path: str = None, data_source: str = 'dev') -> List[Dict[str, str]]:
        """
        Generate role-based SQL dataset for Bird data.
        
        Args:
            role_file_path (str, optional): Path to Bird role assignments JSON file
            data_source (str): Data source type (currently only 'dev' is supported)
            
        Returns:
            List[Dict[str, str]]: Generated dataset
        """
        # Load data
        role_assignments = self.load_role_assignments(role_file_path)
        bird_data = self.load_bird_data(data_source)
        
        if not role_assignments:
            logger.error("No role assignments loaded")
            return []
            
        if not bird_data:
            logger.error("No Bird data loaded")
            return []
        
        # Import prompt templates from configs
        try:
            configs_path = self.project_root / 'configs'
            if str(configs_path) not in sys.path:
                sys.path.insert(0, str(configs_path))
            from prompts import INSTRUCTION_PROMPT, INPUT_PROMPT
        except ImportError:
            logger.warning("Could not import prompt templates from configs, using defaults")
            INSTRUCTION_PROMPT = "I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n##Instruction:\n{}\n"
            INPUT_PROMPT = "###Input:\n{}\n\n###Response:"
        
        # Generate dataset
        dataset = []
        processed_dbs = set()
        skipped_dbs = set()
        denied_count = 0
        allowed_count = 0
        
        for example in bird_data:
            db_id = example['db_id']
            
            # Skip if database has no role assignments
            if db_id not in role_assignments:
                skipped_dbs.add(db_id)
                continue
                
            processed_dbs.add(db_id)
            
            # Use the instruction from bird_dev_data.json directly (it already contains detailed schema)
            base_instruction = example.get('instruction', '')
            
            # Extract tables from query
            query_tables = self.extract_tables_from_bird_query(example['output'], db_id)
            
            # Generate examples for each role
            for role_info in role_assignments[db_id]:
                role_name = role_info['role']
                role_tables = role_info['tables']
                
                # Create dataset entry
                entry = {
                    'db_id': db_id,
                    'instruction': base_instruction,  # Use the rich instruction from bird_dev_data.json
                    'role': role_name,
                    'tables': role_tables,
                    'input': example['input'],  # Already formatted from bird_dev_data.json
                    'output': example['output']  # Will be modified based on permissions
                }
                
                # Check permission and set query
                if self.check_query_permission(query_tables, role_tables):
                    entry['output'] = example['output']
                    allowed_count += 1
                else:
                    entry['output'] = "Sorry, I cannot answer."
                    denied_count += 1
                    logger.debug(f"Access denied for {db_id}, role {role_name}: query uses {query_tables}, role has {role_tables}")
                
                dataset.append(entry)
        
        # Log statistics
        logger.info(f"Bird dataset generation completed for {data_source} data:")
        logger.info(f"- Databases processed: {len(processed_dbs)}")
        logger.info(f"- Databases skipped (no roles): {len(skipped_dbs)}")
        logger.info(f"- Total examples generated: {len(dataset)}")
        logger.info(f"- Queries allowed: {allowed_count}")
        logger.info(f"- Queries denied: {denied_count}")
        logger.info(f"- Denial rate: {denied_count/(allowed_count + denied_count)*100:.2f}%")
        
        if skipped_dbs:
            logger.debug(f"Skipped databases: {sorted(skipped_dbs)}")
        
        return dataset
    
    def save_bird_dataset(self, dataset: List[Dict[str, str]], output_path: str = None, 
                         timestamp: str = None) -> str:
        """
        Save Bird role-based dataset to JSON file.
        
        Args:
            dataset (List[Dict[str, str]]): Dataset to save
            output_path (str, optional): Output file path
            timestamp (str, optional): Timestamp for filename
            
        Returns:
            str: Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        if output_path is None:
            output_path = str(self.output_dir / f'role_sql_dataset_bird_{timestamp}_with_instructions_dev.json')
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Bird dataset saved to: {output_path}")
            logger.info(f"File size: {Path(output_path).stat().st_size:,} bytes")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving Bird dataset: {str(e)}")
            raise