"""
Generate role-based access control text2sql dataset by combining Spider dataset with role assignments.
"""

import json
import sys
import logging
import sqlparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

logger = logging.getLogger('role_sql')

class RoleSQLGenerator:
    """Generate role-based SQL queries dataset based on access control."""
    
    def __init__(self, project_root: Path, output_dir: str = None):
        """
        Initialize generator with project root path and optional output directory.
        
        Args:
            project_root (Path): Path to project root directory
            output_dir (str, optional): Path to output directory for generated files
        """
        self.project_root = project_root
        self.data_dir = project_root / 'data'
        self.output_dir = Path(output_dir) if output_dir else project_root / 'outputs'
        
        # Data source mapping
        self.data_sources = {
            'train': self.data_dir / 'spider_train_data.json',
            'dev': self.data_dir / 'spider_dev_data.json', 
            'test': self.data_dir / 'spider_test_data.json',
            'combined': self.data_dir / 'spider_combined_data.json'
        }
        
        # Tables information files
        self.tables_files = {
            'train_dev': project_root / 'data' / 'spider' / 'tables.json',
            'test': project_root / 'data' / 'spider' / 'test_tables.json'
        }
        
        # Cache for table information
        self._tables_cache = {}
        
    def extract_tables_from_query(self, query: str) -> Set[str]:
        """
        Extract table names from a SQL query.
        
        Args:
            query (str): SQL query string
            
        Returns:
            Set[str]: Set of table names used in the query
        """
        tables = set()
        
        try:
            # Parse the SQL query
            statements = sqlparse.parse(query)
            
            for statement in statements:
                # Use a more targeted approach to find table names
                self._extract_tables_from_statement(statement, tables)
                        
            # Log extraction results
            if tables:
                logger.debug(f"Extracted tables from query: {tables}")
            else:
                logger.warning(f"No tables extracted from query: {query}")
                
        except Exception as e:
            logger.error(f"Error extracting tables from query: {e}")
            logger.error(f"Query: {query}")
            
        return tables
    
    def _extract_tables_from_statement(self, statement, tables):
        """Extract table names from a parsed SQL statement"""
        # Convert statement to string and use regex-based approach for better accuracy
        statement_str = str(statement).strip()
        
        # Use regex to find FROM and JOIN clauses
        import re
        
        # Pattern to match FROM clause: FROM table_name or FROM table AS alias
        from_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?)'
        from_matches = re.finditer(from_pattern, statement_str, re.IGNORECASE)
        
        for match in from_matches:
            table_part = match.group(1).strip()
            table_name = self._clean_table_name(table_part)
            if table_name:
                tables.add(table_name)
                logger.debug(f"Extracted table name from FROM: {table_name}")
        
        # Pattern to match JOIN clauses: JOIN table_name or JOIN table AS alias
        join_pattern = r'\b(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+|OUTER\s+)?JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s+(?:AS\s+)?[a-zA-Z_][a-zA-Z0-9_]*)?)'
        join_matches = re.finditer(join_pattern, statement_str, re.IGNORECASE)
        
        for match in join_matches:
            table_part = match.group(1).strip()
            table_name = self._clean_table_name(table_part)
            if table_name:
                tables.add(table_name)
                logger.debug(f"Extracted table name from JOIN: {table_name}")
        
        # Handle subqueries
        subquery_pattern = r'\(([^()]*SELECT[^()]*)\)'
        subqueries = re.findall(subquery_pattern, statement_str, re.IGNORECASE | re.DOTALL)
        
        for subquery in subqueries:
            subquery_tables = self.extract_tables_from_query(subquery.strip())
            tables.update(subquery_tables)
    
    def _clean_table_name(self, name: str) -> str:
        """Clean and normalize table name"""
        if not name:
            return None
            
        name = name.strip()
        
        # Handle alias cases - split on space and take first part
        if ' ' in name:
            parts = name.split()
            # Check if it's an alias pattern (table_name alias_name)
            if len(parts) >= 2 and parts[1].upper() not in ('AS', 'ON', 'WHERE', 'GROUP', 'ORDER', 'HAVING'):
                name = parts[0]
            else:
                name = parts[0]
        
        # Handle explicit AS aliases
        alias_markers = [' as ', ' AS ', ' As ', ' aS ']
        for marker in alias_markers:
            if marker in name:
                name = name.split(marker)[0]
                break
                
        # Handle schema qualification
        if '.' in name:
            name = name.split('.')[-1]
            
        # Clean up any remaining special characters
        name = name.strip('"`[]() \n\t\r')
        
        # Validate and return - exclude SQL keywords
        excluded_keywords = {
            'select', 'from', 'where', 'group', 'order', 'having', 'join', 'on', 'as',
            'inner', 'left', 'right', 'full', 'outer', 'union', 'intersect', 'except',
            'limit', 'offset', 'distinct', 'all', 'and', 'or', 'not', 'in', 'exists',
            'between', 'like', 'is', 'null', 'true', 'false', 'case', 'when', 'then',
            'else', 'end', 'count', 'sum', 'avg', 'min', 'max'
        }
        
        if name and not name.isspace() and name.lower() not in excluded_keywords:
            return name.lower()
        return None
    
    def _extract_from_complex_token(self, token, tables):
        """Helper method to extract table names from complex tokens"""
        if isinstance(token, sqlparse.sql.Identifier):
            # Handle table identifiers
            table_name = token.get_real_name()
            if table_name:
                clean_name = table_name.strip('"`[]() \n\t\r').lower()
                if clean_name and clean_name not in ('select', 'from', 'where', 'group', 'order', 'having', 'join', 'on', 'as'):
                    tables.add(clean_name)
                    logger.debug(f"Extracted table name from identifier: {clean_name}")
        
        elif isinstance(token, sqlparse.sql.IdentifierList):
            # Handle lists of identifiers
            for identifier in token.get_identifiers():
                table_name = identifier.get_real_name() if hasattr(identifier, 'get_real_name') else str(identifier)
                if table_name:
                    clean_name = table_name.strip('"`[]() \n\t\r').lower()
                    if clean_name and clean_name not in ('select', 'from', 'where', 'group', 'order', 'having', 'join', 'on', 'as'):
                        tables.add(clean_name)
                        logger.debug(f"Extracted table name from identifier list: {clean_name}")
    
    def _extract_from_subqueries(self, query, tables):
        """Helper method to extract table names from subqueries"""
        try:
            # Find subqueries in parentheses
            import re
            subquery_pattern = r'\(([^()]*SELECT[^()]*)\)'
            subqueries = re.findall(subquery_pattern, query, re.IGNORECASE | re.DOTALL)
            
            for subquery in subqueries:
                subquery_tables = self.extract_tables_from_query(subquery.strip())
                tables.update(subquery_tables)
                
        except Exception as e:
            logger.debug(f"Error extracting from subqueries: {e}")
            
        return tables
        
    def load_role_assignments(self, role_file_path: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load role assignments from the specified file path or latest file.
        
        Args:
            role_file_path (str, optional): Path to role assignments JSON file
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Role assignments by database
        """
        try:
            if role_file_path:
                role_file = Path(role_file_path)
            else:
                # Get latest role assignment file
                role_files = list(self.output_dir.glob('role_assignments_*.json'))
                if not role_files:
                    raise FileNotFoundError("No role assignment files found")
                role_file = max(role_files, key=lambda x: x.stat().st_mtime)
                
            with open(role_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['assignments']
                
        except Exception as e:
            logger.error(f"Error loading role assignments: {str(e)}")
            raise
            
    def load_spider_data(self, data_source: str = 'train') -> List[Dict[str, str]]:
        """
        Load Spider data from specified data source.
        
        Args:
            data_source (str): Data source type ('train', 'dev', 'test', 'combined')
        
        Returns:
            List[Dict[str, str]]: List of Spider examples
        """
        if data_source not in self.data_sources:
            raise ValueError(f"Unknown data source: {data_source}")
            
        data_path = self.data_sources[data_source]
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
            
    def load_table_info(self, data_source: str = 'train') -> Dict[str, Dict[str, Any]]:
        """
        Load table information from Spider tables.json files.
        
        Args:
            data_source (str): Data source type ('train', 'dev', 'test', 'combined')
            
        Returns:
            Dict[str, Dict[str, Any]]: Table information by database ID
        """
        # Determine which tables file to use
        if data_source in ['train', 'dev', 'combined']:
            tables_key = 'train_dev'
        else:  # test
            tables_key = 'test'
            
        # Check cache first
        if tables_key in self._tables_cache:
            return self._tables_cache[tables_key]
            
        tables_file = self.tables_files[tables_key]
        
        try:
            with open(tables_file, 'r', encoding='utf-8') as f:
                tables_list = json.load(f)
            
            # Convert to dict indexed by db_id
            tables_dict = {item['db_id']: item for item in tables_list}
            
            # Cache the result
            self._tables_cache[tables_key] = tables_dict
            
            logger.info(f"Loaded table info for {len(tables_dict)} databases from {tables_file}")
            return tables_dict
            
        except FileNotFoundError:
            logger.error(f"Tables file not found: {tables_file}")
            return {}
        except Exception as e:
            logger.error(f"Error loading table info: {str(e)}")
            return {}
    
    def generate_database_instruction(self, db_id: str, table_info: Dict[str, Any]) -> str:
        """
        Generate database instruction similar to decode_json_file function.
        
        Args:
            db_id (str): Database ID
            table_info (Dict[str, Any]): Table information from tables.json
            
        Returns:
            str: Generated instruction string
        """
        try:
            tables = table_info["table_names_original"]
            columns = table_info["column_names_original"][1:]  # Skip the first [-1, "*"] entry
            primary_keys = table_info["primary_keys"]
            foreign_keys = table_info["foreign_keys"]
            
            # Start building the instruction
            instruction = f"{db_id} contains tables such as " + ", ".join(tables) + ". "
            
            # Add column information for each table
            for i, table_name in enumerate(tables):
                table_columns = [column[1] for column in columns if column[0] == i]
                instruction += f"Table {table_name} has columns such as " + ", ".join(table_columns) + ". "
                
                # Add primary key information
                for j in range(len(primary_keys)):
                    if isinstance(primary_keys[j], int):
                        # Single primary key
                        if columns[primary_keys[j] - 1][0] == i:
                            instruction += f"{columns[primary_keys[j] - 1][1]} is the primary key.\n"
                    elif isinstance(primary_keys[j], list):
                        # Composite primary key
                        keys = []
                        for k in range(len(primary_keys[j])):
                            if columns[primary_keys[j][k] - 1][0] == i:
                                keys.append(columns[primary_keys[j][k] - 1][1])
                        if keys:
                            instruction += f"The combination of ({', '.join(keys)}) are the primary key.\n"
            
            # Add foreign key information
            for key in foreign_keys:
                try:
                    foreign_column = columns[key[0] - 1][1]
                    foreign_table = tables[columns[key[0] - 1][0]]
                    reference_column = columns[key[1] - 1][1]
                    reference_table = tables[columns[key[1] - 1][0]]
                    
                    instruction += f"The {foreign_column} of {foreign_table} is the foreign key of {reference_column} of {reference_table}.\n"
                except (IndexError, KeyError) as e:
                    logger.debug(f"Skipping invalid foreign key in {db_id}: {key}, error: {str(e)}")
                    continue
            
            return instruction.strip()
            
        except Exception as e:
            logger.error(f"Error generating instruction for {db_id}: {str(e)}")
            return f"{db_id} database schema information."
            
    def check_query_permission(self, query_tables: Set[str], role_tables: str) -> bool:
        """
        Check if a role has permission to access all tables in a query.
        
        Args:
            query_tables (Set[str]): Tables used in the query
            role_tables (str): Tables accessible by the role
            
        Returns:
            bool: True if role has access to all required tables
        """
        # Convert role_tables string to set and normalize to lower case
        role_tables_set = {t.strip().lower() for t in role_tables.split(',') if t.strip()}
        
        # Convert query tables to lower case
        query_tables_lower = {t.lower() for t in query_tables if t.strip()}
        
        # Log the permission check details
        logger.debug(f"Checking permissions:")
        logger.debug(f"Role has access to: {role_tables_set}")
        logger.debug(f"Query uses tables: {query_tables_lower}")
        
        # Check each required table
        for table in query_tables_lower:
            if table not in role_tables_set:
                logger.info(f"Permission denied: Table '{table}' is not accessible")
                return False
                
        return True
        
    def generate_role_sql_dataset(
        self,
        role_file_path: str = None,
        data_source: str = 'train',
        *,
        spider_data: Optional[List[Dict[str, Any]]] = None,
        instruction_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Generate Spider role-conditioned dataset with minimal fields.

        Args:
            role_file_path: Path to the role assignment JSON file.
            data_source: Spider split to use (train, dev, test, combined).
            spider_data: Optional pre-processed examples; if omitted, the split
                will be loaded from disk.
            instruction_prompt: Optional override for instruction template. When
                omitted, the prompt from ``configs/prompts.py`` (``INSTRUCTION_PROMPT``)
                is used.

        Returns:
            A list of dictionaries. Each entry contains ``db_id``, ``instruction``,
            ``role``, ``tables``, ``input``, and ``output``.
        """

        role_assignments = self.load_role_assignments(role_file_path)
        if not role_assignments:
            logger.error("No role assignments available; aborting dataset generation.")
            return []

        if spider_data is None:
            spider_data = self.load_spider_data(data_source)

        if not spider_data:
            logger.error("Spider source data is empty; aborting dataset generation.")
            return []

        table_info = self.load_table_info(data_source)

        try:
            configs_path = self.project_root / 'configs'
            if str(configs_path) not in sys.path:
                sys.path.insert(0, str(configs_path))
            from prompts import INSTRUCTION_PROMPT  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("Could not import INSTRUCTION_PROMPT from configs; using default template.")
            INSTRUCTION_PROMPT = (
                "I want you to act as a SQL terminal in front of an example database, you need only to "
                "return the sql command to me.Below is an instruction that describes a task, Write a response "
                "that appropriately completes the request.\n##Instruction:\n{}\n"
            )

        instruction_template = instruction_prompt or INSTRUCTION_PROMPT

        dataset: List[Dict[str, str]] = []
        processed_dbs: Set[str] = set()
        skipped_dbs: Set[str] = set()
        allowed_count = 0
        denied_count = 0

        for example in spider_data:
            db_id = example.get('db_id')
            if not db_id:
                logger.debug("Skipping record without db_id: %s", example)
                continue

            if db_id not in role_assignments:
                skipped_dbs.add(db_id)
                continue

            question = example.get('question') or example.get('input')
            sql_query = example.get('query') or example.get('output')

            if question is None or not sql_query:
                logger.debug(
                    "Skipping record for %s due to missing question or SQL (question=%s, sql_present=%s)",
                    db_id,
                    question is not None,
                    bool(sql_query),
                )
                continue

            processed_dbs.add(db_id)

            instruction = example.get('instruction')
            if not instruction:
                if db_id in table_info:
                    db_instruction = self.generate_database_instruction(db_id, table_info[db_id])
                    instruction = instruction_template.format(db_instruction)
                else:
                    logger.warning("No table info found for database: %s", db_id)
                    instruction = instruction_template.format(f"{db_id} database")

            input_text = example.get('input') or question

            query_tables = self.extract_tables_from_query(sql_query)

            for role_info in role_assignments[db_id]:
                role_name = role_info.get('role')
                role_tables = role_info.get('tables', '')

                if not role_name:
                    logger.debug("Skipping role entry without name for database %s", db_id)
                    continue

                has_permission = self.check_query_permission(query_tables, role_tables)
                output_sql = sql_query if has_permission else "Sorry, I cannot answer."

                if has_permission:
                    allowed_count += 1
                else:
                    denied_count += 1

                entry = {
                    'db_id': db_id,
                    'instruction': instruction,
                    'role': role_name,
                    'tables': role_tables,
                    'input': input_text,
                    'output': output_sql,
                }

                dataset.append(entry)

        logger.info("Dataset generation completed for %s data:", data_source)
        logger.info("- Databases processed: %s", len(processed_dbs))
        logger.info("- Databases skipped (no roles): %s", len(skipped_dbs))
        logger.info("- Total examples generated: %s", len(dataset))
        logger.info("- Allowed queries: %s", allowed_count)
        logger.info("- Denied queries: %s", denied_count)

        if skipped_dbs:
            logger.debug("Skipped databases: %s", sorted(skipped_dbs))

        return dataset
        
    def generate_all_datasets(self, role_file_path: str = None, timestamp: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate role-based SQL datasets for all available data sources.
        
        Args:
            role_file_path (str, optional): Path to role assignments JSON file
            timestamp (str, optional): Timestamp for output filenames
            
        Returns:
            Dict[str, Dict[str, Any]]: Results for each data source with statistics
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {}
        
        # Process each data source
        for data_source in ['train', 'dev', 'test']:
            try:
                # Check if data file exists
                if not self.data_sources[data_source].exists():
                    logger.warning(f"Data file for {data_source} not found, skipping")
                    continue
                
                logger.info(f"\nGenerating {data_source} dataset...")
                
                # Generate dataset
                dataset = self.generate_role_sql_dataset(role_file_path, data_source)
                
                if dataset:
                    # Save dataset
                    output_path = self.output_dir / f'role_sql_dataset_{data_source}_{timestamp}.json'
                    self.save_dataset(dataset, str(output_path))
                    
                    # Calculate statistics
                    total_examples = len(dataset)
                    databases = len({example['db_id'] for example in dataset})
                    roles = len({(example['db_id'], example['role']) for example in dataset})
                    denied_queries = sum(1 for example in dataset if example.get('output') == "Sorry, I cannot answer.")
                    
                    results[data_source] = {
                        'total_examples': total_examples,
                        'databases': databases,
                        'roles': roles,
                        'denied_queries': denied_queries,
                        'denial_rate': denied_queries/total_examples*100 if total_examples else 0,
                        'output_path': str(output_path)
                    }
                    
                    logger.info(f"{data_source.title()} dataset statistics:")
                    logger.info(f"- Total examples: {total_examples}")
                    logger.info(f"- Databases: {databases}")
                    logger.info(f"- Unique roles: {roles}")
                    logger.info(f"- Denied queries: {denied_queries} ({denied_queries/total_examples*100:.2f}%)")
                else:
                    logger.warning(f"No dataset generated for {data_source}")
                    results[data_source] = {'error': 'No dataset generated'}
                    
            except Exception as e:
                logger.error(f"Failed to process {data_source} dataset: {str(e)}")
                results[data_source] = {'error': str(e)}
        
        # Generate combined dataset if individual ones exist
        if any('error' not in result for result in results.values()):
            try:
                # Check if combined data file exists
                if self.data_sources['combined'].exists():
                    logger.info(f"\nGenerating combined dataset...")
                    combined_dataset = self.generate_role_sql_dataset(role_file_path, 'combined')
                    
                    if combined_dataset:
                        output_path = self.output_dir / f'role_sql_dataset_combined_{timestamp}.json'
                        self.save_dataset(combined_dataset, str(output_path))
                        
                        total_examples = len(combined_dataset)
                        databases = len({example['db_id'] for example in combined_dataset})
                        roles = len({(example['db_id'], example['role']) for example in combined_dataset})
                        denied_queries = sum(1 for example in combined_dataset if example.get('output') == "Sorry, I cannot answer.")
                        
                        results['combined'] = {
                            'total_examples': total_examples,
                            'databases': databases,
                            'roles': roles,
                            'denied_queries': denied_queries,
                            'denial_rate': denied_queries/total_examples*100 if total_examples else 0,
                            'output_path': str(output_path)
                        }
                        
                        logger.info(f"Combined dataset statistics:")
                        logger.info(f"- Total examples: {total_examples}")
                        logger.info(f"- Databases: {databases}")
                        logger.info(f"- Unique roles: {roles}")
                        logger.info(f"- Denied queries: {denied_queries} ({denied_queries/total_examples*100:.2f}%)")
                
            except Exception as e:
                logger.error(f"Failed to generate combined dataset: {str(e)}")
                results['combined'] = {'error': str(e)}
        
        return results
        
    def save_dataset(self, dataset: List[Dict[str, str]], output_path: str = None) -> None:
        """
        Save generated dataset to file.
        
        Args:
            dataset (List[Dict[str, str]]): Generated dataset
            output_path (str, optional): Path for output file. If not provided,
                                       will save with timestamp in output directory
        """
        try:
            if output_path:
                output_file = Path(output_path)
                # Create parent directories if needed
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Create output directory if needed
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = self.output_dir / f'role_sql_dataset_{timestamp}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Dataset saved to {output_file}")
            logger.info(f"Total examples generated: {len(dataset)}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            raise
            
if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get project root directory
    project_root = Path(__file__).parents[2]
    
    # Initialize generator
    generator = RoleSQLGenerator(
        project_root=project_root,
        output_dir=project_root / 'outputs'
    )
    
    # Example: Generate datasets for all data sources
    role_file = project_root / 'outputs/role_assignments_20250917_111537.json'
    results = generator.generate_all_datasets(role_file_path=str(role_file))
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET GENERATION SUMMARY")
    print("="*50)
    
    for data_source, result in results.items():
        print(f"\n{data_source.upper()} Dataset:")
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✅ Examples: {result['total_examples']}")
            print(f"  📊 Databases: {result['databases']}")
            print(f"  👥 Roles: {result['roles']}")
            print(f"  🚫 Denied: {result['denied_queries']} ({result['denial_rate']:.1f}%)")
            print(f"  📄 Output: {result['output_path']}")
    
    print("\n" + "="*50)
