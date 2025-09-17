"""
Generate role-based access control text2sql dataset by combining Spider dataset with role assignments.
"""

import json
import logging
import sqlparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any

logger = logging.getLogger('role_sql')

class RoleSQLGenerator:
    """Generate role-based SQL queries dataset based on access control."""
    
    def __init__(self, project_root: Path, spider_train_path: str = None, output_dir: str = None):
        """
        Initialize generator with project root path and optional file paths.
        
        Args:
            project_root (Path): Path to project root directory
            spider_train_path (str, optional): Path to spider train data JSON file
            output_dir (str, optional): Path to output directory for generated files
        """
        self.project_root = project_root
        self.spider_train_path = Path(spider_train_path) if spider_train_path else project_root / 'data/spider_train_data.json'
        self.output_dir = Path(output_dir) if output_dir else project_root / 'outputs'
        
    def extract_tables_from_query(self, query: str) -> Set[str]:
        """
        Extract table names from a SQL query.
        
        Args:
            query (str): SQL query string
            
        Returns:
            Set[str]: Set of table names used in the query
        """
        tables = set()
        
        def clean_table_name(name: str) -> str:
            """Clean and normalize table name"""
            name = name.strip()
            
            # Handle alias cases
            alias_markers = [' as ', ' AS ', ' As ', ' aS ']
            for marker in alias_markers:
                if marker in name:
                    name = name.split(marker)[0]
                    break
                    
            # Handle schema qualification
            if '.' in name:
                name = name.split('.')[-1]
                
            # Clean up any remaining special characters
            name = name.strip('"`[] \n\t\r')
            
            # Validate and return
            if name and not name.isspace():
                return name.lower()
            return None
            
        def extract_from_identifiers(token_list, in_join=False):
            """Extract table names from a list of identifiers"""
            for item in token_list:
                if isinstance(item, sqlparse.sql.Identifier):
                    name = clean_table_name(item.value)
                    if name:
                        tables.add(name)
                        # Log extraction for debugging
                        logger.debug(f"Extracted table name: {name} from {item.value}")
                elif isinstance(item, sqlparse.sql.IdentifierList):
                    extract_from_identifiers(item.get_identifiers())
                    
        def process_statement(statement):
            """Process a SQL statement to extract table names"""
            is_after_from_or_join = False
            
            for token in statement.tokens:
                # Skip whitespace and comments
                if token.is_whitespace or token.ttype in (sqlparse.tokens.Comment,):
                    continue
                    
                # Handle FROM and JOIN keywords
                if token.ttype is sqlparse.tokens.Keyword:
                    if token.value.upper() in ('FROM', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN'):
                        is_after_from_or_join = True
                        continue
                    elif token.value.upper() in ('WHERE', 'GROUP', 'ORDER', 'HAVING', 'UNION', 'INTERSECT', 'EXCEPT'):
                        is_after_from_or_join = False
                
                # Extract table names after FROM or JOIN
                if is_after_from_or_join:
                    if isinstance(token, (sqlparse.sql.Identifier, sqlparse.sql.IdentifierList)):
                        extract_from_identifiers([token], True)
                    is_after_from_or_join = False
                    
                # Handle subqueries
                elif isinstance(token, sqlparse.sql.Parenthesis):
                    subquery = token.value[1:-1]  # Remove parentheses
                    if subquery.upper().strip().startswith('SELECT'):
                        tables.update(self.extract_tables_from_query(subquery))

        try:
            # Parse the SQL query
            statements = sqlparse.parse(query)
            
            # Process each statement
            for statement in statements:
                process_statement(statement)
                
            # Log extraction results
            if tables:
                logger.debug(f"Extracted tables from query: {tables}")
            else:
                logger.warning(f"No tables extracted from query: {query}")
                
        except Exception as e:
            logger.error(f"Error extracting tables from query: {e}")
            logger.error(f"Query: {query}")
            
        return tables
            
        try:
            # Parse each statement in the query
            statements = sqlparse.parse(query)
            if not statements:
                return tables
                
            # Process each statement
            for statement in statements:
                expect_table = False
                
                # Process all tokens
                for token in statement.flatten():
                    # Check if we should expect a table name next
                    if expect_table:
                        if isinstance(token, sqlparse.sql.Identifier):
                            name = clean_table_name(token.value)
                            if name:
                                tables.add(name)
                        expect_table = False
                    else:
                        # Check for FROM/JOIN keywords
                        expect_table = process_token(token)
                        
        except Exception as e:
            logger.error(f"Error extracting tables from query: {e}")
            logger.error(f"Query: {query}")
            
        return tables
        
        try:
            # Parse and process the SQL query
            statements = sqlparse.parse(query)
            if not statements:
                return tables
            
            # Process main query and all subqueries
            for statement in statements:
                process_statement(statement)
                    
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            logger.error(f"Query: {query}")
            
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
            
    def load_spider_data(self) -> List[Dict[str, str]]:
        """
        Load Spider training data from specified path.
        
        Returns:
            List[Dict[str, str]]: List of Spider training examples
        """
        try:
            with open(self.spider_train_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading Spider data from {self.spider_train_path}: {str(e)}")
            raise
            
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
        
    def generate_role_sql_dataset(self, role_file_path: str = None) -> List[Dict[str, str]]:
        """
        Generate role-based SQL dataset by combining Spider data with role assignments.
        
        Args:
            role_file_path (str, optional): Path to role assignments JSON file
            
        Returns:
            List[Dict[str, str]]: Generated dataset
        """
        # Load data
        role_assignments = self.load_role_assignments(role_file_path)
        spider_data = self.load_spider_data()
        
        # Generate dataset
        dataset = []
        for example in spider_data:
            db_id = example['db_id']
            
            # Skip if database has no role assignments
            if db_id not in role_assignments:
                continue
                
            # Extract tables from query
            query_tables = self.extract_tables_from_query(example['query'])
            
            # Generate examples for each role
            for role_info in role_assignments[db_id]:
                role_name = role_info['role']
                role_tables = role_info['tables']
                
                # Create dataset entry
                entry = {
                    'db_id': db_id,
                    'role': role_name,
                    'tables': role_tables,
                    'question': example['question']
                }
                
                # Check permission and set query
                if self.check_query_permission(query_tables, role_tables):
                    entry['query'] = example['query']
                else:
                    entry['query'] = "Sorry, I cannot answer."
                    # Add both the denial message and the original query for verification
                    # entry['query'] = f"Sorry, I cannot answer. Original query (for verification): {example['query']}"
                dataset.append(entry)
        
        return dataset
        
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
    
    # Initialize generator with default paths
    generator = RoleSQLGenerator(
        project_root=project_root,
        spider_train_path=project_root / 'data/spider_train_data.json',
        output_dir=project_root / 'outputs'
    )
    
    # Example: Generate and save dataset with custom role assignments file
    role_file = project_root / 'outputs/role_assignments_20250917_111537.json'
    dataset = generator.generate_role_sql_dataset(role_file_path=str(role_file))
    
    # Save with custom output path
    output_path = project_root / 'outputs/role_sql_dataset_custom.json'
    generator.save_dataset(dataset, output_path=str(output_path))
