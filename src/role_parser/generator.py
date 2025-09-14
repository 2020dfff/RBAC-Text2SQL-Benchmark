"""Role generator for database RBAC roles."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

from llm_oracle import Oracle
from .parser import RoleParser
from configs.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class RoleGenerator:
    """Generator for database role assignments using LLM."""
    
    def __init__(self, model: str, api_key: str):
        """Initialize the role generator.
        
        Args:
            model (str): LLM model name
            api_key (str): API key for the LLM service
        """
        self.oracle = Oracle(model=model, apikey=api_key)
        self.parser = RoleParser()
        
    def read_schema_file(self, db_path: Path) -> Optional[str]:
        """Read schema.sql file from a database folder.
        
        Args:
            db_path (Path): Path to the database directory
            
        Returns:
            str: Schema content if file exists, None otherwise
        """
        schema_file = db_path / 'schema.sql'
        if not schema_file.exists():
            return None
        
        with open(schema_file, 'r') as f:
            return f.read()

    def assign_roles_to_schema(self, schema_content: str) -> List[Dict[str, str]]:
        """Assign roles to a database schema using LLM.
        
        Args:
            schema_content (str): Database schema content
            
        Returns:
            list: List of role assignments
        """
        response = self.oracle.query(
            prompt_sys=SYSTEM_PROMPT,
            prompt_user=USER_PROMPT_TEMPLATE.format(schema_content=schema_content),
            temp=0.7,
            top_p=0.9
        )
        
        # Log token usage statistics
        if 'usage' in response:
            usage = response['usage']
            logger.info("API Call Statistics:")
            logger.info(f"  Prompt tokens: {usage.get('prompt_tokens', 0)}")
            logger.info(f"  Completion tokens: {usage.get('completion_tokens', 0)}")
            logger.info(f"  Total tokens: {usage.get('total_tokens', 0)}")
        
        return self.parser.parse_roles(response['answer'])

    def process_database(self, db_path: Path) -> Optional[Dict]:
        """Process a single database and assign roles.
        
        Args:
            db_path (Path): Path to the database directory
            
        Returns:
            dict: Dictionary containing database name, roles, and timestamp
        """
        schema_content = self.read_schema_file(db_path)
        if not schema_content:
            return None
        
        try:
            logger.info(f"Processing schema for database: {db_path.name}")
            roles = self.assign_roles_to_schema(schema_content)
            
            return {
                'database': db_path.name,
                'roles': roles,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing database {db_path.name}: {str(e)}")
            return None

    def save_assignments(self, assignments: Dict, output_dir: Path, timestamp: str) -> Path:
        """Save role assignments to JSON file.
        
        Args:
            assignments (dict): Role assignments data
            output_dir (Path): Directory to save output file
            timestamp (str): Timestamp string for filename
            
        Returns:
            Path: Path to the saved file
        """
        output_file = output_dir / f'role_assignments_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(assignments, f, indent=2)
        logger.info(f"Role assignments saved to {output_file}")
        return output_file
