"""Role generator for database RBAC roles."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

from src.llm_oracle import Oracle
from .parser import RoleParser
from configs.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger('role_assignment')

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

    def __str__(self):
        return f"RoleGenerator(model={self.oracle.model})"
    def __repr__(self):
        return self.__str__()
    def save_assignments(self, assignments: Dict, output_dir: Path, timestamp: str) -> Path:
        output_file = output_dir / f'role_assignments_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(assignments, f, indent=2)
        logger.info(f"Role assignments saved to {output_file}")
        return output_file


class ParallelRoleGenerator:
    """Generator for database role assignments using parallel LLM."""
    def __init__(self, model: str, api_key: str, n_workers: int = 12):
        self.oracle = Oracle(model=model, apikey=api_key)
        self.parser = RoleParser()
        self.n_workers = n_workers

    def read_schema_file(self, db_path: Path) -> Optional[str]:
        schema_file = db_path / 'schema.sql'
        if not schema_file.exists():
            return None
        with open(schema_file, 'r') as f:
            return f.read()

    def assign_roles_to_schema_parallel(self, schema_contents: List[str]) -> List[Dict]:
        prompt_list = [USER_PROMPT_TEMPLATE.format(schema_content=content) for content in schema_contents]
        logger.info(f"Starting parallel API calls with {self.n_workers} workers")
        logger.debug(f"System Prompt: {SYSTEM_PROMPT}")
        try:
            responses = self.oracle.query_all(
                prompt_sys=SYSTEM_PROMPT,
                prompt_user_all=prompt_list,
                workers=self.n_workers,
                temp=0.7,
                top_p=0.9,
                max_completion_tokens=1000
            )
            if not responses:
                logger.error("No responses received from API")
                return []
            self.log_api_statistics(responses)
            return responses
        except Exception as e:
            logger.error(f"Error in parallel API calls: {str(e)}")
            return []

    def log_api_statistics(self, responses: List[Dict]):
        total_prompt, total_completion, total_cached = 0, 0, 0
        for r in responses:
            usage = r.get('usage', {})
            total_prompt += usage.get('prompt_tokens', 0)
            total_completion += usage.get('completion_tokens', 0)
            details = usage.get('prompt_tokens_details', {})
            cached = getattr(details, 'cached_tokens', 0) if hasattr(details, 'cached_tokens') else 0
            total_cached += cached
        hit_rate = (total_cached / total_prompt * 100) if total_prompt else 0
        logger.info(f"Summary Statistics: Total API Calls: {len(responses)}, Total Prompt Tokens: {total_prompt}, Total Completion Tokens: {total_completion}, Total Cached Tokens: {total_cached}, Overall Cache Hit Rate: {hit_rate:.1f}%")

    def process_databases_parallel(self, db_paths: List[Path]) -> List[Dict]:
        """Process multiple databases in parallel using Oracle's query_all functionality.
        
        Args:
            db_paths (List[Path]): List of paths to database directories
            n_workers (int): Number of worker threads to use
        Returns:
            List[Dict]: List of dictionaries containing database names, roles, and timestamps
        """
        schema_contents, valid_dbs, skipped = [], [], []
        for db in db_paths:
            content = self.read_schema_file(db)
            if content:
                schema_contents.append(content)
                valid_dbs.append(db)
            else:
                skipped.append(db.name)
                logger.error(f"Database '{db.name}' skipped: schema.sql file not found")
        if not schema_contents:
            logger.error("No valid schema files found in any database")
            return []
        logger.info(f"Processing {len(valid_dbs)} databases ({len(skipped)} skipped)")
        if skipped:
            logger.info(f"Skipped databases: {', '.join(skipped)}")
        try:
            responses = self.assign_roles_to_schema_parallel(schema_contents)
            results = []
            for db, resp in zip(valid_dbs, responses):
                usage = resp.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                details = usage.get('prompt_tokens_details', {})
                cached = getattr(details, 'cached_tokens', 0) if hasattr(details, 'cached_tokens') else 'N/A'
                non_cached = prompt_tokens - cached if isinstance(cached, int) else 'N/A'
                hit_rate = f"{(cached / prompt_tokens * 100):.1f}%" if isinstance(cached, int) and prompt_tokens else 'N/A'
                roles = self.parser.parse_roles(resp.get('answer', '')) if resp.get('answer') else []
                if roles:
                    results.append({'database': db.name, 'roles': roles, 'timestamp': datetime.now().isoformat()})
                    logger.info(f"Generated {len(roles)} roles for {db.name} - Cached Tokens: {cached}; Uncached tokens: {non_cached}; Hit rates: {hit_rate}")
                else:
                    logger.error(f"No valid roles parsed for database '{db.name}'")
            return results
        except Exception as e:
            logger.error(f"Error in parallel database processing: {str(e)}")
            return []
        
    def save_assignments_parallel(self, assignments: Dict, output_dir: Path, timestamp: str) -> Path:
        output_file = output_dir / f'role_assignments_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(assignments, f, indent=2)
        logger.info(f"Role assignments saved to {output_file}")
        return output_file
