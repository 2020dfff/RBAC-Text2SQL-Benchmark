"""
Generate role-based access control text2sql dataset for Bird dataset.

This module adapts the Spider RoleSQLGenerator for Bird dataset's structure,
which uses detailed schema files instead of tables.json.
"""

import json
import sys
import logging
import re
import sqlparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

from .spider_role_sql_generator import RoleSQLGenerator

logger = logging.getLogger('bird_role_sql')


class BirdRoleSQLGenerator(RoleSQLGenerator):
    """Generate role-based SQL queries dataset for Bird dataset."""

    _CTE_NAME_PATTERN = re.compile(
        r"(?:WITH\s+(?:RECURSIVE\s+)?|,\s*)([A-Za-z_][\w]*)\s*(?:\([^)]*\))?\s+AS\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    _INLINE_SUBQUERY_ALIAS_PATTERN = re.compile(
        r"(?:FROM|JOIN)\s*\(\s*SELECT[\s\S]+?\)\s*(?:AS\s+)?([A-Za-z_][\w]*)",
        re.IGNORECASE,
    )
    _CREATE_OBJECT_PATTERN = re.compile(
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP|TEMPORARY\s+)?(?:TABLE|VIEW|INDEX|TRIGGER)\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w\.]*)",
        re.IGNORECASE,
    )
    _DROP_OBJECT_PATTERN = re.compile(
        r"\bDROP\s+(?:TABLE|VIEW|INDEX|TRIGGER)\s+(?:IF\s+EXISTS\s+)?([A-Za-z_][\w\.]*)",
        re.IGNORECASE,
    )
    FULL_ACCESS_ROLES = {"systemmanager"}
    
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
        self.bird_original_data_file = project_root / 'data' / 'Bird' / 'dev_20240627' / 'dev.json'

        self._difficulty_cache: Dict[str, Dict[Tuple[str, str], str]] = {}
        
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

    def _normalize_question_text(self, question: str) -> str:
        """Normalize question text for consistent lookups."""
        return " ".join(question.strip().split()) if question else ""

    def _load_difficulty_map(self, data_source: str) -> Dict[Tuple[str, str], str]:
        """Load and cache difficulty information from original Bird dataset."""
        if data_source in self._difficulty_cache:
            return self._difficulty_cache[data_source]

        # Currently only dev set is supported
        if data_source != 'dev':
            self._difficulty_cache[data_source] = {}
            return self._difficulty_cache[data_source]

        if not self.bird_original_data_file.exists():
            logger.warning(
                "Original Bird data file not found for difficulty lookup: %s",
                self.bird_original_data_file,
            )
            self._difficulty_cache[data_source] = {}
            return self._difficulty_cache[data_source]

        try:
            with open(self.bird_original_data_file, 'r', encoding='utf-8') as f:
                original_entries = json.load(f)
        except Exception as exc:  # pragma: no cover - logging path
            logger.error("Failed to load Bird original data for difficulty: %s", exc)
            self._difficulty_cache[data_source] = {}
            return self._difficulty_cache[data_source]

        difficulty_map: Dict[Tuple[str, str], str] = {}

        for item in original_entries:
            question = item.get('question')
            db_id = item.get('db_id')
            difficulty = item.get('difficulty')

            if not question or not db_id or not difficulty:
                continue

            key = (db_id, self._normalize_question_text(question))
            difficulty_map[key] = difficulty

        self._difficulty_cache[data_source] = difficulty_map
        logger.info(
            "Loaded difficulty entries for %d Bird questions (%s)",
            len(difficulty_map),
            data_source,
        )
        return self._difficulty_cache[data_source]
    
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
        """Backward-compatible wrapper returning filtered table names only."""

        filtered_tables, _, _ = self._extract_query_tables(query, db_id)
        return filtered_tables

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_query_tables(self, query: str, db_id: str) -> Tuple[Set[str], Set[str], Set[str]]:
        """Return filtered, derived, and raw table references for a query."""

        raw_tables = super().extract_tables_from_query(query)
        if not raw_tables:
            table_info = self.load_bird_table_info(db_id)
            if table_info and table_info.get("table_names"):
                known_tables = {self._normalize_identifier(t) for t in table_info["table_names"]}
                query_lower = query.lower()
                for table in known_tables:
                    if table and re.search(rf"\b{re.escape(table)}\b", query_lower):
                        raw_tables.add(table)
                        logger.debug("Heuristically matched table '%s' in query for %s", table, db_id)

        cte_names = self._extract_cte_names(query)
        inline_aliases = self._extract_inline_aliases(query)
        defined_objects = self._extract_defined_objects(query)
        derived_refs = cte_names | inline_aliases | defined_objects

        filtered_tables = {table for table in raw_tables if table not in derived_refs}
        removed_tables = raw_tables - filtered_tables
        if removed_tables and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Filtered derived references from query on %s: %s", db_id, sorted(removed_tables))

        return filtered_tables, derived_refs, raw_tables

    def _extract_cte_names(self, query: str) -> Set[str]:
        if not query:
            return set()
        return {self._normalize_identifier(match.group(1)) for match in self._CTE_NAME_PATTERN.finditer(query)}

    def _extract_inline_aliases(self, query: str) -> Set[str]:
        if not query:
            return set()
        return {
            self._normalize_identifier(match.group(1))
            for match in self._INLINE_SUBQUERY_ALIAS_PATTERN.finditer(query)
        }

    def _extract_defined_objects(self, query: str) -> Set[str]:
        if not query:
            return set()

        names: Set[str] = set()
        for pattern in (self._CREATE_OBJECT_PATTERN, self._DROP_OBJECT_PATTERN):
            for match in pattern.finditer(query):
                normalized = self._normalize_identifier(match.group(1))
                if normalized:
                    names.add(normalized)
        return names

    @staticmethod
    def _normalize_identifier(name: Optional[str]) -> str:
        if not name:
            return ""
        cleaned = name.strip().strip('"`[]')
        if not cleaned:
            return ""
        if "." in cleaned:
            cleaned = cleaned.split(".")[-1]
        return cleaned.lower()

    def _has_permission(
        self,
        role_name: Optional[str],
        query_tables: Set[str],
        role_tables: str,
        derived_refs: Set[str],
        raw_tables: Set[str],
    ) -> Tuple[bool, Set[str]]:
        role_tables_set = {
            normalized
            for token in role_tables.split(",")
            if (normalized := self._normalize_identifier(token))
        }

        missing_tables = {
            table for table in query_tables if table and table not in role_tables_set
        }
        if not missing_tables:
            return True, set()

        normalized_role = self._normalize_identifier(role_name)
        if normalized_role in self.FULL_ACCESS_ROLES:
            raw_missing = {table for table in raw_tables if table not in role_tables_set}
            unresolved = {table for table in raw_missing if table not in derived_refs}
            if not unresolved:
                logger.debug(
                    "Granting full-access role %s after excluding derived references %s",
                    role_name,
                    sorted(raw_missing),
                )
                return True, set()
            missing_tables = unresolved

        return False, missing_tables
    
    def generate_bird_role_sql_dataset(
        self,
        role_file_path: str = None,
        data_source: str = 'dev',
        bird_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Generate role-based SQL dataset for Bird data.
        
        Args:
            role_file_path (str, optional): Path to Bird role assignments JSON file
            data_source (str): Data source type (currently only 'dev' is supported)
            bird_data (List[Dict[str, Any]], optional): Pre-processed Bird dataset entries
            
        Returns:
            List[Dict[str, str]]: Generated dataset
        """
        # Load data
        role_assignments = self.load_role_assignments(role_file_path)

        if bird_data is None:
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
            from prompts import INSTRUCTION_PROMPT, INPUT_PROMPT  # type: ignore[import-not-found]
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

        difficulty_map = self._load_difficulty_map(data_source)
        
        for example in bird_data:
            db_id = example['db_id']
            
            # Skip if database has no role assignments
            if db_id not in role_assignments:
                skipped_dbs.add(db_id)
                continue
                
            processed_dbs.add(db_id)
            
            # Use the instruction from bird_dev_data.json directly (it already contains detailed schema)
            base_instruction = example.get('instruction', '')
            
            # Extract tables and derived references from query
            query_tables, derived_refs, raw_tables = self._extract_query_tables(example['output'], db_id)
            
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
                    'output': example['output']
                }

                normalized_question = self._normalize_question_text(example.get('input', ''))
                difficulty_lookup = difficulty_map.get((db_id, normalized_question))
                difficulty_value = (
                    difficulty_lookup
                    if difficulty_lookup is not None
                    else example.get('difficulty')
                )

                entry['difficulty'] = difficulty_value

                metadata: Dict[str, Any] = {'gold_sql': example['output']}
                if difficulty_value is not None:
                    metadata['difficulty'] = difficulty_value
                role_description = role_info.get('description')
                if role_description:
                    metadata['role_description'] = role_description
                if raw_tables:
                    metadata['raw_query_tables'] = sorted(raw_tables)
                if query_tables:
                    metadata['query_tables'] = sorted(query_tables)
                if derived_refs:
                    metadata['derived_references'] = sorted(derived_refs)

                allowed, missing_tables = self._has_permission(
                    role_name,
                    query_tables,
                    role_tables,
                    derived_refs,
                    raw_tables,
                )

                if allowed:
                    entry['output'] = example['output']
                    metadata['permission'] = 'allowed'
                    allowed_count += 1
                else:
                    entry['output'] = "Sorry, I cannot answer."
                    metadata['permission'] = 'denied'
                    if missing_tables:
                        metadata['missing_tables'] = sorted(missing_tables)
                    denied_count += 1
                    logger.debug(
                        "Access denied for %s, role %s: query uses %s, role tables %s",
                        db_id,
                        role_name,
                        sorted(query_tables),
                        role_tables,
                    )

                if metadata:
                    entry['metadata'] = metadata

                dataset.append(entry)
        
        # Log statistics
        logger.info(f"Bird dataset generation completed for {data_source} data:")
        logger.info(f"- Databases processed: {len(processed_dbs)}")
        logger.info(f"- Databases skipped (no roles): {len(skipped_dbs)}")
        logger.info(f"- Total examples generated: {len(dataset)}")
        logger.info(f"- Queries allowed: {allowed_count}")
        logger.info(f"- Queries denied: {denied_count}")

        total_decisions = allowed_count + denied_count
        denial_rate = (denied_count / total_decisions * 100) if total_decisions else 0.0
        logger.info(f"- Denial rate: {denial_rate:.2f}%")
        
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
            output_path = str(self.output_dir / f'role_sql_dataset_bird_dev_{timestamp}.json')
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Bird dataset saved to: {output_path}")
            logger.info(f"File size: {Path(output_path).stat().st_size:,} bytes")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving Bird dataset: {str(e)}")
            raise