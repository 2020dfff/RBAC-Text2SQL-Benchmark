"""Role parsing module for database RBAC roles."""

import re
import json
import logging
from typing import List, Dict, Optional, Set
from datetime import datetime
from pathlib import Path
import sqlite3

logger = logging.getLogger(__name__)

class RoleParser:
    """Parser for database role assignments."""
    
    @staticmethod
    def clean_value(value: str) -> str:
        """Clean a field value by removing special characters and normalizing whitespace."""
        value = re.sub(r'\*+', '', value)  # Remove asterisks
        value = re.sub(r'[`\[\]\'"]', '', value)  # Remove backticks, brackets, quotes
        value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
        return value.strip()
    
    @staticmethod
    def extract_role_name(value: str) -> str:
        """Extract clean role name from potentially combined role-permission string."""
        parts = re.split(r'\s*[-–]\s*', value, maxsplit=1)
        return RoleParser.clean_value(parts[0])
    
    @staticmethod
    def normalize_permissions(perm_str: str) -> str:
        """Normalize permission string format."""
        if not perm_str:
            return ""
        
        # Extract permissions from combined role-permission string if needed
        if ' - ' in perm_str and ': ' in perm_str:
            perm_str = perm_str.split(' - ', 1)[1]
        
        # Split into individual permission statements
        permissions = []
        for part in re.split(r'[;,]\s*', perm_str):
            part = part.strip()
            if not part:
                continue
                
            # Check for standard format (table: permission)
            if ': ' in part:
                permissions.append(part)
            # Handle cases where permissions are listed without table
            elif 'READ' in part.upper() or 'WRITE' in part.upper():
                # Try to associate with previous table if exists
                if permissions and ': ' in permissions[-1]:
                    table = permissions[-1].split(': ')[0]
                    permissions.append(f"{table}: {part}")
                else:
                    permissions.append(part)
        
        return '; '.join(permissions)

    @staticmethod
    def validate_tables(db_path: str) -> Set[str]:
        """
        Extract table names from SQLite database.
        
        Args:
            db_path (str): Path to the SQLite database file
            
        Returns:
            Set[str]: Set of valid table names in the database
        
        Raises:
            sqlite3.Error: If there's an error connecting to or querying the database
        """
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                # Query for all table names in the database
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0].lower() for row in cursor.fetchall()}
                return tables
        except sqlite3.Error as e:
            logger.error(f"Error accessing SQLite database at {db_path}: {str(e)}")
            raise

    def parse_roles(self, text: str, db_path: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Parse multiple role assignments from text into structured format (ROLE, DESCRIPTION, TABLES only).
        Optionally validates table names against a SQLite database schema.

        Args:
            text (str): The text containing role definitions
            db_path (Optional[str]): Path to SQLite database for table validation.
                                   If provided, will validate table names.
        
        Returns:
            List[Dict[str, str]]: List of parsed roles with validation markers if needed
        """
        lines = text.strip().split('\n')
        roles = []
        current_role = None
        required_fields = ['role', 'description', 'tables']
        
        # Get valid table names if db_path is provided
        valid_tables = set()
        if db_path:
            try:
                valid_tables = self.validate_tables(db_path)
            except sqlite3.Error as e:
                logger.warning(f"Could not validate tables against database: {str(e)}")
                db_path = None  # Disable validation if database access fails

        for line in lines:
            line = line.strip()
            if not line:
                if current_role and all(k in current_role for k in required_fields):
                    for key in current_role:
                        if key == 'role':
                            current_role[key] = self.extract_role_name(current_role[key])
                        elif key == 'tables':
                            current_role[key] = self.clean_value(current_role[key])
                        else:
                            current_role[key] = self.clean_value(current_role[key])
                    if current_role['role'] and current_role['description'] and current_role['tables']:
                        # Validate tables if db_path is provided
                        if db_path and valid_tables:
                            tables = {t.strip().lower() for t in current_role['tables'].split(',')}
                            invalid_tables = tables - valid_tables
                            if invalid_tables:
                                current_role['tables'] += " NEED HUMAN VERIFICATION (Invalid tables: " + \
                                                        ", ".join(invalid_tables) + ")"
                        roles.append(current_role.copy())
                    current_role = None
                continue

            # Check for new role
            if 'ROLE:' in line.upper() or 'ROLE：' in line.upper():
                if current_role and all(k in current_role for k in required_fields):
                    for key in current_role:
                        if key == 'role':
                            current_role[key] = self.extract_role_name(current_role[key])
                        elif key == 'tables':
                            current_role[key] = self.clean_value(current_role[key])
                        else:
                            current_role[key] = self.clean_value(current_role[key])
                    if current_role['role'] and current_role['description'] and current_role['tables']:
                        roles.append(current_role.copy())
                current_role = {}
                value = re.split('(?i)ROLE[:：]', line)[-1].strip()
                current_role['role'] = value
                continue

            # Process other fields
            if current_role is not None:
                field_mappings = {
                    'DESCRIPTION': 'description',
                    'TABLES': 'tables',
                }
                matched = False
                for field, key in field_mappings.items():
                    if line.upper().startswith(f"{field}:") or line.upper().startswith(f"{field}："):
                        value = re.split(f"(?i){field}[:：]", line)[-1].strip()
                        current_role[key] = value
                        matched = True
                        break
                # If not a new field, append to the last field
                if not matched and current_role:
                    for key in required_fields:
                        if key in current_role:
                            if not any(f.upper() in line.upper() for f in ['ROLE:', 'DESCRIPTION:', 'TABLES:']):
                                current_role[key] += ' ' + line
                                break

        # Process the last role
        if current_role and all(k in current_role for k in required_fields):
            for key in current_role:
                if key == 'role':
                    current_role[key] = self.extract_role_name(current_role[key])
                elif key == 'tables':
                    current_role[key] = self.clean_value(current_role[key])
                else:
                    current_role[key] = self.clean_value(current_role[key])
            
            if current_role['role'] and current_role['description'] and current_role['tables']:
                # Validate tables if db_path is provided
                if db_path and valid_tables:
                    tables = {t.strip().lower() for t in current_role['tables'].split(',')}
                    invalid_tables = tables - valid_tables
                    if invalid_tables:
                        current_role['tables'] += " NEED HUMAN VERIFICATION (Invalid tables: " + \
                                                ", ".join(invalid_tables) + ")"
                roles.append(current_role.copy())
        return roles
