"""Role parsing module for database RBAC roles."""

import re
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

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

    def parse_roles(self, text: str) -> List[Dict[str, str]]:
        """
        Parse multiple role assignments from text into structured format.
        
        Args:
            text (str): Raw text containing role assignments
            
        Returns:
            list: List of dictionaries containing parsed role information
        """
        lines = text.strip().split('\n')
        roles = []
        current_role = None
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_role and all(k in current_role for k in ['role', 'description', 'permissions', 'justification']):
                    for key in current_role:
                        if key == 'role':
                            current_role[key] = self.extract_role_name(current_role[key])
                        elif key == 'permissions':
                            current_role[key] = self.normalize_permissions(current_role[key])
                        else:
                            current_role[key] = self.clean_value(current_role[key])
                    
                    if current_role['role'] and current_role['description']:
                        roles.append(current_role.copy())
                    current_role = None
                continue
            
            # Check for new role
            if 'ROLE:' in line.upper() or 'ROLE：' in line.upper():
                if current_role and all(k in current_role for k in ['role', 'description', 'permissions', 'justification']):
                    for key in current_role:
                        if key == 'role':
                            current_role[key] = self.extract_role_name(current_role[key])
                        elif key == 'permissions':
                            current_role[key] = self.normalize_permissions(current_role[key])
                        else:
                            current_role[key] = self.clean_value(current_role[key])
                    
                    if current_role['role'] and current_role['description']:
                        roles.append(current_role.copy())
                
                current_role = {}
                value = re.split('(?i)ROLE[:：]', line)[-1].strip()
                current_role['role'] = value
                continue
            
            # Process other fields
            if current_role is not None:
                field_mappings = {
                    'DESCRIPTION': 'description',
                    'PERMISSIONS': 'permissions',
                    'JUSTIFICATION': 'justification'
                }
                
                # Check if line starts with any field marker
                matched = False
                for field, key in field_mappings.items():
                    if line.upper().startswith(f"{field}:") or line.upper().startswith(f"{field}："):
                        value = re.split(f"(?i){field}[:：]", line)[-1].strip()
                        current_role[key] = value
                        matched = True
                        break
                
                # If not a new field, append to the last field
                if not matched and current_role:
                    for key in ['role', 'description', 'permissions', 'justification']:
                        if key in current_role:
                            if not any(f.upper() in line.upper() for f in ['ROLE:', 'DESCRIPTION:', 'PERMISSIONS:', 'JUSTIFICATION:']):
                                current_role[key] += ' ' + line
                                break
        
        # Process the last role
        if current_role and all(k in current_role for k in ['role', 'description', 'permissions', 'justification']):
            for key in current_role:
                if key == 'role':
                    current_role[key] = self.extract_role_name(current_role[key])
                elif key == 'permissions':
                    current_role[key] = self.normalize_permissions(current_role[key])
                else:
                    current_role[key] = self.clean_value(current_role[key])
            
            if current_role['role'] and current_role['description']:
                roles.append(current_role.copy())
        
        return roles
