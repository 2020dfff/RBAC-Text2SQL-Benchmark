"""
Process and convert Spider dataset format
"""

import os
import sys
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger('spider_data')

class SpiderDataProcessor:
    """Process Spider dataset and extract relevant information."""
    
    def __init__(self, project_root: Path):
        """
        Initialize processor with project root path.
        
        Args:
            project_root (Path): Path to project root directory
        """
        self.project_root = project_root
        self.spider_dir = project_root / 'data/spider'
        self.spider_db_dir = self.spider_dir / 'database'
        self.output_dir = project_root / 'data'
        
    def get_db_table_count(self, db_path: Path) -> Optional[int]:
        """
        Get number of tables in a SQLite database.
        
        Args:
            db_path (Path): Path to SQLite database file
            
        Returns:
            Optional[int]: Number of tables or None if error occurs
        """
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            logger.error(f"Error accessing database {db_path}: {str(e)}")
            return None
            
    def process_spider_train_data(self) -> None:
        """
        Process Spider train data and save in simplified format.
        Extracts db_id, query, and question fields from train_spider.json.
        """
        input_file = self.spider_dir / 'train_spider.json'
        output_file = self.output_dir / 'spider_train_data.json'
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            processed_data = [
                {
                    'db_id': item.get('db_id', ''),
                    'question': item.get('question', ''),
                    'query': item.get('query', '')
                }
                for item in train_data
                if all(key in item for key in ['db_id', 'query', 'question'])
            ]
            
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully processed {len(processed_data)} records")
            logger.info(f"Saved processed data to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing Spider train data: {str(e)}")
            raise

    def get_db_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all databases in the Spider dataset.
        
        Returns:
            Dict[str, Dict]: Dictionary containing statistics for each database
        """
        stats = {}
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            for db_dir in self.spider_db_dir.iterdir():
                if db_dir.is_dir():
                    db_file = db_dir / f"{db_dir.name}.sqlite"
                    schema_file = db_dir / "schema.sql"

                    db_stats = {
                        'table_count': 0,
                        'has_sqlite': db_file.exists(),
                        'has_schema': schema_file.exists(),
                        'schema_path': str(schema_file) if schema_file.exists() else None
                    }

                    if db_stats['has_sqlite']:
                        table_count = self.get_db_table_count(db_file)
                        if table_count is not None:
                            db_stats['table_count'] = table_count

                    stats[db_dir.name] = db_stats

            # Save detailed statistics to spider_info.json
            spider_info_file = self.output_dir / 'spider_info.json'
            with open(spider_info_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                logger.info(f"Detailed statistics saved to {spider_info_file}")

            return stats
            
        except Exception as e:
            logger.error(f"Error getting database statistics: {str(e)}")
            raise
            
    def get_db_folders(self) -> List[Path]:
        """
        Get list of database folders in Spider dataset.
        
        Returns:
            List[Path]: List of database folder paths
        """
        if not self.spider_db_dir.exists():
            logger.warning(f"Spider database directory not found at {self.spider_db_dir}. Creating it now.")
            self.spider_db_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        return [d for d in self.spider_db_dir.iterdir() if d.is_dir()]

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get project root directory
    project_root = Path(__file__).parents[2]
    
    # Initialize processor
    processor = SpiderDataProcessor(project_root)
    
    # Process train data
    processor.process_spider_train_data()
