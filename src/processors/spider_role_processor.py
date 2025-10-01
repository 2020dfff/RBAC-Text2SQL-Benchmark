"""
Process and convert Spider dataset format
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from tqdm import tqdm

logger = logging.getLogger('spider_data')


def _ensure_list(value: Optional[Iterable[Path]]) -> List[Path]:
    """Coerce an iterable of paths into a sorted list."""

    if value is None:
        return []
    return sorted(list(value))

class SpiderDataProcessor:
    """Process Spider dataset and extract relevant information."""
    
    SPLIT_FILE_MAP = {
        'train': 'train_spider.json',
        'dev': 'dev.json',
        'test': 'test.json',
    }

    def __init__(self, project_root: Path):
        """
        Initialize processor with project root path.
        
        Args:
            project_root (Path): Path to project root directory
        """
        self.project_root = project_root
        self.spider_dir = project_root / 'data/spider'
        self.spider_db_dir = self.spider_dir / 'database'
        self.spider_test_db_dir = self.spider_dir / 'test_database'
        self.output_dir = project_root / 'data'

    # ------------------------------------------------------------------
    # In-memory loading helpers
    # ------------------------------------------------------------------

    def load_raw_split(self, split: str, *, require_query: bool = True) -> List[Dict[str, str]]:
        """Load raw Spider records for a given split without persisting to disk.

        Args:
            split: Dataset split to load ("train", "dev", or "test").
            require_query: When True, rows missing a SQL query are skipped.

        Returns:
            List of dictionaries containing the raw Spider fields (db_id, question, query).
        """

        normalized_split = split.lower()
        if normalized_split not in self.SPLIT_FILE_MAP:
            raise ValueError(f"Unsupported Spider split: {split}")

        filename = self.SPLIT_FILE_MAP[normalized_split]
        input_file = self.spider_dir / filename

        if not input_file.exists():
            raise FileNotFoundError(f"Spider {split} file not found: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed: List[Dict[str, str]] = []
        skipped = 0

        for item in data:
            db_id = item.get('db_id')
            question = item.get('question')
            query = item.get('query')

            if not db_id or question is None:
                skipped += 1
                continue

            if require_query and not query:
                skipped += 1
                continue

            record = {
                'db_id': db_id,
                'question': question,
                'query': query or '',
                'source': normalized_split,
            }
            processed.append(record)

        logger.info(
            "Loaded %s Spider %s records (%s skipped for missing fields)",
            f"{len(processed):,}",
            normalized_split,
            skipped,
        )

        return processed

    def load_raw_splits(self, splits: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
        """Convenience helper returning raw Spider data for multiple splits."""

        datasets: Dict[str, List[Dict[str, str]]] = {}
        for split in splits:
            datasets[split] = self.load_raw_split(split)
        return datasets


class SpiderRoleProcessor:
    """Generate RBAC role assignments for Spider databases using an LLM."""

    def __init__(
        self,
        generator,
        database_dir: Path,
        *,
        dataset_label: str = "spider_dev",
    ) -> None:
        """Initialise the processor.

        Args:
            generator: An instance of :class:`ParallelRoleGenerator` (type hinted
                dynamically to avoid circular imports).
            database_dir: Directory containing Spider database folders.
            dataset_label: Identifier used when persisting assignment artefacts.
        """

        self.generator = generator
        self.database_dir = database_dir
        self.dataset_label = dataset_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_database_dirs(self, db_names: Optional[Iterable[str]] = None) -> List[Path]:
        """Return candidate Spider database directories.

        Args:
            db_names: Optional iterable of specific database names to include.

        Returns:
            A sorted list of directories matching the provided names (or all
            databases when ``db_names`` is ``None``).
        """

        if not self.database_dir.exists():
            logger.error("Spider database directory missing: %s", self.database_dir)
            return []

        if db_names is None:
            return sorted(p for p in self.database_dir.iterdir() if p.is_dir())

        resolved: List[Path] = []
        for name in db_names:
            candidate = self.database_dir / name
            if candidate.exists() and candidate.is_dir():
                resolved.append(candidate)
            else:
                logger.warning("Requested Spider database not found: %s", candidate)
        return sorted(resolved)

    def process_databases(
        self,
        db_names: Optional[Iterable[str]] = None,
        *,
        batch_size: int = 8,
        run_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run parallel role generation for the requested databases."""

        if self.generator is None:
            raise RuntimeError("ParallelRoleGenerator is required to process Spider databases.")

        db_dirs = self.list_database_dirs(db_names)
        if not db_dirs:
            logger.error("No Spider databases available for processing.")
            return self._empty_result(run_timestamp, 0, batch_size, self.dataset_label)

        valid_dirs = self._filter_valid_databases(db_dirs)
        if not valid_dirs:
            logger.error("No valid Spider databases after schema/sqlite validation.")
            return self._empty_result(run_timestamp, len(db_dirs), batch_size, self.dataset_label)

        timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        total_processed = 0
        total_roles = 0
        assignments: Dict[str, List[Dict[str, Any]]] = {}

        n_batches = (len(valid_dirs) + batch_size - 1) // batch_size
        for idx in range(n_batches):
            batch_start = idx * batch_size
            batch_dirs = valid_dirs[batch_start : batch_start + batch_size]
            logger.info(
                "Processing Spider batch %s/%s (%s databases)",
                idx + 1,
                n_batches,
                len(batch_dirs),
            )

            results = self.generator.process_databases_parallel(batch_dirs)
            processed_in_batch = 0
            roles_in_batch = 0

            for result in results:
                if not result:
                    continue
                db_name = result.get("database") or result.get("db_id")
                roles = result.get("roles")
                if not db_name or not roles:
                    continue
                assignments[db_name] = roles
                processed_in_batch += 1
                roles_in_batch += len(roles)
                logger.info("  ✓ %s: %s roles", db_name, len(roles))

            total_processed += processed_in_batch
            total_roles += roles_in_batch

            logger.info(
                "Batch %s complete: %s/%s databases produced roles (%s roles)",
                idx + 1,
                processed_in_batch,
                len(batch_dirs),
                roles_in_batch,
            )

        return {
            "assignments": assignments,
            "metadata": {
                "timestamp": timestamp,
                "total_databases": len(valid_dirs),
                "processed_databases": total_processed,
                "total_roles_generated": total_roles,
                "batch_size": batch_size,
                "dataset_type": self.dataset_label,
            },
        }

    def save_role_assignments(
        self,
        assignments_data: Dict[str, Any],
        output_dir: Path,
        *,
        filename_suffix: Optional[str] = None,
    ) -> Path:
        """Persist generated role assignments to ``output_dir``."""

        timestamp = filename_suffix or assignments_data.get("metadata", {}).get("timestamp")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"role_assignments_{self.dataset_label}_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(assignments_data, f, ensure_ascii=False, indent=2)

        logger.info("Spider role assignments saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _filter_valid_databases(self, db_dirs: Iterable[Path]) -> List[Path]:
        valid: List[Path] = []
        for db_path in _ensure_list(db_dirs):
            schema_file = db_path / "schema.sql"
            sqlite_candidates = list(db_path.glob("*.sqlite"))

            if not schema_file.exists():
                logger.warning("Skipping %s: schema.sql not found", db_path.name)
                continue

            if not sqlite_candidates:
                logger.warning("Skipping %s: no SQLite file found", db_path.name)
                continue

            valid.append(db_path)

        return valid

    @staticmethod
    def _empty_result(
        run_timestamp: Optional[str],
        requested_databases: int,
        batch_size: int,
        dataset_label: str,
    ) -> Dict[str, Any]:
        timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        return {
            "assignments": {},
            "metadata": {
                "timestamp": timestamp,
                "total_databases": requested_databases,
                "processed_databases": 0,
                "total_roles_generated": 0,
                "batch_size": batch_size,
                "dataset_type": dataset_label,
            },
        }

        
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
            
    def _process_spider_data_file(self, input_filename: str, output_filename: str, data_type: str) -> int:
        """
        Generic function to process Spider data files and save in simplified format.
        
        Args:
            input_filename (str): Name of input JSON file
            output_filename (str): Name of output JSON file
            data_type (str): Type of data being processed (train/dev/test)
            
        Returns:
            int: Number of processed records
        """
        input_file = self.spider_dir / input_filename
        output_file = self.output_dir / output_filename
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = [
                {
                    'db_id': item.get('db_id', ''),
                    'question': item.get('question', ''),
                    'query': item.get('query', '')
                }
                for item in data
                if all(key in item for key in ['db_id', 'query', 'question'])
            ]
            
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully processed {len(processed_data)} {data_type} records")
            logger.info(f"Saved {data_type} data to {output_file}")
            
            return len(processed_data)
            
        except Exception as e:
            logger.error(f"Error processing Spider {data_type} data: {str(e)}")
            raise

    def process_spider_train_data(self) -> int:
        """
        Process Spider train data and save in simplified format.
        Extracts db_id, query, and question fields from train_spider.json.
        
        Returns:
            int: Number of processed records
        """
        return self._process_spider_data_file(
            input_filename='train_spider.json',
            output_filename='spider_train_data.json',
            data_type='train'
        )
        
    def process_spider_dev_data(self) -> int:
        """
        Process Spider dev data and save in simplified format.
        Extracts db_id, query, and question fields from dev.json.
        
        Returns:
            int: Number of processed records
        """
        return self._process_spider_data_file(
            input_filename='dev.json',
            output_filename='spider_dev_data.json',
            data_type='dev'
        )
        
    def process_spider_test_data(self) -> int:
        """
        Process Spider test data and save in simplified format.
        Extracts db_id, query, and question fields from test.json.
        
        Returns:
            int: Number of processed records
        """
        return self._process_spider_data_file(
            input_filename='test.json',
            output_filename='spider_test_data.json',
            data_type='test'
        )
        
    def process_all_spider_data(self) -> Dict[str, int]:
        """
        Process all Spider data files (train, dev, test) and optionally create combined dataset.
        
        Returns:
            Dict[str, int]: Dictionary with counts for each dataset type
        """
        results = {}
        
        # Process individual datasets
        try:
            results['train'] = self.process_spider_train_data()
        except Exception as e:
            logger.warning(f"Failed to process train data: {e}")
            results['train'] = 0
            
        try:
            results['dev'] = self.process_spider_dev_data()
        except Exception as e:
            logger.warning(f"Failed to process dev data: {e}")
            results['dev'] = 0
            
        try:
            results['test'] = self.process_spider_test_data()
        except Exception as e:
            logger.warning(f"Failed to process test data: {e}")
            results['test'] = 0
        
        # Create combined dataset if any individual datasets were successful
        if any(count > 0 for count in results.values()):
            try:
                results['combined'] = self._create_combined_dataset(results)
            except Exception as e:
                logger.warning(f"Failed to create combined dataset: {e}")
                results['combined'] = 0
        
        # Log summary
        total_records = sum(results.values()) - results.get('combined', 0)  # Don't double count combined
        logger.info(f"\nSpider Data Processing Summary:")
        logger.info(f"- Train records: {results.get('train', 0)}")
        logger.info(f"- Dev records: {results.get('dev', 0)}")
        logger.info(f"- Test records: {results.get('test', 0)}")
        logger.info(f"- Combined records: {results.get('combined', 0)}")
        logger.info(f"- Total individual records: {total_records}")
        
        return results
        
    def _create_combined_dataset(self, individual_counts: Dict[str, int]) -> int:
        """
        Create a combined dataset from all available individual datasets.
        
        Args:
            individual_counts: Dictionary with record counts for each dataset
            
        Returns:
            int: Number of records in combined dataset
        """
        combined_data = []
        
        # Load and combine all available datasets
        for data_type, filename in [
            ('train', 'spider_train_data.json'),
            ('dev', 'spider_dev_data.json'),
            ('test', 'spider_test_data.json')
        ]:
            if individual_counts.get(data_type, 0) > 0:
                try:
                    file_path = self.output_dir / filename
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Add source information to each record
                        for item in data:
                            item['source'] = data_type
                        combined_data.extend(data)
                        logger.info(f"Added {len(data)} {data_type} records to combined dataset")
                except Exception as e:
                    logger.warning(f"Failed to load {data_type} data for combination: {e}")
        
        if combined_data:
            # Save combined dataset
            output_file = self.output_dir / 'spider_combined_data.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created combined dataset with {len(combined_data)} total records")
            logger.info(f"Saved combined data to {output_file}")
            
        return len(combined_data)

    def get_db_statistics(self, is_test: bool = False) -> Dict[str, Dict]:
        """
        Get statistics for databases in the Spider dataset.
        
        Args:
            is_test (bool): If True, get statistics for test databases; if False, get for train/dev databases
        
        Returns:
            Dict[str, Dict]: Dictionary containing statistics for each database
        """
        stats = {}
        try:
            # Create output directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Get appropriate database folders based on is_test flag
            db_folders = self.get_test_db_folders() if is_test else self.get_db_folders()

            for db_dir in db_folders:
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

            # Save detailed statistics with appropriate filename
            filename = 'spider_test_info.json' if is_test else 'spider_info.json'
            spider_info_file = self.output_dir / filename
            with open(spider_info_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                logger.info(f"Detailed{'test ' if is_test else ' '}statistics saved to {spider_info_file}")

            return stats
            
        except Exception as e:
            logger.error(f"Error getting{'test ' if is_test else ' '}database statistics: {str(e)}")
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
    
    def get_test_db_folders(self) -> List[Path]:
        """
        Get list of test database folders in Spider dataset.
        
        Returns:
            List[Path]: List of test database folder paths
        """
        if not self.spider_test_db_dir.exists():
            logger.warning(f"Spider test database directory not found at {self.spider_test_db_dir}. Creating it now.")
            self.spider_test_db_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        return [d for d in self.spider_test_db_dir.iterdir() if d.is_dir()]

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
