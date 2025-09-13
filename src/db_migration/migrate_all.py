"""
批量迁移Spider SQLite数据库到PostgreSQL
"""
import os
import sys
import logging
import argparse
from sqlite_to_postgres import SQLiteToPostgresConverter

def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def process_database(spider_path: str, db_id: str, output_dir: str):
    """处理单个数据库"""
    sqlite_path = os.path.join(spider_path, 'database', db_id, f'{db_id}.sqlite')
    tables_json_path = os.path.join(spider_path, 'tables.json')
    
    if not os.path.exists(sqlite_path):
        logging.error(f"数据库文件不存在: {sqlite_path}")
        return False
        
    try:
        converter = SQLiteToPostgresConverter(sqlite_path, tables_json_path)
        converter.convert(output_dir)
        return True
    except Exception as e:
        logging.error(f"处理数据库 {db_id} 时出错: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Spider数据库迁移工具')
    parser.add_argument('--spider-path', required=True, help='Spider数据集根目录')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--db-list', help='要处理的数据库ID列表文件')
    args = parser.parse_args()
    
    setup_logging()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取要处理的数据库列表
    if args.db_list:
        with open(args.db_list, 'r') as f:
            db_ids = [line.strip() for line in f if line.strip()]
    else:
        # 处理所有数据库
        db_ids = [d for d in os.listdir(os.path.join(args.spider_path, 'database'))
                 if os.path.isdir(os.path.join(args.spider_path, 'database', d))]
    
    # 统计
    total = len(db_ids)
    success = 0
    failed = []
    
    # 处理每个数据库
    for db_id in db_ids:
        logging.info(f"正在处理数据库: {db_id}")
        if process_database(args.spider_path, db_id, args.output_dir):
            success += 1
        else:
            failed.append(db_id)
            
    # 输出统计信息
    logging.info(f"\n迁移完成:")
    logging.info(f"总数: {total}")
    logging.info(f"成功: {success}")
    logging.info(f"失败: {len(failed)}")
    if failed:
        logging.info("失败的数据库:")
        for db_id in failed:
            logging.info(f"- {db_id}")

if __name__ == '__main__':
    main()
