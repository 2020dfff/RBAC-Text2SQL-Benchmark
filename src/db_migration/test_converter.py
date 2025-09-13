"""
测试数据库转换
"""
import os
import logging
from sqlite_to_postgres import SQLiteToPostgresConverter

def test_conversion():
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据库ID (选择一个较小的数据库进行测试)
    test_db_id = "department_management"
    
    # 路径配置
    spider_path = "../../spider"
    output_dir = "../../data"
    
    # SQLite数据库路径
    sqlite_path = os.path.join(spider_path, "database", test_db_id, f"{test_db_id}.sqlite")
    tables_json_path = os.path.join(spider_path, "tables.json")
    
    if not os.path.exists(sqlite_path):
        logging.error(f"测试数据库不存在: {sqlite_path}")
        return
        
    try:
        # 创建转换器实例
        converter = SQLiteToPostgresConverter(sqlite_path, tables_json_path)
        
        # 执行转换
        converter.convert(output_dir)
        
        logging.info("测试转换完成！")
        logging.info(f"请检查输出目录: {output_dir}")
        
    except Exception as e:
        logging.error(f"转换失败: {str(e)}")

if __name__ == "__main__":
    test_conversion()
