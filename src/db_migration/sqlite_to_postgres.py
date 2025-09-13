"""
SQLite到PostgreSQL的数据库转换器
- 转换表结构
- 处理数据类型映射
- 处理约束和索引
- 生成数据导入脚本
"""
import os
import json
import sqlite3
import logging
from typing import Dict, List, Tuple

class SQLiteToPostgresConverter:
    TYPE_MAPPING = {
        'INTEGER': 'INTEGER',
        'REAL': 'DOUBLE PRECISION',
        'NUMERIC': 'NUMERIC',
        'TEXT': 'TEXT',
        'BLOB': 'BYTEA',
    }
    
    def __init__(self, sqlite_db_path: str, tables_json_path: str):
        self.sqlite_db_path = sqlite_db_path
        self.tables_json_path = tables_json_path
        self.db_id = os.path.basename(sqlite_db_path).replace('.sqlite', '')
        
    def get_schema_info(self) -> Dict:
        """从tables.json获取schema信息"""
        # 读取SQLite数据库获取实际的表结构
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        # 获取所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        schema_info = {
            'db_id': self.db_id,
            'tables': []
        }
        
        for table_name, in tables:
            # 获取表的列信息
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # 获取外键信息
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = cursor.fetchall()
            
            table_info = {
                'table_name': table_name,
                'columns': [
                    {
                        'name': col[1],  # 列名
                        'type': col[2],  # 数据类型
                        'primary': bool(col[5])  # 是否主键
                    }
                    for col in columns
                ],
                'foreign_keys': [
                    {
                        'column_name': fk[3],  # 源列
                        'ref_table': fk[2],    # 引用表
                        'ref_column': fk[4]    # 引用列
                    }
                    for fk in foreign_keys
                ]
            }
            
            schema_info['tables'].append(table_info)
        
        conn.close()
        return schema_info
    
    def extract_sqlite_schema(self) -> List[str]:
        """从SQLite数据库提取schema定义"""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schemas = cursor.fetchall()
        conn.close()
        return [s[0] for s in schemas if s[0]]
    
    def convert_type(self, sqlite_type: str) -> str:
        """转换SQLite数据类型到PostgreSQL类型"""
        base_type = sqlite_type.split('(')[0].upper()
        return self.TYPE_MAPPING.get(base_type, 'TEXT')
    
    def generate_pg_schema(self) -> str:
        """生成PostgreSQL schema脚本"""
        schema_info = self.get_schema_info()
        if not schema_info:
            raise ValueError(f"无法找到数据库 {self.db_id} 的schema信息")
            
        # 创建schema
        pg_script = [
            f'CREATE SCHEMA IF NOT EXISTS "sp_{self.db_id}";',
            f'SET search_path TO "sp_{self.db_id}";',
            '\n-- 禁用约束检查',
            'SET session_replication_role = replica;',
            '\n-- 表定义'
        ]
        
        # 表定义
        for table in schema_info['tables']:
            columns = []
            pk_cols = []
            
            for col in table['columns']:
                col_type = self.convert_type(col['type'])
                col_def = f'"{col["name"]}" {col_type}'
                
                if col.get('primary'):
                    pk_cols.append(f'"{col["name"]}"')
                    
                columns.append(col_def)
                
            if pk_cols:
                columns.append(f"PRIMARY KEY ({', '.join(pk_cols)})")
                
            create_table = f'CREATE TABLE "{table["table_name"]}" (\n  '
            create_table += ',\n  '.join(columns)
            create_table += '\n);'
            pg_script.append(create_table)
            
        # 外键约束
        pg_script.append('\n-- 外键约束')
        for table in schema_info['tables']:
            for fk in table.get('foreign_keys', []):
                fk_def = f'ALTER TABLE "{table["table_name"]}" '
                fk_def += f'ADD CONSTRAINT fk_{table["table_name"]}_{fk["column_name"]} '
                fk_def += f'FOREIGN KEY ("{fk["column_name"]}") '
                fk_def += f'REFERENCES "{fk["ref_table"]}" ("{fk["ref_column"]}");'
                pg_script.append(fk_def)
                
        # 启用约束检查
        pg_script.append('\n-- 启用约束检查')
        pg_script.append('SET session_replication_role = default;')
        
        return '\n'.join(pg_script)
    
    def generate_copy_commands(self) -> Tuple[List[str], List[str]]:
        """生成数据复制命令和验证命令"""
        schema_info = self.get_schema_info()
        copy_commands = []
        verify_commands = []
        
        for table in schema_info['tables']:
            table_name = table['table_name']
            # COPY命令
            copy_cmd = f'\\COPY "{table_name}" FROM \'../data_pg/{self.db_id}/{table_name}.csv\' '
            copy_cmd += 'WITH (FORMAT csv, HEADER true, NULL \'\');'
            copy_commands.append(copy_cmd)
            
            # 验证命令
            verify_cmd = f'-- 验证表 {table_name} 的数据数量\n'
            verify_cmd += f'SELECT COUNT(*) as count_{table_name} FROM "{table_name}";'
            verify_commands.append(verify_cmd)
            
        return copy_commands, verify_commands
    
    def export_data_to_csv(self, output_dir: str):
        """导出数据到CSV文件"""
        conn = sqlite3.connect(self.sqlite_db_path)
        schema_info = self.get_schema_info()
        
        db_output_dir = os.path.join(output_dir, self.db_id)
        os.makedirs(db_output_dir, exist_ok=True)
        
        for table in schema_info['tables']:
            table_name = table['table_name']
            output_path = os.path.join(db_output_dir, f'{table_name}.csv')
            
            # 使用pandas导出为CSV
            import pandas as pd
            query = f'SELECT * FROM "{table_name}";'
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False)
            
        conn.close()
        
    def convert(self, output_base_dir: str):
        """执行完整的转换过程"""
        # 确保输出目录存在
        os.makedirs(os.path.join(output_base_dir, 'schemas_pg'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, 'data_pg'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, 'verify_pg'), exist_ok=True)
        
        # 生成schema文件
        schema_sql = self.generate_pg_schema()
        schema_path = os.path.join(output_base_dir, 'schemas_pg', f'{self.db_id}.sql')
        with open(schema_path, 'w') as f:
            f.write(schema_sql)
            
        # 导出数据到CSV
        self.export_data_to_csv(os.path.join(output_base_dir, 'data_pg'))
        
        # 生成COPY命令
        copy_commands, verify_commands = self.generate_copy_commands()
        
        # 写入数据加载脚本
        data_path = os.path.join(output_base_dir, 'data_pg', f'{self.db_id}.sql')
        with open(data_path, 'w') as f:
            f.write('\n'.join(copy_commands))
            
        # 写入验证脚本
        verify_path = os.path.join(output_base_dir, 'verify_pg', f'{self.db_id}.sql')
        with open(verify_path, 'w') as f:
            f.write('\n'.join(verify_commands))
            
        logging.info(f'数据库 {self.db_id} 转换完成')
