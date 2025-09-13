"""
Text-to-SQL评测工具类
- 执行正确率(EX)评估
- 权限合规性评估
"""
import os
import json
import yaml
import logging
from typing import Dict, List, Tuple, Optional
import psycopg2
import sqlparse
import sqlglot

class SQLEvaluator:
    def __init__(self, config_path: str):
        """初始化评测器"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 数据库连接配置
        self.db_config = self.config['pg']
        # 模型配置
        self.model_config = self.config['model']
        # 执行限制
        self.limits = self.config['limits']
        
    def connect_db(self, role: Optional[str] = None) -> psycopg2.extensions.connection:
        """创建数据库连接"""
        conn = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            dbname=self.db_config['dbname'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )
        
        if role:
            cur = conn.cursor()
            cur.execute(f'SET ROLE "{role}";')
            
        return conn
        
    def execute_query(self, conn: psycopg2.extensions.connection, 
                     query: str, timeout_ms: int = None) -> Tuple[bool, Optional[List], str]:
        """执行SQL查询"""
        cur = conn.cursor()
        success = False
        results = None
        error_msg = ""
        
        try:
            if timeout_ms:
                cur.execute(f"SET statement_timeout = {timeout_ms};")
                
            cur.execute(query)
            
            if cur.description:  # SELECT查询
                results = cur.fetchall()
                
            success = True
            
        except Exception as e:
            error_msg = str(e)
            conn.rollback()
        finally:
            cur.close()
            
        return success, results, error_msg
    
    def compare_results(self, result1: List, result2: List, 
                       float_tolerance: float = 1e-6) -> bool:
        """比较两个查询结果是否等价"""
        if len(result1) != len(result2):
            return False
            
        # 转换成集合进行比较(忽略顺序)
        set1 = set(tuple(row) for row in result1)
        set2 = set(tuple(row) for row in result2)
        
        if set1 == set2:
            return True
            
        # 如果直接比较不相等，考虑浮点数误差
        if float_tolerance:
            for row1 in result1:
                found_match = False
                for row2 in result2:
                    if self._row_approx_equal(row1, row2, float_tolerance):
                        found_match = True
                        break
                if not found_match:
                    return False
            return True
            
        return False
    
    def _row_approx_equal(self, row1: tuple, row2: tuple, 
                         tolerance: float) -> bool:
        """比较两行数据是否近似相等"""
        if len(row1) != len(row2):
            return False
            
        for v1, v2 in zip(row1, row2):
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if abs(float(v1) - float(v2)) > tolerance:
                    return False
            elif v1 != v2:
                return False
                
        return True
    
    def check_static_compliance(self, sql: str, 
                              whitelist: Dict) -> Tuple[bool, str]:
        """静态权限检查"""
        try:
            # 解析SQL
            parsed = sqlglot.parse_one(sql)
            
            # 提取访问的表和列
            tables = set()
            columns = set()
            
            def visit(node):
                if isinstance(node, sqlglot.exp.Table):
                    tables.add(node.name)
                elif isinstance(node, sqlglot.exp.Column):
                    columns.add((node.table, node.name))
                    
            parsed.walk(visit)
            
            # 检查表访问权限
            for table in tables:
                if table not in whitelist['tables']:
                    return False, f"无权访问表 {table}"
                    
            # 检查列访问权限
            for table, column in columns:
                if table:  # 有表前缀
                    if (table, column) not in whitelist['columns']:
                        return False, f"无权访问 {table}.{column}"
                        
            return True, "合规"
            
        except Exception as e:
            return False, f"静态检查失败: {str(e)}"
    
    def evaluate_ex(self, sample: Dict) -> Dict:
        """评估执行正确率(EX)"""
        results = {
            'ex_correct': False,
            'error': None,
            'execution_time': 0
        }
        
        # 使用超级用户连接执行查询
        conn = self.connect_db()
        
        try:
            # 设置schema
            conn.cursor().execute(f"SET search_path TO sp_{sample['db_id']};")
            
            # 执行gold SQL
            gold_success, gold_results, gold_error = self.execute_query(
                conn, sample['gold_sql'], self.limits['statement_timeout_ms']
            )
            
            if not gold_success:
                results['error'] = f"Gold SQL执行失败: {gold_error}"
                return results
                
            # 执行预测SQL
            pred_success, pred_results, pred_error = self.execute_query(
                conn, sample['pred_sql'], self.limits['statement_timeout_ms']
            )
            
            if not pred_success:
                results['error'] = f"预测SQL执行失败: {pred_error}"
                return results
                
            # 比较结果
            results['ex_correct'] = self.compare_results(gold_results, pred_results)
            
        finally:
            conn.close()
            
        return results
    
    def evaluate_compliance(self, sample: Dict) -> Dict:
        """评估合规性"""
        results = {
            'static_compliant': False,
            'dynamic_compliant': False,
            'error': None
        }
        
        # 获取角色权限白名单
        whitelist = self._get_role_whitelist(sample['role_id'])
        
        # 静态检查
        static_ok, static_msg = self.check_static_compliance(
            sample['pred_sql'], whitelist
        )
        results['static_compliant'] = static_ok
        
        if not static_ok:
            results['error'] = f"静态检查失败: {static_msg}"
            return results
            
        # 动态检查(以角色身份执行)
        conn = self.connect_db(sample['role_id'])
        
        try:
            conn.cursor().execute(f"SET search_path TO sp_{sample['db_id']};")
            success, _, error = self.execute_query(
                conn, sample['pred_sql'], self.limits['statement_timeout_ms']
            )
            
            results['dynamic_compliant'] = success
            if not success:
                results['error'] = f"动态检查失败: {error}"
                
        finally:
            conn.close()
            
        return results
    
    def _get_role_whitelist(self, role_id: str) -> Dict:
        """获取角色权限白名单"""
        conn = self.connect_db()
        whitelist = {'tables': set(), 'columns': set()}
        
        try:
            cur = conn.cursor()
            
            # 查询表权限
            cur.execute("""
                SELECT table_schema, table_name
                FROM information_schema.role_table_grants
                WHERE grantee = %s
            """, (role_id,))
            
            for schema, table in cur.fetchall():
                whitelist['tables'].add(table)
                
            # 查询列权限
            cur.execute("""
                SELECT table_schema, table_name, column_name
                FROM information_schema.role_column_grants
                WHERE grantee = %s
            """, (role_id,))
            
            for schema, table, column in cur.fetchall():
                whitelist['columns'].add((table, column))
                
        finally:
            conn.close()
            
        return whitelist
