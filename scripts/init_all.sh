#!/bin/bash

# 设置默认值
export PGHOST=${PGHOST:-localhost}
export PGPORT=${PGPORT:-5432}
export PGDATABASE=${PGDATABASE:-spider_benchmark}
export PGUSER=${PGUSER:-postgres}
export PGPASSWORD=${PGPASSWORD:-postgres}

# 检查环境变量
if [ -z "$PGPASSWORD" ]; then
    echo "请设置PGPASSWORD环境变量"
    exit 1
fi

# 检查PostgreSQL连接
echo "检查PostgreSQL连接..."
if ! ./check_pg_connection.sh; then
    echo "PostgreSQL连接检查失败，请解决上述问题后重试"
    exit 1
fi

# 显示配置信息
echo "PostgreSQL连接配置:"
echo "主机: $PGHOST"
echo "端口: $PGPORT"
echo "数据库: $PGDATABASE"
echo "用户: $PGUSER"

# 创建主数据库（忽略可能的错误）
psql -c "CREATE DATABASE $PGDATABASE;" || true

# 执行所有schema文件
echo "开始执行schema文件..."
for schema_file in ../data/schemas_pg/*.sql; do
    if [ -f "$schema_file" ]; then
        echo "执行schema: $schema_file"
        psql -f "$schema_file"
        if [ $? -ne 0 ]; then
            echo "警告: schema文件 $schema_file 执行失败"
        fi
    fi
done

# 装载数据
echo "开始加载数据..."
for data_file in ../data/data_pg/*.sql; do
    if [ -f "$data_file" ]; then
        echo "加载数据: $data_file"
        psql -f "$data_file"
        if [ $? -ne 0 ]; then
            echo "警告: 数据文件 $data_file 加载失败"
        fi
    fi
done

# 运行验证脚本
echo "开始验证数据..."
for verify_file in ../data/verify_pg/*.sql; do
    if [ -f "$verify_file" ]; then
        echo "验证: $verify_file"
        psql -f "$verify_file"
        if [ $? -ne 0 ]; then
            echo "警告: 验证文件 $verify_file 执行失败"
        fi
    fi
done

echo "Database initialization complete."
