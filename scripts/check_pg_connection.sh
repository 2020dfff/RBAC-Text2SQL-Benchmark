#!/bin/bash

# 检查PostgreSQL连接和权限
echo "正在检查PostgreSQL连接..."

# 1. 测试基本连接
echo "测试数据库连接..."
if psql -c "\conninfo"; then
    echo "基本连接成功"
else
    echo "错误: 无法连接到数据库"
    echo "请检查以下内容："
    echo "1. PostgreSQL 服务是否运行"
    echo "2. 环境变量是否正确设置:"
    echo "   PGHOST=$PGHOST"
    echo "   PGPORT=$PGPORT"
    echo "   PGUSER=$PGUSER"
    echo "   PGDATABASE=$PGDATABASE"
    echo "3. pg_hba.conf 配置是否正确"
    exit 1
fi

# 2. 测试创建数据库权限
echo "测试创建数据库权限..."
if psql -c "CREATE DATABASE test_spider_perm;" && psql -c "DROP DATABASE test_spider_perm;"; then
    echo "数据库创建权限正常"
else
    echo "错误: 没有创建数据库的权限"
    echo "请确保用户 $PGUSER 具有 CREATEDB 权限"
    echo "可以使用以下命令授权："
    echo "   ALTER USER $PGUSER CREATEDB;"
    exit 1
fi

# 3. 测试schema创建权限
echo "测试schema创建权限..."
psql -c "CREATE SCHEMA test_spider_schema;" && psql -c "DROP SCHEMA test_spider_schema;"
if [ $? -eq 0 ]; then
    echo "Schema创建权限正常"
else
    echo "错误: 没有创建schema的权限"
    echo "请确保用户 $PGUSER 具有创建schema的权限"
    exit 1
fi

echo "所有权限检查通过！"
