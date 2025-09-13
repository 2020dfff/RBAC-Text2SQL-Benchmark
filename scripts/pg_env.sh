#!/bin/bash

# PostgreSQL连接配置
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=spider_benchmark
export PGUSER=postgres
export PGPASSWORD=postgres  # 在生产环境中应该更改为安全的密码

# 确保文件权限安全
if [ -f "$0" ]; then
    chmod 600 "$0"
fi
