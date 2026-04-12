#!/bin/bash
# patch_config.sh — 修改服务脚本中 JSON 配置字段的值
#
# 用法: patch_config.sh <文件路径> <key> <新值>
# 例如: patch_config.sh server.sh use_cudagraph true
#        patch_config.sh server.sh use_cudagraph false
#
# 支持 JSON 字段值类型：true/false、数字、带引号字符串

FILE=$1
KEY=$2
VAL=$3

if [ -z "$FILE" ] || [ -z "$KEY" ] || [ -z "$VAL" ]; then
    echo "用法: patch_config.sh <文件路径> <key> <新值>" >&2
    exit 1
fi

if [ ! -f "$FILE" ]; then
    echo "错误: 文件不存在: $FILE" >&2
    exit 1
fi

# 替换 JSON 字段值（匹配 "key": 任意值，直到逗号或右花括号前）
sed -i "s/\"${KEY}\":[[:space:]]*[^,}]*/\"${KEY}\":${VAL}/" "$FILE"

# 验证替换结果
RESULT=$(grep "\"${KEY}\"" "$FILE")
echo "✓ 已修改: ${RESULT}"
