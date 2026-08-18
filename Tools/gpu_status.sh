#!/bin/bash
# 快速查看集群所有机器 GPU 状态
# 用法: bash gpu_status.sh [hostfile]

HOSTFILE="${1:-/root/paddlejob/workspace/hostfile}"

if [ ! -f "$HOSTFILE" ]; then
  echo "hostfile 不存在: $HOSTFILE"
  exit 1
fi

tmpdir=$(mktemp -d)
trap "rm -rf $tmpdir" EXIT

# 并行SSH查询所有机器
for ip in $(awk '{print $1}' "$HOSTFILE" | grep -v '^$'); do
  ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no "$ip" \
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader" \
    > "$tmpdir/$ip" 2>/dev/null &
done
wait

echo "=========================================="
echo "       集群 GPU 状态总览 (H800 80G)"
echo "=========================================="
echo ""

# 收集各类机器IP
full_free_ips=""
partial_free_ips=""
no_free_ips=""
offline_ips=""

for ip in $(awk '{print $1}' "$HOSTFILE" | grep -v '^$'); do
  f="$tmpdir/$ip"
  if [ ! -s "$f" ]; then
    echo "[$ip] 连接失败或无输出"
    echo ""
    offline_ips="$offline_ips $ip"
    continue
  fi

  total_gpus=$(wc -l < "$f")
  free_gpus=$(awk -F', ' '{gsub(/ MiB/,"",$2); if($2+0 < 1000) c++} END{print c+0}' "$f")
  busy_gpus=$(awk -F', ' '{gsub(/ %/,"",$4); if($4+0 >= 50) c++} END{print c+0}' "$f")

  echo "[$ip] 共${total_gpus}卡 | 可用(显存<1G): ${free_gpus}卡 | 高负载(>=50%): ${busy_gpus}卡"

  awk -F', ' '{
    gsub(/ MiB/,"",$2); gsub(/ MiB/,"",$3); gsub(/ %/,"",$4);
    used=$2+0; total=$3+0; util=$4+0;
    if(total>0) pct=int(used*100/total); else pct=0;
    if(used < 1000)     status="空闲";
    else if(util < 50)  status="占用(低负载)";
    else                status="繁忙";
    printf "  GPU%-2s  %5d/%5d MiB (%2d%%)  利用率%3d%%  [%s]\n", $1, used, total, pct, util, status
  }' "$f"
  echo ""

  # 分类统计
  if [ "$free_gpus" -eq "$total_gpus" ]; then
    full_free_ips="$full_free_ips $ip($free_gpus卡)"
  elif [ "$free_gpus" -gt 0 ]; then
    partial_free_ips="$partial_free_ips $ip($free_gpus卡)"
  else
    no_free_ips="$no_free_ips $ip"
  fi
done

# 汇总报告
echo "=========================================="
echo "              汇  总"
echo "=========================================="

full_count=$(echo $full_free_ips | wc -w)
partial_count=$(echo $partial_free_ips | wc -w)
# word count counts "ip(N卡)" as one word each
full_count=$((full_count))
partial_count=$((partial_count))

echo ""
if [ -n "$full_free_ips" ]; then
  echo "[整机空闲 (8卡全部可用)] 共 ${full_count} 台:"
  for entry in $full_free_ips; do
    echo "  $entry"
  done
else
  echo "[整机空闲 (8卡全部可用)] 无"
fi

echo ""
if [ -n "$partial_free_ips" ]; then
  echo "[部分空闲] 共 ${partial_count} 台:"
  for entry in $partial_free_ips; do
    echo "  $entry"
  done
else
  echo "[部分空闲] 无"
fi

echo ""
if [ -n "$no_free_ips" ]; then
  echo "[全部占用]:"
  for entry in $no_free_ips; do
    echo "  $entry"
  done
fi

if [ -n "$offline_ips" ]; then
  echo ""
  echo "[离线/不可达]:"
  for entry in $offline_ips; do
    echo "  $entry"
  done
fi
echo ""
