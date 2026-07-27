# 杀死所有指定端口的进程
for port in 30000 36677 54322 8000 8291 8292 8293 8294 8295 8296 8297 8298 8888 9999 8889 \
8271 8272 8273 5203 5083 5773 3221 6000 6001 6002; do
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

echo "清理5开头的新端口..."
for port in {51001..51008} {52001..52008} {53001..53008} {54001..54008}; do
    if lsof -ti:$port &>/dev/null; then
        echo "  杀死端口 $port 的进程"
        lsof -ti:$port | xargs kill -9 2>/dev/null
    fi
done

fuser -k /dev/nvidia*

set -ex

rm -rf core*
rm -rf log
rm -rf log_*
ps -ef | grep "api_server" | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep "worker_process.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9

fastdeploy_inferernce_pids=$(ps auxww | grep "fastdeploy" | grep -v grep | awk '{print $2}')
echo $fastdeploy_inferernce_pids
for in_pid in ${fastdeploy_inferernce_pids[@]}; do
    kill -9 ${in_pid}
done
echo 'end fastDeploy inference pids'

api_server_pids=$(ps auxww | grep "api_server" | grep -v grep | awk '{print $2}')
echo 'end api server pids:'
echo $api_server_pids

for pid in $api_server_pids; do
    child_pids=$(ps -ef | grep $pid | grep -v grep | awk '{print $2}')
    echo $child_pids
    for in_pid in ${child_pids[@]}; do
        kill -9 ${in_pid}
    done
    echo 'end uvicorn multi workers'
done

lsof /dev/nvidia* | grep -v PID | awk '{print $2}' | xargs kill -9