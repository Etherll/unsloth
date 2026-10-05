# Lane helpers (source after env-linux.sh + `lane N`). Every job waits for a free GPU, logs nvidia-smi and
# load average at start and end, and is retried once; a second failure is logged and skipped.
# The timing lane can hold lanes 6/7 idle: they finish their current job and wait while logs/hold_<N> exists.
jlog() { echo "$(date '+%F %T') [gpu$LANE] $*" | tee -a $RES/jobs.log; }
smistate() {
    echo "loadavg $(cut -d' ' -f1-3 /proc/loadavg) | gpus $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ';') | apps $(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | tr '\n' ';')"
}
job() {
    local name=$1; shift
    mkdir -p $OUT/logs
    for attempt in 1 2; do
        while true; do
            while [ -f $LP_ROOT/logs/hold_$LANE ]; do sleep 30; done
            # Timing-sensitive lanes set LP_JOB_LOAD_MAX: wait while other users load the CPUs.
            while [ -n "$LP_JOB_LOAD_MAX" ] && [ $(cut -d. -f1 /proc/loadavg) -gt $LP_JOB_LOAD_MAX ]; do sleep 60; done
            $PY $S/gpufree.py $LANE || { jlog "ABORT $name: gpufree failed"; return 1; }
            [ -n "$LP_JOB_LOAD_MAX" ] && [ $(cut -d. -f1 /proc/loadavg) -gt $LP_JOB_LOAD_MAX ] && continue
            [ -f $LP_ROOT/logs/hold_$LANE ] || break
        done
        touch $LP_ROOT/logs/running_gpu$LANE
        jlog "START $name (attempt $attempt) $(smistate)"
        # One job per GPU even with two queues on a lane.
        flock $LP_ROOT/logs/lock_gpu$LANE taskset -c "$LP_CPUS" "$@" > $OUT/logs/$name.log 2>&1
        local rc=$?
        rm -f $LP_ROOT/logs/running_gpu$LANE
        jlog "END $name rc=$rc $(smistate)"
        [ $rc -eq 0 ] && return 0
        tail -5 $OUT/logs/$name.log | sed "s/^/    /" | tee -a $RES/jobs.log
    done
    jlog "SKIP $name: failed twice"
    return 1
}
hold() {
    for n in "$@"; do touch $LP_ROOT/logs/hold_$n; done
    for n in "$@"; do while [ -f $LP_ROOT/logs/running_gpu$n ]; do sleep 15; done; done
    jlog "holding lanes $*"
}
release() { for n in "$@"; do rm -f $LP_ROOT/logs/hold_$n; done; jlog "released lanes $*"; }
