import noodles
from noodles.workflow import (get_workflow)
from noodles.run.coroutines import (Connection, coroutine_sink)
from noodles.run.experimental import (logging_worker)
from noodles.display import SimpleDisplay

import subprocess
import sys
import argparse
import json
import time
import shlex
import os

logFolder = None

class Job:
    def __init__(self, task, exclude, state, job, key):
        self.task = task
        self.exclude = exclude
        self.state = state
        self.job = job
        self.key = key


def dynamic_exclusion_worker(display, n_threads):
    """This worker allows mutualy exclusive jobs to start safely. The
    user provides the information on which jobs exclude the simultaneous
    execution of other jobs::

        a = task()
        b = task()

        update_hints(a, {'id': '1', 'exclude': ['2']})
        update_hints(b, {'id': '2', 'exclude': ['1']})

        run(gather(a, b))

    Using this worker, when task ``a`` is sent to the underlying worker,
    task ``b`` is blocked until ``a`` completes, and vice versa.
    """
    result_source, job_sink = logging_worker(n_threads, display).setup()

    jobs = {}
    key_task = {}

    @coroutine_sink
    def pass_job():
        """The scheduler sends jobs to this coroutine. If the 'exclude' key
        is found in the hints, it is run in exclusive mode. We keep an internal
        record of these jobs, and whether they are 'waiting', 'running' or 'done'.
        """
        while True:
            key, job = yield

            if job.hints and 'exclude' in job.hints:
                j = Job(task=job.hints['id'],
                        exclude=job.hints['exclude'],
                        state='waiting',
                        job=job,
                        key=key)
                jobs[j.task] = j
                key_task[key] = j.task
                try_to_start(j.task)

            else:
                job_sink.send((key, job))

    def is_not_running(task):
        """Checks if a task is not running."""
        return not (task in jobs and jobs[task].state == 'running')

    def try_to_start(task):
        """Try to start a task. This only succeeds if the task hasn't already
        run, and no jobs are currently running that is excluded by the task."""
        if jobs[task].state != 'waiting':
            return

        if all(is_not_running(i) for i in jobs[task].exclude):
            jobs[task].state = 'running'
            key, job = jobs[task].key, jobs[task].job
            job_sink.send((key, job))

    def finish(key):
        """Finish a job. This function is called when we recieve a result."""
        task = key_task[key]
        jobs[task].state = 'done'
        for i in jobs[task].exclude:
            try_to_start(i)

    def pass_result():
        """Recieve a result; finish the task in the register and send the result
        back to the scheduler."""
        for key, status, result, err in result_source:
            if key in key_task:
                finish(key)

            yield (key, status, result, err)

    return Connection(pass_result, pass_job)


def run(wf, *, display, n_threads=1):
    """Run the workflow using the dynamic-exclusion worker."""
    worker = dynamic_exclusion_worker(display, n_threads)
    return noodles.Scheduler(error_handler=display.error_handler)\
        .run(worker, get_workflow(wf))


@noodles.schedule_hint(display='{cmd}', confirm=True)
def system_command(cmd, task):
    cmd_split = shlex.split(cmd)  # list(shlex.shlex(cmd))
    p = subprocess.run(
        cmd_split, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, universal_newlines=True)
    p.check_returncode()
    oFile = open(os.path.join(logFolder, task),'w')
    oFile.write(p.stdout)
    oFile.close()
    return p.stdout


def make_job(cmd, task_id, exclude):
    j = system_command(cmd, task_id)
    noodles.update_hints(j, {'id': str(task_id),
                             'exclude': [str(x) for x in exclude]})
    return j


def error_filter(xcptn):
    if isinstance(xcptn, subprocess.CalledProcessError):
        return xcptn.stderr
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SOBA: Run a non-directional exclusion graph job.")
    parser.add_argument(
        '-j', dest='n_threads', type=int, default=1,
        help='number of threads to run simultaneously.')
    parser.add_argument(
        '-dumb', default=False, action='store_true',
        help='print info without special term codes.')
    parser.add_argument(
        'target', type=str,
        help='a JSON file specifying the graph.')
    parser.add_argument(
        'log', type=str,
        help='a log folder.')
    args = parser.parse_args(sys.argv[1:])

    logFolder = args.log
    os.makedirs(logFolder)
    input = json.load(open(args.target, 'r'))
    jobs = [make_job(td['command'],
                     td['id'], td['exclude']) for td in input]
    wf = noodles.gather(*jobs)
    with SimpleDisplay(error_filter) as display:
        run(wf, display=display, n_threads=args.n_threads)
