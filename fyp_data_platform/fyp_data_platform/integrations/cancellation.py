"""Cooperative orchestration cancellation and bounded model process shutdown."""
import os
import signal
import subprocess


class ModelRunStopped(Exception):
    pass


def check_stop(event):
    if event is not None and event.is_set():
        raise ModelRunStopped('Stopped by user.')


def run_cancellable(command, *, cwd, stdout, cancel_event):
    check_stop(cancel_event)
    with subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT,
                          start_new_session=os.name == 'posix') as process:
        while process.poll() is None:
            if not cancel_event.wait(0.2):
                continue
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                process.wait()
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise ModelRunStopped('Stopped by user.')
        check_stop(cancel_event)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
