import subprocess


def run_command(command):
    p = subprocess.Popen([command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate(timeout=10)
    rc = p.returncode

    return out, err, rc
