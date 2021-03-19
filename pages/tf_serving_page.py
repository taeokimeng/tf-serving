import streamlit as st
import subprocess
from modules.command import run_command


def display_terminal(command):
    out, err, rc = run_command(command)
    for line in out.decode('utf-8').split('\n'):
        st.write(line)
    if rc != 0:
        st.write(f"{err.decode('utf-8')}\nReturn code: {rc}")

def display_tensorflow_serving(state):
    st.title(":knife_fork_plate: TensorFlow Serving")
    if state.serving_state is None:
        state.serving_state = 'Unknown'
    display_tensorflow_serving_config(state)

    # Input your configurations
    st.write("---")
    state.host_port = st.text_input("Host port:", state.host_port or "")
    state.container_port = st.text_input("Container port (8501 for REST API):", state.container_port or "8501")

    PORT = state.host_port
    state.container_name = st.text_input("Container name:", state.container_name or "")
    state.source_path = st.text_input("Mount source path:", state.source_path or "")
    state.target_path = st.text_input("Mount target path:", state.target_path or "")
    state.model_config_file_path = st.text_input("Model configuration file path:", state.model_config_file_path or "")

    # Run command
    st.write("---")
    st.subheader("Terminal")
    cmd = st.text_input("Command:")
    if st.button("Enter"):
        display_terminal(cmd)

def display_tensorflow_serving_config(state):
    st.subheader("Docker run configuration")
    st.write("Host port:", state.host_port)
    st.write("Container port:", state.container_port)
    st.write("Container name:", state.container_name)
    st.write("Mount source path:", state.source_path)
    st.write("Mount target path:", state.target_path)
    st.write("Model configuration file path:", state.model_config_file_path)

    cmd = f"docker run --rm -d -p {state.host_port}:{state.container_port} --name {state.container_name} " \
            f"--mount type=bind,source={state.source_path},target={state.target_path} " \
            f"tensorflow/serving --model_config_file={state.model_config_file_path}"

    st.write("Command preview:", cmd)

    st.write("TensorFlow Serving status:", f'**{state.serving_state}**')
    if st.button("TensorFlow Serving status check"):
        cmd = 'docker ps --filter ancestor=tensorflow/serving --format "{{.Status}}"'
        out, err, rc = run_command(cmd)
        if out == b'':
            state.serving_state = "Not running"
        else:
            state.serving_state = out.decode('utf-8').replace('\n', ', ', out.decode('utf-8').count('\n') - 1)

    if st.button("Run Serving"):
        out, err, rc = run_command(cmd)
        if rc != 0:
            st.warning(f"{err.decode('utf-8')}\nReturn code: {rc}")
        else:
            st.success("Serving completed.")

    if st.button("Clear configuration"):
        state.clear()


def run_ssh(state):
    p = subprocess.Popen(["ssh -T tokim@localhost"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate(b"docker run --rm -d -p 8511:8511 --name tf_serving_image_classifiers --mount type=bind,source=/home/tokim/code/tf-serving/models/,target=/models tensorflow/serving --model_config_file=/models/models.config\n",
                             timeout=10)
    print("output:", out.decode("utf-8"), "error:", err.decode("utf-8"))
    print(p.returncode)
    if p.returncode != 0:
        st.warning(f"{err.decode('utf-8')}\nReturn code: {p.returncode}")
    else:
        st.info("Serving completed.")
        state.serving_state = "Running"

# def run_docker_container(state):
#     run_comm = f"docker run --rm -t -p {state.port} --mount type=bind,source={state.source_path},target={state.target_path} " \
#                f"tensorflow/serving & --model_config_file={state.model_config_file_path}"
#     run_comm = f"docker run --rm -t -p 8510:8510 -v $PWD/models:/models tensorflow/serving & --model_config_file=/models/models.config"
#
#     p = subprocess.Popen([run_comm], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#
#     try:
#         # Python 3.7+
#         output = subprocess.run([run_comm], capture_output=True, shell=True, text=True)
#     except TypeError:
#         # Under Python 3.7+
#         output = subprocess.run([run_comm], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
#                                 universal_newlines=True)
#
#     p = subprocess.Popen(["docker run --rm -t tokimeng/tf-st:latest"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#     output, err = p.communicate()
#     rc = p.returncode
#     print(output, err)
#     return p