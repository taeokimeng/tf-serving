# TensorFlow Serving using Docker

## How to use
1. Run TensorFlow Docker Image including Streamlit. (It allows you to use docker command inside a container and to connect with host network.)

* /home/tokim/code/tf-serving should be your directory path.
* tokimeng/tf:0.2 should be the Docker Image name you want to use.

~~~
docker run -it --network host --name serving_manager -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker -v /home/tokim/code/tf-serving:/tf-serving tokimeng/tf:0.2 bash
~~~

2. Go to the directory and run Streamlit application (terminal in the container)
~~~
cd /tf-serving
streamlit run app.py
~~~

### Command examples
Start TensorFlow serving server

* To use a configuration file
~~~
docker run -t -p 8501:8501 --name tf_serving_image_classifiers --mount type=bind,source=/home/tokim/code/tf-serving/models/,target=/models tensorflow/serving --model_config_file=/models/models.config --model_config_file_poll_wait_seconds=60
~~~
* To use model warmup
~~~
docker run -t -p 8501:8501 --name tf_serving_image_classifiers --mount type=bind,source=/home/tokim/code/tf-serving/models/,target=/models tensorflow/serving --model_config_file=/models/models.config --model_config_file_poll_wait_seconds=60 --enable_model_warmup=true
~~~
* To use batching requests
~~~
docker run -t -p 8501:8501 --name tf_serving_image_classifiers --mount type=bind,source=/home/tokim/code/tf-serving/models/,target=/models tensorflow/serving --model_config_file=/models/models.config --model_config_file_poll_wait_seconds=60 --enable_batching --batching_parameters_file=/models/batching.conf
~~~
Above command describes the following:
* REST API port 8501 to our host's port 8501
* Docker container name is tf_serving_vgg16
* Provide the model configuration file
* Check changes in the configuration file every 60 seconds

* To run TensorFlow Docker Image including Streamlit
~~~
docker run -it --name docker_streamlit --network host -v /home/tokim/code/tf-serving:/tf-serving tokimeng/tfvbox:streamlit
~~~
