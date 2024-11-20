# DataCrafter

This is the source code for our paper "Human-Guided Image Generation for Small-Scale Training Datasets Expansion."

## Install environment and run

### Option 1: Docker (recommended)
1. The easiest way to install a environment to run the demo is to use docker. We have packaged all the environment and code into a compressed image package. Please download it from  [datacrafter-run.tar](googledrive). You can load and start the image with the following command.

```sh
$ docker load -i datacrafter-run.tar
```
2. Run the docker image:
   
```sh
$ docker run -it --gpus all -p 24001:8081 24002:8082 datacrafter-run:latest
```

3. Run backend

```sh
$ cd /root/Datacrafter/backend
$ nohup python manager.py run &
```

4. Run frontend

```sh
$ cd /root/Datacrafter/fronted
$ npm install (it will take a while)
$ npm run serve
```

5. Visit http://localhost:24001/ in a browser.



### Option 2: Install with python and node.js
1. This project uses [python 3.10](https://www.python.org/), [cuda 11.8](https://developer.nvidia.com/cuda-toolkit), and [torch 2.3.0](https://pytorch.org/). Go check it out if you don't have them installed.

2. install python package.
```sh
$ pip install -r requirements.txt
$ pip install torch
```
3. Download the repo

4. Download demo data from [here](waittofill), and unpack it in the root folder 

5. Run backend

```sh
$ cd /root/Datacrafter/backend
$ nohup python manager.py run &
```

6. Run frontend

```sh
$ cd /root/Datacrafter/fronted
$ npm install (it will take a while)
$ npm run serve
```

7. Visit http://localhost:8081/ in a browser.



## Contact
If you have any problem about this code, feel free to contact
- feilv@hnu.edu.cn

or describe your problem in Issues.