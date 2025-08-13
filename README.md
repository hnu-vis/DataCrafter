# DataCrafter

This is the source code for our paper "Human-Guided Image Generation for Small-Scale Training Datasets Expansion."

This project uses [python 3.10](https://www.python.org/), [cuda 11.8](https://developer.nvidia.com/cuda-toolkit), and [torch 2.3.0](https://pytorch.org/). Go check it out if you don't have them installed.
## Replicate the results of M2M
If you'd like to reproduce the projection results in the paper, please follow the commands below.
```sh
$ cd M2M/
$ pip install -r requirements.txt
$ python train.py --device cuda
```
The results will be saved in ```/root/DataCrafter/M2M/results/M2M```.
The training was performed on an NVIDIA A100 GPU. The total training time was under 30 minutes, and the peak video memory usage did not exceed 4 GB. 
## Install the environment and run for the system

### Option 1: Docker (recommended)
1. The easiest way to install an environment to run the demo is to use docker. We have packaged all the environment and code into a compressed image package. Please download it from  [datacrafter-run.tar](https://drive.google.com/file/d/1-KQuDaHJ4JtRt-w98Qhw0cefX8iOIX0g/view?usp=drive_link). You can load and start the image with the following command.

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

2. Install python package.
```sh
$ pip install -r requirements.txt
$ pip install torch
```
3. Download the repo

4. Download demo data from [here](https://drive.google.com/file/d/1se-uJddNTuUKAenlMDAu4dL99Xi-YjrT/view?usp=drive_link), and unpack it in the `demo` folder.

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

## Citation 
If you find our work inspiring or use our code in your research, please cite our work:
```sh
@article{chen2025human,
  title={Human-guided image generation for expanding small-scale training image datasets},
  author={Chen, Changjian and Lv, Fei and Guan, Yalong and Wang, Pengcheng and Yu, Shengjie and Zhang, Yifan and Tang, Zhuo},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```
