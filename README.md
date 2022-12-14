# Feature Store
This codebase preprocesses events data every 4 hours and store updated data into Redis Feature store.

# Set-up .env file
There needs to be a `.env` file with following parameters.
```
AWS_ACCESS_KEY_ID='XXXXX'
AWS_SECRET_ACCESS_KEY='XXXXXX'
AWS_REGION='us-east-2'

REDIS_IP = 'localhost'
REDIS_PORT = 6379
ASSET_FS_DB = 8
USER_FS_DB = 9
```

# Instructions (Docker)
1) Ensure there are environment variables or `.env` file, see section above for environment variables.
```
docker-compose build
docker-compose up
```

# Local testing

```
conda env create --file /app/environment.yml
conda activate feature_store
python preprocessing_pipeline.py
```

# Enabling GPU for Docker
To enable the GPU for Docker, make sure Nvidia drivers for the system are installed. [Refer link for details](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)

Commands which can help install Nvidia drivers are:
```
unbuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

Then nvidia-docker2 tools needs to be installed.
To install follow the below instructions.
[Refer link for details](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```
curl https://get.docker.com | sh   && sudo systemctl --now enable docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
# Technical documentation
```
pip install -r mkdocs_requirements.txt
mkdocs serve
```
