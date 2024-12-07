sudo rm -rf keys
mkdir keys
sudo chmod 777 keys
./.github/scripts/init-temp-keys.sh
sudo chmod 777 keys/*
sudo chmod 777 *.pem
docker build -t opentdf:latest .

