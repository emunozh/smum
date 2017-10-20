#!/bin/sh

# get public IP
IP=$(wget -q -O - checkip.dyndns.org|sed -e 's/.*Current IP Address: //' -e 's/<.*$//')
echo $IP

# make certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem

# initiate jupyter notebook
jupyterhub --ip=0.0.0.0 --ssl-cert mycert.pem --ssl-key mykey.key
