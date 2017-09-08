FROM dock0/arch
MAINTAINER emunozh <emunozh@gmail.com>
RUN pacman -S --needed --noconfirm base

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pacman -Syu --noconfirm
RUN pacman -S --needed --noconfirm python python-setuptools
RUN pacman -S --needed --noconfirm python python-pip
RUN pacman -S --needed --noconfirm gcc
RUN pacman -S --needed --noconfirm tk
RUN pacman -S --needed --noconfirm gcc-fortran
RUN pacman -S --needed --noconfirm r
RUN pacman -S --needed --noconfirm python-rpy2
RUN pacman -S --needed --noconfirm python-ipykernel
RUN pacman -S --needed --noconfirm jupyter-notebook
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY urbanmetabolism/ ./urbanmetabolism

RUN ln -s urbanmetabolism/data data
RUN ln -s urbanmetabolism/temp temp

COPY GREGWT_0.7.5.tar.gz ./
RUN R CMD INSTALL GREGWT_0.7.5.tar.gz
#EXPOSE 8888

#CMD [ "python", "./test.py" ]
#RUN ipython3 notebook --port 80
#RUN jupyter notebook
