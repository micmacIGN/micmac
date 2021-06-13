FROM ubuntu:latest
MAINTAINER Ewelina Rupnik <ewelina.rupnik@ign.fr>

ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory
ENV foo /etc/opt/
WORKDIR ${foo}   

#IGN server specifique
#RUN export http_proxy="http://proxy.ign.fr:3128"
#RUN export https_proxy="https://proxy.ign.fr:3128"

#MicMac dependencies
RUN apt-get update && apt-get install -y \
		    build-essential \
		    make \
                    cmake \ 
                    git \
                    proj-bin \
		    exiv2 \
                    exiftool \
                    imagemagick \
		    xorg \
            	    openbox \
                    qt5-default \
                    meshlab \
                    vim


#MicMac clone
#IGN-specific proxy setting
#RUN git config --global http.proxy http://proxy.ign.fr:3128
#RUN git config --global https.proxy https://proxy.ign.fr:3128
RUN git clone https://github.com/micmacIGN/micmac.git

#MicMac build & compile
WORKDIR micmac
RUN mkdir build
WORKDIR build
RUN cmake ../ && make install -j8

#MicMac add environmental variable to executables
ENV PATH=$foo"micmac/bin/:${PATH}"
RUN echo $foo"micmac/bin/:${PATH}"
