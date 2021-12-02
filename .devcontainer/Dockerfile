FROM dealii/dealii:latest

USER root

RUN apt-get update && \
    apt-get install -y \
        libboost-log1.71-dev \
        libgtest-dev \
        python3-dev \
        python3-distutils \
        python3-h5py \
        python3-matplotlib \
        python3-pip \
        python3-pybind11 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python packages for dask
RUN pip3 install bokeh dask distributed jupyter-server-proxy scipy

USER dealii

# Put the ssh deploy key in the right dir; moreover add the github key to the known_hosts files
RUN mkdir /home/dealii/.ssh && \
    chmod 700 /home/dealii/.ssh && \
    echo -----BEGIN OPENSSH PRIVATE KEY----- > /home/dealii/.ssh/id_rsa && \
    echo b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn >> /home/dealii/.ssh/id_rsa && \
    echo NhAAAAAwEAAQAAAYEA1XQUz4y00qRj09fmIF646hC/tWPq5QHKiu7GjOUWOubq55STDS8r >> /home/dealii/.ssh/id_rsa && \
    echo snewBVNSnSTPLz4ipCo2W2s8XWJZD/UrEBVuXHYIM8nvplaVSH5y+e52iTeHMkLE+MYeAO >> /home/dealii/.ssh/id_rsa && \
    echo cvVA2PTlQhkqn2tIPwB8rtKXUYxIZBb0meTbOVppAvoPDNO4kSJtfP4N7lHNGgC6xOZt1B >> /home/dealii/.ssh/id_rsa && \
    echo M1bcFjglJkJuGgND90rGtfs4JhbXbtg3KM3VWew7mU/bhObHTaLOUSzNTL/yb3AJSpE5xO >> /home/dealii/.ssh/id_rsa && \
    echo 70JtAGLJTJURuzshv4xBip90EAjBbNkU9WspbtORXU2nfvtrcHHHT8AssDimMXN6kKsGbK >> /home/dealii/.ssh/id_rsa && \
    echo 167wn1ZYulKdk4RziJWVB4lJTBAx8qWDToNnXa7OaZKoAyGsuRm7MfKNakvdl/9bmaxW4t >> /home/dealii/.ssh/id_rsa && \
    echo r64LEvMyDgkEwc5fc3kvpDHH9YjY7IVxD9K2rRZ7V7edDvVl/yTzJiyRHLAEAHO+syTarl >> /home/dealii/.ssh/id_rsa && \
    echo zZuXDHvLMr/P+1h3VZ7/yFD/BcK+9eob8vGhTVKtAAAFgHdZ5BV3WeQVAAAAB3NzaC1yc2 >> /home/dealii/.ssh/id_rsa && \
    echo EAAAGBANV0FM+MtNKkY9PX5iBeuOoQv7Vj6uUByoruxozlFjrm6ueUkw0vK7J3sAVTUp0k >> /home/dealii/.ssh/id_rsa && \
    echo zy8+IqQqNltrPF1iWQ/1KxAVblx2CDPJ76ZWlUh+cvnudok3hzJCxPjGHgDnL1QNj05UIZ >> /home/dealii/.ssh/id_rsa && \
    echo Kp9rSD8AfK7Sl1GMSGQW9Jnk2zlaaQL6DwzTuJEibXz+De5RzRoAusTmbdQTNW3BY4JSZC >> /home/dealii/.ssh/id_rsa && \
    echo bhoDQ/dKxrX7OCYW127YNyjN1VnsO5lP24Tmx02izlEszUy/8m9wCUqROcTu9CbQBiyUyV >> /home/dealii/.ssh/id_rsa && \
    echo Ebs7Ib+MQYqfdBAIwWzZFPVrKW7TkV1Np377a3Bxx0/ALLA4pjFzepCrBmyteu8J9WWLpS >> /home/dealii/.ssh/id_rsa && \
    echo nZOEc4iVlQeJSUwQMfKlg06DZ12uzmmSqAMhrLkZuzHyjWpL3Zf/W5msVuLa+uCxLzMg4J >> /home/dealii/.ssh/id_rsa && \
    echo BMHOX3N5L6Qxx/WI2OyFcQ/Stq0We1e3nQ71Zf8k8yYskRywBABzvrMk2q5c2blwx7yzK/ >> /home/dealii/.ssh/id_rsa && \
    echo z/tYd1We/8hQ/wXCvvXqG/LxoU1SrQAAAAMBAAEAAAGBAK1IHjjbnlO+Nnhctaojm3QRAs >> /home/dealii/.ssh/id_rsa && \
    echo Cri547lQSykKcCtsg8fAhh+frjB1NLjK7WFKQTpMpSY+bnidcvOAMLewfUXRAihcQz4bbB >> /home/dealii/.ssh/id_rsa && \
    echo QXbEpU4DsS0qpDoCxRNNVwAqPWM4SpiupV7ON68LoYar8SkyWNOmBuRLpUVF+2hS2LMuLC >> /home/dealii/.ssh/id_rsa && \
    echo w4MDQbs4uq+UwxRsfdeVLgn4ksJukZo3O6QpZLqXwemIdhwe0sM0ngkyWeCzFp38CvYTa1 >> /home/dealii/.ssh/id_rsa && \
    echo uJysiLGC8IC/oS6kSJik1rmeDwcgv39JRK4k5tffnlow9SmYz+D/LnGjsoiykUtC5BVxyZ >> /home/dealii/.ssh/id_rsa && \
    echo F6XxMTVzBUM4Qtk7yX8mxjqIwtfHwwuFhr9aAWxFf0SL335jXRKmqSNCXjc0GDsSKDtFU9 >> /home/dealii/.ssh/id_rsa && \
    echo ZtihO91ubNkfwDeJQn3VVOePZY6v398HY6AI/9TEFMt6B22NBYx+yZ2J+qN1qkKg1jKtFg >> /home/dealii/.ssh/id_rsa && \
    echo mXoZnD+EEMGNduANCN/GAG1l8VwO+QcibUiR1Qfr02YWGtVTx/f6siuCjZItOzWcbqAQAA >> /home/dealii/.ssh/id_rsa && \
    echo AMBcySe80ASiylYojL6Akkv07OHE6MYWIC4qGaOmb1CYaSuBrkd5Cekl0/gDfD5Zm5zOf8 >> /home/dealii/.ssh/id_rsa && \
    echo R+23Li+0yz2XBXbM40c1eIOJL/3raCXqzas7H8Dh0u5Q6CesMDU0yoxoDHk3LJU4kJPuWc >> /home/dealii/.ssh/id_rsa && \
    echo ASL6vASI2Hj+jkubxH1DDAS7JflsMZhqvp1Y0pLv+kg7PlXdnC/tbWEyVepIHcx0xNfX+j >> /home/dealii/.ssh/id_rsa && \
    echo YS+uGhsPbkVmYizZyCMmTvjle9fTJzojs608x09Ltfypn0IIcAAADBAPr0X7szthVzZn8b >> /home/dealii/.ssh/id_rsa && \
    echo aGo6kmPhvSXFgF55Zzts3P5zh95VvFkwD2gM/hHxrPCuEGnXvuoUdohsei31AC0bbp4sa/ >> /home/dealii/.ssh/id_rsa && \
    echo Q+RVQRe4DEoHO0YEdBIa16/AvZLRWzV85r7JcU5DZsoqPo83fuGwcFWsOsb4LhhzLV0mDt >> /home/dealii/.ssh/id_rsa && \
    echo UoBEKIJlyHe28kwK9zm0R7pvzI0F8W1nXvOTEigsjPjkKTcJCZfgVWjvPCD1Y+9mkGV8ll >> /home/dealii/.ssh/id_rsa && \
    echo gnX3K/pVwnp4qJNMEvHDV464MQtgASgQAAAMEA2b6xy99duqGIeS1ZFXbyq8HHRu8Objbu >> /home/dealii/.ssh/id_rsa && \
    echo ZcDKOhONXHSVaijD7Na7+dc+8poSFbuwF1jFUdNU5b/ZgeCao9uEXjOKxczNEedFfk1UA6 >> /home/dealii/.ssh/id_rsa && \
    echo pcXiGnn905cClG2/eK8/yOPKKCOmCcWer2AOnSlL246YL70tCTh1xw6z7flIiNTWPCybT3 >> /home/dealii/.ssh/id_rsa && \
    echo x7GaAZVtMA9Yw85HX+hwnZwHBs3S2e6nxJoMEs0y88y7yljf/Q+/4rSFT+nyy5Kwub8+Xi >> /home/dealii/.ssh/id_rsa && \
    echo T/YBcOkV4ZfRItAAAACXN0ZXBAWm9ybgE= >> /home/dealii/.ssh/id_rsa && \
    echo -----END OPENSSH PRIVATE KEY----- >> /home/dealii/.ssh/id_rsa && \
    chmod 600 /home/dealii/.ssh/id_rsa && \
    echo github.com ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAq2A7hRGmdnm9tUDbO9IDSwBK6TbQa+PXYPCPy6rbTrTtw7PHkccKrpp0yVhp5HdEIcKr6pLlVDBfOLX9QUsyCOV0wzfjIJNlGEYsdlLJizHhbn2mUjvSAHQqZETYP81eFzLQNnPHt4EVVUh7VfDESU84KezmD5QlWpXLmvU31/yMf+Se8xhHTvKSCZIFImWwoG6mbUoWf9nzpIoaSjB+weqqUUmpaaasXVal72J+UX2B+2RPW3RcT0eOzQgqlJL3RKrTJvdsjE3JEAvGq3lGHSZXy28G3skua2SmVi/w4yCE6gbODqnTWlg7+wC604ydGXA8VJiS5ap43JXiUFFAaQ== > /home/dealii/.ssh/known_hosts && \
    chmod 644 /home/dealii/.ssh/known_hosts
