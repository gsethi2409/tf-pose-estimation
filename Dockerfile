FROM idock.daumkakao.io/kakaobrain/deepcloud-sshd:openpose-preprocess

COPY ./*.py /root/tf-openpose/
WORKDIR /root/tf-openpose/

RUN cd /root/tf-openpose/ && pip3 install -r requirements.txt

ENTRYPOINT ["python3", "pose_dataworker.py"]
