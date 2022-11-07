#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import sys
import cv2
from numpy.lib.function_base import copy
import torch

from feature_extractor import to_segments
from utils.utils import build_transforms
from utils.load_model import load_models

# env
# ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
# SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
client = boto3.client('kinesisvideo', region_name='ap-northeast-1')

# response = client.list_streams()
# print(response)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


transforms = build_transforms(mode='c3d')


anomaly_detector, feature_extractor = load_models(
        feature_extractor_path = 'pretrained/c3d.pickle',
        ad_model_path = 'exps/c3d/models/epoch_80000.pt',
        features_method = 'c3d',
        device = device,
)


# In[2]:


def get_data_endpoint(stream_name: str, api_name: str) -> str:
    """
    call GetDataEndpoint API
    https://docs.aws.amazon.com/kinesisvideostreams/latest/dg/API_GetDataEndpoint.html
    """
    kinesis_video = boto3.client("kinesisvideo", region_name='ap-northeast-1')
    res = kinesis_video.get_data_endpoint(StreamName=stream_name, APIName=api_name)
    return res["DataEndpoint"]


def get_hls_streaming_session_url(stream_name: str) -> str:
    """
    call GetHLSStreamingSessionURL API
    https://docs.aws.amazon.com/kinesisvideostreams/latest/dg/API_reader_GetHLSStreamingSessionURL.html
    """
    res = boto3.client(
        "kinesis-video-archived-media", endpoint_url=get_data_endpoint(stream_name, "GET_HLS_STREAMING_SESSION_URL"), region_name='ap-northeast-1'
    ).get_hls_streaming_session_url(
        StreamName=stream_name,
        PlaybackMode="LIVE",
        Expires=43200
    )
    return res["HLSStreamingSessionURL"]


# In[3]:


def extract_features(frames):
    frames = torch.tensor(frames)
    frames = transforms(frames).to(device)
    data = frames[:, range(0, frames.shape[1], 1), ...]
    data = data.unsqueeze(0)
    with torch.no_grad():
        outputs = feature_extractor(data.to(device)).detach().cpu()

    return to_segments(outputs.numpy(), 1)


def predict_anomaly_score(features):
    features = torch.tensor(features).to(device)
    with torch.no_grad():
        preds = anomaly_detector(features)

    return preds.detach().cpu().numpy().flatten()


# In[6]:


stream_name = 'ry-itano-vstream'
hls_url = get_hls_streaming_session_url(stream_name)
cap = cv2.VideoCapture(hls_url)

# cap = cv2.VideoCapture(0)

data_stream = boto3.client('kinesis', region_name='ap-northeast-1')

# window_name = 'LIVE Camera'
delay = 1
clip_len = 16
queue = []

if not cap.isOpened():
    sys.exit()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        queue.append(frame)

        if len(queue) == clip_len:

            frames = copy(queue)
            features = extract_features(frames=frames)
            score = predict_anomaly_score(features=features)
            res = data_stream.put_record(StreamName='ry-itano-anomaly-detection-stream', Data='{:.2f}'.format(score[0]*100), PartitionKey='123')
            print('{:.2f}'.format(score[0]*100), res['ResponseMetadata']['HTTPStatusCode'])
            queue.clear()
            
        # cv2.imshow(window_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        print('Update HLS URL')
        hls_url = get_hls_streaming_session_url(stream_name)
        cap = cv2.VideoCapture(hls_url)
        continue
        

# cv2.destroyWindow(window_name)


# In[5]:


#-------------------run-only-in-ipynb-environment-------------------------------

# ※CAUTION※ Save this file before executing the following code!!!
# Generate py from ipynb and save it automatically

if 'get_ipython' in globals():
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '*.ipynb'])
    print('Saved!')
# End of if 'if 'get_ipython' in globals():'

#-------------------run-only-in-ipynb-environment-------------------------------

