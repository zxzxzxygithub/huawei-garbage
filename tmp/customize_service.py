# -*- coding: utf-8 -*-
# Copyright 2019 ModelArts Authors from Huawei Cloud. All Rights Reserved.
# https://www.huaweicloud.com/product/modelarts.html
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import ast
import io
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.util import make_tensor_proto

import h5py
from model_service.tfserving_model_service import TfServingBaseService
from PIL import Image

EPS = np.finfo(float).eps


def normalize_feature(feature):
  assert feature is not None
  return feature / (np.linalg.norm(feature) + EPS)


def get_topk_result(query,
                    all_data,
                    query_included=True,
                    metric_type='cosine_similarity',
                    top_k=1):
  """Get the nearset data to query in all_data

  :param query: having shape of (1, k) where k is the number of feature dimension.
  :param all_data: having shape of (m, k) where k is the number of feature dimension and m is the number of samples.
  :param query_included: whether the query is included in all_data.
  :param metric_type: metric on how to compute the similarities between any two smaples.
  :param top_k: the top k similar examples will be returned.

  :return an index or a list of index indicating the location of the most similar samples in all_data.
  """
  if metric_type == 'cosine_similarity':
    similarity_scores = np.dot(np.array(query), (np.array(all_data)).T)
    rankings = np.argsort(
        similarity_scores
    )[::-1]  # larger score means the samples are more similar
  elif metric_type == 'euclidean_distance':
    distances = [np.sqrt(np.sum(np.square(query, data))) for data in all_data]
    rankings = np.argsort(
        distances)  # smaller distance means the samples are more similar
  else:
    print('No metric_type is provided!')
  if top_k > 1:
    if query_included:
      return rankings[1:top_k + 1]
    return rankings[0:top_k]
  elif top_k == 1:
    if query_included:
      return rankings[1]
    return rankings[0]
  else:
    print('Please set valid top_k number!')


class cnn_service(TfServingBaseService):

  def _preprocess(self, data):
    preprocessed_data = {}
    for k, v in data.items():
      for file_name, file_content in v.items():
        image = Image.open(file_content)
        image = image.convert('RGB')
        image = np.asarray(image, dtype=np.float32)
        image = image[np.newaxis, :, :, :]
        preprocessed_data[k] = image
    return preprocessed_data

  def _postprocess(self, data):
    h5f = h5py.File(os.path.join(self.model_path, 'index'), 'r')
    labels_list = h5f['labels_list'][:]
    labels_list = [label.decode('utf-8') if isinstance(label, bytes) else label for label in labels_list]
    is_multilabel = h5f['is_multilabel'].value
    h5f.close()
    outputs = {}

    def softmax(x):
      x = np.array(x)
      orig_shape = x.shape

      if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / (np.sum(x) + EPS)
        x = np.apply_along_axis(exp_minmax, 1, x)
        denominator = np.apply_along_axis(denom, 1, x)
        if len(denominator.shape) == 1:
          denominator = denominator.reshape((denominator.shape[0], 1))
        x = x * denominator
      else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / (np.sum(numerator) + EPS)
        x = numerator.dot(denominator)
      assert x.shape == orig_shape

      return x
    self.label_id_name_dict = \
            {
                "0": "工艺品/仿唐三彩",
                "1": "工艺品/仿宋木叶盏",
    "2": "工艺品/布贴绣",
    "3": "工艺品/景泰蓝",
    "4": "工艺品/木马勺脸谱",
    "5": "工艺品/柳编",
    "6": "工艺品/葡萄花鸟纹银香囊",
    "7": "工艺品/西安剪纸",
    "8": "工艺品/陕历博唐妞系列",
    "9": "景点/关中书院",
    "10": "景点/兵马俑",
    "11": "景点/南五台",
    "12": "景点/大兴善寺",
    "13": "景点/大观楼",
    "14": "景点/大雁塔",
    "15": "景点/小雁塔",
    "16": "景点/未央宫城墙遗址",
    "17": "景点/水陆庵壁塑",
    "18": "景点/汉长安城遗址",
    "19": "景点/西安城墙",
    "20": "景点/钟楼",
    "21": "景点/长安华严寺",
    "22": "景点/阿房宫遗址",
    "23": "民俗/唢呐",
    "24": "民俗/皮影",
    "25": "特产/临潼火晶柿子",
    "26": "特产/山茱萸",
    "27": "特产/玉器",
    "28": "特产/阎良甜瓜",
    "29": "特产/陕北红小豆",
    "30": "特产/高陵冬枣",
    "31": "美食/八宝玫瑰镜糕",
    "32": "美食/凉皮",
    "33": "美食/凉鱼",
    "34": "美食/德懋恭水晶饼",
    "35": "美食/搅团",
    "36": "美食/枸杞炖银耳",
    "37": "美食/柿子饼",
    "38": "美食/浆水面",
    "39": "美食/灌汤包",
    "40": "美食/烧肘子",
    "41": "美食/石子饼",
    "42": "美食/神仙粉",
    "43": "美食/粉汤羊血",
    "44": "美食/羊肉泡馍",
    "45": "美食/肉夹馍",
    "46": "美食/荞面饸饹",
    "47": "美食/菠菜面",
    "48": "美食/蜂蜜凉粽子",
    "49": "美食/蜜饯张口酥饺",
    "50": "美食/西安油茶",
    "51": "美食/贵妃鸡翅",
    "52": "美食/醪糟",
    "53": "美食/金线油塔"
            }  

    if is_multilabel:
      predictions_list = [1 / (1 + np.exp(-p)) for p in data['logits'][0]]
    else:
      predictions_list = softmax(data['logits'][0])
    predictions_list = ['%.3f' % p for p in predictions_list]

    scores = dict(zip(labels_list, predictions_list))
    scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(labels_list) > 5:
      scores = scores[:5]
    label_index = predictions_list.index(max(predictions_list))
    predicted_label = str(labels_list[label_index])
    outputs['result'] =  self.label_id_name_dict[predicted_label]
    return outputs