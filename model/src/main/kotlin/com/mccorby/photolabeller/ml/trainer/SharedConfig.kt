package com.mccorby.photolabeller.ml.trainer

// image size, channels, batchSize를 갖는 데이터 구조를 정의 ->
data class SharedConfig(val imageSize: Int, val channels: Int, val batchSize: Int, val featureLayerIndex: Int = 3)