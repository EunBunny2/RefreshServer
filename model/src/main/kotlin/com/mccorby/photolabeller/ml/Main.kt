package com.mccorby.photolabeller.ml

import com.mccorby.photolabeller.ml.trainer.CifarTrainer
import com.mccorby.photolabeller.ml.trainer.SharedConfig
//import org.bytedeco.javacpp.opencv_core
import org.datavec.image.loader.CifarLoader // dl4j에서 제공하는 데이터 로더 사용하는듯
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.deeplearning4j.util.ModelSerializer
import java.io.File
import java.util.*

// 초기 모델 학습

// main은 세 개의 매개변수를 받는다.
// args[0] : 프로세스를 트리거하는 문자열 train
// args[1] : 결과 모델을 저장할 유효한 디렉토리
// args[2] : 모델의 위치 또는 분류할 이미지의 위치. web으로 DL4J에서 제공하는 UI monitor를 시작할 수 있다. 그리고 예측을 실행하여 모델을 테스트할 수 있다.

fun main(args: Array<String>) {
    val config = SharedConfig(32, 3, 100)
    val trainer = CifarTrainer(config)
    var model =trainer.createModel(123)
    val numEpochs = 10
    val numSamples = 10000
    model = trainer.train(model, numSamples, numEpochs)

//    if (args.isNotEmpty() && args[0] == "train") {
//        println("args[0] : "+ args[0])
//        val seed = 123
//        val iterations = 1
//        val numLabels = CifarLoader.NUM_LABELS // 라이브러리에 포함된 메서드. 시팔 데이터의 label 가져옴
//        val saveFile = "cifar_federated-${Date().time}.zip" // 모델 파일 저장
//
//        val numEpochs = 50
//        val numSamples = 10000
//
//        val config = SharedConfig(32, 3, 100)
//        val trainer = CifarTrainer(config)
//        var model = trainer.createModel(seed, iterations, numLabels)
//        model = trainer.train(model, numSamples, numEpochs, getVisualization(args.getOrNull(2)))
//
//        if (args[1].isNotEmpty()) {
//            println("args[0] : "+ args[1])
//            println("Saving model to ${args[1]}")
//            trainer.saveModel(model, args[1] + "/$saveFile")
//        }
//
//        val eval = trainer.eval(model, numSamples)
//        println(eval.stats())
//
//    } else {
//        predict(args[0], args[1])
//    }
}

//fun predict(modelFile: String, imageFile: String) {
//    val config = SharedConfig(32, 3, 100)
//    val trainer = CifarTrainer(config)
//
//    val model = ModelSerializer.restoreMultiLayerNetwork(modelFile)
//
//    val eval = trainer.eval(model, 100)
//    println(eval.stats())
//
//    val file = File(imageFile)
//    val resizedImage = opencv_core.Mat()
//    val sz = opencv_core.Size(32, 32)
//    val opencvImage = org.bytedeco.javacpp.opencv_imgcodecs.imread(file.absolutePath)
//    org.bytedeco.javacpp.opencv_imgproc.resize(opencvImage, resizedImage, sz)
//
//    val nativeImageLoader = NativeImageLoader()
//    val image = nativeImageLoader.asMatrix(resizedImage)
//    val reshapedImage = image.reshape(1, 3, 32, 32)
//    val result = model.predict(reshapedImage)
//    println(result.joinToString(", ", prefix = "[", postfix = "]"))
//}

private fun getVisualization(visualization: String?): IterationListener {
    return when (visualization) {
        "web" -> {  // args[2]가 web
            //Initialize the user interface backend(사용자 인터페이스 백엔드 초기화)
            val uiServer = UIServer.getInstance()

            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            // 네트워크 정보(gradients, score vs. time etc)를 저장할 위치를 구성.
            val statsStorage = InMemoryStatsStorage()         //Alternative: new FileStatsStorage(File), for saving and loading later
                                                            // 다른 방법 : new FileStatsStorage(File), 저장과 이후에 로드를 위해

            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            // UI에 StatsStorage 인스턴스 연결 : StatsStorage의 내용을 시각화할 수 있다.
            uiServer.attach(statsStorage)

            //Then add the StatsListener to collect this information from the network, as it trains
            //StatsListener를 추가하여 네트워크에서 이 정보를 수집한다.
            StatsListener(statsStorage)
        }
        // args[2]가 null
        else -> ScoreIterationListener(50)
    }
}