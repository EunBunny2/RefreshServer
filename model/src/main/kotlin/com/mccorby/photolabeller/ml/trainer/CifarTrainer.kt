package com.mccorby.photolabeller.ml.trainer

import org.datavec.image.loader.CifarLoader
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File

class CifarTrainer(private val config: SharedConfig) {
    // 모델 생성
//    fun createModel(seed: Int, iterations: Int, numLabels: Int): MultiLayerNetwork {
//        val modelConf = NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .updater(Updater.ADAM)
//                .iterations(iterations)
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .l1(1e-4)
//                .regularization(true)
//                .l2(5 * 1e-4)
//                .list()
//                .layer(0, ConvolutionLayer.Builder(intArrayOf(4, 4), intArrayOf(1, 1), intArrayOf(0, 0))
//                        .name("cnn1")
//                        .convolutionMode(ConvolutionMode.Same)
//                        .nIn(3)
//                        .nOut(32)
//                        .weightInit(WeightInit.XAVIER_UNIFORM)
//                        .activation(Activation.RELU)
//                        .learningRate(1e-2)
//                        .biasInit(1e-2)
//                        .biasLearningRate(1e-2 * 2)
//                        .build())
//                .layer(1, SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, intArrayOf(3, 3))
//                        .name("pool1")
//                        .build())
//                .layer(2, LocalResponseNormalization.Builder(3.0, 5e-05, 0.75)
//                        .build())
//                .layer(3, DenseLayer.Builder()
//                        .name("ffn1")
//                        .nOut(64)
//                        .dropOut(0.5)
//                        .build())
//                .layer(4, OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .nOut(numLabels)
//                        .weightInit(WeightInit.XAVIER)
//                        .activation(Activation.SOFTMAX)
//                        .build())
//                .backprop(true)
//                .pretrain(false)
//                .setInputType(InputType.convolutional(config.imageSize, config.imageSize, config.channels))
//                .build()
//
//        return MultiLayerNetwork(modelConf).also { it.init() }
//    }

    fun createModel(seed: Int): MultiLayerNetwork {
        val HIDDEN_LAYER_WIDTH = 100;
        var modelConf = NeuralNetConfiguration.Builder()
                .seed(seed)
                .biasInit(1e-2)
                .miniBatch(false)
                .updater(Updater.ADAM)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, GravesLSTM.Builder()
                        .nIn(24) // input size
                        .nOut(HIDDEN_LAYER_WIDTH) // output size
                        .activation(Activation.TANH)
                        .build())
                .layer(1, GravesLSTM.Builder()
                        .nIn(HIDDEN_LAYER_WIDTH)
                        .nOut(HIDDEN_LAYER_WIDTH)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(HIDDEN_LAYER_WIDTH)
                        .nOut(24)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build()
        return MultiLayerNetwork(modelConf).also { it.init() }
    }

    // 학습
//    fun train(model: MultiLayerNetwork, numSamples: Int, epochs: Int, scoreListener: IterationListener): MultiLayerNetwork {
    fun train(model: MultiLayerNetwork, numSamples: Int, epochs: Int): MultiLayerNetwork {
        //        model.setListeners(scoreListener)
//        val cifar = CifarDataSetIterator(config.batchSize, numSamples,
//                intArrayOf(config.imageSize, config.imageSize, config.channels),
//                CifarLoader.NUM_LABELS,
//                null,
//                false,
//                true)
//
//        for (i in 0 until epochs) {
//            println("Epoch=====================$i")
//            model.fit(cifar)
//        }
        val path = "model/src/main/data/input.csv"
        // 모델 만들고 zip 파일로 저장해서 안드로이드 asset 파일에 저장하기
        try{
            val file = File(path)
            var lines = file.readLines()
            var line_len = lines.size
            lines = lines.subList(1, line_len - line_len % 24)
            line_len = lines.size
            var steps = Array<IntArray>((line_len/24)+1, {IntArray(24)})
            var max_step_indexs = Array<Int>((line_len/24),{0})
            var i = 0
            var j = 0

            for (line in lines){
                if(j == 24){
                    i++
                    j = 0
                }
                steps[i][j++] = (line.split(",")[2]).toInt()
            }
            // train_x (제일 마지막 데이터 삭제 -> 왜냐면 다음날 데이터를 정답 데이터로 사용할거니까)
//            var train_x = steps.sliceArray(0..steps.size-2)
            var train_x = Array<FloatArray>((line_len/24), {FloatArray(24)})

            // normalization
            var x_max = steps[0].max()!!.toFloat()
            var x_min = steps[0].min()!!.toFloat()

            for(i in 1..train_x.size-1){
                if(train_x[i].max()!! > x_max){
                    x_max = steps[i].max()!!.toFloat()
                }
                if(train_x[i].min()!! > x_min){
                    x_min = steps[i].min()!!.toFloat()
                }
            }


            for(i in 0..train_x.size-1){
                for(j in 0..train_x[i].size-1){
                    train_x[i][j] = ((steps[i][j]-x_min)/(x_max-x_min))
                }
            }

            // train_y (one-hot 안된)
            for(i in 0..max_step_indexs.size-1) {
                max_step_indexs[i] = steps[i+1].indexOf(steps[i+1].max()!!) // !! : null이 들어올 수 없다
            }

            var one_hot = Array<FloatArray>(24, {FloatArray(24)})
            for(i in 0..23){
                one_hot[i][i] = 1.0f
            }
            var train_y = Array<FloatArray>(max_step_indexs.size, {FloatArray(24)})

            // train_y (one-hot 된)
            for(i in 0..train_y.size-1){
                train_y[i] = one_hot[max_step_indexs[i]]
            }

            var nd4j_train_x = Nd4j.create(train_x[0])
            var nd4j_train_y = Nd4j.create(train_y[0])
//            model.setListeners(scoreListener)
//            model.setListeners(StatsListener(statsStorage))
            for (i in 0 until epochs) {
                println("Epoch=====================$i")
                // TODO
                // model fit 해야됨
                model.fit(nd4j_train_x, nd4j_train_y)
            }
        }catch(e:Exception){
            println(e.message)
        }
        return model
    }

    fun eval(model: MultiLayerNetwork, numSamples: Int): Evaluation {
        val cifarEval = CifarDataSetIterator(config.batchSize, numSamples,
                intArrayOf(config.imageSize, config.imageSize, config.channels),
                CifarLoader.NUM_LABELS,
                null,
                false,
                false)

        println("=====eval model========")
        val eval = Evaluation(cifarEval.labels)
        while (cifarEval.hasNext()) {
            val testDS = cifarEval.next(config.batchSize)
            val output = model.output(testDS.featureMatrix)
            eval.eval(testDS.labels, output)
        }
        return eval
    }

    fun saveModel(model: MultiLayerNetwork, location: String) {
        ModelSerializer.writeModel(model, File(location), true)
    }
}