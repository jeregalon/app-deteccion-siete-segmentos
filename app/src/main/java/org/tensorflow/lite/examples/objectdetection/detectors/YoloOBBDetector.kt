package org.tensorflow.lite.examples.objectdetection.detectors

import android.content.Context
import android.graphics.RectF
import android.util.Log
import com.ultralytics.yolo.ImageProcessing
import com.ultralytics.yolo.models.LocalYoloModel
import com.ultralytics.yolo.predict.detect.DetectedObject
import com.ultralytics.yolo.predict.detect.TfliteDetector
import org.tensorflow.lite.support.image.TensorImage


class YoloOBBDetector(
    var confidenceThreshold: Float = 0.5f,
    var iouThreshold: Float = 0.3f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context
): ObjectDetector {

    private var yolo1: TfliteDetector
    private var yolo2: TfliteDetector
    private var ip: ImageProcessing

    init {

    // Modelo de detección de la unidad de medida
        yolo1 = TfliteDetector(context)
        yolo1.setIouThreshold(iouThreshold)
        yolo1.setConfidenceThreshold(confidenceThreshold)

        val modelPath1 = "digital_characters_obb_model_float32.tflite"
        val metadataPath1 = "digital_characters_obb_metadata.yaml"

        val config1 = LocalYoloModel(
            "obb",
            "tflite",
            modelPath1,
            metadataPath1,
        )

    // Modelo de detección de la medición
        yolo2 = TfliteDetector(context)
        yolo2.setIouThreshold(iouThreshold)
        yolo2.setConfidenceThreshold(confidenceThreshold)

        val modelPath2 = "separated_characters_model_float32.tflite"
        val metadataPath2 = "separated_characters_metadata.yaml"

        val config2 = LocalYoloModel(
            "obb",
            "detect",
            modelPath2,
            metadataPath2,
        )

        val useGPU = currentDelegate == 0
        yolo1.loadModel(
            config1,
            useGPU
        )
        yolo2.loadModel(
            config1,
            useGPU
        )
        ip = ImageProcessing()

    }

    override fun detect(image: TensorImage, imageRotation: Int): DetectionResult  {

        val bitmap = image.bitmap

        // Preprocesado
        val ppImage = yolo1.preprocess(bitmap)

        // ASPECT_RATIO = 4:3
        // => imgW = imgH * 3/4
//        var imgH: Int
//        var imgW: Int
//        if (imageRotation == 90 || imageRotation == 270) {
//            imgH = ppImage.height
//            imgW = imgH * 3 / 4
//        }
//        else {
//            imgW = ppImage.width
//            imgH = imgW * 3 / 4
//
//        }

        var imgH = bitmap.height
        var imgW = bitmap.width

        Log.d("ANCHO Y ALTURA:", "ANCHO: $imgW, ALTURA: $imgH")

    // Detección con el primer modelo
        val results1 = yolo1.predict(ppImage)
        val detections1 = ArrayList<ObjectDetection>()

        for (result: DetectedObject in results1) {
            val category = Category(
                result.label,
                result.confidence,
            )
            val yoloBox = result.boundingBox

            val left = yoloBox.left * imgW
            val top = yoloBox.top * imgH
            val right = yoloBox.right * imgW
            val bottom = yoloBox.bottom * imgH

            val bbox = RectF(
                left,
                top,
                right,
                bottom
            )
            val detection = ObjectDetection(
                bbox,
                category
            )
            detections1.add(detection)
        }

    // Detección con el segundo modelo
        val results2 = yolo2.predict(ppImage)
        val detections2 = ArrayList<ObjectDetection>()

        for (result: DetectedObject in results2) {
            val category = Category(
                result.label,
                result.confidence,
            )
            val yoloBox = result.boundingBox

            val left = yoloBox.left * imgW
            val top = yoloBox.top * imgH
            val right = yoloBox.right * imgW
            val bottom = yoloBox.bottom * imgH

            val bbox = RectF(
                left,
                top,
                right,
                bottom
            )
            val detection = ObjectDetection(
                bbox,
                category
            )
            detections2.add(detection)
        }

        val ret = DetectionResult(bitmap, detections1 + detections2)
        ret.info = yolo2.stats
        return ret



    }


}