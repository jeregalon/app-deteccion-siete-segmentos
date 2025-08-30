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

    private var yolo: TfliteDetector
    private var ip: ImageProcessing

    init {

        yolo = TfliteDetector(context)
        yolo.setIouThreshold(iouThreshold)
        yolo.setConfidenceThreshold(confidenceThreshold)

        val modelPath = "digital_characters_obb_model_float32.tflite"
        val metadataPath = "digital_characters_obb_metadata.yaml"

        val config = LocalYoloModel(
            "obb",
            "tflite",
            modelPath,
            metadataPath,
        )

        val useGPU = currentDelegate == 0
        yolo.loadModel(
            config,
            useGPU
        )

        ip = ImageProcessing()

    }

    override fun detect(image: TensorImage, imageRotation: Int): DetectionResult  {

        val bitmap = image.bitmap

        val ppImage = yolo.preprocess(bitmap)
        val results = yolo.predict(ppImage)

//        val TAG = "Yolo Detector"

//        Log.d(TAG, "Resultados brutos: $results")
//
//        for ((i, res) in results.withIndex()) {
//            Log.d("YoloDetector", "===== Objeto #$i =====")
//            // Listar todas las propiedades por reflexión
//            for (field in res.javaClass.declaredFields) {
//                field.isAccessible = true
//                val value = field.get(res)
//                Log.d("YoloDetector", "${field.name} = $value")
//            }
//        }

        val detections = ArrayList<ObjectDetection>()

        // ASPECT_RATIO = 4:3
        // => imgW = imgH * 3/4
        var imgH: Int
        var imgW: Int
        if (imageRotation == 90 || imageRotation == 270) {
            imgH = ppImage.height
            imgW = imgH * 3 / 4
        }
        else {
            imgW = ppImage.width
            imgH = imgW * 3 / 4

        }


        for (result: DetectedObject in results) {
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
            detections.add(detection)
        }

        val ret = DetectionResult(ppImage, detections)
        ret.info = yolo.stats
        return ret

    }


}