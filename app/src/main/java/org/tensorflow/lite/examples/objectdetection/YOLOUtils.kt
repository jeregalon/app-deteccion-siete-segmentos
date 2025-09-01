package org.tensorflow.lite.examples.objectdetection

import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection

object YOLOUtils {
    fun processDetections(results: List<ObjectDetection>): Pair<String, String> {
        // Filtrar medición
        val measurementResults = results.filter { det ->
            val lbl = det.category.label.trim()
            lbl in listOf("0","1","2","3","4","5","6","7","8","9",".")
        }

        // Filtrar unidades
        val unitResults = results.filter { det ->
            val lbl = det.category.label.trim()
            lbl in listOf("Lb","Kg","OZ","jin")
        }

        // Construir medición en orden de izquierda a derecha
        val sortedChars = measurementResults.sortedBy { it.boundingBox.left }
        val sb = StringBuilder()
        for (det in sortedChars) {
            sb.append(det.category.label.trim())
        }
        val readingStr = sb.toString().ifEmpty { "—" }

        // Unidad más confiable
        val bestUnit = unitResults.maxByOrNull { it.category.confidence }
        val unitLabel = bestUnit?.category?.label ?: ""

        return Pair(readingStr, unitLabel)
    }

}