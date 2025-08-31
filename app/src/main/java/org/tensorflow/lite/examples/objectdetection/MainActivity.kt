package org.tensorflow.lite.examples.objectdetection

/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.OrientationEventListener
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.objectdetection.databinding.ActivityMainBinding
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity(), ObjectDetectorHelper.DetectorListener {

    private val TAG = "MainActivity"
    private lateinit var activityMainBinding: ActivityMainBinding

    private var selectedImageUri: Uri? = null
    private var selectedBitmap: Bitmap? = null

    private var UnitDetectorHelper: ObjectDetectorHelper? = null
    private var MeasurementDetectorHelper: ObjectDetectorHelper? = null

    // Ejecutor para realizar inferencias en background
    private val detectionExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // Estado para saber qué resultado esperamos
    private enum class WaitState { NONE, WAIT_UNIT, WAIT_MEASUREMENT }
    @Volatile private var waitState = WaitState.NONE

    // Guardar resultados temporales
    private var unitResults: List<ObjectDetection> = emptyList()
    private var measurementResults: List<ObjectDetection> = emptyList()

    // Bitmap que se usará para las inferencias
    private var bitmapForDetection: Bitmap? = null

    // Lanzadores
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data: Intent? = result.data
            selectedImageUri = data?.data
            selectedImageUri?.let { uri ->
                // Cargar bitmap y mostrar
                val bmp = getBitmapFromUri(uri)
                if (bmp != null) {
                    selectedBitmap = bmp
                    activityMainBinding.imageView.setImageBitmap(bmp)
                } else {
                    Log.w(TAG, "No se pudo cargar bitmap de la URI")
                }
            }
        }
    }

    private val takePhotoLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data: Intent? = result.data
            val bitmap = data?.extras?.get("data") as? Bitmap
            bitmap?.let {
                selectedBitmap = it.copy(Bitmap.Config.ARGB_8888, true)
                activityMainBinding.imageView.setImageBitmap(selectedBitmap)
                // opcional guardar si quieres un Uri
                // val uri = saveBitmapToMediaStore(selectedBitmap)
                // selectedImageUri = uri
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        // Inicializar los dos detectores (modelos distintos)
        UnitDetectorHelper = ObjectDetectorHelper(
            threshold = 0.5f,
            numThreads = 2,
            maxResults = 3,
            currentDelegate = ObjectDetectorHelper.DELEGATE_CPU,
            currentModel = ObjectDetectorHelper.MODEL_YOLO_OBB,
            context = this,
            objectDetectorListener = this
        )

        MeasurementDetectorHelper = ObjectDetectorHelper(
            threshold = 0.5f,
            numThreads = 2,
            maxResults = 20,
            currentDelegate = ObjectDetectorHelper.DELEGATE_CPU,
            currentModel = ObjectDetectorHelper.MODEL_SEPARATED,
            context = this,
            objectDetectorListener = this
        )

        // Upload button
        activityMainBinding.btnUpload.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK).apply { type = "image/*" }
            pickImageLauncher.launch(intent)
        }

        // Camera button (small bitmap from extras)
        activityMainBinding.btnCamera.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            takePhotoLauncher.launch(intent)
        }

        // Predict button: ejecutar ambos detectores secuencialmente en background
        activityMainBinding.btnPredict.setOnClickListener {
            val bmp = selectedBitmap ?: run {
                println("⚠️ Selecciona una imagen antes de predecir")
                return@setOnClickListener
            }

            // Guardar bitmap para usar en detectores
            bitmapForDetection = bmp

            // Disable buttons para evitar re-entradas
            setUiEnabled(false)
            activityMainBinding.tvResult.text = "Detectando..."

            // Lanzar trabajo en background: primero UNIT, luego MEASUREMENT
            detectionExecutor.execute {
                try {
                    // Preparar estado
                    waitState = WaitState.WAIT_UNIT
                    unitResults = emptyList()
                    measurementResults = emptyList()

                    // Ejecutar primer detector (esto invocará back onResults cuando termine)
                    UnitDetectorHelper?.detect(bmp, 0)
                    // Nota: cuando termine, onResults (en background) lanzará el siguiente detector
                    // y al final procesará y actualizará la UI.
                } catch (e: Exception) {
                    Log.e(TAG, "Error en predicción", e)
                    runOnUiThread {
                        activityMainBinding.tvResult.text = "Error en detección"
                        setUiEnabled(true)
                    }
                }
            }
        }
    }

    // ---- DetectorListener callbacks ----
    override fun onError(error: String) {
        Log.e(TAG, "Detector error: $error")
        runOnUiThread {
            activityMainBinding.tvResult.text = "Error: $error"
            setUiEnabled(true)
        }
    }

    override fun onResults(
        results: List<ObjectDetection>,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        // Este método puede ser invocado desde el hilo del detector (nuestro executor).
        Log.d(TAG, "onResults recibidos. waitState=$waitState  resultsSize=${results.size}")

        when (waitState) {
            WaitState.WAIT_UNIT -> {
                // Guardamos los resultados de UnitDetector
                unitResults = results
                Log.d(TAG, "Unit results: ${unitResults.size}")

                // Lanzar ahora el detector de caracteres (measurement) en el mismo executor
                val bmp = bitmapForDetection
                if (bmp == null) {
                    Log.w(TAG, "No hay bitmap para la siguiente inferencia")
                    runOnUiThread {
                        activityMainBinding.tvResult.text = "Error: bitmap faltante"
                        setUiEnabled(true)
                    }
                    waitState = WaitState.NONE
                    return
                }

                // Cambiamos estado antes de invocar el siguiente detect
                waitState = WaitState.WAIT_MEASUREMENT
                MeasurementDetectorHelper?.detect(bmp, 0)
                // Cuando el measurement termine, caeremos en la rama WAIT_MEASUREMENT
            }

            WaitState.WAIT_MEASUREMENT -> {
                // Guardamos measurement results y procesamos todo
                measurementResults = results
                Log.d(TAG, "Measurement results: ${measurementResults.size}")

                // Volver al estado NONE
                waitState = WaitState.NONE

                // Procesar measurementResults:
                val sortedChars = measurementResults.sortedBy { it.boundingBox.left }

                // Construir cadena de lectura
                val sb = StringBuilder()
                for (det in sortedChars) {
                    val lbl = det.category.label.trim()
                    // map labels to chars robustamente
                    val ch = when {
                        lbl == "." || lbl == "dot" || lbl.equals("point", true) -> "."
                        lbl.length == 1 && lbl[0].isDigit() -> lbl
                        // si la etiqueta es "10" o "7." etc, intentamos extraer dígitos/dot
                        else -> {
                            val filtered = lbl.filter { c -> c.isDigit() || c == '.' }
                            if (filtered.isNotEmpty()) filtered else ""
                        }
                    }
                    sb.append(ch)
                }
                val readingStr = sb.toString().ifEmpty { "—" }

                // Filtrar solo resultados que no sean Measurement ni Lock
                val filteredUnitResults = unitResults.filter {
                    it.category.label != "Measurement" && it.category.label != "Lock"
                }

                // Elegir unidad con mayor confianza (entre las filtradas)
                val bestUnit = filteredUnitResults.maxByOrNull { it.category.confidence }
                val unitLabel = bestUnit?.category?.label ?: ""

                // Preparar texto final
                val resultText = if (unitLabel.isNotEmpty()) {
                    "$readingStr $unitLabel"
                } else {
                    readingStr
                }

                // Mostrar y dibujar overlay (UI thread)
                runOnUiThread {
                    activityMainBinding.tvResult.text = resultText

                    // Para dibujar cajas juntas: unimos ambas listas
                    val union = ArrayList<ObjectDetection>()
                    union.addAll(unitResults)
                    union.addAll(measurementResults)

                    // Calcular displayRect para el overlay
                    val rect = getImageDisplayRect(activityMainBinding.imageView)
                    if (rect != null) {
                        activityMainBinding.overlay.setResults(union, imageHeight, imageWidth, rect)
                        activityMainBinding.overlay.invalidate()
                    } else {
                        // fallback: si no hay displayRect, limpias o pasas sólo measurement
                        activityMainBinding.overlay.clear()
                    }

                    // Rehabilitar UI
                    setUiEnabled(true)
                }
            }

            else -> {
                // Si no estamos esperando nada, ignoramos (o logueamos)
                Log.w(TAG, "onResults recibido pero waitState = NONE")
            }
        }
    }

    // ---- utilidades ----

    private fun setUiEnabled(enabled: Boolean) {
        activityMainBinding.btnUpload.isEnabled = enabled
        activityMainBinding.btnCamera.isEnabled = enabled
        activityMainBinding.btnPredict.isEnabled = enabled
    }

    // Convierte un Uri a Bitmap (igual que tenías)
    private fun getBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= 28) {
                val source = ImageDecoder.createSource(this.contentResolver, uri)
                ImageDecoder.decodeBitmap(source).copy(Bitmap.Config.ARGB_8888, true)
            } else {
                MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                    .copy(Bitmap.Config.ARGB_8888, true)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }

    // Extra: calcula display rect del ImageView (como ya tenías)
    private fun getImageDisplayRect(imageView: ImageView): RectF? {
        val d = imageView.drawable ?: return null
        val matrix = imageView.imageMatrix
        val drawableRect = RectF(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        val viewRect = RectF()
        matrix.mapRect(viewRect, drawableRect)
        return viewRect
    }

    override fun onDestroy() {
        super.onDestroy()
        detectionExecutor.shutdownNow()
    }
}
