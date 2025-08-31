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

package org.tensorflow.lite.examples.objectdetection

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
import android.view.Surface
import android.view.WindowManager
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.examples.objectdetection.databinding.ActivityMainBinding
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import android.view.OrientationEventListener
import android.widget.ImageView

/**
 * Main entry point into our app. This app follows the single-activity pattern, and all
 * functionality is implemented in the form of fragments.
 */
class MainActivity : AppCompatActivity(), ObjectDetectorHelper.DetectorListener {

    private lateinit var activityMainBinding: ActivityMainBinding
    private var selectedImageUri: Uri? = null
    private var detectorHelper: ObjectDetectorHelper? = null

//    private var currentDeviceRotation: Int = 0
//    private var rotationWhenPhotoWasTaken: Int = 0

//    private val orientationListener by lazy {
//        object : OrientationEventListener(this) {
//            override fun onOrientationChanged(orientation: Int) {
//                if (orientation == ORIENTATION_UNKNOWN) return
//
//                // Normalizamos a múltiplos de 90° (0, 90, 180, 270)
//                val rounded = when {
//                    orientation in 45..134 -> 90
//                    orientation in 135..224 -> 180
//                    orientation in 225..314 -> 270
//                    else -> 0
//                }
//
//                currentDeviceRotation = rounded
//            }
//        }
//    }

    // Lanzador para elegir imagen de la galería
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data: Intent? = result.data
            selectedImageUri = data?.data
            activityMainBinding.imageView.setImageURI(selectedImageUri) // mostrar la imagen en el ImageView
        }
    }

    // Lanzador para tomar foto con la cámara
    private val takePhotoLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data: Intent? = result.data
            val bitmap = data?.extras?.get("data") as? Bitmap
            bitmap?.let {
                activityMainBinding.imageView.setImageBitmap(it)
                // Guardar temporalmente en MediaStore para tener un Uri
                val uri = saveBitmapToMediaStore(it)
                selectedImageUri = uri
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        activityMainBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(activityMainBinding.root)

        // Inicializar el detector YOLO
        detectorHelper = ObjectDetectorHelper(
            threshold = 0.5f,
            numThreads = 2,
            maxResults = 3,
            currentDelegate = ObjectDetectorHelper.DELEGATE_CPU,
            currentModel = ObjectDetectorHelper.MODEL_YOLO,
            context = this,
            objectDetectorListener = this
        )

        // Botón para subir imagen
        activityMainBinding.btnUpload.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK).apply {
                type = "image/*"
            }
            pickImageLauncher.launch(intent)
        }

        // Botón para abrir la cámara
        activityMainBinding.btnCamera.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            takePhotoLauncher.launch(intent)
        }

        // Botón para predecir cajas delimitadoras
        activityMainBinding.btnPredict.setOnClickListener {
            selectedImageUri?.let { uri ->
                val bitmap = getBitmapFromUri(uri)
                bitmap?.let {
                    detectorHelper?.detect(it, 0)
                }
            } ?: run {
                println("⚠️ Selecciona una imagen antes de predecir")
            }
        }
    }

    // Convierte un Uri a Bitmap
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

    private fun getImageDisplayRect(imageView: ImageView): RectF? {
        val d = imageView.drawable ?: return null
        val matrix = imageView.imageMatrix

        val drawableRect = RectF(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        val viewRect = RectF()
        matrix.mapRect(viewRect, drawableRect)
        // viewRect está en coordenadas del ImageView; si el ImageView está en (0,0) del FrameLayout,
        // coincide con coordenadas del Overlay. Si hubiese padding/desplazamiento, habría que sumarlo.
        return viewRect
    }

    // Guarda un Bitmap en MediaStore y devuelve un Uri
    private fun saveBitmapToMediaStore(bitmap: Bitmap): Uri? {
        val path = MediaStore.Images.Media.insertImage(contentResolver, bitmap, "captured_image", null)
        return Uri.parse(path)
    }

    // Implementación de DetectorListener
    override fun onError(error: String) {
        println("🚨 Detector error: $error")
    }

    override fun onResults(
        results: List<ObjectDetection>,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        runOnUiThread {
            // Espera al próximo loop de UI para asegurar que ImageView ya tiene drawable y tamaño
            activityMainBinding.imageView.post {
                val rect = getImageDisplayRect(activityMainBinding.imageView)
                if (rect != null) {
                    activityMainBinding.overlay.setResults(
                        results = results,
                        imgHeight = imageHeight,
                        imgWidth = imageWidth,
                        displayRect = rect
                    )
                } else {
                    // Si no hay drawable aún, limpia overlay
                    activityMainBinding.overlay.clear()
                    Log.w("MainActivity", "No hay drawable en el ImageView; no se puede dibujar overlay.")
                }
            }
        }
        println("✅ Detección completada en $inferenceTime ms")
        println("Altura: $imageHeight")
        println("Ancho: $imageWidth")
        for (det in results) {
            println("Objeto detectado: ${det.category.label} (${det.category.confidence})")
            println("BoundingBox: ${det.boundingBox}")
        }
    }

//    override fun onResume() {
//        super.onResume()
//        orientationListener.enable()
//    }
//
//    override fun onPause() {
//        super.onPause()
//        orientationListener.disable()
//    }


    override fun onBackPressed() {
        if (Build.VERSION.SDK_INT == Build.VERSION_CODES.Q) {
            // Workaround for Android Q memory leak issue in IRequestFinishCallback$Stub.
            // (https://issuetracker.google.com/issues/139738913)
            finishAfterTransition()
        } else {
            super.onBackPressed()
        }
    }
}
