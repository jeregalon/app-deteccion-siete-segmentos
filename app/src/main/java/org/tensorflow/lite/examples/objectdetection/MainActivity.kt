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

/**
 * Main entry point into our app. This app follows the single-activity pattern, and all
 * functionality is implemented in the form of fragments.
 */
class MainActivity : AppCompatActivity(), ObjectDetectorHelper.DetectorListener {

    private lateinit var activityMainBinding: ActivityMainBinding
    private var selectedImageUri: Uri? = null
    private var detectorHelper: ObjectDetectorHelper? = null

    // Registrar el lanzador para elegir imagen de la galería
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
                // Calcular rotación del teléfono
                val rotation = getDeviceRotationInRadians()
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
                    // Por ahora rotación = 0, puedes adaptarlo si quieres
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

    // Guarda un Bitmap en MediaStore y devuelve un Uri
    private fun saveBitmapToMediaStore(bitmap: Bitmap): Uri? {
        val path = MediaStore.Images.Media.insertImage(contentResolver, bitmap, "captured_image", null)
        return Uri.parse(path)
    }

    // Obtiene la rotación del dispositivo en radianes
    private fun getDeviceRotationInRadians(): Float {
        val rotation = (getSystemService(WINDOW_SERVICE) as WindowManager).defaultDisplay.rotation
        return when (rotation) {
            Surface.ROTATION_0 -> 0f
            Surface.ROTATION_90 -> Math.PI.toFloat() / 2f
            Surface.ROTATION_180 -> Math.PI.toFloat()
            Surface.ROTATION_270 -> 3f * Math.PI.toFloat() / 2f
            else -> 0f
        }
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
        println("✅ Detección completada en $inferenceTime ms")
        for (det in results) {
            println("Objeto detectado: ${det.category.label} (${det.category.confidence})")
            println("BoundingBox: ${det.boundingBox}")
        }
    }

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
