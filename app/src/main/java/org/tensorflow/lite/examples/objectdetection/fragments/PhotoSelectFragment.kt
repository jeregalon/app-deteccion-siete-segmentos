package org.tensorflow.lite.examples.objectdetection.fragments

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import org.tensorflow.lite.examples.objectdetection.R
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.graphics.RectF
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import android.view.OrientationEventListener
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.databinding.DataBindingUtil.setContentView
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.examples.objectdetection.YOLOUtils.processDetections
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentPhotoSelectBinding
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService
import kotlin.concurrent.thread

class PhotoSelectFragment : Fragment(), ObjectDetectorHelper.DetectorListener {

    private val TAG = "PhotoSelectFragment"
    private lateinit var fragmentPhotoSelectBinding: FragmentPhotoSelectBinding

    private var selectedImageUri: Uri? = null
    private var selectedBitmap: Bitmap? = null

    private var objectDetectorHelper: ObjectDetectorHelper? = null

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
                    fragmentPhotoSelectBinding.imageView.setImageBitmap(bmp)
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
                fragmentPhotoSelectBinding.imageView.setImageBitmap(selectedBitmap)
                // opcional guardar si quieres un Uri
                // val uri = saveBitmapToMediaStore(selectedBitmap)
                // selectedImageUri = uri
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        fragmentPhotoSelectBinding = FragmentPhotoSelectBinding.inflate(inflater, container, false)

        // Inicializar detector
        objectDetectorHelper = ObjectDetectorHelper(
            threshold = 0.5f,
            numThreads = 2,
            maxResults = 3,
            currentDelegate = ObjectDetectorHelper.DELEGATE_CPU,
            currentModel = ObjectDetectorHelper.MODEL_YOLO_COMBINED,
            context = requireContext(),
            objectDetectorListener = this
        )

        // Upload button
        fragmentPhotoSelectBinding.btnUpload.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK).apply { type = "image/*" }
            pickImageLauncher.launch(intent)
        }

        // Camera button (small bitmap from extras)
        fragmentPhotoSelectBinding.btnCamera.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            takePhotoLauncher.launch(intent)
        }

        // Botón para predecir cajas delimitadoras
        fragmentPhotoSelectBinding.btnPredict.setOnClickListener {
            selectedImageUri?.let { uri ->
                val bitmap = getBitmapFromUri(uri)
                bitmap?.let {
                    objectDetectorHelper?.detect(it, 0)
                }
            } ?: run {
                println("⚠️ Selecciona una imagen antes de predecir")
            }
        }
        return fragmentPhotoSelectBinding.root
    }

    // ---- DetectorListener callbacks ----
    override fun onError(error: String) {
        Log.e(TAG, "Detector error: $error")
        requireActivity().runOnUiThread {
            fragmentPhotoSelectBinding.tvResult.text = "Error: $error"
            setUiEnabled(true)
        }
    }

    override fun onResults(
        results: List<ObjectDetection>,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        Log.d(TAG, "onResults recibidos: total=${results.size}")

        val (readingStr, unitLabel) = processDetections(results)

        val resultText = if (unitLabel.isNotEmpty()) {
            "$readingStr $unitLabel"
        } else {
            readingStr
        }

        requireActivity().runOnUiThread {
            fragmentPhotoSelectBinding.tvResult.text = resultText

            val overlayList = ArrayList<ObjectDetection>()

            overlayList.addAll(
                results.filter {
                    it.category.label.trim() in listOf("0","1","2","3","4","5","6","7","8","9",".")
                }
            )

            results.maxByOrNull {
                if (it.category.label.trim() in listOf("Lb","Kg","OZ","jin"))
                    it.category.confidence
                else -1f
            }?.let { overlayList.add(it) }

            val rect = getImageDisplayRect(fragmentPhotoSelectBinding.imageView)
            if (rect != null) {
                fragmentPhotoSelectBinding.overlay.setResults(overlayList, imageHeight, imageWidth, rect)
                fragmentPhotoSelectBinding.overlay.invalidate()
            } else {
                fragmentPhotoSelectBinding.overlay.clear()
            }

            setUiEnabled(true)
        }
    }

    // ---- utilidades ----

    private fun setUiEnabled(enabled: Boolean) {
        fragmentPhotoSelectBinding.btnUpload.isEnabled = enabled
        fragmentPhotoSelectBinding.btnCamera.isEnabled = enabled
        fragmentPhotoSelectBinding.btnPredict.isEnabled = enabled
    }

    // Convierte un Uri a Bitmap (igual que tenías)
    private fun getBitmapFromUri(uri: Uri): Bitmap? {
        return try {
            if (Build.VERSION.SDK_INT >= 28) {
                val source = ImageDecoder.createSource(requireActivity().contentResolver, uri)
                ImageDecoder.decodeBitmap(source).copy(Bitmap.Config.ARGB_8888, true)
            } else {
                MediaStore.Images.Media.getBitmap(requireActivity().contentResolver, uri)
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



    companion object {
        /**
         * Use this factory method to create a new instance of
         * this fragment using the provided parameters.
         *
         * @param param1 Parameter 1.
         * @param param2 Parameter 2.
         * @return A new instance of fragment PhotoSelectFragment.
         */
        // TODO: Rename and change types and number of parameters
        @JvmStatic
        fun newInstance(param1: String, param2: String) =
            PhotoSelectFragment().apply {
                arguments = Bundle().apply {
//                    putString(ARG_PARAM1, param1)
//                    putString(ARG_PARAM2, param2)
                }
            }
    }
}