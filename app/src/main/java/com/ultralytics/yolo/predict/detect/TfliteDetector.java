//
// Detector basado en YOLO exportado a TFLite
//
package com.ultralytics.yolo.predict.detect;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;

import com.ultralytics.yolo.ImageProcessing;
import com.ultralytics.yolo.models.LocalYoloModel;
import com.ultralytics.yolo.models.YoloModel;
import com.ultralytics.yolo.predict.PredictorException;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


/**
 * Detector que usa TensorFlow Lite para correr un modelo YOLO en Android.
 * - Carga el modelo y etiquetas desde assets
 * - Preprocesa bitmaps de entrada
 * - Ejecuta inferencia con TFLite (CPU o GPU)
 * - Postprocesa resultados (NMS, filtrado por confianza, etc.)
 */
public class TfliteDetector extends Detector {

    // Constantes
    private static final long FPS_INTERVAL_MS = 1000;   // Intervalo de actualización de FPS
    private static final int NUM_BYTES_PER_CHANNEL = 4; // 4 bytes = float32
    private static final float Nanos2Millis = 1 / 1e6f; // conversión nanosegundos → milisegundos

    // Handler para medir FPS en el hilo principal
    private final Handler handler = new Handler(Looper.getMainLooper());

    // Frame temporal (no se usa en este fragmento, pero reservado)
    private final Bitmap pendingBitmapFrame;

    // Configuración del modelo
    private int numClasses;
    private double confidenceThreshold = 0.25f; // confianza mínima
    private double iouThreshold = 0.45f;        // umbral IoU para NMS
    private int numItemsThreshold = 30;         // máximo de objetos por frame

    // Motor de inferencia TFLite
    private Interpreter interpreter;

    // Buffers de entrada y salida
    private Object[] inputArray;    // arreglo de objetos que contendrá los tensores de entrada del modelo
    private Map<Integer, Object> outputMap; // mapa que relaciona índices de salida del modelo con sus buffers asociados.
    private ByteBuffer imgData;   // buffer de entrada
    private ByteBuffer outData;   // buffer de salida
    private int[] intValues;      // pixeles ARGB temporales

    // Forma de salida del modelo
    private int outputShape2;
    private int outputShape3;
    private float[][] output;     // copia de salida en float[][]

    // Callbacks opcionales
    private ObjectDetectionResultCallback objectDetectionResultCallback;
    private FloatResultCallback inferenceTimeCallback;
    private FloatResultCallback fpsRateCallback;

    // Estadísticas de tiempos
    public class Stats {
        private float imageSetupTime;  // tiempo de preprocesado
        private float inferenceTime;   // tiempo de inferencia
        private float postProcessTime; // tiempo de postprocesado
    }
    public Stats stats;

    // Utilidad de preprocesado (conversión ARGB → RGB normalizado)
    private ImageProcessing ip;


    public TfliteDetector(Context context) {
        super(context);

        // Se crea un Bitmap cuadrado del tamaño de entrada del modelo
        pendingBitmapFrame = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);

        stats = new Stats();
        imgData = null;
        intValues = null;
        outData = null;

        ip = new ImageProcessing();
    }

    /**
     * Carga el modelo YOLO desde assets y prepara el intérprete TFLite
     */
    @Override
    public void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception {
        if (yoloModel instanceof LocalYoloModel) {
            final LocalYoloModel localYoloModel = (LocalYoloModel) yoloModel;

            // Verifica que las rutas de modelo y metadatos sean válidas
            if (localYoloModel.modelPath == null || localYoloModel.modelPath.isEmpty() ||
                    localYoloModel.metadataPath == null || localYoloModel.metadataPath.isEmpty()) {
                throw new Exception("Rutas de modelo o metadatos inválidas");
            }

            final AssetManager assetManager = context.getAssets();

            // Cargar etiquetas de clases
            loadLabels(assetManager, localYoloModel.metadataPath);
            numClasses = labels.size();

            try {
                // Mapea el archivo .tflite a memoria
                MappedByteBuffer modelFile = loadModelFile(assetManager, localYoloModel.modelPath);
                // Inicializa el intérprete (GPU o CPU)
                initDelegate(modelFile, useGpu);
            } catch (Exception e) {
                throw new PredictorException("Error cargando modelo");
            }
        }
    }

    /**
     * Redimensiona un Bitmap al tamaño de entrada del modelo
     */
    public Bitmap preprocess(Bitmap bitmap) {
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
    }

    /**
     * Ejecuta inferencia en una imagen
     */
    @Override
    public ArrayList<DetectedObject> predict(Bitmap bitmap) {
        try {
            long startTime = System.nanoTime();

            // Preprocesado → genera el tensor de entrada
            setInputOptim(bitmap);

            stats.imageSetupTime = (System.nanoTime() - startTime) * Nanos2Millis;

            // Corre la inferencia y devuelve detecciones
            return runInference();
        } catch (Exception e) {
            // Si algo falla, devuelve lista vacía
            return new ArrayList<>();
        }
    }

    // Métodos para ajustar parámetros de detección
    @Override
    public void setConfidenceThreshold(float confidence) { this.confidenceThreshold = confidence; }
    @Override
    public void setIouThreshold(float iou) { this.iouThreshold = iou; }
    @Override
    public void setNumItemsThreshold(int numItems) { this.numItemsThreshold = numItems; }

    // Métodos para registrar callbacks
    @Override
    public void setObjectDetectionResultCallback(ObjectDetectionResultCallback callback) { objectDetectionResultCallback = callback; }
    @Override
    public void setInferenceTimeCallback(FloatResultCallback callback) { inferenceTimeCallback = callback; }
    @Override
    public void setFpsRateCallback(FloatResultCallback callback) { fpsRateCallback = callback; }

    /**
     * Carga el archivo .tflite desde assets a memoria
     */
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Inicializa el intérprete TFLite, con GPU delegate si es posible
     */
    private void initDelegate(MappedByteBuffer buffer, boolean useGpu) {
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        try {
            CompatibilityList compatibilityList = new CompatibilityList();
            if (useGpu && compatibilityList.isDelegateSupportedOnThisDevice()) {
                // Si hay soporte GPU, usar GpuDelegate
                GpuDelegateFactory.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions.setQuantizedModelsAllowed(true));
                interpreterOptions.addDelegate(gpuDelegate);
            } else {
                // Caso contrario, usar CPU con 4 hilos
                interpreterOptions.setNumThreads(4);
            }
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        } catch (Exception e) {
            // Si falla GPU, fallback a CPU
            interpreterOptions = new Interpreter.Options();
            interpreterOptions.setNumThreads(4);
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        }

        // Leer forma de salida del modelo
        int[] outputShape = interpreter.getOutputTensor(0).shape();
        outputShape2 = outputShape[1];
        outputShape3 = outputShape[2];
        output = new float[outputShape2][outputShape3];
    }

    /**
     * Preprocesa un Bitmap y lo mete en un ByteBuffer optimizado
     */
    private void setInputOptim(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // Inicializar buffers solo una vez
        if (intValues == null) {
            intValues = new int[INPUT_SIZE * INPUT_SIZE];  // ⚠️ Asume bitmap ya redimensionado a INPUT_SIZE
            int batchSize = 1;
            int RGB = 3;
            int numPixels = INPUT_SIZE * INPUT_SIZE;
            int bufferSize = batchSize * RGB * numPixels * NUM_BYTES_PER_CHANNEL;

            imgData = ByteBuffer.allocateDirect(bufferSize);
            imgData.order(ByteOrder.nativeOrder());

            outData = ByteBuffer.allocateDirect(outputShape2 * outputShape3 * NUM_BYTES_PER_CHANNEL);
            outData.order(ByteOrder.nativeOrder());
        }

        // Extrae pixeles ARGB
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        // Convierte ARGB → RGB normalizado y ajusta a formato YOLO
        ip.argb2yolo(intValues, imgData, width, height);

        imgData.rewind();

        // Asigna input y output al intérprete
        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        outData.rewind();
        outputMap.put(0, outData);
    }

    /**
     * Ejecuta inferencia con TFLite y aplica postprocesado
     */
    private ArrayList<DetectedObject> runInference() {
        if (interpreter != null) {
            long startTime = System.nanoTime();

            // Ejecución de inferencia
            interpreter.runForMultipleInputsOutputs(inputArray, outputMap);
            stats.inferenceTime = (System.nanoTime() - startTime) * Nanos2Millis;

            // Recupera resultados desde ByteBuffer
            ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
            if (byteBuffer != null) {
                byteBuffer.rewind();

                // Copiar a arreglo float[][] para postprocesado
                for (int j = 0; j < outputShape2; ++j) {
                    for (int k = 0; k < outputShape3; ++k) {
                        output[j][k] = byteBuffer.getFloat();
                    }
                }

                // Postprocesado: aplica confianza, IoU y NMS
                startTime = System.nanoTime();
                ArrayList<DetectedObject> ret = PostProcessUtils.postprocess(
                        output,
                        outputShape3,
                        outputShape2,
                        (float) confidenceThreshold,
                        (float) iouThreshold,
                        numItemsThreshold,
                        numClasses,
                        labels
                );
                stats.postProcessTime = (System.nanoTime() - startTime) * Nanos2Millis;

                return ret;
            }
        }
        return new ArrayList<>();
    }
}
