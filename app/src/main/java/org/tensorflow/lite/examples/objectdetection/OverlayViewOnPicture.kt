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

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import java.util.LinkedList
import kotlin.math.max
import org.tensorflow.lite.task.vision.detector.Detection
import kotlin.math.roundToInt

class OverlayViewOnPicture(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.GREEN
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(180, 0, 0, 0)
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
    }

    private var results: List<ObjectDetection> = emptyList()
    private var imageWidth = 0
    private var imageHeight = 0

    // Rectángulo donde la imagen está realmente dibujada dentro del ImageView (en coords del View)
    private val imageDisplayRect = RectF()

    fun clear() {
        results = emptyList()
        imageWidth = 0
        imageHeight = 0
        imageDisplayRect.setEmpty()
        invalidate()
    }

    /**
     * results: cajas en coordenadas de la imagen (px)
     * imgH/W: dimensiones originales de la imagen con la que se infirió
     * displayRect: dónde está dibujada esa imagen dentro del contenedor (FrameLayout/ImageView)
     */
    fun setResults(
        results: List<ObjectDetection>,
        imgHeight: Int,
        imgWidth: Int,
        displayRect: RectF
    ) {
        this.results = results
        this.imageHeight = imgHeight
        this.imageWidth = imgWidth
        this.imageDisplayRect.set(displayRect)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (results.isEmpty() || imageWidth == 0 || imageHeight == 0 || imageDisplayRect.isEmpty) {
            return
        }

        val sx = imageDisplayRect.width() / imageWidth.toFloat()
        val sy = imageDisplayRect.height() / imageHeight.toFloat()

        for (det in results) {
            val r = det.boundingBox
            // Transformar de coords imagen -> coords del Overlay (ajustadas al displayRect)
            val left = imageDisplayRect.left + r.left * sx
            val top = imageDisplayRect.top + r.top * sy
            val right = imageDisplayRect.left + r.right * sx
            val bottom = imageDisplayRect.top + r.bottom * sy

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val label = "${det.category.label} ${(det.category.confidence * 100).roundToInt()}%"
            val textW = textPaint.measureText(label)
            val textH = textPaint.fontMetrics.let { it.descent - it.ascent }
            val pad = 6f

            val bgRect = RectF(
                left,
                top - textH - 2 * pad,
                left + textW + 2 * pad,
                top
            )

            canvas.drawRoundRect(bgRect, 8f, 8f, textBgPaint)
            canvas.drawText(label, bgRect.left + pad, bgRect.bottom - pad, textPaint)
        }
    }
}
