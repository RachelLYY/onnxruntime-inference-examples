/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.onnxruntime.example.imageclassifier

import android.graphics.*
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
const val DIM_BATCH_SIZE = 1;
const val DIM_PIXEL_SIZE = 3;
const val IMAGE_SIZE_X = 192;
const val IMAGE_SIZE_Y = 256;
// 224 224
fun preProcess(bitmap: Bitmap): FloatBuffer {
    val imgData = FloatBuffer.allocate(
        DIM_BATCH_SIZE
                * DIM_PIXEL_SIZE
                * IMAGE_SIZE_X
                * IMAGE_SIZE_Y
    )
    //var bitmap=resizeAndPadBitmap(bitmap)
    var bitmap = Bitmap.createBitmap(192, 256, Bitmap.Config.ARGB_8888)
    imgData.rewind()
    val stride = IMAGE_SIZE_X * IMAGE_SIZE_Y
    val bmpData = IntArray(stride)
    bitmap.getPixels(bmpData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
    for (i in 0..IMAGE_SIZE_X - 1) {
        for (j in 0..IMAGE_SIZE_Y - 1) {
            val idx = IMAGE_SIZE_Y * i + j
            val pixelValue = bmpData[idx]
            imgData.put(idx, (((pixelValue shr 16 and 0xFF) / 255f - 0.485f) / 0.229f))
            imgData.put(idx + stride, (((pixelValue shr 8 and 0xFF) / 255f - 0.456f) / 0.224f))
            imgData.put(idx + stride * 2, (((pixelValue and 0xFF) / 255f - 0.406f) / 0.225f))
        }
    }

    imgData.rewind()
    return imgData
}
/*fun resizeAndPadBitmap(img: Bitmap): Bitmap {
    val width = img.width
    val height = img.height

    // 计算缩放比例
    val scale = minOf(192f / width, 256f / height)

    // 计算缩放后的新宽和新高
    val newWidth = (width * scale).toInt()
    val newHeight = (height * scale).toInt()

    // 缩放图片
    val scaledBitmap = Bitmap.createScaledBitmap(img, newWidth, newHeight, true)
    var mat = Mat()
    var mat1=Mat()
    Utils.bitmapToMat(scaledBitmap, mat)
    // 计算需要填充的像素数
    val leftPadding = (192 - newWidth) / 2
    val rightPadding = 192 - newWidth - leftPadding
    val topPadding = (256 - newHeight) / 2
    val bottomPadding = 256 - newHeight - topPadding
    Core.copyMakeBorder(mat,mat1,topPadding,bottomPadding, leftPadding,rightPadding,Core.BORDER_CONSTANT)
    // 添加 padding
    val paddedBitmap = Bitmap.createBitmap(192, 256, Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(mat1, paddedBitmap)
    return paddedBitmap
}*/


/*fun getSimccMaximum(simcc_x: Array<FloatArray>, simcc_y: Array<FloatArray>): Pair<Array<FloatArray>, FloatArray> {

    var N: Int? = null
    var K: Int
    var Wx: Int

    if (simcc_x.size == 3) {
        simcc_x.size
        N = simcc_x.shape[0]
        K = simcc_x.shape[1]
        Wx = simcc_x.shape[2]
        simcc_x = simcc_x.reshape(N * K, -1)
        simcc_y = simcc_y.reshape(N * K, -1)
    } else {
        K = simcc_x.shape[0]
        Wx = simcc_x.shape[1]
    }

    val x_locs = simcc_x.argmax(axis=1)
    val y_locs = simcc_y.argmax(axis=1)
    val locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    val max_val_x = simcc_x.amax(axis=1)
    val max_val_y = simcc_y.amax(axis=1)

    val mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    val vals = max_val_x
    locs[vals <= 0.] = -1

    if (N != null) {
        locs = locs.reshape(N, K, 2)
        vals = vals.reshape(N, K)
    }

    return Pair(locs, vals)
}

fun decode(simcc_x: Array<FloatArray>, simcc_y: Array<FloatArray>): Pair<Array<FloatArray>, FloatArray> {
    var (keypoints, scores) = getSimccMaximum(simcc_x, simcc_y)

    val sigma = arrayOf(6, 6)

    keypoints = keypoints/2

    return Pair(keypoints, scores)
}*/