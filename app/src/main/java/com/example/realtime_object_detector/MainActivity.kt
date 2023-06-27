package com.example.realtime_object_detector

// Kütüphaneleri ve modülleri içe aktarma
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.FileUtils
import android.os.Handler
import android.os.HandlerThread
import android.provider.MediaStore
import android.view.Surface
import android.view.TextureView
import android.view.TextureView.SurfaceTextureListener
import android.widget.ImageView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.realtime_object_detector.databinding.ActivityMainBinding
import com.example.realtime_object_detector.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.lang.reflect.Field

// Ana aktivite sınıfımızı tanımlıyoruz
class MainActivity : AppCompatActivity() {

    // Gerekli değişkenleri tanımlama
    private val paint = Paint()
    private lateinit var imageProcessor : ImageProcessor
    private val CAMERA_REQUEST_CODE = 100
    private lateinit var cameraManager : CameraManager
    private lateinit var handler : Handler
    private lateinit var cameraDevice : CameraDevice
    private lateinit var textureView : TextureView
    private lateinit var imageView : ImageView
    private lateinit var bitmap : Bitmap
    private lateinit var model : SsdMobilenetV11Metadata1
    private var colors = listOf<Int>(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    private lateinit var labels : List<String>

    // Aktivite oluşturulduğunda çalışacak olan kod
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)  // Üst sınıfın onCreate metodunu çağırma
        setContentView(R.layout.activity_main)  // Layout dosyasını belirleme

        // TensorFlow modeli ve etiketlerin yüklenmesi
        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = SsdMobilenetV11Metadata1.newInstance(this@MainActivity)

        // Arka plan işlemcisi oluşturma
        val handlerThread = HandlerThread("camera_thread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        // Kamera ve görüntüleme nesnelerini başlatma
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        textureView = findViewById(R.id.textureView)
        imageView = findViewById(R.id.imageView)

        // Kamera çıktılarını textureView'a yüklemek için SurfaceTextureListener oluşturma
        textureView.surfaceTextureListener = object : SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                // Kamera izni kontrol etme
                checkPermission()
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {

            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            // Kamera çıktısını TensorFlow modeline besleyip sonuçları çizme
            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!
                var image = TensorImage.fromBitmap(bitmap)
                image = imageProcessor.process(image)
                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

                val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width

                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f
                var x = 0
                scores.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    if (fl > 0.5){
                        paint.setColor(colors.get(index))
                        paint.style = Paint.Style.STROKE
                        canvas.drawRect(RectF(locations.get(x + 1) * w, locations.get(x) * h, locations.get(x + 3) * w, locations.get(x + 2) * h), paint)
                        paint.style = Paint.Style.FILL
                        canvas.drawText(labels.get(classes.get(index).toInt()) + " " + fl.toString(), locations.get(x + 1) * w, locations.get(x) * h, paint)
                    }
                }
                imageView.setImageBitmap(mutable)
            }
        }
    }

    // Kamerayı açma fonksiyonu
    @SuppressLint("MissingPermission")
    private fun openCamera() {
        cameraManager.openCamera(
            cameraManager.cameraIdList[0],
            object : CameraDevice.StateCallback() {
                override fun onOpened(p0: CameraDevice) {
                    cameraDevice = p0
                    var surfaceTexture = textureView.surfaceTexture
                    var surface = Surface(surfaceTexture)
                    var captureRequest =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    captureRequest.addTarget(surface)
                    cameraDevice.createCaptureSession(
                        listOf(surface),
                        object : CameraCaptureSession.StateCallback() {
                            override fun onConfigured(p0: CameraCaptureSession) {
                                p0.setRepeatingRequest(captureRequest.build(), null, null)
                            }

                            override fun onConfigureFailed(p0: CameraCaptureSession) {

                            }
                        },
                        handler
                    )
                }

                override fun onDisconnected(p0: CameraDevice) {

                }

                override fun onError(p0: CameraDevice, p1: Int) {

                }

            },
            handler
        )
    }

    // Kamera iznini kontrol etme fonksiyonu
    private fun checkPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestCameraPermission()  // İzin verilmemişse, izin isteme
        } else {
            openCamera()  // İzin verilmişse, kamerayı açma
        }
    }

    // Kamera iznini isteme fonksiyonu
    private fun requestCameraPermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(
                this,
                Manifest.permission.CAMERA
            )
        ) {
            // Kullanıcıya neden kamera iznine ihtiyaç duyduğunuzu açıklayın.
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_REQUEST_CODE
            )
        }
    }

    // İzin isteme sonucunu işleme fonksiyonu
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            CAMERA_REQUEST_CODE -> {
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    openCamera()  // İzin verildiğinde kamerayı açma
                } else {
                    // İzin verilmediğinde ne yapılacağına dair kodları burada yazabilirsiniz.
                }
                return
            }

            else -> {
                // Diğer izin taleplerini ele alır.
            }
        }
    }

    // Aktivite yok edilirken modeli kapatma
    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
}
