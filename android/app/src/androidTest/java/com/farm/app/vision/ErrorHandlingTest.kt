package com.farm.app.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer

/**
 * A8 done-when: a forced native error returns an error code instead of
 * crashing. Every failure path of the JNI boundary is exercised; the engine and
 * the process keep working afterwards.
 */
@RunWith(AndroidJUnit4::class)
class ErrorHandlingTest {
    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    @Test fun forcedExceptionsBecomeErrorCodes() {
        assertEquals(VisionNative.OK, VisionNative.nativeForceError(0))
        assertEquals(VisionNative.NATIVE_EXCEPTION, VisionNative.nativeForceError(1))  // std::runtime_error
        assertEquals(VisionNative.OUT_OF_MEMORY, VisionNative.nativeForceError(2))     // std::bad_alloc
        assertEquals(VisionNative.NATIVE_EXCEPTION, VisionNative.nativeForceError(3))  // throw 42
    }

    @Test fun badHandlesAreRejectedNotDereferenced() {
        val out = FloatArray(VisionNative.PROBS + 12)
        assertEquals(VisionNative.INVALID_HANDLE, VisionNative.nativeAnalyzeRgb(0L, ByteArray(12), 2, 2, 0, 12, out))
        assertEquals(VisionNative.INVALID_HANDLE, VisionNative.nativeAnalyzeRgb(0x1234_5678L, ByteArray(12), 2, 2, 0, 12, out))
        assertEquals(VisionNative.INVALID_HANDLE, VisionNative.nativeRelease(0xdeadL))
        assertEquals(VisionNative.INVALID_HANDLE, VisionNative.nativeInfo(0L, FloatArray(2)))

        val engine = VisionEngine.create(appContext, allowGpu = false)
        engine.close()
        engine.close()  // idempotent
        val r = engine.analyzeRgb(ByteArray(224 * 224 * 3), 224, 224)
        assertEquals(VisionOutcome.Failed(VisionNative.INVALID_HANDLE), r)
    }

    @Test fun badArgumentsReturnCodesAndEngineKeepsWorking() {
        VisionEngine.create(appContext, allowGpu = false).use { engine ->
            val handle = VisionEngine::class.java.getDeclaredField("handle").apply { isAccessible = true }.getLong(engine)
            val out = FloatArray(VisionNative.PROBS + 12)
            // Too few bytes for the stated size, zero size, null input, small result array.
            assertEquals(VisionNative.INVALID_ARGUMENT, VisionNative.nativeAnalyzeRgb(handle, ByteArray(10), 224, 224, 0, 12, out))
            assertEquals(VisionNative.INVALID_ARGUMENT, VisionNative.nativeAnalyzeRgb(handle, ByteArray(10), 0, 224, 0, 12, out))
            assertEquals(VisionNative.INVALID_ARGUMENT, VisionNative.nativeAnalyzeRgb(handle, null, 224, 224, 0, 12, out))
            assertEquals(VisionNative.INVALID_ARGUMENT,
                VisionNative.nativeAnalyzeRgb(handle, ByteArray(224 * 224 * 3), 224, 224, 0, 12, FloatArray(3)))
            // Heap (non-direct) buffer where a direct one is required; too-small direct buffer.
            assertEquals(VisionNative.BUFFER_NOT_DIRECT,
                VisionNative.nativeCheckQuality(handle, ByteBuffer.allocate(640 * 480), 640, 480, 640, FloatArray(4)))
            assertEquals(VisionNative.INVALID_ARGUMENT,
                VisionNative.nativeCheckQuality(handle, ByteBuffer.allocateDirect(100), 640, 480, 640, FloatArray(4)))
            assertEquals(VisionNative.INVALID_ARGUMENT, VisionNative.nativeCheckQuality(handle, null, 640, 480, 640, FloatArray(4)))

            // After all that, a real image still analyzes.
            val g = "01_early_blight"
            val rgb = testContext.assets.open("$g.rgb").use { it.readBytes() }
            val r = engine.analyzeRgb(rgb, 360, 480).resultOrNull()
            assertNotNull("engine still works after errors", r)
            assertEquals("early_blight", r!!.top?.id)
        }
    }

    @Test fun corruptModelFailsCreationWithAnException() {
        val garbage = ByteBuffer.allocateDirect(1024).apply { repeat(1024) { put(it.toByte()) }; rewind() }
        try {
            VisionNative.nativeCreate(null, garbage, FloatArray(8) { 1f }, 12, 0, 224, 1, false, BooleanArray(12))
            fail("expected IllegalStateException")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!, e.message!!.contains("vision engine"))
        }
        // Non-direct model buffer.
        try {
            VisionNative.nativeCreate(null, ByteBuffer.allocate(10), FloatArray(8), 12, 0, 224, 1, false, BooleanArray(12))
            fail("expected IllegalStateException")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!, e.message!!.contains("code ${VisionNative.BUFFER_NOT_DIRECT}"))
        }
    }
}
