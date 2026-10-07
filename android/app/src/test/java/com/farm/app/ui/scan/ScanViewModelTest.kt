package com.farm.app.ui.scan

import com.farm.app.vision.Decision
import com.farm.app.vision.Label
import com.farm.app.vision.QualityVerdict
import com.farm.app.vision.VisionResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Test

class ScanViewModelTest {
    private fun frame(decision: Decision, sharpness: Float, label: String = "early_blight") = VisionResult(
        decision, Label(label, label, label, "disease"), 0.9f, FloatArray(12), sharpness, 120f,
        if (decision == Decision.RETAKE) QualityVerdict.BLURRY else QualityVerdict.OK, 1f, 1f, 1f,
    )

    @Test fun burstKeepsSharpestFrameThatPassedTheGate() {
        val best = ScanViewModel.best(
            listOf(frame(Decision.RETAKE, 0.95f), frame(Decision.CONFIDENT, 0.6f), frame(Decision.UNCERTAIN, 0.7f, "late_blight")),
        )
        assertEquals("late_blight", best.top?.id)
    }

    @Test fun burstOfBlurryFramesIsARetake() {
        assertEquals(Decision.RETAKE, ScanViewModel.best(listOf(frame(Decision.RETAKE, 0.3f), frame(Decision.RETAKE, 0.4f))).decision)
    }

    @Test fun takePhotoCollectsThreeFrames() {
        val vm = ScanViewModel()
        vm.onFrame(frame(Decision.CONFIDENT, 0.8f))
        assertNull(vm.state.value.captured)  // live only
        vm.takePhoto()
        vm.onFrame(frame(Decision.CONFIDENT, 0.6f))
        vm.onFrame(frame(Decision.CONFIDENT, 0.7f, "late_blight"))
        assertNull(vm.state.value.captured)
        vm.onFrame(frame(Decision.RETAKE, 0.9f))
        assertEquals("late_blight", vm.state.value.captured?.top?.id)
        assertFalse(vm.state.value.capturing)
    }
}
