package com.farm.app.vision

import org.junit.Assert.assertEquals
import org.junit.Test

class ThreadsTest {
    @Test fun bigLittleUsesOnlyTheBigCores() {
        // POCO M6 Plus (SM4450): 6 x 458 + 2 x 1024.
        assertEquals(2, VisionEngine.threadsFor(List(6) { 458L } + List(2) { 1024L }, emptyList(), 8))
    }

    @Test fun primePlusBigCountsBoth() {
        assertEquals(4, VisionEngine.threadsFor(listOf(1024L, 870, 870, 870, 400, 400, 400, 400), emptyList(), 8))
    }

    @Test fun uniformCoresUseAllUpToFour() {
        assertEquals(4, VisionEngine.threadsFor(List(8) { 1024L }, emptyList(), 8))
        assertEquals(2, VisionEngine.threadsFor(List(2) { 1024L }, emptyList(), 2))
    }

    @Test fun fallsBackToFrequencyThenCoreCount() {
        assertEquals(2, VisionEngine.threadsFor(emptyList(), List(6) { 1958400L } + List(2) { 2304000L }, 8))
        assertEquals(4, VisionEngine.threadsFor(emptyList(), emptyList(), 8))
        assertEquals(1, VisionEngine.threadsFor(emptyList(), emptyList(), 1))
    }
}
