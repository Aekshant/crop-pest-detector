package com.farm.app.ui

import com.farm.app.ui.laya.LayaViewModel
import com.farm.app.ui.status.StatusViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class ViewModelTest {
    private val main = StandardTestDispatcher()

    @Before fun setUp() = Dispatchers.setMain(main)
    @After fun tearDown() = Dispatchers.resetMain()

    @Test fun layaSendAddsUserMessageThenReply() = runTest(main) {
        val vm = LayaViewModel(reply = { delay(100); "echo: $it" })
        vm.onDraftChange("  kya karu?  ")
        vm.send()
        assertTrue(vm.state.value.thinking)
        assertEquals("", vm.state.value.draft)
        assertEquals(1, vm.state.value.messages.size)
        advanceUntilIdle()
        val s = vm.state.value
        assertFalse(s.thinking)
        assertEquals(listOf(true to "kya karu?", false to "echo: kya karu?"), s.messages.map { it.fromUser to it.text })
    }

    @Test fun layaIgnoresBlankAndDoubleSend() = runTest(main) {
        val vm = LayaViewModel(reply = { delay(100); "ok" })
        vm.send()
        assertTrue(vm.state.value.messages.isEmpty())
        vm.onDraftChange("one"); vm.send()
        vm.onDraftChange("two"); vm.send() // still thinking about "one"
        advanceUntilIdle()
        assertEquals(2, vm.state.value.messages.size)
    }

    @Test fun statusLoadsCoreVersion() = runTest(main) {
        val vm = StatusViewModel(coreVersion = { "farm-core test" })
        assertEquals(null, vm.state.value.coreVersion)
        advanceUntilIdle()
        // Loaded on Dispatchers.Default; wait for it to land.
        repeat(50) { if (vm.state.value.coreVersion == null) Thread.sleep(10); advanceUntilIdle() }
        assertEquals("farm-core test", vm.state.value.coreVersion)
    }
}
