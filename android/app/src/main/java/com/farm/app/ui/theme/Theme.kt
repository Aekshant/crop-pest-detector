package com.farm.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Leaf = Color(0xFF2E7D32)
private val LeafLight = Color(0xFF81C784)
private val Soil = Color(0xFF6D4C41)
private val Tomato = Color(0xFFC62828)

private val LightColors = lightColorScheme(
    primary = Leaf,
    primaryContainer = Color(0xFFC8E6C9),
    onPrimaryContainer = Color(0xFF0B3D10),
    secondary = Soil,
    secondaryContainer = Color(0xFFDCEDC8),
    onSecondaryContainer = Color(0xFF1B3A0E),
    tertiary = Tomato,
    background = Color(0xFFFAFAF7),
    surface = Color(0xFFFAFAF7),
    surfaceVariant = Color(0xFFECEFE8),
    surfaceContainer = Color(0xFFF0F3EC),
)
private val DarkColors = darkColorScheme(
    primary = LeafLight,
    primaryContainer = Color(0xFF1B5E20),
    secondary = Color(0xFFBCAAA4),
    secondaryContainer = Color(0xFF33691E),
    tertiary = Color(0xFFEF9A9A),
)

@Composable
fun FarmLensTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (darkTheme) DarkColors else LightColors, content = content)
}
