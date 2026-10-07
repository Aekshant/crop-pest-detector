package com.farm.app.ui

import androidx.annotation.StringRes
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.farm.app.R
import com.farm.app.ui.history.HistoryScreen
import com.farm.app.ui.laya.LayaScreen
import com.farm.app.ui.scan.ScanScreen
import com.farm.app.ui.status.StatusScreen

/** Top-level destinations shown in the bottom navigation bar. */
enum class Destination(val route: String, @StringRes val label: Int, val icon: ImageVector) {
    Scan("scan", R.string.nav_scan, Icons.Filled.Search),
    Laya("laya", R.string.nav_laya, Icons.Filled.Person),
    History("history", R.string.nav_history, Icons.Filled.DateRange),
    Status("status", R.string.nav_status, Icons.Filled.Info),
}

@Composable
fun FarmLensApp() {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = backStackEntry?.destination

    Scaffold(
        bottomBar = {
            NavigationBar {
                Destination.entries.forEach { dest ->
                    NavigationBarItem(
                        selected = currentDestination?.hierarchy?.any { it.route == dest.route } == true,
                        onClick = {
                            navController.navigate(dest.route) {
                                // One copy of each tab; keep each tab's state when switching.
                                popUpTo(navController.graph.findStartDestination().id) { saveState = true }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = { Icon(dest.icon, contentDescription = null) },
                        label = { Text(stringResource(dest.label)) },
                    )
                }
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = Destination.Scan.route,
            modifier = Modifier.padding(padding),
        ) {
            composable(Destination.Scan.route) { ScanScreen() }
            composable(Destination.Laya.route) { LayaScreen() }
            composable(Destination.History.route) { HistoryScreen() }
            composable(Destination.Status.route) { StatusScreen() }
        }
    }
}
