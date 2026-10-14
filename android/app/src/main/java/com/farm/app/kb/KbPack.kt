package com.farm.app.kb

import android.database.sqlite.SQLiteDatabase
import com.farm.app.laya.KbRow
import java.io.File

/**
 * The KB pack from the model bundle (kb/build_pack.py), opened read-only with
 * Android's SQLite. Android can't load sqlite-vec, so vectors are read from
 * kb_chunk.embedding and searched by [KbRetriever] in memory.
 */
class KbPack private constructor(
    val meta: Map<String, String>,
    val chunks: List<KbChunk>,
    val treatments: List<KbRow>,
) {
    companion object {
        fun open(file: File): KbPack {
            val db = SQLiteDatabase.openDatabase(file.path, null, SQLiteDatabase.OPEN_READONLY)
            try {
                val meta = db.rawQuery("SELECT key, value FROM meta", null).use { c ->
                    buildMap { while (c.moveToNext()) put(c.getString(0), c.getString(1)) }
                }
                val chunks = db.rawQuery("SELECT id, label, text, embedding FROM kb_chunk ORDER BY id", null).use { c ->
                    buildList {
                        while (c.moveToNext()) add(KbChunk(c.getInt(0), labels(c.getString(1)), c.getString(2), KbRetriever.vector(c.getBlob(3))))
                    }
                }
                val rows = db.rawQuery("SELECT label, product, dose, phi_days, registration_status FROM kb_treatment ORDER BY id", null).use { c ->
                    buildList { while (c.moveToNext()) add(KbRow(c.getString(0), c.getString(1), c.getString(2), c.getInt(3), c.getString(4) == "registered")) }
                }
                return KbPack(meta, chunks, rows)
            } finally {
                db.close()
            }
        }

        fun labels(column: String?): Set<String> = column.orEmpty().split(';').filter { it.isNotBlank() }.toSet()
    }
}
