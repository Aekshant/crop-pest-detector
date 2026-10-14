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
        /**
         * [includeUnreviewed]: also let Laya use rows still pending agronomist review
         * (debug builds, for testing before the review is done). Rejected rows are never used.
         */
        fun open(file: File, includeUnreviewed: Boolean = false): KbPack {
            val db = SQLiteDatabase.openDatabase(file.path, null, SQLiteDatabase.OPEN_READONLY)
            try {
                val meta = db.rawQuery("SELECT key, value FROM meta", null).use { c ->
                    buildMap { while (c.moveToNext()) put(c.getString(0), c.getString(1)) }
                }
                val publishers = db.rawQuery("SELECT url, publisher FROM kb_source", null).use { c ->
                    buildMap { while (c.moveToNext()) put(c.getString(0), shortPublisher(c.getString(1))) }
                }
                val chunks = db.rawQuery("SELECT id, label, text, embedding, source_url FROM kb_chunk ORDER BY id", null).use { c ->
                    buildList {
                        while (c.moveToNext()) add(KbChunk(c.getInt(0), labels(c.getString(1)), c.getString(2), KbRetriever.vector(c.getBlob(3)),
                                                           c.getString(4), publishers[c.getString(4)].orEmpty()))
                    }
                }
                val rows = db.rawQuery(
                    "SELECT label, product, dose, phi_days, registration_status, review_status, display_rank FROM kb_treatment ORDER BY id", null,
                ).use { c ->
                    buildList {
                        while (c.moveToNext()) add(KbRow(
                            c.getString(0), c.getString(1), c.getString(2), c.getInt(3), c.getString(4) == "registered",
                            reviewed = c.getString(5) == "approved" || (includeUnreviewed && c.getString(5) == "pending"),
                            rank = c.getString(6)?.toIntOrNull() ?: Int.MAX_VALUE,
                        ))
                    }
                }
                return KbPack(meta, chunks, rows)
            } finally {
                db.close()
            }
        }

        fun labels(column: String?): Set<String> = column.orEmpty().split(';').filter { it.isNotBlank() }.toSet()

        /** "TNAU Agritech Portal (Tamil Nadu Agricultural University)" -> "TNAU Agritech Portal". */
        fun shortPublisher(publisher: String): String = publisher.substringBefore(" (").trim()
    }
}
