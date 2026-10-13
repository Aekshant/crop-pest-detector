package com.farm.app.laya

import org.json.JSONObject

data class ArgSpec(val type: String, val required: Boolean, val minimum: Long? = null, val maximum: Long? = null)
data class ToolSpec(val name: String, val needsNetwork: Boolean, val sideEffect: Boolean, val args: Map<String, ArgSpec>)

/** ml/bundle/tools_schema_v1.json: the tools Laya may call and their argument rules. */
class ToolSchema(val version: Int, val labels: List<String>, val respondKinds: Set<String>, val tools: Map<String, ToolSpec>) {

    /** Why [call] breaks the schema, or null when it is valid (design: CALL_TOOL validates args first). */
    fun validate(call: Understanding.Call): String? {
        val spec = tools[call.tool] ?: return "unknown tool ${call.tool}"
        for ((name, value) in call.args) {
            val arg = spec.args[name] ?: return "unknown arg $name"
            when (arg.type) {
                "integer" -> {
                    val n = (value as? Number)?.toLong() ?: return "$name must be an integer"
                    if (arg.minimum != null && n < arg.minimum) return "$name below ${arg.minimum}"
                    if (arg.maximum != null && n > arg.maximum) return "$name above ${arg.maximum}"
                }
                "label" -> if (value !in labels) return "$name '$value' is not a label"
                "datetime" -> if (value !is String || !DATETIME.matches(value)) return "$name is not ISO 8601"
                else -> if (value !is String || value.isBlank()) return "$name must be text"
            }
        }
        spec.args.filter { it.value.required && it.key !in call.args }.keys.firstOrNull()?.let { return "missing $it" }
        return null
    }

    companion object {
        private val DATETIME = Regex("""\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)""")

        fun parse(json: String): ToolSchema {
            val o = JSONObject(json)
            val toolsJson = o.getJSONObject("tools")
            val tools = toolsJson.keys().asSequence().associateWith { name ->
                val t = toolsJson.getJSONObject(name)
                val argsJson = t.getJSONObject("args")
                ToolSpec(
                    name, t.getBoolean("needsNetwork"), t.getBoolean("sideEffect"),
                    argsJson.keys().asSequence().associateWith { a ->
                        val s = argsJson.getJSONObject(a)
                        ArgSpec(
                            s.getString("type"), s.getBoolean("required"),
                            if (s.has("minimum")) s.getLong("minimum") else null,
                            if (s.has("maximum")) s.getLong("maximum") else null,
                        )
                    },
                )
            }
            val labels = o.getJSONArray("labels").let { a -> List(a.length()) { a.getString(it) } }
            val kinds = o.getJSONArray("respond_kinds").let { a -> List(a.length()) { a.getString(it) }.toSet() }
            return ToolSchema(o.getInt("schema_version"), labels, kinds, tools)
        }
    }
}
