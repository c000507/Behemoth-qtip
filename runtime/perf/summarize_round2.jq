def millis: . * 1000 | round / 1000;
["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] as $order
| [.results[] | select(.status == "measured")] as $rows
| "== \(input_filename) rows=\($rows[0].rows // "none") cross_path_failures=\([$rows[] | select(.cross_path_pass == false)] | length)",
  ($rows[] | select(.cross_path_pass == false)
   | "   FAIL \(.code_layout) \(.projection) \(.mode) max_abs_error=\(.max_abs_error)"),
  ($rows | group_by(.code_layout)[]
   | .[0].code_layout as $layout
   | (reduce .[] as $r ({}; .[$r.projection][$r.mode] = $r.median_ms)) as $projections
   | [$order[] as $name
      | ($projections[$name] // {} | to_entries | map(select(.key != "auto")))
      | select(length > 0) | min_by(.value) as $best
      | {name: $name, best: $best, auto: $projections[$name].auto}] as $cells
   | "  \($layout) layer best \($cells | map(.best.value) | add // 0 | millis) ms  auto \(if any($cells[]; .auto == null) then "missing" else ($cells | map(.auto) | add // 0 | millis) end) ms | \($cells | map("\(.name | split(".")[-1] | sub("_proj$"; "")) \(.best.value | millis) (\(.best.key))") | join("; "))",
     ([$projections | to_entries[] | .value | to_entries[]]
      | group_by(.key) | map(select(length == ($order | length)) | {mode: .[0].key, ms: (map(.value) | add)})
      | sort_by(.ms) | .[:8] | "    layer sums: \(map("\(.mode) \(.ms | millis)") | join(", "))"))
