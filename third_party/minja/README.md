Vendored `include/minja/minja.hpp` from https://github.com/google/minja.
Git blob: `5ed0556b92acbc50ed3454b105a23449b221e014`.
Retrieved 2026-11-04. MIT license is in LICENSE.
Local change: parse Jinja numeric attributes (`items.0`), required by the official
GLM-5.3-Flash template. Equivalent to the already supported `items[0]` syntax.
Also expose `capitalize` as a Jinja filter, with ASCII lowercasing of the suffix
and unsigned-character ctype calls. Used for GLM's `low`/`high`/`max` labels.
Trailing whitespace and the extra final blank line were removed.

Used by the GLM backend to apply each checkpoint's actual Jinja chat template.
