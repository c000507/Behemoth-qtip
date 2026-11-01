// JSON Schema -> GBNF compiler for OpenAI-style structured outputs
// (response_format json_object / json_schema).
//
// Supported: type (string or list), properties/required/additionalProperties,
// items/prefixItems/minItems/maxItems, enum/const, anyOf/oneOf (oneOf is
// treated as anyOf), allOf with one schema, $ref to "#", #/$defs/...,
// #/definitions/..., nullable (OpenAPI), minLength/maxLength, format (date,
// time, date-time, uuid enforced; other formats are annotations), integer
// minimum/maximum/exclusiveMinimum/exclusiveMaximum (exact digit-range
// grammars), number bounds with integer-valued limits. Annotation keywords
// (title, description, default, examples, $schema, $id, $comment, ...) are
// ignored. Every other keyword, and every combination that would change the
// accepted language in a way the compiler cannot express, throws SchemaError
// naming the keyword so the server answers HTTP 400.
#pragma once

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace behemoth::grammar {

struct SchemaError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// Grammar whose root is any JSON object (OpenAI {"type":"json_object"}).
std::string json_object_grammar();

// Grammar for a JSON Schema (OpenAI {"type":"json_schema"}).
std::string json_schema_grammar(const nlohmann::ordered_json &schema);

// GBNF for the decimal integers in [lo, hi] (inclusive; at most 16 digits,
// i.e. |x| < 1e16). Exposed for tests.
std::string integer_range_gbnf(long long lo, bool has_lo, long long hi, bool has_hi);

} // namespace behemoth::grammar
