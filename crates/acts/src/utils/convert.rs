use crate::{ActError, Context, Result, Vars, scheduler::Task};
use regex::Regex;
use serde_json::Value as JsonValue;
use std::sync::{Arc, LazyLock};

/// Evaluate every `${{ ... }}` expression inside a node's params.
///
/// Returns `Err` as soon as one expression cannot be evaluated.
///
/// **Never swallow that error.** The previous implementation logged it and substituted
/// `JsonValue::Null` — and for a string holding several placeholders it substituted the
/// literal text `null` — so the node kept running with *partially filled params*. When
/// such a payload cannot be deserialized by the receiver it never answers, so the sender
/// waits out the whole request timeout (observed: 600s stalled, wedging the build queue)
/// while the only trace was a single `eprintln!`. Callers (`Task::params`,
/// `Context::emit_message`) fail the node instead.
pub fn fill_params(params: &JsonValue, ctx: &Context) -> Result<JsonValue> {
    match params {
        JsonValue::String(source) => {
            let exprs = get_exprs(source);
            if !exprs.is_empty() {
                let mut value = source.clone();
                for (range, expr, content) in &exprs {
                    let result =
                        Context::scope(ctx, move || ctx.runtime.env().eval::<JsonValue>(content))
                            .map_err(|err| ActError::Exception {
                                ecode: "param_expr".to_string(),
                                message: format!(
                                    "failed to evaluate `{expr}` in params string `{source}`: {err}"
                                ),
                            })?;
                    // just return json for only one express
                    if range.start == 0 && range.end == source.len() {
                        return Ok(result);
                    }

                    match result {
                        JsonValue::Bool(v) => {
                            value = value.replace(expr, &v.to_string());
                        }
                        JsonValue::Number(v) => {
                            value = value.replace(expr, &v.to_string());
                        }
                        JsonValue::String(v) => {
                            value = value.replace(expr, &v);
                        }
                        v => {
                            value = value.replace(expr, &v.to_string());
                        }
                    }
                }
                // return string json for multiple expressions
                return Ok(JsonValue::String(value));
            }

            // return params itself for no expression
            Ok(params.clone())
        }
        JsonValue::Array(values) => {
            let mut arr = Vec::new();
            for value in values {
                arr.push(fill_params(value, ctx)?);
            }
            Ok(JsonValue::Array(arr))
        }
        JsonValue::Object(map) => {
            let mut obj = serde_json::Map::new();
            for (k, value) in map {
                obj.insert(k.clone(), fill_params(value, ctx)?);
            }
            Ok(JsonValue::Object(obj))
        }
        v => Ok(v.clone()),
    }
}

/// fill the vars
/// 1. if the inputs is an expression, just calculate it
///    or insert the input itself
pub fn fill_inputs(inputs: &Vars, ctx: &Context) -> Vars {
    let mut ret = Vars::new();
    for (k, v) in inputs.iter() {
        if let JsonValue::String(value) = v {
            if let Some(expr) = get_expr(value) {
                let result =
                    Context::scope(ctx, move || ctx.runtime.env().eval::<JsonValue>(&expr));

                let new_value = result.unwrap_or_else(|err| {
                    eprintln!("fill_inputs: expr:{value}, err={err}");
                    JsonValue::Null
                });

                // satisfies the rule 1
                ret.insert(k.clone(), new_value);
                continue;
            }
        } else if let JsonValue::Object(obj) = v {
            ret.insert(k.clone(), fill_inputs(&Vars::from(obj.clone()), ctx).into());
            continue;
        }
        ret.insert(k.clone(), v.clone());
    }

    ret
}

/// fill the outputs
/// 1. if the outputs is an expression, just calculate it
/// 2. if the env and the outputs both has the same key, using the local outputs
pub fn fill_outputs(outputs: &Vars, ctx: &Context) -> Vars {
    // println!("fill_outputs: outputs={outputs}");
    let mut ret = Vars::new();
    for (k, v) in outputs.iter() {
        if let JsonValue::String(string) = v
            && let Some(expr) = get_expr(string)
        {
            let result = Context::scope(ctx, move || ctx.runtime.env().eval::<JsonValue>(&expr));
            let new_value = result.unwrap_or_else(|err| {
                eprintln!("fill_outputs: expr:{string}, err={err}");
                JsonValue::Null
            });

            // satisfies the rule 1
            ret.insert(k.clone(), new_value);
            continue;
        }

        // rule 2
        if v.is_null() {
            // the env value
            match ctx.task().find(k) {
                Some(v) => ret.insert(k.clone(), v),
                None => ret.insert(k.clone(), v.clone()),
            };
        } else {
            // insert the orign value
            ret.insert(k.clone(), v.clone());
        }
    }

    ret
}

pub fn fill_proc_vars(task: &Arc<Task>, values: &Vars, ctx: &Context) -> Vars {
    let mut ret = Vars::new();
    for (k, v) in values.iter() {
        if let JsonValue::String(string) = v
            && let Some(expr) = get_expr(string)
        {
            let result = Context::scope(ctx, || ctx.runtime.env().eval::<JsonValue>(&expr));
            let new_value = result.unwrap_or(JsonValue::Null);

            // satisfies the rule 1
            ret.insert(k.clone(), new_value);

            continue;
        }

        // rule 2
        match task.find::<JsonValue>(k) {
            Some(v) => ret.insert(k.clone(), v.clone()),
            None => ret.insert(k.clone(), v.clone()),
        };
    }
    ret
}

pub fn get_expr(text: &str) -> Option<String> {
    // Only a string that is **exactly one** placeholder is a bare expression.
    //
    // Counting with the lazy `get_exprs` scanner matters: an anchored greedy regex
    // (`^\$\{\{(.+)\}\}$`) happily matches `${{ a }}/${{ b }}` as a single expression and
    // captures `a }}/${{ b`, whose evaluation always fails. (Making that quantifier lazy
    // does not help either — the `$` anchor still forces it to swallow the second
    // placeholder.)
    let exprs = get_exprs(text);
    if exprs.len() != 1 || exprs[0].0.start != 0 || exprs[0].0.end != text.len() {
        return None;
    }
    Some(exprs[0].2.trim().to_string())
}

pub fn get_exprs(text: &str) -> Vec<(core::ops::Range<usize>, String, String)> {
    // ⚠️ `(.*?)` is **required**: with the greedy `(.*)` a string like
    // `${{ release_prefix }}/${{ app }}` matched once, from the first `${{` to the last
    // `}}`, capturing `release_prefix }}/${{ app` as the expression — evaluating it always
    // failed (`unexpected token in expression: '}'`).
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\$\{\{(.*?)\}\}").unwrap());
    RE.captures_iter(text)
        .map(|caps| {
            let input = caps.get(0).unwrap();
            let content = caps.get(1).unwrap();
            (
                input.range(),
                input.as_str().to_string(),
                content.as_str().to_string(),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{get_expr, get_exprs};

    /// A string holding two placeholders must yield **two** expressions.
    ///
    /// With the old greedy `\$\{\{(.*)\}\}` it yielded one expression spanning from the
    /// first `${{` to the last `}}` (`release_prefix }}/${{ app`), whose evaluation always
    /// failed with `unexpected token in expression: '}'`.
    #[test]
    fn get_exprs_handles_several_placeholders_in_one_string() {
        let exprs = get_exprs("${{ release_prefix }}/${{ alias.last_output }}");
        assert_eq!(exprs.len(), 2, "expected two expressions, got {exprs:?}");
        assert_eq!(exprs[0].2.trim(), "release_prefix");
        assert_eq!(exprs[1].2.trim(), "alias.last_output");
        assert_eq!(exprs[0].0, 0..21);
    }

    #[test]
    fn get_exprs_keeps_single_placeholder_behaviour() {
        let exprs = get_exprs("${{ project }}");
        assert_eq!(exprs.len(), 1);
        assert_eq!(exprs[0].0, 0..14);
        assert_eq!(exprs[0].2.trim(), "project");
    }

    #[test]
    fn get_expr_only_accepts_a_string_that_is_exactly_one_placeholder() {
        assert_eq!(get_expr("${{ a }}").as_deref(), Some("a"));
        // Not "one expression spanning both placeholders" — no match at all.
        assert_eq!(get_expr("${{ a }}/${{ b }}"), None);
        assert_eq!(get_expr("plain"), None);
    }
}
