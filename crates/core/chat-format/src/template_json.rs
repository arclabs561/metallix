//! The `tojson` filter as Hugging Face chat templates expect it.
//!
//! transformers renders chat templates with its own filter,
//! `json.dumps(x, ensure_ascii=False, indent=indent, separators=separators,
//! sort_keys=sort_keys)`. minijinja's built-in instead writes compact JSON,
//! escapes HTML characters and refuses `ensure_ascii`, so a template such as
//! `MiniCPM5`'s `tool | tojson(ensure_ascii=False)` fails to render, and others
//! render tool schemas the model never saw in training. Keyword arguments
//! other than `ensure_ascii`, `indent` and `sort_keys` are refused. Floats
//! are spelled as Python's `repr` spells them (`1e-05`, `1e+16`).
//!
//! Neither `serde_json` nor minijinja is built with `preserve_order` here, so
//! every map a template sees, from request JSON or a template literal, is
//! already key-sorted: `sort_keys` is accepted and changes nothing, and
//! output matches Python's only where the source dict was in sorted order.

use std::io;

use minijinja::{Error, ErrorKind, Value, value::Kwargs};
use serde::Serialize;
use serde_json::ser::Formatter;

#[allow(
    clippy::needless_pass_by_value,
    reason = "minijinja hands filters their keyword arguments by value"
)]
pub(crate) fn tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let indent = kwargs.get::<Option<usize>>("indent")?;
    let _sorted_already = kwargs.get::<Option<bool>>("sort_keys")?;
    kwargs.assert_all_used()?;
    let formatter = PythonJson {
        indent: indent.map(|width| vec![b' '; width]),
        depth: 0,
        has_value: false,
        ensure_ascii,
    };
    let mut output = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut output, formatter);
    value.serialize(&mut serializer).map_err(|error| {
        Error::new(ErrorKind::BadSerialization, "tojson could not serialize").with_source(error)
    })?;
    let text = String::from_utf8(output)
        .map_err(|_| Error::new(ErrorKind::BadSerialization, "tojson produced invalid UTF-8"))?;
    Ok(Value::from_safe_string(text))
}

/// Python `json.dumps` layout: `", "` and `": "` separators without an
/// indent; with one, `","` plus a newline per item, as Python 3.4+ does.
struct PythonJson {
    indent: Option<Vec<u8>>,
    depth: usize,
    has_value: bool,
    ensure_ascii: bool,
}

impl PythonJson {
    fn separate<W: ?Sized + io::Write>(&self, writer: &mut W, first: bool) -> io::Result<()> {
        if !first {
            writer.write_all(b",")?;
        }
        match &self.indent {
            Some(indent) => {
                writer.write_all(b"\n")?;
                (0..self.depth).try_for_each(|_| writer.write_all(indent))
            }
            None if first => Ok(()),
            None => writer.write_all(b" "),
        }
    }

    fn close<W: ?Sized + io::Write>(&mut self, writer: &mut W, bracket: &[u8]) -> io::Result<()> {
        self.depth -= 1;
        if let Some(indent) = &self.indent {
            if self.has_value {
                writer.write_all(b"\n")?;
                (0..self.depth).try_for_each(|_| writer.write_all(indent))?;
            }
        }
        self.has_value = true;
        writer.write_all(bracket)
    }
}

impl Formatter for PythonJson {
    fn begin_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.depth += 1;
        self.has_value = false;
        writer.write_all(b"[")
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"]")
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.separate(writer, first)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.depth += 1;
        self.has_value = false;
        writer.write_all(b"{")
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.close(writer, b"}")
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.separate(writer, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, _writer: &mut W) -> io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn write_f64<W: ?Sized + io::Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        writer.write_all(python_float_repr(value).as_bytes())
    }

    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }
        for character in fragment.chars() {
            if character.is_ascii() {
                writer.write_all(&[character as u8])?;
            } else {
                let mut units = [0_u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}

/// Python's `float.__repr__` for a finite value: the shortest digits that
/// round-trip (as Rust's `{:e}` writes them), positional with at least one
/// fractional digit when the decimal exponent is in `-4..16`, otherwise
/// `d.ddde+XX` with a signed exponent of at least two digits.
fn python_float_repr(value: f64) -> String {
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("`{:e}` always writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes a decimal exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    if !(-4..16).contains(&exponent) {
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return format!(
            "{sign}{mantissa}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        );
    }
    let digits = mantissa.replace('.', "");
    let point = usize::try_from(exponent + 1).unwrap_or(0);
    if point == 0 {
        let zeros = "0".repeat(usize::try_from(-exponent - 1).unwrap_or(0));
        return format!("{sign}0.{zeros}{digits}");
    }
    if digits.len() <= point {
        let zeros = "0".repeat(point - digits.len());
        return format!("{sign}{digits}{zeros}.0");
    }
    format!("{sign}{}.{}", &digits[..point], &digits[point..])
}

#[cfg(test)]
mod tests {
    use minijinja::{Environment, context};
    use serde_json::json;

    fn render(source: &str, value: &serde_json::Value) -> Result<String, minijinja::Error> {
        let mut environment = Environment::new();
        environment.add_filter("tojson", super::tojson);
        environment.render_str(source, context! { value => value })
    }

    #[test]
    fn matches_python_json_dumps() {
        let value =
            json!({"b": [1, 2.5, true, null], "a": {"é": "中 <&> \"q\"\n"}, "e": {}, "f": []});
        // Expected strings from CPython 3.11 `json.dumps` on the same
        // (key-sorted) object, as transformers' filter calls it.
        assert_eq!(
            render("{{ value | tojson }}", &value).unwrap(),
            r#"{"a": {"é": "中 <&> \"q\"\n"}, "b": [1, 2.5, true, null], "e": {}, "f": []}"#
        );
        assert_eq!(
            render("{{ value | tojson(ensure_ascii=True) }}", &value).unwrap(),
            r#"{"a": {"\u00e9": "\u4e2d <&> \"q\"\n"}, "b": [1, 2.5, true, null], "e": {}, "f": []}"#
        );
        assert_eq!(
            render("{{ value | tojson(indent=2) }}", &value).unwrap(),
            "{\n  \"a\": {\n    \"é\": \"中 <&> \\\"q\\\"\\n\"\n  },\n  \"b\": [\n    1,\n    2.5,\n    true,\n    null\n  ],\n  \"e\": {},\n  \"f\": []\n}"
        );
        assert_eq!(
            render("{{ value | tojson(ensure_ascii=False) }}", &json!("😀")).unwrap(),
            "\"😀\""
        );
        assert_eq!(
            render("{{ value | tojson(ensure_ascii=True) }}", &json!("😀")).unwrap(),
            r#""\ud83d\ude00""#
        );
    }

    #[test]
    fn floats_use_python_repr_spelling() {
        let value = json!([
            1e-05,
            1e16,
            1.5e-7,
            0.0001,
            2.5,
            1e15,
            123_456_789_012_345_680.0,
            -0.0,
            1.0,
            -12.5,
            5e-324,
            1.797_693_134_862_315_7e308
        ]);
        // python3 -c 'import json;print(json.dumps([1e-05, 1e16, 1.5e-7, 0.0001,
        //   2.5, 1e15, 123456789012345680.0, -0.0, 1.0, -12.5, 5e-324,
        //   1.7976931348623157e308]))'
        assert_eq!(
            render("{{ value | tojson }}", &value).unwrap(),
            "[1e-05, 1e+16, 1.5e-07, 0.0001, 2.5, 1000000000000000.0, \
             1.2345678901234568e+17, -0.0, 1.0, -12.5, 5e-324, 1.7976931348623157e+308]"
        );
    }

    #[test]
    fn template_maps_arrive_key_sorted() {
        let source =
            "{{ {'z': 1, 'a': 2} | tojson }} {{ {'z': 1, 'a': 2} | tojson(sort_keys=True) }}";
        assert_eq!(
            render(source, &json!(null)).unwrap(),
            r#"{"a": 2, "z": 1} {"a": 2, "z": 1}"#
        );
    }

    #[test]
    fn refuses_unsupported_arguments() {
        assert!(render("{{ value | tojson(separators=[',', ':']) }}", &json!(1)).is_err());
        assert!(render("{{ value | tojson(indent='x') }}", &json!(1)).is_err());
    }
}
