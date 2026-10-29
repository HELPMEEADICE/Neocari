use serde::{Deserialize, Deserializer};

use crate::{Error, Result};

const FORMAT: &str = "exp3.json";
/// Default fade-in time used when an expression omits `FadeInTime`.
pub const DEFAULT_EXPRESSION_FADE_IN_TIME: f32 = 1.0;
/// Default fade-out time used when an expression omits `FadeOutTime`.
pub const DEFAULT_EXPRESSION_FADE_OUT_TIME: f32 = 1.0;

#[derive(Debug, Clone, PartialEq)]
/// Parsed Cubism `exp3.json` expression data.
pub struct Expression3 {
    kind: String,
    fade_in_time: Option<f32>,
    fade_out_time: Option<f32>,
    parameters: Vec<ExpressionParameter>,
}

impl Expression3 {
    /// Parses an expression JSON document from a string.
    pub fn from_json_str(source: &str) -> Result<Self> {
        let raw: RawExpression3 =
            serde_json::from_str(source).map_err(|error| Error::InvalidJson {
                format: FORMAT,
                message: error.to_string(),
            })?;

        Ok(Self {
            kind: raw.kind,
            fade_in_time: raw.fade_in_time,
            fade_out_time: raw.fade_out_time,
            parameters: raw.parameters,
        })
    }

    /// Returns the expression type string from the JSON file.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Returns the optional fade-in time declared by the file.
    pub fn fade_in_time(&self) -> Option<f32> {
        self.fade_in_time
    }

    /// Returns the fade-in time with Cubism defaults applied.
    pub fn resolved_fade_in_time(&self) -> f32 {
        resolved_expression_fade_in_time(self.fade_in_time)
    }

    /// Returns the optional fade-out time declared by the file.
    pub fn fade_out_time(&self) -> Option<f32> {
        self.fade_out_time
    }

    /// Returns the fade-out time with Cubism defaults applied.
    pub fn resolved_fade_out_time(&self) -> f32 {
        resolved_expression_fade_out_time(self.fade_out_time)
    }

    /// Returns parameter operations in this expression.
    pub fn parameters(&self) -> &[ExpressionParameter] {
        &self.parameters
    }
}

/// Resolves an optional expression fade-in time to a non-negative value.
pub fn resolved_expression_fade_in_time(fade_in_time: Option<f32>) -> f32 {
    fade_in_time
        .unwrap_or(DEFAULT_EXPRESSION_FADE_IN_TIME)
        .max(0.0)
}

/// Resolves an optional expression fade-out time to a non-negative value.
pub fn resolved_expression_fade_out_time(fade_out_time: Option<f32>) -> f32 {
    fade_out_time
        .unwrap_or(DEFAULT_EXPRESSION_FADE_OUT_TIME)
        .max(0.0)
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct RawExpression3 {
    #[serde(rename = "Type")]
    kind: String,
    #[serde(rename = "FadeInTime", default)]
    fade_in_time: Option<f32>,
    #[serde(rename = "FadeOutTime", default)]
    fade_out_time: Option<f32>,
    #[serde(rename = "Parameters", default)]
    parameters: Vec<ExpressionParameter>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
/// One parameter operation in an expression.
pub struct ExpressionParameter {
    #[serde(rename = "Id")]
    id: String,
    #[serde(rename = "Value")]
    value: f32,
    #[serde(
        rename = "Blend",
        default,
        deserialize_with = "deserialize_expression_blend"
    )]
    blend: ExpressionBlend,
}

impl ExpressionParameter {
    /// Returns the target parameter id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the value used by this parameter operation.
    pub fn value(&self) -> f32 {
        self.value
    }

    /// Returns how the value is blended with the current parameter.
    pub fn blend(&self) -> ExpressionBlend {
        self.blend
    }
}

/// Applies one expression parameter operation to a current value.
pub fn apply_expression_parameter(
    current: f32,
    parameter: &ExpressionParameter,
    weight: f32,
) -> f32 {
    apply_expression_blend(current, parameter.value, parameter.blend, weight)
}

/// Applies an expression blend operation to a current value.
pub fn apply_expression_blend(
    current: f32,
    value: f32,
    blend: ExpressionBlend,
    weight: f32,
) -> f32 {
    match blend {
        ExpressionBlend::Add => current + (value * weight),
        ExpressionBlend::Multiply => current * (1.0 + (value - 1.0) * weight),
        ExpressionBlend::Overwrite if weight == 1.0 => value,
        ExpressionBlend::Overwrite => (current * (1.0 - weight)) + (value * weight),
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
/// Blend mode for an expression parameter.
pub enum ExpressionBlend {
    /// Adds the expression value to the current parameter.
    #[default]
    Add,
    /// Multiplies the current parameter by the expression value.
    Multiply,
    /// Interpolates from the current parameter to the expression value.
    Overwrite,
}

impl ExpressionBlend {
    fn from_raw(value: Option<&str>) -> Self {
        match value {
            Some("Multiply") => Self::Multiply,
            Some("Overwrite") => Self::Overwrite,
            _ => Self::Add,
        }
    }
}

fn deserialize_expression_blend<'de, D>(
    deserializer: D,
) -> std::result::Result<ExpressionBlend, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error as _;

    // Serialized Unity expressions use CubismParameterBlendMode:
    // 0 = Override, 1 = Additive, 2 = Multiply. Standard exp3.json uses names.
    // This differs from the native SDK's expression blend enum ordering.
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_json::Value::Null) => Ok(ExpressionBlend::Add),
        Some(serde_json::Value::String(name)) => Ok(ExpressionBlend::from_raw(Some(&name))),
        Some(serde_json::Value::Number(number)) => match number.as_u64() {
            Some(0) => Ok(ExpressionBlend::Overwrite),
            Some(1) => Ok(ExpressionBlend::Add),
            Some(2) => Ok(ExpressionBlend::Multiply),
            _ => Err(D::Error::custom(
                "expression Blend must be 0 (overwrite), 1 (add) or 2 (multiply)",
            )),
        },
        Some(_) => Err(D::Error::custom(
            "expression Blend must be an integer or a blend name",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blends(source: &str) -> Vec<ExpressionBlend> {
        Expression3::from_json_str(source)
            .unwrap()
            .parameters()
            .iter()
            .map(ExpressionParameter::blend)
            .collect()
    }

    #[test]
    fn integer_blend_codes_follow_the_unity_serialized_enum() {
        let parsed = blends(
            r#"{"Type":"Live2D Expression","Parameters":[
                {"Id":"A","Value":1.0,"Blend":0},
                {"Id":"B","Value":2.0,"Blend":1},
                {"Id":"C","Value":3.0,"Blend":2},
                {"Id":"D","Value":4.0}
            ]}"#,
        );
        assert_eq!(
            parsed,
            [
                ExpressionBlend::Overwrite,
                ExpressionBlend::Add,
                ExpressionBlend::Multiply,
                ExpressionBlend::Add,
            ]
        );
    }

    #[test]
    fn integer_and_named_blends_apply_identically_during_fades() {
        for (code, name) in [(0, "Overwrite"), (1, "Add"), (2, "Multiply")] {
            let numeric = Expression3::from_json_str(&format!(
                r#"{{"Type":"Live2D Expression","Parameters":[{{"Id":"A","Value":0.25,"Blend":{code}}}]}}"#
            ))
            .unwrap();
            let named = Expression3::from_json_str(&format!(
                r#"{{"Type":"Live2D Expression","Parameters":[{{"Id":"A","Value":0.25,"Blend":"{name}"}}]}}"#
            ))
            .unwrap();
            for current in [0.0, 0.5, 1.0] {
                for weight in [0.0, 0.25, 0.5, 1.0] {
                    assert_eq!(
                        apply_expression_parameter(current, &numeric.parameters()[0], weight),
                        apply_expression_parameter(current, &named.parameters()[0], weight),
                        "Blend {code} ({name}), current {current}, weight {weight}"
                    );
                }
            }
        }
    }

    #[test]
    fn named_blends_still_parse() {
        let parsed = blends(
            r#"{"Type":"Live2D Expression","Parameters":[
                {"Id":"A","Value":1.0,"Blend":"Overwrite"}
            ]}"#,
        );
        assert_eq!(parsed, [ExpressionBlend::Overwrite]);
    }

    #[test]
    fn an_unknown_blend_code_is_rejected() {
        let error = Expression3::from_json_str(
            r#"{"Type":"Live2D Expression","Parameters":[{"Id":"A","Value":1.0,"Blend":7}]}"#,
        );
        assert!(error.is_err());
    }
}
