//! Cubism 2 expression playback.

use std::sync::Arc;

use super::{
    json::{Moc2Expression, Moc2ExpressionBlend},
    motion::easing_sine,
    runtime::Moc2Runtime,
};

/// One active expression.
#[derive(Debug)]
struct Moc2ExpressionEntry {
    expression: Arc<Moc2Expression>,
    start_time: Option<f32>,
    fade_in_start_time: f32,
    targets: Vec<Option<usize>>,
}

impl Moc2ExpressionEntry {
    fn new(expression: Arc<Moc2Expression>, runtime: &mut Moc2Runtime) -> Self {
        let targets = expression
            .parameters()
            .iter()
            .map(|parameter| Some(runtime.ensure_parameter_index(parameter.id())))
            .collect();
        Self {
            expression,
            start_time: None,
            fade_in_start_time: 0.0,
            targets,
        }
    }

    fn fade_weight(&self, now: f32) -> f32 {
        let fade_in = self.expression.fade_in_ms();
        if fade_in <= 0.0 {
            1.0
        } else {
            easing_sine((now - self.fade_in_start_time) * 1000.0 / fade_in)
        }
    }

    fn update(&mut self, runtime: &mut Moc2Runtime, now: f32) {
        if self.start_time.is_none() {
            self.start_time = Some(now);
            self.fade_in_start_time = now;
        }

        let weight = self.fade_weight(now);
        // Cubism 2 applies expression parameters from last to first so that the
        // first entry wins when two entries target the same parameter.
        for (parameter, target) in self.expression.parameters().iter().zip(&self.targets).rev() {
            let Some(parameter_index) = *target else {
                continue;
            };
            match parameter.blend() {
                Moc2ExpressionBlend::Add => {
                    runtime.add_parameter_value(parameter_index, parameter.value(), weight)
                }
                Moc2ExpressionBlend::Multiply => {
                    runtime.multiply_parameter_value(parameter_index, parameter.value(), weight)
                }
                Moc2ExpressionBlend::Overwrite => {
                    runtime.write_parameter_weighted(parameter_index, parameter.value(), weight)
                }
            }
        }
    }
}

/// Applies the currently selected Cubism 2 expression.
///
/// Cubism 2 expressions run until they are replaced or stopped, and cross-fade
/// through their own `fade_in` value.
#[derive(Debug, Default)]
pub struct Moc2ExpressionManager {
    entries: Vec<Moc2ExpressionEntry>,
}

impl Moc2ExpressionManager {
    /// Creates an empty expression manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether no expression is active.
    pub fn is_finished(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of active expressions.
    pub fn active_expressions(&self) -> usize {
        self.entries.len()
    }

    /// Starts an expression, replacing any active one.
    pub(crate) fn start(
        &mut self,
        expression: Arc<Moc2Expression>,
        runtime: &mut Moc2Runtime,
        now: f32,
    ) {
        self.entries.clear();
        let mut entry = Moc2ExpressionEntry::new(expression, runtime);
        entry.start_time = Some(now);
        entry.fade_in_start_time = now;
        self.entries.push(entry);
    }

    /// Stops all active expressions.
    pub fn stop_all(&mut self) {
        self.entries.clear();
    }

    /// Applies every active expression.
    pub(crate) fn update(&mut self, runtime: &mut Moc2Runtime, now: f32) {
        for entry in &mut self.entries {
            entry.update(runtime, now);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::moc2::{Moc2Expression, Moc2ExpressionBlend};

    #[test]
    fn expressions_normalise_blend_types() {
        let expression = Moc2Expression::from_json_str(
            r#"{"fade_in": 0, "params": [
                {"id": "PARAM_A", "val": 2, "calc": "mult", "def": 4},
                {"id": "PARAM_B", "val": 3, "calc": "set"},
                {"id": "PARAM_C", "val": 5, "def": 2}
            ]}"#,
        )
        .unwrap();

        assert!((expression.parameters()[0].value() - 0.5).abs() < 1e-6);
        assert_eq!(expression.parameters()[1].value(), 3.0);
        assert_eq!(
            expression.parameters()[1].blend(),
            Moc2ExpressionBlend::Overwrite
        );
        assert_eq!(expression.parameters()[2].value(), 3.0);
    }
}
