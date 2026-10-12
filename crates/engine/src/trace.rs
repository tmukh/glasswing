/// Intermediate tensors recorded during a forward pass, in the order they were computed.
/// Names match the Python reference (`layers.0.q`, `logits`, ...). This is what the parity
/// test compares now, and what the visualization will draw later.
#[derive(Debug, Default)]
pub struct Trace {
    pub tensors: Vec<TracedTensor>,
}

#[derive(Debug, Clone)]
pub struct TracedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Trace {
    pub fn record(&mut self, name: &str, shape: &[usize], data: &[f32]) {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
        self.tensors.push(TracedTensor {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.to_vec(),
        });
    }

    pub fn get(&self, name: &str) -> Option<&TracedTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
}
