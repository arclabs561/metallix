//! V4.1-shaped request composition over live numerical operands.
//!
//! A request is startup, a validated list of typed layers, and the final head.
//! Each layer is one of six fixed kinds (window-only, ratio-two owner or
//! consumer, ratio-one owner, candidate indexer or consumer) with an optional
//! Engram before its block. [`RequestSession::step`] walks that list in one
//! visible loop; consumers borrow the latest same-step publication of the
//! producer their attention layout names. The reduced five-block fixture and
//! the 40-layer checkpoint schedule are two lists of the same kinds. This
//! module has no checkpoint loader, fixture decoder, callback graph, or
//! general graph interpreter.

mod definition;
mod error;
mod model;
mod output;
mod session;

pub use definition::{
    BlockDefinition, EngramDefinition, LayerFourDefinition, LayerKind, LayerOneDefinition,
    LayerThreeDefinition, ReusedAttentionDefinition, ScheduledLayer, StartupDefinition,
};
pub use error::{RequestError, ScheduleError};
pub use model::{HeadPositions, RequestModel};
pub use output::{LayerStepOutput, RequestStepOutput, ScheduledAttentionOutput};
pub use session::{RequestSession, StepSources};

const MAX_REQUEST_ELEMENTS: usize = 1 << 20;

fn reserve<T>(elements: usize, field: &'static str) -> Result<Vec<T>, RequestError> {
    if elements > MAX_REQUEST_ELEMENTS {
        return Err(RequestError::ElementLimit { elements });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| RequestError::Allocation { field, elements })?;
    Ok(values)
}
