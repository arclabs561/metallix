//! Prompts whose placeholder tokens stand for encoder output rows.
//!
//! An audio or image model puts one placeholder token (`<|audio_pad|>`,
//! `<|image_pad|>`) in the prompt per encoder output row; the decoder's input
//! at those positions is the row, not the placeholder's token embedding.
//! These types say where the rows go and what they are. They are generic
//! over the row type, so they hold no tensors of their own and every
//! decoder crate can share them.
//!
//! Positions are not part of a span. Each model derives them from the span
//! kind: audio and text advance one position per token, while an image's
//! positions follow its family's multi-axis rule. A caller therefore cannot
//! give a span another family's positions.

#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc)]

use std::ops::Range;

/// What a span's rows encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpanKind {
    /// Audio encoder frames, one per placeholder, in time order.
    Audio,
    /// Image patches after merging, row-major over a `grid_h` by `grid_w`
    /// grid.
    Image {
        /// Rows of merged patches.
        grid_h: u32,
        /// Columns of merged patches.
        grid_w: u32,
    },
}

/// One run of placeholder positions and the rows that replace them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedSpan<R> {
    start: usize,
    len: usize,
    kind: SpanKind,
    digest: [u8; 32],
    rows: R,
}

impl<R> EmbeddedSpan<R> {
    /// A span replacing positions `start..start + len` with `rows`.
    ///
    /// `len` is authoritative: the decoder checks that `rows` holds exactly
    /// `len` rows of its hidden width. `digest` identifies the media and its
    /// preprocessing (for example SHA-256 of the decoded samples or pixels
    /// and the preprocessor settings); the caller computes it, and anything
    /// keyed on the prompt, such as a prefix cache, must include it rather
    /// than treating the placeholders as ordinary tokens.
    #[must_use]
    pub const fn new(start: usize, len: usize, kind: SpanKind, digest: [u8; 32], rows: R) -> Self {
        Self {
            start,
            len,
            kind,
            digest,
            rows,
        }
    }

    /// The first replaced position.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Number of replaced positions and rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the span is empty, which [`EmbeddedPrompt::new`] refuses.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// One past the last replaced position.
    ///
    /// # Panics
    ///
    /// Panics if the unvalidated span's end overflows `usize`. Spans admitted
    /// by [`EmbeddedPrompt::new`] cannot overflow.
    #[must_use]
    pub const fn end(&self) -> usize {
        match self.start.checked_add(self.len) {
            Some(end) => end,
            None => panic!("embedded span end overflows usize"),
        }
    }

    /// What the rows encode.
    #[must_use]
    pub const fn kind(&self) -> SpanKind {
        self.kind
    }

    /// The media digest.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// The replacement rows.
    #[must_use]
    pub const fn rows(&self) -> &R {
        &self.rows
    }
}

/// Prompt ids, placeholders included, with validated spans.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedPrompt<'a, R> {
    ids: &'a [i32],
    spans: Vec<EmbeddedSpan<R>>,
}

impl<'a, R> EmbeddedPrompt<'a, R> {
    /// Checks that the spans are non-empty, ascending, disjoint and inside
    /// `ids`, and that an image span has one row per grid cell.
    ///
    /// # Errors
    ///
    /// Returns [`SpanError`] for the first span that fails.
    pub fn new(ids: &'a [i32], spans: Vec<EmbeddedSpan<R>>) -> Result<Self, SpanError> {
        let mut free_from = 0;
        for (index, span) in spans.iter().enumerate() {
            let reject = |reason| SpanError { index, reason };
            if span.len == 0 {
                return Err(reject(SpanProblem::Empty));
            }
            if span.start < free_from {
                return Err(reject(SpanProblem::Overlap));
            }
            let end = span
                .start
                .checked_add(span.len)
                .filter(|&end| end <= ids.len())
                .ok_or_else(|| reject(SpanProblem::PastEnd))?;
            if let SpanKind::Image { grid_h, grid_w } = span.kind {
                let cells = usize::try_from(u64::from(grid_h) * u64::from(grid_w)).ok();
                if cells != Some(span.len) {
                    return Err(reject(SpanProblem::GridMismatch));
                }
            }
            free_from = end;
        }
        Ok(Self { ids, spans })
    }

    /// Every prompt id, placeholders included.
    #[must_use]
    pub const fn ids(&self) -> &'a [i32] {
        self.ids
    }

    /// The spans, in position order.
    #[must_use]
    pub fn spans(&self) -> &[EmbeddedSpan<R>] {
        &self.spans
    }

    /// The part of the prompt in `positions`, for prefilling in chunks.
    /// A span that crosses the chunk's edges contributes only its rows
    /// inside the chunk. An empty range yields no span parts.
    ///
    /// # Panics
    ///
    /// Panics if `positions` is not inside the prompt.
    #[must_use]
    pub fn chunk(&self, positions: Range<usize>) -> PromptChunk<'_, R> {
        assert!(
            positions.start <= positions.end && positions.end <= self.ids.len(),
            "chunk {positions:?} is outside a {}-token prompt",
            self.ids.len()
        );
        let spans = self
            .spans
            .iter()
            .filter(|span| {
                !positions.is_empty() && span.start < positions.end && span.end() > positions.start
            })
            .map(|span| {
                let first = span.start.max(positions.start);
                let last = span.end().min(positions.end);
                ChunkSpan {
                    start: first - positions.start,
                    row_offset: first - span.start,
                    len: last - first,
                    span,
                }
            })
            .collect();
        PromptChunk {
            ids: &self.ids[positions.clone()],
            offset: positions.start,
            spans,
        }
    }
}

/// A contiguous part of an [`EmbeddedPrompt`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptChunk<'p, R> {
    ids: &'p [i32],
    offset: usize,
    spans: Vec<ChunkSpan<'p, R>>,
}

impl<'p, R> PromptChunk<'p, R> {
    /// The chunk's ids.
    #[must_use]
    pub const fn ids(&self) -> &'p [i32] {
        self.ids
    }

    /// The chunk's first position in the whole prompt.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// The parts of spans inside the chunk, in position order.
    #[must_use]
    pub fn spans(&self) -> &[ChunkSpan<'p, R>] {
        &self.spans
    }
}

/// The part of one span inside a chunk: chunk positions
/// `start..start + len` take the span's rows `row_offset..row_offset + len`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkSpan<'p, R> {
    /// First position, relative to the chunk.
    pub start: usize,
    /// First row of the span used here.
    pub row_offset: usize,
    /// Positions and rows used here.
    pub len: usize,
    /// The whole span, for its kind, digest and rows.
    pub span: &'p EmbeddedSpan<R>,
}

/// Why [`EmbeddedPrompt::new`] refused a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("embedded span {index}: {reason}")]
pub struct SpanError {
    /// Index of the span in the list.
    pub index: usize,
    /// What is wrong with it.
    pub reason: SpanProblem,
}

/// What is wrong with a span.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SpanProblem {
    /// The span has no rows.
    #[error("a span needs at least one row")]
    Empty,
    /// The span starts before the previous one ends.
    #[error("spans must be ascending and disjoint")]
    Overlap,
    /// The span ends past the prompt.
    #[error("the span ends past the prompt")]
    PastEnd,
    /// An image span's length is not `grid_h * grid_w`.
    #[error("an image span needs grid_h * grid_w rows")]
    GridMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const IDS: [i32; 12] = [0; 12];

    fn audio(start: usize, len: usize) -> EmbeddedSpan<Vec<usize>> {
        // Rows are labelled by span position so a test can follow them.
        EmbeddedSpan::new(start, len, SpanKind::Audio, [7; 32], (0..len).collect())
    }

    #[test]
    fn accepts_ordered_spans_and_refuses_bad_ones() {
        assert!(EmbeddedPrompt::new(&IDS, vec![audio(1, 3), audio(4, 2)]).is_ok());
        let problem = |spans| EmbeddedPrompt::new(&IDS, spans).unwrap_err();
        assert_eq!(
            problem(vec![audio(1, 3), audio(3, 2)]),
            SpanError {
                index: 1,
                reason: SpanProblem::Overlap
            }
        );
        assert_eq!(problem(vec![audio(10, 3)]).reason, SpanProblem::PastEnd);
        assert_eq!(problem(vec![audio(0, 0)]).reason, SpanProblem::Empty);
        let image = EmbeddedSpan::new(
            0,
            6,
            SpanKind::Image {
                grid_h: 2,
                grid_w: 2,
            },
            [0; 32],
            (),
        );
        assert_eq!(
            EmbeddedPrompt::new(&IDS, vec![image]).unwrap_err().reason,
            SpanProblem::GridMismatch
        );
    }

    #[test]
    fn a_chunk_takes_only_its_part_of_a_straddling_span() {
        let prompt = EmbeddedPrompt::new(&IDS, vec![audio(2, 6)]).unwrap();
        let chunk = prompt.chunk(5..10);
        assert_eq!((chunk.offset(), chunk.ids().len()), (5, 5));
        let [part] = chunk.spans() else {
            panic!("one span part expected")
        };
        assert_eq!((part.start, part.row_offset, part.len), (0, 3, 3));
        assert!(prompt.chunk(8..12).spans().is_empty());
    }

    #[test]
    fn an_empty_chunk_inside_a_span_has_no_rows() {
        let prompt = EmbeddedPrompt::new(&IDS, vec![audio(2, 6)]).unwrap();
        let chunk = prompt.chunk(4..4);
        assert!(chunk.ids().is_empty());
        assert!(chunk.spans().is_empty());
    }

    #[test]
    fn overflowing_span_is_rejected_before_chunking() {
        let span = EmbeddedSpan::new(usize::MAX, 2, SpanKind::Audio, [0; 32], ());
        assert_eq!(
            EmbeddedPrompt::new(&IDS, vec![span]).unwrap_err().reason,
            SpanProblem::PastEnd
        );
    }

    proptest! {
        // However a prompt is cut into chunks, every span row lands at its
        // own position exactly once.
        #[test]
        fn chunks_place_every_row_exactly_once(
            lengths in proptest::collection::vec((0_usize..3, 1_usize..5), 0..4),
            cuts in proptest::collection::btree_set(1_usize..40, 0..6),
        ) {
            let mut spans = Vec::new();
            let mut position = 0;
            for (gap, len) in lengths {
                spans.push(audio(position + gap, len));
                position += gap + len;
            }
            let ids = vec![0; position + 2];
            let prompt = EmbeddedPrompt::new(&ids, spans).unwrap();
            let mut edges: Vec<usize> = cuts.into_iter().filter(|&cut| cut < ids.len()).collect();
            edges.insert(0, 0);
            edges.push(ids.len());
            let mut placed = vec![None; ids.len()];
            for window in edges.windows(2) {
                let chunk = prompt.chunk(window[0]..window[1]);
                for part in chunk.spans() {
                    for step in 0..part.len {
                        let at = chunk.offset() + part.start + step;
                        let row = part.span.rows()[part.row_offset + step];
                        prop_assert!(placed[at].is_none());
                        placed[at] = Some((part.span.start(), row));
                    }
                }
            }
            for span in prompt.spans() {
                for row in 0..span.len() {
                    prop_assert_eq!(placed[span.start() + row], Some((span.start(), row)));
                }
            }
            let covered = prompt.spans().iter().map(EmbeddedSpan::len).sum::<usize>();
            prop_assert_eq!(placed.iter().flatten().count(), covered);
        }
    }
}
