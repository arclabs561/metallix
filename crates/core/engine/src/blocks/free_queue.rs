//! Intrusive doubly linked list of free blocks.

/// Marks a missing neighbour.
const NIL: u32 = u32::MAX;

/// Free blocks in eviction order: the front is taken first.
///
/// Links live in arrays indexed by block, so push, pop and removal of an
/// arbitrary block (a prefix hit on a free cached block) are O(1) without
/// allocation.
#[derive(Clone, Debug)]
pub(super) struct FreeQueue {
    prev: Vec<u32>,
    next: Vec<u32>,
    queued: Vec<bool>,
    head: u32,
    tail: u32,
    len: usize,
}

impl FreeQueue {
    /// Creates a queue holding blocks `0..blocks` in index order.
    pub(super) fn full(blocks: u32) -> Self {
        let mut queue = Self {
            prev: vec![NIL; blocks as usize],
            next: vec![NIL; blocks as usize],
            queued: vec![false; blocks as usize],
            head: NIL,
            tail: NIL,
            len: 0,
        };
        for block in 0..blocks {
            queue.push_back(block);
        }
        queue
    }

    pub(super) const fn len(&self) -> usize {
        self.len
    }

    pub(super) fn contains(&self, block: u32) -> bool {
        self.queued[block as usize]
    }

    pub(super) fn push_back(&mut self, block: u32) {
        debug_assert!(!self.contains(block));
        let index = block as usize;
        self.prev[index] = self.tail;
        self.next[index] = NIL;
        if self.tail == NIL {
            self.head = block;
        } else {
            self.next[self.tail as usize] = block;
        }
        self.tail = block;
        self.queued[index] = true;
        self.len += 1;
    }

    pub(super) fn push_front(&mut self, block: u32) {
        debug_assert!(!self.contains(block));
        let index = block as usize;
        self.prev[index] = NIL;
        self.next[index] = self.head;
        if self.head == NIL {
            self.tail = block;
        } else {
            self.prev[self.head as usize] = block;
        }
        self.head = block;
        self.queued[index] = true;
        self.len += 1;
    }

    pub(super) fn pop_front(&mut self) -> Option<u32> {
        if self.head == NIL {
            return None;
        }
        let block = self.head;
        self.remove(block);
        Some(block)
    }

    /// Unlinks `block`; a block that is not queued is left alone.
    pub(super) fn remove(&mut self, block: u32) {
        let index = block as usize;
        if !self.queued[index] {
            return;
        }
        let (prev, next) = (self.prev[index], self.next[index]);
        if prev == NIL {
            self.head = next;
        } else {
            self.next[prev as usize] = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.prev[next as usize] = prev;
        }
        self.prev[index] = NIL;
        self.next[index] = NIL;
        self.queued[index] = false;
        self.len -= 1;
    }

    /// Iterates from the front (next to be taken) to the back.
    #[cfg(test)]
    pub(super) fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        let mut cursor = self.head;
        std::iter::from_fn(move || {
            if cursor == NIL {
                return None;
            }
            let block = cursor;
            cursor = self.next[block as usize];
            Some(block)
        })
    }
}
