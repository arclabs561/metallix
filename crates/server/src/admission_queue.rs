//! Per-model FIFO admission for the `mx serve` front process: the waiting
//! queue of a future iteration-level scheduler.
//!
//! Requests that arrive while a model is at its running limit wait here in
//! arrival order, up to a bounded depth and wait deadline, instead of being
//! refused. Three layers stay separate so batching can replace the run loop
//! without touching policy or state:
//!
//! - [`QueueSettings`]: admission policy (how many may wait, for how long).
//! - [`Scheduler`]: queue state and its transitions, with no locking or
//!   blocking; the caller holds the lock.
//! - [`ModelQueue`]: the blocking loop that parks a request until the
//!   scheduler starts it, its deadline passes, or its client leaves.
//!
//! The queue lives in the front process because that process owns the client
//! socket (so a queued client that leaves can be dropped), the idle stopper
//! (which already treats a counted request as activity), and on-demand startup
//! (so requests can wait while a child starts).

use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
    time::{Duration, Instant},
};

/// How often a waiting request checks whether its client is still there.
const PROBE_INTERVAL: Duration = Duration::from_millis(200);
const MAX_RETRY_AFTER_SECS: u64 = 60;

/// Admission policy for one model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct QueueSettings {
    /// Requests that may wait behind the running ones; 0 refuses any overlap.
    pub(crate) depth: usize,
    /// Longest a request waits for its turn before it is refused.
    pub(crate) wait: Duration,
    /// Requests the model runs at once: its batch size for a batching model,
    /// one otherwise.
    pub(crate) max_running: usize,
}

impl Default for QueueSettings {
    fn default() -> Self {
        Self {
            depth: 8,
            wait: Duration::from_secs(60),
            max_running: 1,
        }
    }
}

/// Why a request was not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Refusal {
    /// The queue already holds `depth` waiting requests.
    Full,
    /// The request waited the whole deadline without reaching the model.
    Expired,
    /// The client closed its connection while waiting.
    Gone,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Refused {
    pub(crate) refusal: Refusal,
    pub(crate) waited: Duration,
    /// Seconds a client should wait before retrying: the recent service time.
    pub(crate) retry_after_secs: u64,
}

/// Counters a metrics endpoint can read; names follow vLLM's where one exists.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "read by the metrics endpoint the scheduler adds")
)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct QueueMetrics {
    /// `vllm:num_requests_waiting`.
    pub(crate) num_requests_waiting: usize,
    /// `vllm:num_requests_running`.
    pub(crate) num_requests_running: usize,
    /// Sum and count of `vllm:request_queue_time_seconds` over started requests.
    pub(crate) request_queue_time_seconds_sum: f64,
    pub(crate) request_queue_time_seconds_count: u64,
    /// Requests refused because the queue was full, their wait expired, or
    /// their client left.
    pub(crate) refused_full: u64,
    pub(crate) refused_expired: u64,
    pub(crate) refused_gone: u64,
}

/// One waiting request.
struct Waiting {
    ticket: u64,
    arrived: Instant,
}

/// Waiting and running requests for one model, and the transitions between
/// them. Pure state: callers hold the lock and decide when to block.
struct Scheduler {
    waiting: VecDeque<Waiting>,
    running: usize,
    max_running: usize,
    next_ticket: u64,
    /// Moving average of how long started requests ran.
    service: Option<Duration>,
    metrics: QueueMetrics,
}

impl Scheduler {
    const fn new(max_running: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: 0,
            max_running,
            next_ticket: 0,
            service: None,
            metrics: QueueMetrics {
                num_requests_waiting: 0,
                num_requests_running: 0,
                request_queue_time_seconds_sum: 0.0,
                request_queue_time_seconds_count: 0,
                refused_full: 0,
                refused_expired: 0,
                refused_gone: 0,
            },
        }
    }

    /// Starts a request at once when nothing waits and a slot is free.
    fn start_now(&mut self) -> bool {
        let free = self.waiting.is_empty() && self.running < self.max_running;
        if free {
            self.started(Duration::ZERO);
        }
        free
    }

    /// Queues a request behind the others, or refuses it when `depth` already
    /// wait. Returns its ticket and the number waiting, itself included.
    fn enqueue(&mut self, depth: usize, arrived: Instant) -> Option<(u64, usize)> {
        if self.waiting.len() >= depth {
            self.metrics.refused_full += 1;
            return None;
        }
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.waiting.push_back(Waiting { ticket, arrived });
        Some((ticket, self.waiting.len()))
    }

    /// Starts `ticket` if it is first in line and a slot is free.
    fn try_start(&mut self, ticket: u64) -> bool {
        let Some(first) = self.waiting.front() else {
            return false;
        };
        if first.ticket != ticket || self.running >= self.max_running {
            return false;
        }
        let waited = first.arrived.elapsed();
        self.waiting.pop_front();
        self.started(waited);
        true
    }

    fn started(&mut self, waited: Duration) {
        self.running += 1;
        self.metrics.request_queue_time_seconds_sum += waited.as_secs_f64();
        self.metrics.request_queue_time_seconds_count += 1;
    }

    /// Removes a waiting request that will not run.
    fn cancel(&mut self, ticket: u64, refusal: Refusal) {
        self.waiting.retain(|waiting| waiting.ticket != ticket);
        match refusal {
            Refusal::Full => self.metrics.refused_full += 1,
            Refusal::Expired => self.metrics.refused_expired += 1,
            Refusal::Gone => self.metrics.refused_gone += 1,
        }
    }

    /// Frees the slot of a request that ran for `held`.
    fn finish(&mut self, held: Duration) {
        self.running -= 1;
        self.service = Some(self.service.map_or(held, |old| (old * 3 + held) / 4));
    }

    /// The average service time shared by the running slots: with several
    /// requests running, one slot frees that much sooner.
    fn retry_after_secs(&self) -> u64 {
        let slots = u128::try_from(self.max_running).unwrap_or(u128::MAX);
        self.service
            .map_or(1, |service| {
                u64::try_from(service.as_millis().div_ceil(1000 * slots)).unwrap_or(u64::MAX)
            })
            .clamp(1, MAX_RETRY_AFTER_SECS)
    }

    fn metrics(&self) -> QueueMetrics {
        QueueMetrics {
            num_requests_waiting: self.waiting.len(),
            num_requests_running: self.running,
            ..self.metrics
        }
    }
}

pub(crate) struct ModelQueue {
    settings: QueueSettings,
    scheduler: Mutex<Scheduler>,
    turn: Condvar,
}

/// A running slot on the model, held until the value is dropped.
pub(crate) struct Admitted<'a> {
    queue: &'a ModelQueue,
    started: Instant,
    /// Time spent waiting for the model.
    pub(crate) waited: Duration,
    /// Requests waiting at arrival, this one included; 0 when started at once.
    pub(crate) depth: usize,
}

impl Drop for Admitted<'_> {
    fn drop(&mut self) {
        let held = self.started.elapsed();
        self.queue.lock().finish(held);
        self.queue.turn.notify_all();
    }
}

impl ModelQueue {
    pub(crate) fn new(settings: QueueSettings) -> Self {
        Self {
            settings,
            scheduler: Mutex::new(Scheduler::new(settings.max_running.max(1))),
            turn: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Scheduler> {
        self.scheduler.lock().expect("queue lock")
    }

    /// Requests waiting for this model now, excluding running ones.
    pub(crate) fn waiting(&self) -> usize {
        self.lock().waiting.len()
    }

    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "read by the metrics endpoint the scheduler adds")
    )]
    pub(crate) fn metrics(&self) -> QueueMetrics {
        self.lock().metrics()
    }

    /// Waits in arrival order for a running slot. `gone` is polled while
    /// waiting, outside the queue lock, and reports whether the client left.
    pub(crate) fn admit(&self, mut gone: impl FnMut() -> bool) -> Result<Admitted<'_>, Refused> {
        let arrived = Instant::now();
        let deadline = arrived + self.settings.wait;
        let mut scheduler = self.lock();
        if scheduler.start_now() {
            return Ok(Admitted {
                queue: self,
                started: Instant::now(),
                waited: Duration::ZERO,
                depth: 0,
            });
        }
        let Some((ticket, depth)) = scheduler.enqueue(self.settings.depth, arrived) else {
            return Err(Refused {
                refusal: Refusal::Full,
                waited: Duration::ZERO,
                retry_after_secs: scheduler.retry_after_secs(),
            });
        };
        loop {
            if scheduler.try_start(ticket) {
                return Ok(Admitted {
                    queue: self,
                    started: Instant::now(),
                    waited: arrived.elapsed(),
                    depth,
                });
            }
            let now = Instant::now();
            let refusal = if now >= deadline {
                Some(Refusal::Expired)
            } else {
                let timeout = (deadline - now).min(PROBE_INTERVAL);
                scheduler = self
                    .turn
                    .wait_timeout(scheduler, timeout)
                    .expect("queue lock")
                    .0;
                drop(scheduler);
                let left = gone();
                scheduler = self.lock();
                left.then_some(Refusal::Gone)
            };
            if let Some(refusal) = refusal {
                scheduler.cancel(ticket, refusal);
                let retry_after_secs = scheduler.retry_after_secs();
                drop(scheduler);
                // The request behind this one may now be first in line.
                self.turn.notify_all();
                return Err(Refused {
                    refusal,
                    waited: arrived.elapsed(),
                    retry_after_secs,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, mpsc::channel},
        thread,
    };

    use super::*;

    fn queue(depth: usize, wait_ms: u64) -> Arc<ModelQueue> {
        Arc::new(ModelQueue::new(QueueSettings {
            depth,
            wait: Duration::from_millis(wait_ms),
            max_running: 1,
        }))
    }

    #[test]
    fn a_batching_model_runs_its_limit_at_once_and_queues_the_rest() {
        let mut scheduler = Scheduler::new(3);
        let now = Instant::now();
        for _ in 0..3 {
            assert!(scheduler.start_now(), "a free batch slot starts at once");
        }
        assert!(!scheduler.start_now(), "the fourth request must wait");
        let (ticket, depth) = scheduler.enqueue(8, now).expect("room to wait");
        assert_eq!(depth, 1);
        assert!(!scheduler.try_start(ticket));
        scheduler.finish(Duration::from_secs(6));
        assert!(
            scheduler.try_start(ticket),
            "a finished slot frees the waiter"
        );
        // Six seconds of service shared by three slots: retry in two.
        assert_eq!(scheduler.retry_after_secs(), 2);
    }

    /// Waits until `count` requests are queued, so arrival order is fixed.
    fn until_waiting(queue: &ModelQueue, count: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while queue.waiting() != count {
            assert!(Instant::now() < deadline, "queue never reached {count}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn waiters_are_admitted_one_at_a_time_in_arrival_order() {
        let queue = queue(8, 10_000);
        let first = queue.admit(|| false).expect("free model admits at once");
        assert_eq!((first.depth, first.waited), (0, Duration::ZERO));
        let (order, admitted) = channel();
        let mut waiters = Vec::new();
        for index in 0..5 {
            let waiter = Arc::clone(&queue);
            let order = order.clone();
            waiters.push(thread::spawn(move || {
                let admitted = waiter.admit(|| false).expect("admitted after waiting");
                // While held, no later waiter may run.
                order.send((index, admitted.depth)).unwrap();
                thread::sleep(Duration::from_millis(5));
            }));
            until_waiting(&queue, index + 1);
        }
        drop(first);
        for waiter in waiters {
            waiter.join().unwrap();
        }
        drop(order);
        let seen: Vec<(usize, usize)> = admitted.iter().collect();
        assert_eq!(seen, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        assert_eq!(queue.waiting(), 0);
        let metrics = queue.metrics();
        assert_eq!(metrics.request_queue_time_seconds_count, 6);
        assert!(
            metrics.request_queue_time_seconds_sum > 0.0,
            "waiters queued"
        );
        assert!(queue.admit(|| false).is_ok(), "the model is free again");
    }

    #[test]
    fn a_full_queue_refuses_at_once_with_a_retry_hint() {
        let queue = queue(1, 10_000);
        let held = queue.admit(|| false).unwrap();
        let waiter = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || queue.admit(|| false).map(|admitted| admitted.depth))
        };
        until_waiting(&queue, 1);
        let refused = queue.admit(|| false).err().expect("queue is full");
        assert_eq!(refused.refusal, Refusal::Full);
        assert_eq!(refused.waited, Duration::ZERO);
        assert_eq!(refused.retry_after_secs, 1, "no service time measured yet");
        drop(held);
        assert_eq!(waiter.join().unwrap(), Ok(1));
    }

    #[test]
    fn depth_zero_refuses_any_overlap() {
        let queue = queue(0, 10_000);
        let _held = queue.admit(|| false).unwrap();
        assert_eq!(
            queue.admit(|| false).err().map(|refused| refused.refusal),
            Some(Refusal::Full)
        );
    }

    #[test]
    fn a_waiter_past_its_deadline_is_refused_and_the_next_moves_up() {
        let queue = queue(8, 150);
        let held = queue.admit(|| false).unwrap();
        let started = Instant::now();
        let refused = queue.admit(|| false).err().expect("deadline passes");
        assert_eq!(refused.refusal, Refusal::Expired);
        assert!(refused.waited >= Duration::from_millis(150));
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(queue.waiting(), 0, "the expired request left the queue");
        drop(held);
        assert_eq!(
            queue.admit(|| false).map(|admitted| admitted.depth).ok(),
            Some(0)
        );
    }

    #[test]
    fn a_departed_client_leaves_the_queue_without_being_admitted() {
        let queue = queue(8, 10_000);
        let held = queue.admit(|| false).unwrap();
        let (leave, left) = channel::<()>();
        let departing = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || queue.admit(|| left.try_recv().is_ok()).err())
        };
        until_waiting(&queue, 1);
        let behind = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || queue.admit(|| false).map(|admitted| admitted.depth))
        };
        until_waiting(&queue, 2);
        leave.send(()).unwrap();
        let refused = departing.join().unwrap().expect("departed client refused");
        assert_eq!(refused.refusal, Refusal::Gone);
        until_waiting(&queue, 1);
        drop(held);
        // The request behind the departed one runs next.
        assert_eq!(behind.join().unwrap(), Ok(2));
    }

    #[test]
    fn retry_hint_follows_measured_service_time() {
        let queue = queue(0, 1000);
        let held = queue.admit(|| false).unwrap();
        thread::sleep(Duration::from_millis(1100));
        drop(held);
        let _held = queue.admit(|| false).unwrap();
        assert_eq!(queue.admit(|| false).err().unwrap().retry_after_secs, 2);
    }

    #[test]
    fn metrics_count_waiting_running_queue_time_and_refusals() {
        let queue = queue(1, 150);
        assert_eq!(queue.metrics(), QueueMetrics::default());
        let held = queue.admit(|| false).unwrap();
        let waiter = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || queue.admit(|| false).err().map(|refused| refused.refusal))
        };
        until_waiting(&queue, 1);
        let metrics = queue.metrics();
        assert_eq!(
            (metrics.num_requests_waiting, metrics.num_requests_running),
            (1, 1)
        );
        assert_eq!(
            queue.admit(|| false).err().map(|refused| refused.refusal),
            Some(Refusal::Full)
        );
        assert_eq!(waiter.join().unwrap(), Some(Refusal::Expired));
        drop(held);
        let started = queue.admit(|| false).unwrap();
        drop(started);
        let metrics = queue.metrics();
        assert_eq!(
            (
                metrics.num_requests_waiting,
                metrics.num_requests_running,
                metrics.request_queue_time_seconds_count,
                metrics.refused_full,
                metrics.refused_expired,
                metrics.refused_gone,
            ),
            (0, 0, 2, 1, 1, 0)
        );
        // Both started requests found the model free, so neither queued.
        assert!(metrics.request_queue_time_seconds_sum.abs() < f64::EPSILON);
    }
}
