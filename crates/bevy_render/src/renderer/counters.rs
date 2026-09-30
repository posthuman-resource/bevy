//! Renderer activity counters: how many GPU objects and uploads the renderer
//! asked wgpu for since they were last read.
//!
//! Every [`RenderDevice`](super::RenderDevice) creation and every
//! [`RenderQueue`](super::RenderQueue) write adds to one process-wide set of
//! relaxed atomics, which [`RenderCounters::take`] reads and resets. A
//! profiler reads them once per frame to name what a slow frame did (a burst
//! of pipeline creation, a large upload, thousands of bind groups). Calls made
//! on the raw `wgpu` device or queue, bypassing these wrappers, are not
//! counted, so each number is a lower bound.

use core::sync::atomic::{AtomicU64, Ordering};

/// The process-wide counters. See the module documentation.
pub static RENDER_COUNTERS: RenderCounters = RenderCounters::new();

/// Relaxed atomic counters, one per kind of renderer activity.
pub struct RenderCounters {
    bind_groups: AtomicU64,
    buffers_created: AtomicU64,
    buffer_bytes_created: AtomicU64,
    buffer_writes: AtomicU64,
    buffer_write_bytes: AtomicU64,
    texture_writes: AtomicU64,
    texture_write_bytes: AtomicU64,
    textures_created: AtomicU64,
    pipelines_created: AtomicU64,
    shader_modules_created: AtomicU64,
}

/// One reading of [`RenderCounters`]: the activity since the previous reading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderCountersFrame {
    /// Bind groups created.
    pub bind_groups: u64,
    /// Buffers created, and their total size in bytes.
    pub buffers_created: u64,
    pub buffer_bytes_created: u64,
    /// Queue buffer writes (`write_buffer` and `write_buffer_with`), and bytes.
    pub buffer_writes: u64,
    pub buffer_write_bytes: u64,
    /// Queue texture writes, including textures created with data, and bytes.
    pub texture_writes: u64,
    pub texture_write_bytes: u64,
    /// Textures created.
    pub textures_created: u64,
    /// Render, compute and mesh pipelines created.
    pub pipelines_created: u64,
    /// Shader modules created.
    pub shader_modules_created: u64,
}

impl RenderCounters {
    const fn new() -> Self {
        Self {
            bind_groups: AtomicU64::new(0),
            buffers_created: AtomicU64::new(0),
            buffer_bytes_created: AtomicU64::new(0),
            buffer_writes: AtomicU64::new(0),
            buffer_write_bytes: AtomicU64::new(0),
            texture_writes: AtomicU64::new(0),
            texture_write_bytes: AtomicU64::new(0),
            textures_created: AtomicU64::new(0),
            pipelines_created: AtomicU64::new(0),
            shader_modules_created: AtomicU64::new(0),
        }
    }

    /// The activity since the previous call, resetting every counter.
    pub fn take(&self) -> RenderCountersFrame {
        let t = |c: &AtomicU64| c.swap(0, Ordering::Relaxed);
        RenderCountersFrame {
            bind_groups: t(&self.bind_groups),
            buffers_created: t(&self.buffers_created),
            buffer_bytes_created: t(&self.buffer_bytes_created),
            buffer_writes: t(&self.buffer_writes),
            buffer_write_bytes: t(&self.buffer_write_bytes),
            texture_writes: t(&self.texture_writes),
            texture_write_bytes: t(&self.texture_write_bytes),
            textures_created: t(&self.textures_created),
            pipelines_created: t(&self.pipelines_created),
            shader_modules_created: t(&self.shader_modules_created),
        }
    }

    #[inline]
    pub(crate) fn bind_group(&self) {
        self.bind_groups.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn buffer_created(&self, bytes: u64) {
        self.buffers_created.fetch_add(1, Ordering::Relaxed);
        self.buffer_bytes_created
            .fetch_add(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn buffer_write(&self, bytes: u64) {
        self.buffer_writes.fetch_add(1, Ordering::Relaxed);
        self.buffer_write_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn texture_write(&self, bytes: u64) {
        self.texture_writes.fetch_add(1, Ordering::Relaxed);
        self.texture_write_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn texture_created(&self) {
        self.textures_created.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn pipeline(&self) {
        self.pipelines_created.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub(crate) fn shader_module(&self) {
        self.shader_modules_created.fetch_add(1, Ordering::Relaxed);
    }
}
