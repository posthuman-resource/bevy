//! Opt-in debug groups and diagnostic spans around the draws of each pipeline group in a render
//! phase.

use alloc::{borrow::Cow, boxed::Box};
use core::any::{Any, TypeId};

use bevy_ecs::{resource::Resource, world::World};
use bevy_material::descriptor::CachedRenderPipelineId;
use bevy_platform::collections::HashMap;

use super::{CachedRenderPipelinePhaseItem, PhaseItem, TrackedRenderPass};
use crate::diagnostic::{DiagnosticsRecorder, RecordDiagnostics};
use crate::render_resource::PipelineCache;

/// Which render phases group their draws by pipeline, for diagnostics and graphics debuggers.
///
/// A phase enabled here wraps each run of consecutive draws whose pipelines share a name
/// ([`PipelineCache::render_pipeline_name`], the shader's asset path) in a debug group of that
/// name, which a graphics debugger shows. With
/// [`RenderDiagnosticsPlugin`](crate::diagnostic::RenderDiagnosticsPlugin) the run is also a pass
/// span nested in the pass's own: its GPU time and pipeline statistics are recorded as
/// `render/<pass>/<pipeline name>/...`, and a name drawn in several runs records several values in
/// a frame. Spans inside a pass need [`wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES`] for GPU
/// time.
///
/// ```ignore
/// render_app.world_mut().resource_mut::<PhaseDrawSpans>().enable::<Opaque3d>();
/// ```
#[derive(Resource, Default)]
pub struct PhaseDrawSpans {
    phases: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

type PipelineOf<I> = fn(&I) -> CachedRenderPipelineId;

impl PhaseDrawSpans {
    /// Time the draws of phase `I` by pipeline.
    pub fn enable<I: CachedRenderPipelinePhaseItem>(&mut self) -> &mut Self {
        let pipeline_of: PipelineOf<I> = <I as CachedRenderPipelinePhaseItem>::cached_pipeline;
        self.phases.insert(TypeId::of::<I>(), Box::new(pipeline_of));
        self
    }

    /// Stop timing the draws of phase `I`.
    pub fn disable<I: PhaseItem>(&mut self) -> &mut Self {
        self.phases.remove(&TypeId::of::<I>());
        self
    }

    fn pipeline_of<I: PhaseItem>(&self) -> Option<PipelineOf<I>> {
        self.phases
            .get(&TypeId::of::<I>())?
            .downcast_ref::<PipelineOf<I>>()
            .copied()
    }
}

/// The open draw group of one phase's render, when that phase is grouped.
pub(super) struct DrawSpans<'w, I> {
    recorder: Option<&'w DiagnosticsRecorder>,
    cache: &'w PipelineCache,
    pipeline_of: PipelineOf<I>,
    open: Option<(CachedRenderPipelineId, Cow<'static, str>)>,
}

impl<'w, I: PhaseItem> DrawSpans<'w, I> {
    /// `None` unless phase `I` is enabled in [`PhaseDrawSpans`].
    pub(super) fn new(world: &'w World) -> Option<Self> {
        let pipeline_of = world.get_resource::<PhaseDrawSpans>()?.pipeline_of::<I>()?;
        Some(Self {
            recorder: world.get_resource::<DiagnosticsRecorder>(),
            cache: world.get_resource::<PipelineCache>()?,
            pipeline_of,
            open: None,
        })
    }

    /// Called before `item` is drawn: opens its group's span unless that group is open.
    pub(super) fn before(&mut self, pass: &mut TrackedRenderPass<'_>, item: &I) {
        let id = (self.pipeline_of)(item);
        if self.open.as_ref().is_some_and(|(open, _)| *open == id) {
            return;
        }
        let name = self.cache.render_pipeline_name(id);
        if let Some((open_id, open_name)) = &mut self.open
            && *open_name == name
        {
            *open_id = id;
            return;
        }
        self.end(pass);
        pass.push_debug_group(&name);
        self.recorder.begin_pass_span(pass, name.clone());
        self.open = Some((id, name));
    }

    /// Closes the open group's span, if any.
    pub(super) fn end(&mut self, pass: &mut TrackedRenderPass<'_>) {
        if self.open.take().is_some() {
            self.recorder.end_pass_span(pass);
            pass.pop_debug_group();
        }
    }
}

/// Calls [`DrawSpans::before`] when the phase is grouped.
pub(super) fn before<I: PhaseItem>(
    spans: &mut Option<DrawSpans<'_, I>>,
    pass: &mut TrackedRenderPass<'_>,
    item: &I,
) {
    if let Some(spans) = spans {
        spans.before(pass, item);
    }
}

/// Calls [`DrawSpans::end`] when the phase is grouped.
pub(super) fn end<I: PhaseItem>(
    spans: &mut Option<DrawSpans<'_, I>>,
    pass: &mut TrackedRenderPass<'_>,
) {
    if let Some(spans) = spans {
        spans.end(pass);
    }
}
