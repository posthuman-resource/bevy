//! Volumetric fog and volumetric lighting, also known as light shafts or god
//! rays.
//!
//! This module implements a more physically-accurate, but slower, form of fog
//! than the [`crate::fog`] module does. Notably, this *volumetric fog* allows
//! for light beams from directional lights to shine through, creating what is
//! known as *light shafts* or *god rays*.
//!
//! To add volumetric fog to a scene, add [`bevy_light::VolumetricFog`] to the
//! camera, and add [`bevy_light::VolumetricLight`] to directional lights that you wish to
//! be volumetric. [`bevy_light::VolumetricFog`] feature numerous settings that
//! allow you to define the accuracy of the simulation, as well as the look of
//! the fog. Currently, only interaction with directional lights that have
//! shadow maps is supported. Note that the overhead of the effect scales
//! directly with the number of directional lights in use, so apply
//! [`bevy_light::VolumetricLight`] sparingly for the best results.
//!
//! The overall algorithm, which is implemented as a postprocessing effect, is a
//! combination of the techniques described in [Scratchapixel] and [this blog
//! post]. It uses raymarching in screen space, transformed into shadow map
//! space for sampling and combined with physically-based modeling of absorption
//! and scattering. Bevy employs the widely-used [Henyey-Greenstein phase
//! function] to model asymmetry; this essentially allows light shafts to fade
//! into and out of existence as the user views them.
//!
//! [Scratchapixel]: https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/intro-volume-rendering.html
//!
//! [this blog post]: https://www.alexandre-pestana.com/volumetric-lights/
//!
//! [Henyey-Greenstein phase function]: https://www.pbr-book.org/4ed/Volume_Scattering/Phase_Functions#TheHenyeyndashGreensteinPhaseFunction

use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, Assets, Handle};
use bevy_core_pipeline::{
    core_3d::prepare_core_3d_depth_textures,
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{component::Component, resource::Resource, schedule::IntoScheduleConfigs as _};
use bevy_light::FogVolume;
use bevy_math::{
    primitives::{Cuboid, Plane3d},
    Vec2, Vec3,
};
use bevy_mesh::{Mesh, Meshable};
use bevy_render::{
    render_resource::SpecializedRenderPipelines,
    sync_component::{SyncComponent, SyncComponentPlugin},
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use render::{
    volumetric_fog, VolumetricFogPipeline, VolumetricFogUniformBuffer,
    VolumetricFogUpsamplePipeline,
};

use crate::{volumetric_fog::render::init_volumetric_fog_pipeline, MeshPipelineSystems};

pub mod render;

/// A plugin that implements volumetric fog.
pub struct VolumetricFogPlugin;

/// **Fork addition (posthuman-resource/bevy, branch `phase-shift/half-res-fog`).**
///
/// Marches this camera's volumetric fog at `1 / divisor` of the view's
/// resolution in each axis, then composites it onto the view at full
/// resolution with a depth-aware (joint bilateral) upsample. Upstream Bevy
/// always marches one ray per pixel, which at 3840×2054 is most of the frame.
///
/// Put it on the camera beside [`bevy_light::VolumetricFog`]. `divisor` 1 (or
/// no component) is upstream's full-resolution path, byte for byte; 2 is half
/// resolution (a quarter of the rays); values are clamped to
/// `1..=`[`Self::MAX_DIVISOR`]. Assumes a perspective, reverse-Z projection
/// for the depth weights (Bevy's default); an orthographic camera still
/// renders, with softer edges.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct VolumetricFogResolution {
    /// The fog is marched at `ceil(width / divisor) × ceil(height / divisor)`.
    pub divisor: u32,
}

impl VolumetricFogResolution {
    /// The largest divisor honoured.
    pub const MAX_DIVISOR: u32 = 8;

    /// The divisor actually used, `1..=MAX_DIVISOR`.
    pub fn effective_divisor(&self) -> u32 {
        self.divisor.clamp(1, Self::MAX_DIVISOR)
    }
}

#[derive(Resource)]
pub struct FogAssets {
    plane_mesh: Handle<Mesh>,
    cube_mesh: Handle<Mesh>,
}

impl Plugin for VolumetricFogPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "volumetric_fog.wgsl");
        // Fork: the low-resolution fog's composite (`VolumetricFogResolution`).
        embedded_asset!(app, "volumetric_fog_upsample.wgsl");

        let mut meshes = app.world_mut().resource_mut::<Assets<Mesh>>();
        let plane_mesh = meshes.add(Plane3d::new(Vec3::Z, Vec2::ONE).mesh());
        let cube_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0).mesh());

        app.add_plugins(SyncComponentPlugin::<FogVolume, Self>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .insert_resource(FogAssets {
                plane_mesh,
                cube_mesh,
            })
            .init_gpu_resource::<SpecializedRenderPipelines<VolumetricFogPipeline>>()
            // Fork: `VolumetricFogResolution`'s composite pipelines.
            .init_gpu_resource::<SpecializedRenderPipelines<VolumetricFogUpsamplePipeline>>()
            .init_gpu_resource::<VolumetricFogUniformBuffer>()
            .add_systems(
                RenderStartup,
                init_volumetric_fog_pipeline.after(MeshPipelineSystems),
            )
            .add_systems(ExtractSchedule, render::extract_volumetric_fog)
            .add_systems(
                Render,
                (
                    render::prepare_volumetric_fog_pipelines.in_set(RenderSystems::Prepare),
                    render::prepare_volumetric_fog_uniforms.in_set(RenderSystems::Prepare),
                    render::prepare_view_depth_textures_for_volumetric_fog
                        .in_set(RenderSystems::Prepare)
                        .before(prepare_core_3d_depth_textures),
                    // Fork: `VolumetricFogResolution`'s low-resolution target.
                    render::prepare_volumetric_fog_low_res_textures
                        .in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_systems(
                Core3d,
                volumetric_fog
                    .after(Core3dSystems::MainPass)
                    .before(Core3dSystems::EarlyPostProcess),
            );
    }
}

impl SyncComponent<VolumetricFogPlugin> for FogVolume {
    type Target = Self;
}
