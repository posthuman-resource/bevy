//! Rendering of fog volumes.

use core::array;

use bevy_asset::{load_embedded_asset, AssetId, AssetServer, Handle};
use bevy_camera::Camera3d;
use bevy_color::ColorToComponents as _;
use bevy_core_pipeline::FullscreenShader;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    resource::Resource,
    system::{Commands, Local, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_light::{FogVolume, VolumetricFog, VolumetricLight};
use bevy_math::{vec4, Affine3A, Mat4, Vec3, Vec3A, Vec4};
use bevy_mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy_render::{
    camera::ExtractedCamera,
    diagnostic::RecordDiagnostics,
    mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{
            sampler, texture_2d, texture_3d, texture_depth_2d, texture_depth_2d_multisampled,
            uniform_buffer,
        },
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BindingResource,
        BlendComponent, BlendFactor, BlendOperation, BlendState, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, DynamicBindGroupEntries, DynamicUniformBuffer, Extent3d,
        Face, FragmentState, LoadOp, Operations, PipelineCache, PrimitiveState,
        RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
        SamplerBindingType, ShaderStages, ShaderType, SpecializedRenderPipeline,
        SpecializedRenderPipelines, StoreOp, TextureDescriptor, TextureDimension, TextureFormat,
        TextureSampleType, TextureUsages, VertexState,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
    sync_world::RenderEntity,
    texture::{CachedTexture, GpuImage, TextureCache},
    view::{ExtractedView, Msaa, ViewDepthTexture, ViewTarget},
    Extract,
};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_transform::components::GlobalTransform;
use bevy_utils::prelude::default;
use bitflags::bitflags;

use crate::{MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, MeshViewBindGroup, ViewKeyCache};

use super::{FogAssets, VolumetricFogResolution};

/// Fork: the format of `VolumetricFogResolution`'s low-resolution target. It
/// holds premultiplied in-scattered light and `1 - transmittance`, so it needs
/// alpha and HDR range whatever the view's own format is.
const LOW_RES_FOG_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

bitflags! {
    /// Flags that describe the bind group layout used to render volumetric fog.
    #[derive(Clone, Copy, PartialEq)]
    struct VolumetricFogBindGroupLayoutKey: u8 {
        /// The framebuffer is multisampled.
        const MULTISAMPLED = 0x1;
        /// The volumetric fog has a 3D voxel density texture.
        const DENSITY_TEXTURE = 0x2;
    }
}

/// The total number of bind group layouts.
///
/// This is the total number of combinations of all
/// [`VolumetricFogBindGroupLayoutKey`] flags.
const VOLUMETRIC_FOG_BIND_GROUP_LAYOUT_COUNT: usize =
    VolumetricFogBindGroupLayoutKey::all().bits() as usize + 1;

/// A matrix that converts from local 1×1×1 space to UVW 3D density texture
/// space.
static UVW_FROM_LOCAL: Mat4 = Mat4::from_cols(
    vec4(1.0, 0.0, 0.0, 0.0),
    vec4(0.0, 1.0, 0.0, 0.0),
    vec4(0.0, 0.0, 1.0, 0.0),
    vec4(0.5, 0.5, 0.5, 1.0),
);

/// The GPU pipeline for the volumetric fog postprocessing effect.
#[derive(Resource)]
pub struct VolumetricFogPipeline {
    /// A reference to the shared set of mesh pipeline view layouts.
    mesh_view_layouts: MeshPipelineViewLayouts,

    /// All bind group layouts.
    ///
    /// Since there aren't too many of these, we precompile them all.
    volumetric_view_bind_group_layouts:
        [BindGroupLayoutDescriptor; VOLUMETRIC_FOG_BIND_GROUP_LAYOUT_COUNT],

    // The shader asset handle.
    shader: Handle<Shader>,
}

/// Fork: the pipeline that composites `VolumetricFogResolution`'s
/// low-resolution fog onto the view with a depth-aware upsample.
#[derive(Resource)]
pub struct VolumetricFogUpsamplePipeline {
    /// Indexed by "the depth buffer is multisampled".
    bind_group_layouts: [BindGroupLayoutDescriptor; 2],
    fullscreen_shader: FullscreenShader,
    shader: Handle<Shader>,
}

/// Fork: identifies one specialization of the upsample composite.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct VolumetricFogUpsamplePipelineKey {
    target_format: TextureFormat,
    multisampled: bool,
    divisor: u32,
}

/// The two render pipelines that we use for fog volumes: one for when a 3D
/// density texture is present and one for when it isn't.
#[derive(Component)]
pub struct ViewVolumetricFogPipelines {
    /// The render pipeline that we use when no density texture is present, and
    /// the density distribution is uniform.
    pub textureless: CachedRenderPipelineId,
    /// The render pipeline that we use when a density texture is present.
    pub textured: CachedRenderPipelineId,
    /// Fork: the composite when the fog is marched at low resolution
    /// (`VolumetricFogResolution`); `None` on upstream's full-resolution path.
    pub upsample: Option<CachedRenderPipelineId>,
}

/// Fork: the low-resolution target `VolumetricFogResolution` marches into.
#[derive(Component, Deref)]
pub struct ViewVolumetricFogLowRes(CachedTexture);

/// Identifies a single specialization of the volumetric fog shader.
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct VolumetricFogPipelineKey {
    /// The layout of the view, which is needed for the raymarching.
    mesh_pipeline_view_key: MeshPipelineViewLayoutKey,

    /// The vertex buffer layout of the primitive.
    ///
    /// Both planes (used when the camera is inside the fog volume) and cubes
    /// (used when the camera is outside the fog volume) use identical vertex
    /// buffer layouts, so we only need one of them.
    vertex_buffer_layout: MeshVertexBufferLayoutRef,

    /// Texture format of the view target
    target_format: TextureFormat,

    /// The volumetric fog has a 3D voxel density texture.
    has_density_texture: bool,

    /// Fork: `VolumetricFogResolution`'s divisor; 1 is upstream's path.
    divisor: u32,
}

/// The same as [`VolumetricFog`] and [`FogVolume`], but formatted for
/// the GPU.
///
/// See the documentation of those structures for more information on these
/// fields.
#[derive(ShaderType)]
pub struct VolumetricFogUniform {
    clip_from_local: Mat4,

    /// The transform from world space to 3D density texture UVW space.
    uvw_from_world: Mat4,

    /// View-space plane equations of the far faces of the fog volume cuboid.
    ///
    /// The vector takes the form V = (N, -N⋅Q), where N is the normal of the
    /// plane and Q is any point in it, in view space. The equation of the plane
    /// for homogeneous point P = (Px, Py, Pz, Pw) is V⋅P = 0.
    far_planes: [Vec4; 6],

    fog_color: Vec3,
    light_tint: Vec3,
    ambient_color: Vec3,
    ambient_intensity: f32,
    step_count: u32,

    /// The radius of a sphere that bounds the fog volume in view space.
    bounding_radius: f32,

    absorption: f32,
    scattering: f32,
    density: f32,
    density_texture_offset: Vec3,
    scattering_asymmetry: f32,
    light_intensity: f32,
    jitter_strength: f32,
}

/// Specifies the offset within the [`VolumetricFogUniformBuffer`] of the
/// [`VolumetricFogUniform`] for a specific view.
#[derive(Component, Deref, DerefMut)]
pub struct ViewVolumetricFog(Vec<ViewFogVolume>);

/// Information that the render world needs to maintain about each fog volume.
pub struct ViewFogVolume {
    /// The 3D voxel density texture for this volume, if present.
    density_texture: Option<AssetId<Image>>,
    /// The offset of this view's [`VolumetricFogUniform`] structure within the
    /// [`VolumetricFogUniformBuffer`].
    uniform_buffer_offset: u32,
    /// True if the camera is outside the fog volume; false if it's inside the
    /// fog volume.
    exterior: bool,
}

/// The GPU buffer that stores the [`VolumetricFogUniform`] data.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct VolumetricFogUniformBuffer(pub DynamicUniformBuffer<VolumetricFogUniform>);

pub fn init_volumetric_fog_pipeline(
    mut commands: Commands,
    mesh_view_layouts: Res<MeshPipelineViewLayouts>,
    asset_server: Res<AssetServer>,
    fullscreen_shader: Res<FullscreenShader>,
) {
    // Fork: the upsample composite's layouts: the low-resolution fog and the
    // full-resolution depth, one layout per depth sample count.
    let upsample_layout = |multisampled: bool| {
        BindGroupLayoutDescriptor::new(
            "volumetric fog upsample bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    if multisampled {
                        texture_depth_2d_multisampled()
                    } else {
                        texture_depth_2d()
                    },
                ),
            ),
        )
    };
    commands.insert_resource(VolumetricFogUpsamplePipeline {
        bind_group_layouts: [upsample_layout(false), upsample_layout(true)],
        fullscreen_shader: fullscreen_shader.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "volumetric_fog_upsample.wgsl"),
    });

    // Create the bind group layout entries common to all bind group
    // layouts.
    let base_bind_group_layout_entries = &BindGroupLayoutEntries::single(
        ShaderStages::VERTEX_FRAGMENT,
        // `volumetric_fog`
        uniform_buffer::<VolumetricFogUniform>(true),
    );

    // For every combination of `VolumetricFogBindGroupLayoutKey` bits,
    // create a bind group layout.
    let bind_group_layouts = array::from_fn(|bits| {
        let flags = VolumetricFogBindGroupLayoutKey::from_bits_retain(bits as u8);

        let mut bind_group_layout_entries = base_bind_group_layout_entries.to_vec();

        // `depth_texture`
        bind_group_layout_entries.extend_from_slice(&BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            ((
                1,
                if flags.contains(VolumetricFogBindGroupLayoutKey::MULTISAMPLED) {
                    texture_depth_2d_multisampled()
                } else {
                    texture_depth_2d()
                },
            ),),
        ));

        // `density_texture` and `density_sampler`
        if flags.contains(VolumetricFogBindGroupLayoutKey::DENSITY_TEXTURE) {
            bind_group_layout_entries.extend_from_slice(&BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (2, texture_3d(TextureSampleType::Float { filterable: true })),
                    (3, sampler(SamplerBindingType::Filtering)),
                ),
            ));
        }

        // Create the bind group layout.
        let description = flags.bind_group_layout_description();
        BindGroupLayoutDescriptor::new(description, &bind_group_layout_entries)
    });

    commands.insert_resource(VolumetricFogPipeline {
        mesh_view_layouts: mesh_view_layouts.clone(),
        volumetric_view_bind_group_layouts: bind_group_layouts,
        shader: load_embedded_asset!(asset_server.as_ref(), "volumetric_fog.wgsl"),
    });
}

/// Extracts [`VolumetricFog`], [`FogVolume`], and [`VolumetricLight`]s
/// from the main world to the render world.
pub fn extract_volumetric_fog(
    mut commands: Commands,
    view_targets: Extract<
        Query<(
            RenderEntity,
            &VolumetricFog,
            Option<&VolumetricFogResolution>,
        )>,
    >,
    fog_volumes: Extract<Query<(RenderEntity, &FogVolume, &GlobalTransform)>>,
    volumetric_lights: Extract<Query<(RenderEntity, &VolumetricLight)>>,
) {
    if volumetric_lights.is_empty() {
        // TODO: needs better way to handle clean up in render world
        for (entity, ..) in view_targets.iter() {
            commands.entity(entity).remove::<(
                VolumetricFog,
                ViewVolumetricFogPipelines,
                ViewVolumetricFog,
                // Fork: `VolumetricFogResolution` and its target.
                VolumetricFogResolution,
                ViewVolumetricFogLowRes,
            )>();
        }
        for (entity, ..) in fog_volumes.iter() {
            commands.entity(entity).remove::<FogVolume>();
        }
        return;
    }

    for (entity, volumetric_fog, resolution) in view_targets.iter() {
        let mut view = commands
            .get_entity(entity)
            .expect("Volumetric fog entity wasn't synced.");
        view.insert(*volumetric_fog);
        // Fork: the render entity is retained, so an absent resolution is
        // removed rather than left from an earlier frame.
        match resolution {
            Some(resolution) if resolution.effective_divisor() > 1 => {
                view.insert(*resolution);
            }
            _ => {
                view.remove::<(VolumetricFogResolution, ViewVolumetricFogLowRes)>();
            }
        }
    }

    for (entity, fog_volume, fog_transform) in fog_volumes.iter() {
        commands
            .get_entity(entity)
            .expect("Fog volume entity wasn't synced.")
            .insert((*fog_volume).clone())
            .insert(*fog_transform);
    }

    for (entity, volumetric_light) in volumetric_lights.iter() {
        commands
            .get_entity(entity)
            .expect("Volumetric light entity wasn't synced.")
            .insert(*volumetric_light);
    }
}

pub fn volumetric_fog(
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthTexture,
        &ViewVolumetricFogPipelines,
        &ViewVolumetricFog,
        &MeshViewBindGroup,
        &Msaa,
        // Fork: `VolumetricFogResolution`'s target, when the fog is low-res.
        Option<&ViewVolumetricFogLowRes>,
    )>,
    pipeline_cache: Res<PipelineCache>,
    volumetric_lighting_pipeline: Res<VolumetricFogPipeline>,
    volumetric_lighting_uniform_buffers: Res<VolumetricFogUniformBuffer>,
    // Fork: the composite for the low-res path.
    upsample_pipeline: Res<VolumetricFogUpsamplePipeline>,
    image_assets: Res<RenderAssets<GpuImage>>,
    mesh_allocator: Res<MeshAllocator>,
    fog_assets: Res<FogAssets>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut ctx: RenderContext,
) {
    let (
        view_target,
        view_depth_texture,
        view_volumetric_lighting_pipelines,
        view_fog_volumes,
        view_bind_group,
        msaa,
        low_res,
    ) = view.into_inner();

    // Fetch the uniform buffer and binding.
    let (
        Some(textureless_pipeline),
        Some(textured_pipeline),
        Some(volumetric_lighting_uniform_buffer_binding),
    ) = (
        pipeline_cache.get_render_pipeline(view_volumetric_lighting_pipelines.textureless),
        pipeline_cache.get_render_pipeline(view_volumetric_lighting_pipelines.textured),
        volumetric_lighting_uniform_buffers.binding(),
    )
    else {
        return;
    };

    // Fork: on the low-res path the volumes are marched into the low-res
    // target and composited afterwards. The march pipelines were specialized
    // for that target's format, so without the target or the composite
    // pipeline nothing is drawn this frame.
    let low_res = match view_volumetric_lighting_pipelines.upsample {
        None => None,
        Some(upsample_id) => {
            let (Some(low_res), Some(upsample)) =
                (low_res, pipeline_cache.get_render_pipeline(upsample_id))
            else {
                return;
            };
            Some((low_res, upsample))
        }
    };

    // Fork: a GPU span, so the fog is a named zone in Tracy and in
    // `RenderDiagnosticsPlugin` rather than the gap between two others.
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let time_span = diagnostics.time_span(ctx.command_encoder(), "volumetric_fog");

    let command_encoder = ctx.command_encoder();
    command_encoder.push_debug_group("volumetric_lighting");

    // Fork: the target the volumes are drawn into, and its first load: the
    // low-res target starts transparent every frame; the view is blended onto.
    let (fog_target, mut fog_load) = match low_res {
        Some((low_res, _)) => (&low_res.default_view, LoadOp::Clear(default())),
        None => (view_target.main_texture_view(), LoadOp::Load),
    };
    let mut marched = false;

    for view_fog_volume in view_fog_volumes.iter() {
        // If the camera is outside the fog volume, pick the cube mesh;
        // otherwise, pick the plane mesh. In the latter case we'll be
        // effectively rendering a full-screen quad.
        let mesh_handle = if view_fog_volume.exterior {
            fog_assets.cube_mesh.clone()
        } else {
            fog_assets.plane_mesh.clone()
        };

        let Some(vertex_buffer_slice) = mesh_allocator.mesh_vertex_slice(&mesh_handle.id()) else {
            continue;
        };

        let density_image = view_fog_volume
            .density_texture
            .and_then(|density_texture| image_assets.get(density_texture));

        // Pick the right pipeline, depending on whether a density texture
        // is present or not.
        let pipeline = if density_image.is_some() {
            textured_pipeline
        } else {
            textureless_pipeline
        };

        // This should always succeed, but if the asset was unloaded don't
        // panic.
        let Some(render_mesh) = render_meshes.get(&mesh_handle) else {
            // Fork: `break`, not `return`, so the debug group and the GPU
            // span below are closed.
            break;
        };

        // Create the bind group for the view.
        //
        // TODO: Cache this.

        let mut bind_group_layout_key = VolumetricFogBindGroupLayoutKey::empty();
        bind_group_layout_key.set(
            VolumetricFogBindGroupLayoutKey::MULTISAMPLED,
            !matches!(*msaa, Msaa::Off),
        );

        // Create the bind group entries. The ones relating to the density
        // texture will only be filled in if that texture is present.
        let mut bind_group_entries = DynamicBindGroupEntries::sequential((
            volumetric_lighting_uniform_buffer_binding.clone(),
            BindingResource::TextureView(view_depth_texture.view()),
        ));
        if let Some(density_image) = density_image {
            bind_group_layout_key.insert(VolumetricFogBindGroupLayoutKey::DENSITY_TEXTURE);
            bind_group_entries = bind_group_entries.extend_sequential((
                BindingResource::TextureView(&density_image.texture_view),
                BindingResource::Sampler(&density_image.sampler),
            ));
        }

        let volumetric_view_bind_group_layout = &volumetric_lighting_pipeline
            .volumetric_view_bind_group_layouts[bind_group_layout_key.bits() as usize];

        let volumetric_view_bind_group = ctx.render_device().create_bind_group(
            None,
            &pipeline_cache.get_bind_group_layout(volumetric_view_bind_group_layout),
            &bind_group_entries,
        );

        let render_pass_descriptor = RenderPassDescriptor {
            label: Some("volumetric lighting pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                // Fork: the low-res target on that path (upstream: the view).
                view: fog_target,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: fog_load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        // Fork: only the first volume clears the low-res target.
        fog_load = LoadOp::Load;
        marched = true;

        let command_encoder = ctx.command_encoder();
        let mut render_pass = command_encoder.begin_render_pass(&render_pass_descriptor);

        render_pass.set_vertex_buffer(0, *vertex_buffer_slice.buffer.slice(..));
        render_pass.set_pipeline(pipeline);

        render_pass.set_bind_group(0, &view_bind_group.main, &view_bind_group.main_offsets);
        render_pass.set_bind_group(
            1,
            &volumetric_view_bind_group,
            &[view_fog_volume.uniform_buffer_offset],
        );

        // Draw elements or arrays, as appropriate.
        match &render_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_buffer_slice) = mesh_allocator.mesh_index_slice(&mesh_handle.id())
                else {
                    continue;
                };

                render_pass.set_index_buffer(*index_buffer_slice.buffer.slice(..), *index_format);
                render_pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    0..1,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                render_pass.draw(vertex_buffer_slice.range, 0..1);
            }
        }
    }

    // Fork: composite the low-res fog onto the view, once, if anything was
    // marched into it this frame (it starts transparent, so an empty target
    // would be a no-op anyway, but an uncleared one would not be).
    if let Some((low_res, upsample)) = low_res
        && marched
    {
        let multisampled = !matches!(*msaa, Msaa::Off);
        let bind_group = ctx.render_device().create_bind_group(
            "volumetric fog upsample bind group",
            &pipeline_cache
                .get_bind_group_layout(&upsample_pipeline.bind_group_layouts[multisampled as usize]),
            &BindGroupEntries::sequential((&low_res.default_view, view_depth_texture.view())),
        );
        let mut composite = ctx.command_encoder().begin_render_pass(&RenderPassDescriptor {
            label: Some("volumetric fog upsample pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view_target.main_texture_view(),
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        composite.set_pipeline(upsample);
        composite.set_bind_group(0, &bind_group, &[]);
        composite.draw(0..3, 0..1);
    }

    ctx.command_encoder().pop_debug_group();
    time_span.end(ctx.command_encoder());
}

impl SpecializedRenderPipeline for VolumetricFogPipeline {
    type Key = VolumetricFogPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // We always use hardware 2x2 filtering for sampling the shadow map; the
        // more accurate versions with percentage-closer filtering aren't worth
        // the overhead.
        let mut shader_defs = vec!["SHADOW_FILTER_METHOD_HARDWARE_2X2".into()];

        // We need a separate layout for MSAA and non-MSAA, as well as one for
        // the presence or absence of the density texture.
        let mut bind_group_layout_key = VolumetricFogBindGroupLayoutKey::empty();
        bind_group_layout_key.set(
            VolumetricFogBindGroupLayoutKey::MULTISAMPLED,
            key.mesh_pipeline_view_key
                .contains(MeshPipelineViewLayoutKey::MULTISAMPLED),
        );
        bind_group_layout_key.set(
            VolumetricFogBindGroupLayoutKey::DENSITY_TEXTURE,
            key.has_density_texture,
        );

        let volumetric_view_bind_group_layout =
            self.volumetric_view_bind_group_layouts[bind_group_layout_key.bits() as usize].clone();

        // Both the cube and plane have the same vertex layout, so we don't need
        // to distinguish between the two.
        let vertex_format = key
            .vertex_buffer_layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])
            .expect("Failed to get vertex layout for volumetric fog hull");

        if key
            .mesh_pipeline_view_key
            .contains(MeshPipelineViewLayoutKey::MULTISAMPLED)
        {
            shader_defs.push("MULTISAMPLED".into());
        }

        if key
            .mesh_pipeline_view_key
            .contains(MeshPipelineViewLayoutKey::ATMOSPHERE)
        {
            shader_defs.push("ATMOSPHERE".into());
        }

        if key.has_density_texture {
            shader_defs.push("DENSITY_TEXTURE".into());
        }

        // Fork: march one ray per `divisor × divisor` block of the view
        // (`VolumetricFogResolution`).
        if key.divisor > 1 {
            shader_defs.push("VOLUMETRIC_FOG_LOW_RES".into());
            shader_defs.push(ShaderDefVal::UInt(
                "VOLUMETRIC_FOG_DIVISOR".into(),
                key.divisor,
            ));
        }

        let layout = self
            .mesh_view_layouts
            .get_view_layout(key.mesh_pipeline_view_key);
        let layout = vec![
            layout.main_layout,
            volumetric_view_bind_group_layout.clone(),
        ];

        RenderPipelineDescriptor {
            label: Some("volumetric lighting pipeline".into()),
            layout,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_format],
                ..default()
            },
            primitive: PrimitiveState {
                cull_mode: Some(Face::Back),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    // Blend on top of what's already in the framebuffer. Doing
                    // the alpha blending with the hardware blender allows us to
                    // avoid having to use intermediate render targets.
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        // Fork: the low-res target accumulates coverage too,
                        // `1 - Π transmittance`, for the composite to blend
                        // with; the view's own alpha is left alone as upstream.
                        alpha: if key.divisor > 1 {
                            BlendComponent {
                                src_factor: BlendFactor::One,
                                dst_factor: BlendFactor::OneMinusSrcAlpha,
                                operation: BlendOperation::Add,
                            }
                        } else {
                            BlendComponent {
                                src_factor: BlendFactor::Zero,
                                dst_factor: BlendFactor::One,
                                operation: BlendOperation::Add,
                            }
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

/// Specializes volumetric fog pipelines for all views with that effect enabled.
pub fn prepare_volumetric_fog_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<VolumetricFogPipeline>>,
    volumetric_lighting_pipeline: Res<VolumetricFogPipeline>,
    // Fork: the low-res composite (`VolumetricFogResolution`).
    mut upsample_pipelines: ResMut<SpecializedRenderPipelines<VolumetricFogUpsamplePipeline>>,
    upsample_pipeline: Res<VolumetricFogUpsamplePipeline>,
    fog_assets: Res<FogAssets>,
    view_targets: Query<
        (Entity, &ExtractedView, Option<&VolumetricFogResolution>),
        With<VolumetricFog>,
    >,
    meshes: Res<RenderAssets<RenderMesh>>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let Some(plane_mesh) = meshes.get(&fog_assets.plane_mesh) else {
        // There's an off chance that the mesh won't be prepared yet if `RenderAssetBytesPerFrame` limiting is in use.
        return;
    };

    for (entity, view, resolution) in view_targets.iter() {
        let Some(mesh_pipeline_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        // Fork: at a divisor above 1 the volumes are marched into the
        // low-res target, and a composite puts them on the view.
        let divisor = resolution.map_or(1, VolumetricFogResolution::effective_divisor);
        let mesh_pipeline_view_key: MeshPipelineViewLayoutKey = (*mesh_pipeline_key).into();
        let upsample = (divisor > 1).then(|| {
            upsample_pipelines.specialize(
                &pipeline_cache,
                &upsample_pipeline,
                VolumetricFogUpsamplePipelineKey {
                    target_format: view.target_format,
                    multisampled: mesh_pipeline_view_key
                        .contains(MeshPipelineViewLayoutKey::MULTISAMPLED),
                    divisor,
                },
            )
        });

        // Specialize the pipeline.
        let textureless_pipeline_key = VolumetricFogPipelineKey {
            mesh_pipeline_view_key,
            vertex_buffer_layout: plane_mesh.layout.clone(),
            target_format: if divisor > 1 {
                LOW_RES_FOG_FORMAT
            } else {
                view.target_format
            },
            has_density_texture: false,
            divisor,
        };
        let textureless_pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &volumetric_lighting_pipeline,
            textureless_pipeline_key.clone(),
        );
        let textured_pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &volumetric_lighting_pipeline,
            VolumetricFogPipelineKey {
                has_density_texture: true,
                ..textureless_pipeline_key
            },
        );

        commands.entity(entity).insert(ViewVolumetricFogPipelines {
            textureless: textureless_pipeline_id,
            textured: textured_pipeline_id,
            upsample,
        });
    }
}

/// Fork: allocates `VolumetricFogResolution`'s low-resolution target,
/// `ceil(width / divisor) × ceil(height / divisor)` of the camera's target.
pub fn prepare_volumetric_fog_low_res_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &VolumetricFogResolution), With<VolumetricFog>>,
) {
    for (entity, camera, resolution) in &views {
        let Some(size) = camera.physical_target_size else {
            continue;
        };
        let divisor = resolution.effective_divisor();
        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("volumetric_fog_low_res_texture"),
                size: Extent3d {
                    width: size.x.div_ceil(divisor).max(1),
                    height: size.y.div_ceil(divisor).max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: LOW_RES_FOG_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );
        commands
            .entity(entity)
            .insert(ViewVolumetricFogLowRes(texture));
    }
}

impl SpecializedRenderPipeline for VolumetricFogUpsamplePipeline {
    type Key = VolumetricFogUpsamplePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = vec![ShaderDefVal::UInt(
            "VOLUMETRIC_FOG_DIVISOR".into(),
            key.divisor,
        )];
        if key.multisampled {
            shader_defs.push("MULTISAMPLED".into());
        }
        RenderPipelineDescriptor {
            label: Some("volumetric fog upsample pipeline".into()),
            layout: vec![self.bind_group_layouts[key.multisampled as usize].clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    // Premultiplied fog over the view, exactly as upstream's
                    // full-resolution pass blends it.
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

/// A system that converts [`VolumetricFog`] into [`VolumetricFogUniform`]s.
pub fn prepare_volumetric_fog_uniforms(
    mut commands: Commands,
    mut volumetric_lighting_uniform_buffer: ResMut<VolumetricFogUniformBuffer>,
    view_targets: Query<(Entity, &ExtractedView, &VolumetricFog)>,
    fog_volumes: Query<(Entity, &FogVolume, &GlobalTransform)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut local_from_world_matrices: Local<Vec<Affine3A>>,
) {
    // Do this up front to avoid O(n^2) matrix inversion.
    local_from_world_matrices.clear();
    for (_, _, fog_transform) in fog_volumes.iter() {
        local_from_world_matrices.push(fog_transform.affine().inverse());
    }

    let uniform_count = view_targets.iter().len() * local_from_world_matrices.len();

    let Some(mut writer) =
        volumetric_lighting_uniform_buffer.get_writer(uniform_count, &render_device, &render_queue)
    else {
        return;
    };

    for (view_entity, extracted_view, volumetric_fog) in view_targets.iter() {
        let world_from_view = extracted_view.world_from_view.affine();

        let mut view_fog_volumes = vec![];

        for ((_, fog_volume, _), local_from_world) in
            fog_volumes.iter().zip(local_from_world_matrices.iter())
        {
            // Calculate the transforms to and from 1×1×1 local space.
            let local_from_view = *local_from_world * world_from_view;
            let view_from_local = local_from_view.inverse();

            // Determine whether the camera is inside or outside the volume, and
            // calculate the clip space transform.
            let interior = camera_is_inside_fog_volume(&local_from_view);
            let hull_clip_from_local = calculate_fog_volume_clip_from_local_transforms(
                interior,
                &extracted_view.clip_from_view,
                &view_from_local,
            );

            // Calculate the radius of the sphere that bounds the fog volume.
            let bounding_radius = view_from_local
                .transform_vector3a(Vec3A::splat(0.5))
                .length();

            // Write out our uniform.
            let uniform_buffer_offset = writer.write(&VolumetricFogUniform {
                clip_from_local: hull_clip_from_local,
                uvw_from_world: UVW_FROM_LOCAL * *local_from_world,
                far_planes: get_far_planes(&view_from_local),
                fog_color: fog_volume.fog_color.to_linear().to_vec3(),
                light_tint: fog_volume.light_tint.to_linear().to_vec3(),
                ambient_color: volumetric_fog.ambient_color.to_linear().to_vec3(),
                ambient_intensity: volumetric_fog.ambient_intensity,
                step_count: volumetric_fog.step_count,
                bounding_radius,
                absorption: fog_volume.absorption,
                scattering: fog_volume.scattering,
                density: fog_volume.density_factor,
                density_texture_offset: fog_volume.density_texture_offset,
                scattering_asymmetry: fog_volume.scattering_asymmetry,
                light_intensity: fog_volume.light_intensity,
                jitter_strength: volumetric_fog.jitter,
            });

            view_fog_volumes.push(ViewFogVolume {
                uniform_buffer_offset,
                exterior: !interior,
                density_texture: fog_volume.density_texture.as_ref().map(Handle::id),
            });
        }

        commands
            .entity(view_entity)
            .insert(ViewVolumetricFog(view_fog_volumes));
    }
}

/// A system that marks all view depth textures as readable in shaders.
///
/// The volumetric lighting pass needs to do this, and it doesn't happen by
/// default.
pub fn prepare_view_depth_textures_for_volumetric_fog(
    mut view_targets: Query<&mut Camera3d>,
    fog_volumes: Query<&VolumetricFog>,
) {
    if fog_volumes.is_empty() {
        return;
    }

    for mut camera in view_targets.iter_mut() {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}

fn get_far_planes(view_from_local: &Affine3A) -> [Vec4; 6] {
    let (mut far_planes, mut next_index) = ([Vec4::ZERO; 6], 0);

    for &local_normal in &[
        Vec3A::X,
        Vec3A::NEG_X,
        Vec3A::Y,
        Vec3A::NEG_Y,
        Vec3A::Z,
        Vec3A::NEG_Z,
    ] {
        let view_normal = view_from_local
            .transform_vector3a(local_normal)
            .normalize_or_zero();

        let view_position = view_from_local.transform_point3a(-local_normal * 0.5);
        let plane_coords = view_normal.extend(-view_normal.dot(view_position));

        // Filter planes that are facing away from the camera.
        if plane_coords.w <= 0.0 {
            // When planes are filtered here, the `far_planes` array will be padded with
            // one or more "zero" planes: (0.0, 0.0, 0.0, 0.0), these planes will be
            // correctly ignored by the shader in the plane sorting step.
            continue;
        }

        far_planes[next_index] = plane_coords;
        next_index += 1;
    }

    far_planes
}

impl VolumetricFogBindGroupLayoutKey {
    /// Creates an appropriate debug description for the bind group layout with
    /// these flags.
    fn bind_group_layout_description(&self) -> String {
        if self.is_empty() {
            return "volumetric lighting view bind group layout".to_owned();
        }

        format!(
            "volumetric lighting view bind group layout ({})",
            self.iter()
                .filter_map(|flag| {
                    if flag == VolumetricFogBindGroupLayoutKey::DENSITY_TEXTURE {
                        Some("density texture")
                    } else if flag == VolumetricFogBindGroupLayoutKey::MULTISAMPLED {
                        Some("multisampled")
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Given the transform from the view to the 1×1×1 cube in local fog volume
/// space, returns true if the camera is inside the volume.
fn camera_is_inside_fog_volume(local_from_view: &Affine3A) -> bool {
    local_from_view
        .translation
        .abs()
        .cmple(Vec3A::splat(0.5))
        .all()
}

/// Given the local transforms, returns the matrix that transforms model space
/// to clip space.
fn calculate_fog_volume_clip_from_local_transforms(
    interior: bool,
    clip_from_view: &Mat4,
    view_from_local: &Affine3A,
) -> Mat4 {
    if !interior {
        return *clip_from_view * Mat4::from(*view_from_local);
    }

    // If the camera is inside the fog volume, then we'll be rendering a full
    // screen quad. The shader will start its raymarch at the fragment depth
    // value, however, so we need to make sure that the depth of the full screen
    // quad is at the near clip plane `z_near`.
    let z_near = clip_from_view.w_axis[2];
    Mat4::from_cols(
        vec4(z_near, 0.0, 0.0, 0.0),
        vec4(0.0, z_near, 0.0, 0.0),
        vec4(0.0, 0.0, 0.0, 0.0),
        vec4(0.0, 0.0, z_near, z_near),
    )
}
