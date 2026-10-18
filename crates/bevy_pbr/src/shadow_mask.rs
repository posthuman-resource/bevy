//! Directional shadow masks: a per-view, screen-space shadow term that opaque surfaces take in place of a
//! directional light's shadow maps.
//!
//! The mask is written before the main pass by whatever makes it, for example a pass that traces a shadow ray from
//! each pixel of the depth prepass. Bevy only reads it: a view that has [`DirectionalShadowMask`] binds the
//! [`ViewDirectionalShadowMask`] texture its producer inserts on the render-world view, and a directional light that
//! opts in with [`DirectionalLightShadowMask::Receive`] takes its shadow on that view's opaque surfaces from the mask.
//! Blended surfaces, the volumetric fog and views without a mask keep the light's shadow maps.
//!
//! What the mask cannot see can still cast: a light marked [`DirectionalLightShadowMask::Casters`] adds no light, and
//! its shadow maps (usually holding only the casters the mask's producer does not handle) are folded into the mask by
//! minimum.

use bevy_app::{App, Plugin};
use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_resource::TextureView,
    RenderApp,
};

/// Registers the shadow-mask components and extracts [`DirectionalShadowMask`].
pub struct DirectionalShadowMaskPlugin;

impl Plugin for DirectionalShadowMaskPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<DirectionalShadowMask>()
            .register_type::<DirectionalLightShadowMask>()
            .add_plugins(ExtractComponentPlugin::<DirectionalShadowMask>::default());
    }
}

/// Add to a camera to give its opaque surfaces a directional shadow mask.
///
/// The mask is made from the depth prepass, which this requires. Until its producer inserts a
/// [`ViewDirectionalShadowMask`] on the view, the mask reads lit.
#[derive(Component, Clone, Copy, Debug, Default, Reflect, ExtractComponent)]
#[reflect(Component, Default, Clone)]
#[require(bevy_core_pipeline::prepass::DepthPrepass)]
#[extract_app(RenderApp)]
pub struct DirectionalShadowMask {
    /// Opaque surfaces show the shadow term they take from the mask (white lit, black shadowed) instead of their
    /// shading, to inspect the mask.
    pub debug_view: bool,
}

/// How a [`DirectionalLight`](bevy_light::DirectionalLight) uses the views' directional shadow masks.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
#[reflect(Component, Default, Clone, PartialEq)]
pub enum DirectionalLightShadowMask {
    /// Opaque surfaces in a view with a mask take this light's shadow from the mask in place of its shadow maps.
    #[default]
    Receive,
    /// This light adds no light of its own. On opaque surfaces in a view with a mask, its shadow maps are folded into
    /// the mask by minimum, for every light that receives the mask. Give it the receiving light's direction and only
    /// the casters the mask does not see (with render layers).
    Casters,
}

/// A view's directional shadow mask, at the main pass's size: red is 1 where lit and 0 where shadowed. Inserted on the
/// render-world view by the pass that writes it, before the main pass.
#[derive(Component, Clone)]
pub struct ViewDirectionalShadowMask(pub TextureView);
