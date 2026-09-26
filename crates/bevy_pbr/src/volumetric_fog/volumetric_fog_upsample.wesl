// Fork addition (posthuman-resource/bevy, branch `phase-shift/half-res-fog`).
//
// Composites volumetric fog that was marched at `1 / divisor` resolution
// (`VolumetricFogResolution`) onto the full-resolution view. Each full-res
// pixel takes the four nearest low-res texels, weighted bilinearly and by how
// close each texel's depth is to its own (a joint bilateral upsample), so fog
// does not bleed across the silhouette of a rock against the sky. A texel's
// depth is read at the full-res pixel its ray was marched from; this must
// match `low_res_representative_pixel` in `volumetric_fog.wesl`.
//
// The low-res target holds premultiplied in-scattered light in rgb and
// `1 - transmittance` in alpha; the output is blended onto the view with
// (One, OneMinusSrcAlpha), as upstream's full-resolution pass is.

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;

@group(0) @binding(0) var low_res_fog: texture_2d<f32>;
@if(MULTISAMPLED)
@group(0) @binding(1) var depth_texture: texture_depth_multisampled_2d;
@else
@group(0) @binding(1) var depth_texture: texture_depth_2d;

// How far apart two depths may be, relative to the nearer one, before a
// texel stops counting as the same surface. A few percent keeps sloped ground
// smooth and still separates a rock from the terrain or sky behind it.
const RELATIVE_DEPTH_TOLERANCE: f32 = 0.02;

fn load_depth(pixel: vec2<u32>) -> f32 {
    return textureLoad(depth_texture, vec2<i32>(pixel), 0);
}

// Relative separation of two reverse-Z depths. Bevy's perspective projection
// is infinite reverse-Z, so view distance is `near / depth` and the relative
// difference of the distances reduces to `|a - b| / min(a, b)`; the sky
// (depth 0) is infinitely far from anything that is not the sky.
fn relative_separation(a: f32, b: f32) -> f32 {
    return abs(a - b) / max(min(a, b), 1e-9);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let divisor = u32(constants::VOLUMETRIC_FOG_DIVISOR);
    let full_size = textureDimensions(depth_texture);
    let low_size = vec2<i32>(textureDimensions(low_res_fog));
    let pixel = min(vec2<u32>(floor(in.position.xy)), full_size - vec2(1u));
    let depth = load_depth(pixel);

    // Low-res texel `i` was marched from full-res pixel `i * divisor + half`;
    // put this pixel's centre in that lattice.
    let half = f32(divisor / 2u);
    let lattice = (in.position.xy - 0.5 - half) / f32(divisor);
    let base = vec2<i32>(floor(lattice));
    let f = lattice - floor(lattice);

    var sum = vec4(0.0);
    var weight_sum = 0.0;
    var nearest = vec4(0.0);
    var nearest_separation = 1e30;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let texel = clamp(base + vec2(i, j), vec2(0), low_size - vec2(1));
            let bilinear = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            let source = min(vec2<u32>(texel) * divisor + vec2(divisor / 2u), full_size - vec2(1u));
            let separation = relative_separation(depth, load_depth(source));
            let fog = textureLoad(low_res_fog, texel, 0);
            let w = bilinear / (RELATIVE_DEPTH_TOLERANCE + separation);
            sum += fog * w;
            weight_sum += w;
            if (separation < nearest_separation) {
                nearest_separation = separation;
                nearest = fog;
            }
        }
    }

    // Every neighbour on another surface (a one-pixel sliver): take the one
    // whose depth is closest rather than a blend of the wrong ones.
    if (weight_sum < 1e-4) {
        return nearest;
    }
    return sum / weight_sum;
}
