#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var fullres_color: texture_2d<f32>;
@group(0) @binding(1) var fullres_depth: texture_depth_2d;
@group(0) @binding(2) var lowres_color: texture_2d<f32>;
@group(0) @binding(3) var lowres_depth: texture_depth_2d;
@group(0) @binding(4) var nearest_sampler: sampler;

struct CompositorSettings {
    depth_bias: f32,
}
@group(0) @binding(5) var<uniform> settings: CompositorSettings;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let fr_color = textureSample(fullres_color, nearest_sampler, in.uv);
    let fr_depth = textureSample(fullres_depth, nearest_sampler, in.uv);
    let lr_color = textureSample(lowres_color, nearest_sampler, in.uv);
    let lr_depth = textureSample(lowres_depth, nearest_sampler, in.uv);

    // Bevy reversed-Z: 1.0 = near, 0.0 = far
    // Scale bias by depth — near objects (d≈1) get full bias,
    // far objects (d≈0) get proportionally less, matching the
    // depth-buffer mismatch between lowres and fullres.
    let effective_bias = settings.depth_bias * fr_depth;

    if lr_color.a > 0.01 && lr_depth >= fr_depth - effective_bias {
        return lr_color;
    }
    return fr_color;
}
