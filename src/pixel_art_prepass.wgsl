//! Custom prepass fragment for pixel art models: writes alpha=1.0 to the normal
//! prepass texture so the edge detection shader draws outlines on these pixels.
//! (Mirrors terrain_prepass.wgsl but with alpha=1.0 instead of 0.0)

#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_prepass_functions,
}

#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    pbr_prepass_functions::prepass_alpha_discard(in);

    var out: FragmentOutput;

#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.unclipped_depth;
#endif

#ifdef NORMAL_PREPASS
    // Write correct normal (world space, packed to [0,1]) with alpha = 1.0
    // Alpha 1.0 tells the edge detection shader to draw outlines on this pixel
    out.normal = vec4(in.world_normal * 0.5 + vec3(0.5), 1.0);
#endif

#ifdef MOTION_VECTOR_PREPASS
    out.motion_vector = pbr_prepass_functions::calculate_motion_vector(
        in.world_position, in.previous_world_position
    );
#endif

    return out;
}
#else
@fragment
fn fragment(in: VertexOutput) {
    pbr_prepass_functions::prepass_alpha_discard(in);
}
#endif
