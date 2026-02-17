//! Holdout prepass: writes depth (for occlusion) and normal alpha=0.0
//! so edge detection ignores holdout geometry.

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
    // Alpha 0.0 → edge detection suppressed on holdout pixels
    out.normal = vec4(in.world_normal * 0.5 + vec3(0.5), 0.0);
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
