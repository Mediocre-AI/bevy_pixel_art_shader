//! Holdout material: outputs transparent color while writing depth.
//! Used as an invisible occluder on the low-res layer to clip pixel art
//! entities behind full-res geometry (terrain, walls).

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::forward_io::{VertexOutput, FragmentOutput}
#endif

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
#ifdef PREPASS_PIPELINE
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    // Fully transparent — depth is written by the opaque pass
    out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
#endif
    return out;
}
