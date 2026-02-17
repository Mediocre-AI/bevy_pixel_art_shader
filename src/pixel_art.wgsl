//! Pixel art fragment shader for 3D models.
//!
//! Integrates with Bevy's full PBR lighting pipeline, then post-processes:
//!   1. Toon quantize the PBR lighting result (hard band edges)
//!   2. CIELAB palette quantization
//!   3. World-space Bayer dithering (only at band boundaries)
//!
//! debug_stage controls which stages are applied:
//!   0 = full pipeline, 1 = PBR only, 2 = +toon, 3 = +palette, 4 = +dither

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

// ============================================================================
// Extension uniform (binding 100)
// ============================================================================

struct PixelArtParams {
    base_tint: vec4<f32>,
    toon_bands: f32,
    toon_softness: f32,
    toon_shadow_floor: f32,
    dither_density: f32,
    palette_count: u32,
    palette_strength: f32,
    dither_strength: f32,
    debug_stage: u32,              // 0=full, 1=PBR, 2=+toon, 3=+palette, 4=+dither
    palette_colors: array<vec4<f32>, 32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> pixel_art: PixelArtParams;

// ============================================================================
// Toon quantization (hard edge version)
// ============================================================================

fn toon_quantize(value: f32, bands: f32, softness: f32) -> f32 {
    if (softness < 0.001) {
        // Hard edge: snap to nearest band
        return round(value * bands) / bands;
    }
    let nearest = round(value * bands) / bands;
    return smoothstep(nearest - softness, nearest + softness, value);
}

// ============================================================================
// CIELAB color conversion
// ============================================================================

fn linear_rgb_to_xyz(rgb: vec3<f32>) -> vec3<f32> {
    let x = dot(vec3<f32>(0.4124564, 0.3575761, 0.1804375), rgb);
    let y = dot(vec3<f32>(0.2126729, 0.7151522, 0.0721750), rgb);
    let z = dot(vec3<f32>(0.0193339, 0.1191920, 0.9503041), rgb);
    return vec3<f32>(x, y, z);
}

fn lab_f(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if (t > delta * delta * delta) {
        return pow(t, 1.0 / 3.0);
    } else {
        return t / (3.0 * delta * delta) + 4.0 / 29.0;
    }
}

fn xyz_to_lab(xyz: vec3<f32>) -> vec3<f32> {
    let white = vec3<f32>(0.95047, 1.00000, 1.08883);
    let scaled = xyz / white;
    let fx = lab_f(scaled.x);
    let fy = lab_f(scaled.y);
    let fz = lab_f(scaled.z);
    return vec3<f32>(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

fn linear_rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    return xyz_to_lab(linear_rgb_to_xyz(rgb));
}

// ============================================================================
// Palette matching (CIELAB nearest-neighbor)
// ============================================================================

struct PaletteMatch {
    nearest_rgb: vec3<f32>,
    second_rgb: vec3<f32>,
    blend: f32,
}

fn find_palette_match(color: vec3<f32>) -> PaletteMatch {
    let lab = linear_rgb_to_lab(color);

    var d1: f32 = 1e10;
    var d2: f32 = 1e10;
    var c1: vec3<f32> = color;
    var c2: vec3<f32> = color;

    let count = pixel_art.palette_count;
    for (var i: u32 = 0u; i < count; i++) {
        let pal_rgb = pixel_art.palette_colors[i].rgb;
        let pal_lab = linear_rgb_to_lab(pal_rgb);
        let dist = distance(lab, pal_lab);

        if (dist < d1) {
            d2 = d1;
            c2 = c1;
            d1 = dist;
            c1 = pal_rgb;
        } else if (dist < d2) {
            d2 = dist;
            c2 = pal_rgb;
        }
    }

    var result: PaletteMatch;
    result.nearest_rgb = c1;
    result.second_rgb = c2;
    let total = d1 + d2;
    if (total > 0.001) {
        result.blend = d1 / total;
    } else {
        result.blend = 0.0;
    }
    return result;
}

// ============================================================================
// 4x4 Bayer dithering matrix
// ============================================================================

fn bayer4x4(pos: vec2<f32>) -> f32 {
    let x = u32(pos.x) % 4u;
    let y = u32(pos.y) % 4u;
    var matrix = array<array<f32, 4>, 4>(
        array<f32, 4>( 0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0),
        array<f32, 4>(12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0),
        array<f32, 4>( 3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0),
        array<f32, 4>(15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0),
    );
    return matrix[y][x];
}

// ============================================================================
// Main fragment
// ============================================================================

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // --- 1. Build PBR input from base StandardMaterial ---
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Override base color with our tint
    pbr_input.material.base_color = pixel_art.base_tint;

    // Alpha discard
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material,
        pbr_input.material.base_color,
    );

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;

    // --- 2. Bevy PBR lighting (all scene lights, shadows, IBL) ---
    out.color = apply_pbr_lighting(pbr_input);
    var color = out.color.rgb;

    // Stage 1: PBR only — stop here
    if (pixel_art.debug_stage == 1u) {
        out.color = vec4<f32>(color, out.color.a);
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
        return out;
    }

    // --- 3. Toon quantize the lit result (hard band edges) ---
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (luminance > 0.001) {
        let toon_lum = toon_quantize(luminance, pixel_art.toon_bands, pixel_art.toon_softness);
        let final_lum = mix(pixel_art.toon_shadow_floor, 1.0, toon_lum);
        color = color * (final_lum / luminance);
    } else {
        color = pixel_art.base_tint.rgb * pixel_art.toon_shadow_floor;
    }
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));

    // Stage 2: PBR + Toon — stop here
    if (pixel_art.debug_stage == 2u) {
        out.color = vec4<f32>(color, out.color.a);
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
        return out;
    }

    // --- 4. CIELAB palette quantization ---
    if (pixel_art.palette_count > 0u) {
        let pm = find_palette_match(color);
        var quantized = pm.nearest_rgb;

        // Stage 3: +Palette (no dither) — skip dithering
        if (pixel_art.debug_stage != 3u && pixel_art.dither_strength > 0.0) {
            let boundary_mask = smoothstep(0.1, 0.35, pm.blend);
            let effective_dither = boundary_mask * pixel_art.dither_strength;

            if (effective_dither > 0.0) {
                let threshold = bayer4x4(floor(in.world_position.xz * pixel_art.dither_density));
                if (pm.blend * effective_dither > threshold) {
                    quantized = pm.second_rgb;
                }
            }
        }

        color = mix(color, quantized, pixel_art.palette_strength);
    }

    out.color = vec4<f32>(color, out.color.a);

    // --- 5. Post-lighting (fog, tonemapping, etc.) ---
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
