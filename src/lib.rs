use bevy::asset::embedded_asset;
use bevy::pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{AsBindGroup, ShaderType};
use bevy::shader::ShaderRef;

// ============================================================================
// Public types
// ============================================================================

/// Material type alias: StandardMaterial + PixelArtExtension.
pub type PixelArtMaterial = ExtendedMaterial<StandardMaterial, PixelArtExtension>;

/// Material type alias: StandardMaterial + HoldoutExtension.
/// Invisible occluder that writes depth but outputs fully transparent color.
/// Use on duplicated geometry in the low-res layer to occlude pixel art entities
/// behind full-res scene geometry (terrain, walls).
pub type HoldoutMaterial = ExtendedMaterial<StandardMaterial, HoldoutExtension>;

/// Material extension for pixel art rendering of 3D models.
/// Integrates with Bevy's full PBR lighting, then post-processes:
///   - Toon quantize the PBR lighting result
///   - CIELAB palette quantization + screen-space Bayer dithering
/// Prepass writes alpha=1.0 so edge detection outlines are enabled.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct PixelArtExtension {
    #[uniform(100)]
    pub params: PixelArtShaderParams,
}

impl MaterialExtension for PixelArtExtension {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_pixel_art_shader/pixel_art.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "embedded://bevy_pixel_art_shader/pixel_art_prepass.wgsl".into()
    }
}

/// Material extension for holdout/occluder rendering.
/// Writes depth to the depth buffer while outputting fully transparent color.
/// The prepass writes alpha=0.0 so edge detection ignores holdout geometry.
#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct HoldoutExtension {}

impl MaterialExtension for HoldoutExtension {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_pixel_art_shader/holdout.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "embedded://bevy_pixel_art_shader/holdout_prepass.wgsl".into()
    }
}

/// GPU-side pixel art parameters. Must match the WGSL struct layout exactly.
#[derive(Clone, Debug, ShaderType)]
pub struct PixelArtShaderParams {
    /// Base tint color (linear RGBA). Replaces the model's base_color.
    pub base_tint: Vec4,
    /// Number of toon shading bands (default: 3.0).
    pub toon_bands: f32,
    /// Softness of toon band transitions (default: 0.05).
    pub toon_softness: f32,
    /// Minimum brightness in shadow areas (default: 0.3).
    pub toon_shadow_floor: f32,
    /// Screen-space dither pattern scale: 1.0 = 1 Bayer cell per pixel (default).
    pub dither_density: f32,
    /// Number of active palette colors (0 = disable quantization, max 64).
    pub palette_count: u32,
    /// Blend strength toward palette colors (0.0..1.0, default: 1.0).
    pub palette_strength: f32,
    /// Bayer dither strength (0 = off, 1.0 = full, default: 0.3).
    pub dither_strength: f32,
    /// Debug visualization stage (0=full, 1=PBR only, 2=+toon, 3=+palette, 4=+dither).
    pub debug_stage: u32,
    /// Palette colors in linear RGB (max 64 entries, stored as Vec4 for alignment).
    pub palette_colors: [Vec4; 64],
}

impl Default for PixelArtShaderParams {
    fn default() -> Self {
        let (palette, count) = default_pixel_art_palette();
        Self {
            base_tint: Vec4::new(1.0, 1.0, 1.0, 1.0),
            toon_bands: 10.0,
            toon_softness: 0.0,
            toon_shadow_floor: 0.1,
            dither_density: 1.0,
            palette_count: count,
            palette_strength: 0.25,
            dither_strength: 0.3,
            debug_stage: 0,
            palette_colors: palette,
        }
    }
}

// ============================================================================
// Plugin
// ============================================================================

pub struct PixelArtShaderPlugin;

impl Plugin for PixelArtShaderPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "pixel_art.wgsl");
        embedded_asset!(app, "pixel_art_prepass.wgsl");
        embedded_asset!(app, "holdout.wgsl");
        embedded_asset!(app, "holdout_prepass.wgsl");

        app.add_plugins(MaterialPlugin::<PixelArtMaterial>::default());
        app.add_plugins(MaterialPlugin::<HoldoutMaterial>::default());
    }
}

// ============================================================================
// Palette helpers
// ============================================================================

/// Convert an sRGB u8 triplet to linear RGB Vec4 (alpha = 1.0).
fn srgb_to_linear_vec4(r: u8, g: u8, b: u8) -> Vec4 {
    let c = Color::srgb_u8(r, g, b).to_linear();
    Vec4::new(c.red, c.green, c.blue, 1.0)
}

/// 64-color pixel art palette (PICO-8 32 + DB32-inspired 32) for rendering.
/// Returns (palette_colors array, active count).
pub fn default_pixel_art_palette() -> ([Vec4; 64], u32) {
    let mut colors = [Vec4::ZERO; 64];

    // PICO-8 base palette (16 colors)
    colors[0] = srgb_to_linear_vec4(0, 0, 0);         // black
    colors[1] = srgb_to_linear_vec4(29, 43, 83);      // dark blue
    colors[2] = srgb_to_linear_vec4(126, 37, 83);     // dark purple
    colors[3] = srgb_to_linear_vec4(0, 135, 81);      // dark green
    colors[4] = srgb_to_linear_vec4(171, 82, 54);     // brown
    colors[5] = srgb_to_linear_vec4(95, 87, 79);      // dark grey
    colors[6] = srgb_to_linear_vec4(194, 195, 199);   // light grey
    colors[7] = srgb_to_linear_vec4(255, 241, 232);   // white
    colors[8] = srgb_to_linear_vec4(255, 0, 77);      // red
    colors[9] = srgb_to_linear_vec4(255, 163, 0);     // orange
    colors[10] = srgb_to_linear_vec4(255, 236, 39);   // yellow
    colors[11] = srgb_to_linear_vec4(0, 228, 54);     // green
    colors[12] = srgb_to_linear_vec4(41, 173, 255);   // blue
    colors[13] = srgb_to_linear_vec4(131, 118, 156);  // lavender
    colors[14] = srgb_to_linear_vec4(255, 119, 168);  // pink
    colors[15] = srgb_to_linear_vec4(255, 204, 170);  // peach

    // PICO-8 extended palette (16 colors)
    colors[16] = srgb_to_linear_vec4(41, 24, 20);     // dark brown
    colors[17] = srgb_to_linear_vec4(17, 29, 53);     // darker blue
    colors[18] = srgb_to_linear_vec4(66, 33, 54);     // dark magenta
    colors[19] = srgb_to_linear_vec4(18, 83, 89);     // dark teal
    colors[20] = srgb_to_linear_vec4(116, 47, 41);    // rust
    colors[21] = srgb_to_linear_vec4(73, 51, 59);     // mauve grey
    colors[22] = srgb_to_linear_vec4(162, 136, 121);  // tan
    colors[23] = srgb_to_linear_vec4(243, 239, 125);  // light yellow
    colors[24] = srgb_to_linear_vec4(190, 18, 80);    // crimson
    colors[25] = srgb_to_linear_vec4(255, 108, 36);   // bright orange
    colors[26] = srgb_to_linear_vec4(168, 231, 46);   // lime
    colors[27] = srgb_to_linear_vec4(0, 181, 67);     // forest green
    colors[28] = srgb_to_linear_vec4(6, 90, 181);     // royal blue
    colors[29] = srgb_to_linear_vec4(117, 70, 101);   // plum
    colors[30] = srgb_to_linear_vec4(255, 110, 89);   // salmon
    colors[31] = srgb_to_linear_vec4(255, 157, 129);  // light salmon

    // DB32-inspired extras (32 colors): earth tones, skin, sky, foliage, metal
    colors[32] = srgb_to_linear_vec4(34, 32, 52);     // void purple
    colors[33] = srgb_to_linear_vec4(69, 40, 60);     // wine
    colors[34] = srgb_to_linear_vec4(102, 57, 49);    // sienna
    colors[35] = srgb_to_linear_vec4(143, 86, 59);    // copper
    colors[36] = srgb_to_linear_vec4(223, 113, 38);   // tangerine
    colors[37] = srgb_to_linear_vec4(217, 160, 102);  // sand
    colors[38] = srgb_to_linear_vec4(238, 195, 154);  // skin light
    colors[39] = srgb_to_linear_vec4(251, 242, 54);   // lemon
    colors[40] = srgb_to_linear_vec4(153, 229, 80);   // grass
    colors[41] = srgb_to_linear_vec4(106, 190, 48);   // leaf
    colors[42] = srgb_to_linear_vec4(55, 148, 110);   // jade
    colors[43] = srgb_to_linear_vec4(75, 105, 47);    // moss
    colors[44] = srgb_to_linear_vec4(82, 75, 36);     // olive
    colors[45] = srgb_to_linear_vec4(50, 60, 57);     // slate green
    colors[46] = srgb_to_linear_vec4(63, 63, 116);    // storm blue
    colors[47] = srgb_to_linear_vec4(48, 96, 130);    // steel blue
    colors[48] = srgb_to_linear_vec4(91, 110, 225);   // cornflower
    colors[49] = srgb_to_linear_vec4(99, 155, 255);   // sky
    colors[50] = srgb_to_linear_vec4(95, 205, 228);   // cyan
    colors[51] = srgb_to_linear_vec4(203, 219, 252);  // ice
    colors[52] = srgb_to_linear_vec4(155, 173, 183);  // silver
    colors[53] = srgb_to_linear_vec4(132, 126, 135);  // pewter
    colors[54] = srgb_to_linear_vec4(105, 106, 106);  // iron
    colors[55] = srgb_to_linear_vec4(89, 86, 82);     // graphite
    colors[56] = srgb_to_linear_vec4(118, 66, 138);   // amethyst
    colors[57] = srgb_to_linear_vec4(172, 50, 50);    // brick red
    colors[58] = srgb_to_linear_vec4(217, 87, 99);    // rose
    colors[59] = srgb_to_linear_vec4(215, 123, 186);  // bubblegum
    colors[60] = srgb_to_linear_vec4(143, 151, 74);   // khaki
    colors[61] = srgb_to_linear_vec4(138, 111, 48);   // bronze
    colors[62] = srgb_to_linear_vec4(75, 47, 55);     // maroon
    colors[63] = srgb_to_linear_vec4(45, 45, 45);     // charcoal

    (colors, 64)
}
