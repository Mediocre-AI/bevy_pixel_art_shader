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
///   - CIELAB palette quantization + world-space Bayer dithering
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
    /// World-space dither pattern density: cells per world unit (default: 8.0).
    pub dither_density: f32,
    /// Number of active palette colors (0 = disable quantization, max 32).
    pub palette_count: u32,
    /// Blend strength toward palette colors (0.0..1.0, default: 1.0).
    pub palette_strength: f32,
    /// Bayer dither strength (0 = off, 1.0 = full, default: 0.3).
    pub dither_strength: f32,
    /// Padding for 16-byte alignment.
    pub _pad: f32,
    /// Palette colors in linear RGB (max 32 entries, stored as Vec4 for alignment).
    pub palette_colors: [Vec4; 32],
}

impl Default for PixelArtShaderParams {
    fn default() -> Self {
        let (palette, count) = default_pixel_art_palette();
        Self {
            base_tint: Vec4::new(1.0, 1.0, 1.0, 1.0),
            toon_bands: 3.0,
            toon_softness: 0.0,    // hard band edges (pixel art look)
            toon_shadow_floor: 0.2,
            dither_density: 8.0,    // 8 dither cells per world unit
            palette_count: count,
            palette_strength: 1.0,
            dither_strength: 0.3,   // subtle, boundary-only dithering
            _pad: 0.0,
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

/// PICO-8 inspired 16-color palette for pixel art rendering.
/// Returns (palette_colors array, active count).
pub fn default_pixel_art_palette() -> ([Vec4; 32], u32) {
    let mut colors = [Vec4::ZERO; 32];

    // PICO-8 palette (16 colors)
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

    (colors, 16)
}
