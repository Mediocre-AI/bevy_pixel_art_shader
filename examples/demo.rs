//! Standalone demo: 3-camera pixel art pipeline with holdout occluders.
//!
//! Architecture:
//!   Full-res 3D Camera (layer 0) → window         (terrain, reference objects, UI overlay)
//!   Low-res 3D Camera  (layer 1) → 320×180 texture (pixel art + holdout + EdgeDetection)
//!   UI ImageNode                  → canvas overlay  (nearest upscale on top of full-res scene)
//!
//! Holdout ground on layer 1 writes depth but outputs transparent,
//! so pixel art entities behind terrain are properly occluded.
//!
//! Run:  cargo run --example demo

use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::image::ImageSampler;
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy_edge_detection_outline::{EdgeDetection, EdgeDetectionPlugin};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use bevy_pixel_art_shader::{
    HoldoutExtension, HoldoutMaterial, PixelArtExtension, PixelArtMaterial, PixelArtShaderParams,
    PixelArtShaderPlugin, default_pixel_art_palette,
};

/// Low-resolution canvas dimensions.
const RES_WIDTH: u32 = 320;
const RES_HEIGHT: u32 = 180;

/// Layer 1: low-res pixel art entities + holdout occluders.
const PIXEL_ART_LAYER: RenderLayers = RenderLayers::layer(1);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Pixel Art Shader Demo".to_string(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(PixelArtShaderPlugin)
        .add_plugins(EdgeDetectionPlugin::default())
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate_models, swap_glb_materials, debug_ui))
        .run();
}

/// Tag for spinning models.
#[derive(Component)]
struct Spinning;

/// Marker: GLB scene needs StandardMaterial → PixelArtMaterial swap.
#[derive(Component)]
struct NeedsMaterialSwap;

/// Marker for the low-res pixel art camera.
#[derive(Component)]
struct PixelArtCamera;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut pixel_materials: ResMut<Assets<PixelArtMaterial>>,
    mut holdout_materials: ResMut<Assets<HoldoutMaterial>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let (palette, palette_count) = default_pixel_art_palette();

    // ---- Low-res render target (320×180, nearest-neighbor, transparent) ----
    let mut canvas_image = Image::new_target_texture(
        RES_WIDTH,
        RES_HEIGHT,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    canvas_image.sampler = ImageSampler::nearest();
    let image_handle = images.add(canvas_image);

    // --- Helper: create a PixelArtMaterial with a given tint ---
    let make_pixel_mat = |mats: &mut Assets<PixelArtMaterial>, color: Color| {
        let linear = color.to_linear();
        mats.add(ExtendedMaterial {
            base: StandardMaterial {
                perceptual_roughness: 1.0,
                reflectance: 0.0,
                ..default()
            },
            extension: PixelArtExtension {
                params: PixelArtShaderParams {
                    base_tint: Vec4::new(linear.red, linear.green, linear.blue, 1.0),
                    palette_colors: palette,
                    palette_count,
                    ..default()
                },
            },
        })
    };

    // ================================================================
    //  Layer 0: full-res scene (terrain, reference objects)
    // ================================================================

    // Ground plane (visible, full-res)
    let ground_mat = std_materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.35, 0.35),
        ..default()
    });
    let ground_mesh = meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(20.0)));
    commands.spawn((
        Name::new("Ground"),
        Mesh3d(ground_mesh.clone()),
        MeshMaterial3d(ground_mat),
        Transform::from_xyz(1.5, -0.01, -2.0),
    ));

    // Reference objects (standard PBR, full-res)
    let ref_sphere_mat = std_materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.2, 0.2),
        ..default()
    });
    commands.spawn((
        Name::new("Ref Sphere"),
        Mesh3d(meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap())),
        MeshMaterial3d(ref_sphere_mat),
        Transform::from_xyz(0.0, 1.0, -4.0),
        Spinning,
    ));

    let ref_cube_mat = std_materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.8, 0.3),
        ..default()
    });
    commands.spawn((
        Name::new("Ref Cube"),
        Mesh3d(meshes.add(Cuboid::new(1.5, 1.5, 1.5))),
        MeshMaterial3d(ref_cube_mat),
        Transform::from_xyz(3.0, 0.75, -4.0),
        Spinning,
    ));

    let ref_torus_mat = std_materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.4, 0.9),
        ..default()
    });
    commands.spawn((
        Name::new("Ref Torus"),
        Mesh3d(meshes.add(Torus::new(0.4, 0.8))),
        MeshMaterial3d(ref_torus_mat),
        Transform::from_xyz(6.0, 1.0, -4.0),
        Spinning,
    ));

    // GLB reference (standard PBR, full-res)
    let glb_handle_ref: Handle<Scene> = asset_server.load("demo_model.glb#Scene0");
    commands.spawn((
        Name::new("GLB Reference"),
        SceneRoot(glb_handle_ref),
        Transform::from_xyz(-3.0, 0.0, -4.0).with_scale(Vec3::splat(1.5)),
        Spinning,
    ));

    // ================================================================
    //  Layer 1: low-res pixel art + holdout occluders
    // ================================================================

    // Holdout ground (invisible occluder — same mesh/position as real ground)
    let holdout_mat = holdout_materials.add(ExtendedMaterial {
        base: StandardMaterial::default(),
        extension: HoldoutExtension {},
    });
    commands.spawn((
        Name::new("Holdout Ground"),
        Mesh3d(ground_mesh),
        MeshMaterial3d(holdout_mat),
        Transform::from_xyz(1.5, -0.01, -2.0),
        PIXEL_ART_LAYER,
    ));

    // Pixel art primitives
    let sphere_mat = make_pixel_mat(&mut pixel_materials, Color::srgb(0.9, 0.2, 0.2));
    commands.spawn((
        Name::new("Sphere"),
        Mesh3d(meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap())),
        MeshMaterial3d(sphere_mat),
        Transform::from_xyz(0.0, 1.0, 0.0),
        Spinning,
        PIXEL_ART_LAYER,
    ));

    let cube_mat = make_pixel_mat(&mut pixel_materials, Color::srgb(0.2, 0.8, 0.3));
    commands.spawn((
        Name::new("Cube"),
        Mesh3d(meshes.add(Cuboid::new(1.5, 1.5, 1.5))),
        MeshMaterial3d(cube_mat),
        Transform::from_xyz(3.0, 0.75, 0.0),
        Spinning,
        PIXEL_ART_LAYER,
    ));

    let torus_mat = make_pixel_mat(&mut pixel_materials, Color::srgb(0.2, 0.4, 0.9));
    commands.spawn((
        Name::new("Torus"),
        Mesh3d(meshes.add(Torus::new(0.4, 0.8))),
        MeshMaterial3d(torus_mat),
        Transform::from_xyz(6.0, 1.0, 0.0),
        Spinning,
        PIXEL_ART_LAYER,
    ));

    // GLB model (pixel art — material swap happens async)
    let glb_handle: Handle<Scene> = asset_server.load("demo_model.glb#Scene0");
    commands.spawn((
        Name::new("GLB PixelArt"),
        SceneRoot(glb_handle),
        Transform::from_xyz(-3.0, 0.0, 0.0).with_scale(Vec3::splat(1.5)),
        Spinning,
        NeedsMaterialSwap,
        PIXEL_ART_LAYER,
    ));

    // ================================================================
    //  Light (visible to both layers)
    // ================================================================
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.5, 0.0)),
        RenderLayers::layer(0).with(1),
    ));

    // ================================================================
    //  Cameras
    // ================================================================

    let camera_transform =
        Transform::from_xyz(3.0, 5.0, 10.0).looking_at(Vec3::new(1.5, 1.0, -2.0), Vec3::Y);

    // Low-res 3D camera → 320×180 texture (transparent clear for compositing)
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -1,
            clear_color: Color::srgba(0.0, 0.0, 0.0, 0.0).into(),
            ..default()
        },
        RenderTarget::Image(image_handle.clone().into()),
        Msaa::Off,
        camera_transform,
        PIXEL_ART_LAYER,
        EdgeDetection::default(),
        PixelArtCamera,
    ));

    // Full-res 3D camera → window (layer 0, default)
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            clear_color: Color::srgb(0.1, 0.1, 0.1).into(),
            ..default()
        },
        camera_transform,
    ));

    // ================================================================
    //  UI overlay: canvas image on top of full-res scene
    // ================================================================
    commands.spawn((
        ImageNode::new(image_handle),
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            position_type: PositionType::Absolute,
            ..default()
        },
    ));
}

/// Slowly rotate all Spinning entities.
fn rotate_models(time: Res<Time>, mut q: Query<&mut Transform, With<Spinning>>) {
    let angle = time.delta_secs() * 0.3;
    for mut t in q.iter_mut() {
        t.rotate_y(angle);
    }
}

/// Swap GLB scene's StandardMaterial children to PixelArtMaterial + move to pixel art layer.
fn swap_glb_materials(
    mut commands: Commands,
    query: Query<(Entity, &Children), With<NeedsMaterialSwap>>,
    children_query: Query<&Children>,
    mat_query: Query<&MeshMaterial3d<StandardMaterial>>,
    mut pixel_materials: ResMut<Assets<PixelArtMaterial>>,
) {
    let (palette, palette_count) = default_pixel_art_palette();

    for (entity, top_children) in query.iter() {
        let mut swapped_any = false;

        let mut stack: Vec<Entity> = top_children.iter().collect();
        while let Some(child) = stack.pop() {
            if mat_query.get(child).is_ok() {
                let tint = Color::srgb(0.8, 0.5, 0.2);
                let linear = tint.to_linear();
                commands
                    .entity(child)
                    .remove::<MeshMaterial3d<StandardMaterial>>()
                    .insert((
                        MeshMaterial3d(pixel_materials.add(ExtendedMaterial {
                            base: StandardMaterial {
                                perceptual_roughness: 1.0,
                                reflectance: 0.0,
                                ..default()
                            },
                            extension: PixelArtExtension {
                                params: PixelArtShaderParams {
                                    base_tint: Vec4::new(
                                        linear.red,
                                        linear.green,
                                        linear.blue,
                                        1.0,
                                    ),
                                    palette_colors: palette,
                                    palette_count,
                                    ..default()
                                },
                            },
                        })),
                        PIXEL_ART_LAYER,
                    ));
                swapped_any = true;
            }
            if let Ok(grandchildren) = children_query.get(child) {
                stack.extend(grandchildren.iter());
            }
        }

        if swapped_any {
            commands.entity(entity).remove::<NeedsMaterialSwap>();
        }
    }
}

/// egui debug panel for live shader parameter tweaking.
fn debug_ui(
    mut contexts: EguiContexts,
    mut pixel_materials: ResMut<Assets<PixelArtMaterial>>,
    mut edge_q: Query<&mut EdgeDetection, With<PixelArtCamera>>,
) {
    egui::Window::new("Pixel Art Settings")
        .default_width(300.0)
        .show(contexts.ctx_mut().unwrap(), |ui| {
            // ── PixelArt Material params ──
            ui.collapsing("Pixel Art Shader", |ui| {
                // Collect handles first to avoid borrow conflict
                let handles: Vec<_> = pixel_materials
                    .iter()
                    .map(|(id, _)| AssetId::from(id))
                    .collect();

                if let Some(first_id) = handles.first() {
                    // Read current values from first material
                    let params = pixel_materials
                        .get(*first_id)
                        .unwrap()
                        .extension
                        .params
                        .clone();

                    let mut toon_bands = params.toon_bands;
                    let mut toon_softness = params.toon_softness;
                    let mut toon_shadow_floor = params.toon_shadow_floor;
                    let mut dither_density = params.dither_density;
                    let mut palette_strength = params.palette_strength;
                    let mut dither_strength = params.dither_strength;
                    let mut palette_count = params.palette_count;

                    let mut changed = false;

                    ui.label("Toon Shading");
                    changed |= ui
                        .add(egui::Slider::new(&mut toon_bands, 1.0..=10.0).text("Bands"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut toon_softness, 0.0..=0.5).text("Softness"))
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut toon_shadow_floor, 0.0..=1.0)
                                .text("Shadow Floor"),
                        )
                        .changed();

                    ui.separator();
                    ui.label("Palette & Dithering");
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut palette_count, 0..=16).text("Palette Colors"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut palette_strength, 0.0..=1.0)
                                .text("Palette Strength"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut dither_density, 1.0..=32.0)
                                .text("Dither Density"),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut dither_strength, 0.0..=1.0)
                                .text("Dither Strength"),
                        )
                        .changed();

                    if changed {
                        for id in &handles {
                            if let Some(mat) = pixel_materials.get_mut(*id) {
                                mat.extension.params.toon_bands = toon_bands;
                                mat.extension.params.toon_softness = toon_softness;
                                mat.extension.params.toon_shadow_floor = toon_shadow_floor;
                                mat.extension.params.dither_density = dither_density;
                                mat.extension.params.palette_strength = palette_strength;
                                mat.extension.params.dither_strength = dither_strength;
                                mat.extension.params.palette_count = palette_count;
                            }
                        }
                    }
                }
            });

            // ── EdgeDetection params ──
            ui.collapsing("Edge Detection", |ui| {
                if let Ok(mut ed) = edge_q.single_mut() {
                    ui.label("Thresholds");
                    ui.add(
                        egui::Slider::new(&mut ed.depth_threshold, 0.0..=5.0)
                            .text("Depth Threshold"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.normal_threshold, 0.0..=2.0)
                            .text("Normal Threshold"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.color_threshold, 0.0..=1.0)
                            .text("Color Threshold"),
                    );

                    ui.separator();
                    ui.label("Thickness");
                    ui.add(
                        egui::Slider::new(&mut ed.depth_thickness, 0.0..=5.0)
                            .text("Depth Thickness"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.normal_thickness, 0.0..=5.0)
                            .text("Normal Thickness"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.color_thickness, 0.0..=5.0)
                            .text("Color Thickness"),
                    );

                    ui.separator();
                    ui.label("Steep Angle");
                    ui.add(
                        egui::Slider::new(&mut ed.steep_angle_threshold, 0.0..=1.0)
                            .text("Threshold"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.steep_angle_multiplier, 0.0..=5.0)
                            .text("Multiplier"),
                    );

                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut ed.block_pixel, 1..=4).text("Block Pixel"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.flat_rejection_threshold, 0.0..=1.0)
                            .text("Flat Rejection"),
                    );

                    ui.separator();
                    let mut color = [
                        ed.edge_color.to_srgba().red,
                        ed.edge_color.to_srgba().green,
                        ed.edge_color.to_srgba().blue,
                    ];
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        ed.edge_color = Color::srgb(color[0], color[1], color[2]);
                    }
                    ui.label("Edge Color");

                    ui.separator();
                    ui.checkbox(&mut ed.enable_depth, "Enable Depth Edges");
                    ui.checkbox(&mut ed.enable_normal, "Enable Normal Edges");
                    ui.checkbox(&mut ed.enable_color, "Enable Color Edges");
                }
            });
        });
}
