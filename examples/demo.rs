//! Standalone demo: pixel art pipeline with holdout occluders + debug stage viewer.
//!
//! Architecture:
//!   Full-res 3D Camera (layer 0) → window         (terrain, reference objects)
//!   Low-res 3D Camera  (layer 1) → 320×180 texture (pixel art + holdout + EdgeDetection)
//!   UI ImageNode                  → canvas overlay  (nearest upscale on top of full-res scene)
//!
//! Run:  cargo run --example demo

use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::image::ImageSampler;
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy_edge_detection_outline::{EdgeDetection, EdgeDetectionPlugin};
use bevy_egui::{
    EguiContext, EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass,
    PrimaryEguiContext, egui,
};
use bevy_pixel_art_shader::{
    HoldoutExtension, HoldoutMaterial, PixelArtExtension, PixelArtMaterial, PixelArtShaderParams,
    PixelArtShaderPlugin, default_pixel_art_palette,
};

const RES_WIDTH: u32 = 320;
const RES_HEIGHT: u32 = 180;
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
        // Prevent bevy_egui from auto-attaching to the first camera (the low-res one).
        // We manually add PrimaryEguiContext to the window camera instead.
        .insert_resource(EguiGlobalSettings {
            auto_create_primary_context: false,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate_models, swap_glb_materials))
        .add_systems(EguiPrimaryContextPass, debug_ui)
        .run();
}

#[derive(Component)]
struct Spinning;

#[derive(Component)]
struct NeedsMaterialSwap;

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

    let mut canvas_image = Image::new_target_texture(
        RES_WIDTH,
        RES_HEIGHT,
        TextureFormat::Rgba8Unorm,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    canvas_image.sampler = ImageSampler::nearest();
    let image_handle = images.add(canvas_image);

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

    let sphere_mesh = meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap());

    // ================================================================
    //  Layer 0: full-res scene
    // ================================================================

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

    // GLB reference (standard PBR, full-res)
    let glb_handle_ref: Handle<Scene> = asset_server.load("demo_model.glb#Scene0");
    commands.spawn((
        Name::new("GLB Reference"),
        SceneRoot(glb_handle_ref),
        Transform::from_xyz(-3.0, 0.0, -4.0).with_scale(Vec3::splat(1.5)),
        Spinning,
    ));

    // ================================================================
    //  Layer 1: pixel art sphere cluster + holdout
    // ================================================================

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

    // Overlapping sphere cluster — various colors and sizes
    let cluster = [
        // (position, radius_scale, color)
        (Vec3::new(0.0, 1.2, 0.0), 1.2, Color::srgb(0.9, 0.15, 0.15)),  // red (center)
        (Vec3::new(1.3, 0.9, 0.3), 0.9, Color::srgb(0.15, 0.7, 0.2)),   // green
        (Vec3::new(-1.1, 1.0, 0.5), 1.0, Color::srgb(0.2, 0.3, 0.9)),   // blue
        (Vec3::new(0.5, 1.8, -0.3), 0.7, Color::srgb(0.95, 0.8, 0.1)),  // yellow
        (Vec3::new(-0.5, 0.6, 1.0), 0.6, Color::srgb(0.8, 0.3, 0.8)),   // purple
        (Vec3::new(0.8, 0.5, -0.8), 0.5, Color::srgb(0.1, 0.8, 0.8)),   // cyan
        (Vec3::new(-0.3, 2.2, 0.2), 0.55, Color::srgb(0.95, 0.5, 0.1)), // orange
        (Vec3::new(1.5, 1.5, -0.5), 0.65, Color::srgb(0.9, 0.4, 0.6)),  // pink
    ];

    for (i, (pos, scale, color)) in cluster.iter().enumerate() {
        let mat = make_pixel_mat(&mut pixel_materials, *color);
        commands.spawn((
            Name::new(format!("Sphere {i}")),
            Mesh3d(sphere_mesh.clone()),
            MeshMaterial3d(mat),
            Transform::from_translation(*pos).with_scale(Vec3::splat(*scale)),
            Spinning,
            PIXEL_ART_LAYER,
        ));
    }

    // Cube and torus for shape variety
    let cube_mat = make_pixel_mat(&mut pixel_materials, Color::srgb(0.2, 0.8, 0.3));
    commands.spawn((
        Name::new("Cube"),
        Mesh3d(meshes.add(Cuboid::new(1.5, 1.5, 1.5))),
        MeshMaterial3d(cube_mat),
        Transform::from_xyz(3.5, 0.75, 0.0),
        Spinning,
        PIXEL_ART_LAYER,
    ));

    let torus_mat = make_pixel_mat(&mut pixel_materials, Color::srgb(0.2, 0.4, 0.9));
    commands.spawn((
        Name::new("Torus"),
        Mesh3d(meshes.add(Torus::new(0.4, 0.8))),
        MeshMaterial3d(torus_mat),
        Transform::from_xyz(5.5, 1.0, 0.0),
        Spinning,
        PIXEL_ART_LAYER,
    ));

    // GLB model (pixel art)
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
    //  Light (both layers)
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

    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            clear_color: Color::srgb(0.2, 0.05, 0.3).into(), // purple bg for debug visibility
            ..default()
        },
        camera_transform,
        // Attach egui to the window camera (not the low-res render-to-texture one).
        EguiContext::default(),
        PrimaryEguiContext,
    ));

    // ================================================================
    //  UI overlay
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

fn rotate_models(time: Res<Time>, mut q: Query<&mut Transform, With<Spinning>>) {
    let angle = time.delta_secs() * 0.3;
    for mut t in q.iter_mut() {
        t.rotate_y(angle);
    }
}

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

// ============================================================================
//  Debug UI
// ============================================================================

const STAGE_LABELS: &[&str] = &[
    "0: Full Pipeline",
    "1: PBR Only",
    "2: PBR + Toon",
    "3: PBR + Toon + Palette",
    "4: PBR + Toon + Palette + Dither",
];

fn debug_ui(
    mut contexts: EguiContexts,
    mut pixel_materials: ResMut<Assets<PixelArtMaterial>>,
    mut edge_q: Query<&mut EdgeDetection, With<PixelArtCamera>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("Pixel Art Settings")
        .default_width(280.0)
        .default_pos([10.0, 10.0])
        .collapsible(true)
        .show(ctx, |ui| {
            // ── Pipeline Stage Selector ──
            ui.heading("Pipeline Stage");
            ui.label("Select which stages are applied:");

            let handles: Vec<_> = pixel_materials
                .iter()
                .map(|(id, _)| AssetId::from(id))
                .collect();

            let current_stage = handles
                .first()
                .and_then(|id| pixel_materials.get(*id))
                .map(|m| m.extension.params.debug_stage)
                .unwrap_or(0);

            let mut selected = current_stage;
            for (i, label) in STAGE_LABELS.iter().enumerate() {
                ui.radio_value(&mut selected, i as u32, *label);
            }

            if selected != current_stage {
                for id in &handles {
                    if let Some(mat) = pixel_materials.get_mut(*id) {
                        mat.extension.params.debug_stage = selected;
                    }
                }
            }

            // Edge detection toggle (stage 5 effectively)
            if let Ok(mut ed) = edge_q.single_mut() {
                let mut edges_on = ed.enable_depth || ed.enable_normal;
                if ui.checkbox(&mut edges_on, "Edge Detection Outlines").changed() {
                    ed.enable_depth = edges_on;
                    ed.enable_normal = edges_on;
                }
            }

            ui.separator();

            // ── Parameter sliders ──
            ui.collapsing("Pixel Art Params", |ui| {
                if let Some(first_id) = handles.first() {
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

            ui.collapsing("Edge Detection Params", |ui| {
                if let Ok(mut ed) = edge_q.single_mut() {
                    ui.add(
                        egui::Slider::new(&mut ed.depth_threshold, 0.0..=5.0)
                            .text("Depth Threshold"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.normal_threshold, 0.0..=2.0)
                            .text("Normal Threshold"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.depth_thickness, 0.0..=5.0)
                            .text("Depth Thickness"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.normal_thickness, 0.0..=5.0)
                            .text("Normal Thickness"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.steep_angle_threshold, 0.0..=1.0)
                            .text("Steep Threshold"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.steep_angle_multiplier, 0.0..=5.0)
                            .text("Steep Multiplier"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.block_pixel, 1..=4).text("Block Pixel"),
                    );
                    ui.add(
                        egui::Slider::new(&mut ed.flat_rejection_threshold, 0.0..=1.0)
                            .text("Flat Rejection"),
                    );

                    let mut color = [
                        ed.edge_color.to_srgba().red,
                        ed.edge_color.to_srgba().green,
                        ed.edge_color.to_srgba().blue,
                    ];
                    if ui.color_edit_button_rgb(&mut color).changed() {
                        ed.edge_color = Color::srgb(color[0], color[1], color[2]);
                    }
                    ui.label("Edge Color");
                }
            });
        });
}
