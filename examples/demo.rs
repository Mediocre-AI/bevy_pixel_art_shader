//! Standalone demo: pixel art pipeline with holdout occluders + debug stage viewer.
//!
//! Architecture:
//!   Full-res 3D Camera (layer 0) → window         (terrain, standard PBR comparison)
//!   Low-res 3D Camera  (layer 1) → 320×180 texture (pixel art + holdout + EdgeDetection)
//!   UI ImageNode                  → canvas overlay  (nearest upscale on top of full-res scene)
//!
//! Controls: left-drag = orbit, right-drag = pan, scroll = zoom
//!
//! Run:  cargo run --example demo

use bevy::camera::visibility::RenderLayers;
use bevy::camera::RenderTarget;
use bevy::image::ImageSampler;
use bevy::pbr::ExtendedMaterial;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy_edge_detection_outline::{EdgeDetection, EdgeDetectionPlugin, EdgeOperator};
use bevy_egui::{
    EguiContext, EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass,
    PrimaryEguiContext, egui,
};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
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
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(EguiPlugin::default())
        .insert_resource(EguiGlobalSettings {
            auto_create_primary_context: false,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate_models, swap_glb_materials, sync_pixel_art_camera))
        .add_systems(EguiPrimaryContextPass, debug_ui)
        .run();
}

#[derive(Component)]
struct Spinning;

#[derive(Component)]
struct NeedsMaterialSwap;

#[derive(Component)]
struct PixelArtCamera;

#[derive(Component)]
struct WindowCamera;

fn setup(
    mut commands: Commands,
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
    let cube_mesh = meshes.add(Cuboid::new(1.5, 1.5, 1.5));
    let torus_mesh = meshes.add(Torus::new(0.4, 0.8));

    // ================================================================
    //  Layer 0: full-res standard PBR comparison objects
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
    ));

    // Standard PBR: sphere, cube, torus (z = -3, right side)
    let std_shapes: [(Color, &str, Handle<Mesh>, Vec3); 3] = [
        (Color::srgb(0.9, 0.15, 0.15), "Std Sphere", sphere_mesh.clone(), Vec3::new(4.0, 1.0, -3.0)),
        (Color::srgb(0.2, 0.8, 0.3),  "Std Cube",   cube_mesh.clone(),   Vec3::new(7.0, 0.75, -3.0)),
        (Color::srgb(0.2, 0.4, 0.9),  "Std Torus",  torus_mesh.clone(),  Vec3::new(10.0, 1.0, -3.0)),
    ];
    for (color, name, mesh, pos) in &std_shapes {
        commands.spawn((
            Name::new(*name),
            Mesh3d(mesh.clone()),
            MeshMaterial3d(std_materials.add(StandardMaterial {
                base_color: *color,
                perceptual_roughness: 1.0,
                reflectance: 0.0,
                ..default()
            })),
            Transform::from_translation(*pos),
            Spinning,
        ));
    }

    // ================================================================
    //  Layer 1: pixel art objects + holdout
    // ================================================================

    let holdout_mat = holdout_materials.add(ExtendedMaterial {
        base: StandardMaterial::default(),
        extension: HoldoutExtension {},
    });
    commands.spawn((
        Name::new("Holdout Ground"),
        Mesh3d(ground_mesh),
        MeshMaterial3d(holdout_mat),
        PIXEL_ART_LAYER,
    ));

    // Sphere cluster — far left
    let cluster_offset = Vec3::new(-8.0, 0.0, 0.0);
    let cluster = [
        (Vec3::new(0.0, 1.2, 0.0), 1.2, Color::srgb(0.9, 0.15, 0.15)),
        (Vec3::new(1.3, 0.9, 0.3), 0.9, Color::srgb(0.15, 0.7, 0.2)),
        (Vec3::new(-1.1, 1.0, 0.5), 1.0, Color::srgb(0.2, 0.3, 0.9)),
        (Vec3::new(0.5, 1.8, -0.3), 0.7, Color::srgb(0.95, 0.8, 0.1)),
        (Vec3::new(-0.5, 0.6, 1.0), 0.6, Color::srgb(0.8, 0.3, 0.8)),
        (Vec3::new(0.8, 0.5, -0.8), 0.5, Color::srgb(0.1, 0.8, 0.8)),
        (Vec3::new(-0.3, 2.2, 0.2), 0.55, Color::srgb(0.95, 0.5, 0.1)),
        (Vec3::new(1.5, 1.5, -0.5), 0.65, Color::srgb(0.9, 0.4, 0.6)),
    ];
    for (i, (pos, scale, color)) in cluster.iter().enumerate() {
        commands.spawn((
            Name::new(format!("Sphere {i}")),
            Mesh3d(sphere_mesh.clone()),
            MeshMaterial3d(make_pixel_mat(&mut pixel_materials, *color)),
            Transform::from_translation(*pos + cluster_offset).with_scale(Vec3::splat(*scale)),
            Spinning,
            PIXEL_ART_LAYER,
        ));
    }

    // Pixel art: sphere, cube, torus (z = 0, same x as standard PBR)
    let pa_shapes: [(Color, &str, Handle<Mesh>, Vec3); 3] = [
        (Color::srgb(0.9, 0.15, 0.15), "PA Sphere", sphere_mesh.clone(), Vec3::new(4.0, 1.0, 0.0)),
        (Color::srgb(0.2, 0.8, 0.3),  "PA Cube",   cube_mesh.clone(),   Vec3::new(7.0, 0.75, 0.0)),
        (Color::srgb(0.2, 0.4, 0.9),  "PA Torus",  torus_mesh.clone(),  Vec3::new(10.0, 1.0, 0.0)),
    ];
    for (color, name, mesh, pos) in &pa_shapes {
        commands.spawn((
            Name::new(*name),
            Mesh3d(mesh.clone()),
            MeshMaterial3d(make_pixel_mat(&mut pixel_materials, *color)),
            Transform::from_translation(*pos),
            Spinning,
            PIXEL_ART_LAYER,
        ));
    }

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
    let cam_transform =
        Transform::from_xyz(5.0, 6.0, 12.0).looking_at(Vec3::new(3.0, 0.5, -1.5), Vec3::Y);

    // Low-res pixel art camera (render-to-texture)
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: -1,
            clear_color: Color::srgba(0.0, 0.0, 0.0, 0.0).into(),
            ..default()
        },
        RenderTarget::Image(image_handle.clone().into()),
        Msaa::Off,
        cam_transform,
        PIXEL_ART_LAYER,
        EdgeDetection::default(),
        PixelArtCamera,
    ));

    // Full-res window camera (with orbit controls + egui)
    commands.spawn((
        Camera3d::default(),
        Camera {
            order: 0,
            clear_color: Color::srgb(0.15, 0.15, 0.18).into(),
            ..default()
        },
        cam_transform,
        PanOrbitCamera {
            focus: Vec3::new(3.0, 0.5, -1.5),
            radius: Some(14.0),
            ..default()
        },
        EguiContext::default(),
        PrimaryEguiContext,
        WindowCamera,
    ));

    // ================================================================
    //  UI overlay (low-res texture on top)
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

// Sync pixel art camera transform with the window camera's orbit state.
fn sync_pixel_art_camera(
    window_q: Query<&Transform, (With<WindowCamera>, Changed<Transform>)>,
    mut pa_q: Query<&mut Transform, (With<PixelArtCamera>, Without<WindowCamera>)>,
) {
    let Ok(win_tf) = window_q.single() else {
        return;
    };
    if let Ok(mut pa_tf) = pa_q.single_mut() {
        *pa_tf = *win_tf;
    }
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
            ui.heading("Pipeline Stage");

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

            if let Ok(mut ed) = edge_q.single_mut() {
                let mut edges_on = ed.enable_depth || ed.enable_normal;
                if ui.checkbox(&mut edges_on, "Edge Detection Outlines").changed() {
                    ed.enable_depth = edges_on;
                    ed.enable_normal = edges_on;
                }
            }

            ui.separator();

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
                    changed |= ui
                        .add(
                            egui::Slider::new(&mut palette_count, 0..=64).text("Palette Colors"),
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
                    ui.horizontal(|ui| {
                        ui.label("Operator:");
                        let is_sobel = ed.operator == EdgeOperator::Sobel;
                        if ui.selectable_label(is_sobel, "Sobel").clicked() {
                            ed.operator = EdgeOperator::Sobel;
                        }
                        if ui.selectable_label(!is_sobel, "Roberts Cross").clicked() {
                            ed.operator = EdgeOperator::RobertsCross;
                        }
                    });
                    ui.separator();
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
