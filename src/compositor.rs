use bevy::{
    asset::{embedded_asset, load_embedded_asset},
    core_pipeline::{
        FullscreenShader,
        core_3d::{
            DEPTH_TEXTURE_SAMPLING_SUPPORTED,
            graph::{Core3d, Node3d},
        },
        prepass::{DepthPrepass, ViewPrepassTextures},
    },
    ecs::query::QueryState,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{
            Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel,
        },
        render_resource::{
            binding_types::{sampler, texture_2d, texture_depth_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::RenderEntity,
        texture::GpuImage,
        view::ViewTarget,
    },
};

// ──────────────────────────────────────────────
//  Public components
// ──────────────────────────────────────────────

/// Marker for the low-res pixel art camera. Synced to the render world
/// so the compositor node can query its prepass textures.
#[derive(Component, Clone, Default, Reflect)]
#[reflect(Component)]
pub struct LowResPixelArtCamera;

/// Attach to the full-res camera to enable depth-aware compositing.
/// Automatically requires `DepthPrepass` on the same entity.
#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
#[require(DepthPrepass)]
pub struct PixelArtCompositor {
    pub lowres_image: Handle<Image>,
    /// Depth bias for the lowres vs fullres comparison.
    /// Compensates for precision mismatch between the two depth buffers.
    pub depth_bias: f32,
}

// ──────────────────────────────────────────────
//  GPU uniform
// ──────────────────────────────────────────────

#[derive(Component, Clone, Copy, ShaderType)]
pub struct CompositorUniform {
    pub depth_bias: f32,
}

impl ExtractComponent for CompositorUniform {
    type QueryData = &'static PixelArtCompositor;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        compositor: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(CompositorUniform {
            depth_bias: compositor.depth_bias,
        })
    }
}

// ──────────────────────────────────────────────
//  Plugin
// ──────────────────────────────────────────────

pub struct PixelArtCompositorPlugin;

impl Plugin for PixelArtCompositorPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "compositor.wgsl");

        app.register_type::<PixelArtCompositor>();
        app.register_type::<LowResPixelArtCamera>();
        app.add_plugins((
            ExtractComponentPlugin::<CompositorUniform>::default(),
            UniformComponentPlugin::<CompositorUniform>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<CompositorPipeline>>()
            .add_systems(ExtractSchedule, extract_compositor)
            .add_systems(
                Render,
                prepare_compositor_pipelines.in_set(RenderSystems::Prepare),
            )
            .add_render_graph_node::<CompositorNode>(Core3d, CompositorLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::PostProcessing, CompositorLabel, Node3d::Fxaa),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<CompositorPipeline>();
    }
}

// ──────────────────────────────────────────────
//  Render-world types
// ──────────────────────────────────────────────

/// Extracted each frame from `PixelArtCompositor`.
#[derive(Component, Clone)]
pub struct ExtractedCompositor {
    pub lowres_image: Handle<Image>,
}

/// Per-view cached pipeline id.
#[derive(Component, Clone, Copy)]
pub struct CompositorPipelineId(CachedRenderPipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct CompositorLabel;

// ──────────────────────────────────────────────
//  Pipeline resource
// ──────────────────────────────────────────────

#[derive(Resource)]
pub struct CompositorPipeline {
    pub shader: Handle<Shader>,
    pub nearest_sampler: Sampler,
    pub layout: BindGroupLayoutDescriptor,
    pub fullscreen_shader: FullscreenShader,
}

impl FromWorld for CompositorPipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = load_embedded_asset!(world, "compositor.wgsl");

        let layout = BindGroupLayoutDescriptor::new(
            "pixel_art_compositor: bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // 0: fullres color
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // 1: fullres depth
                    texture_depth_2d(),
                    // 2: lowres color
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    // 3: lowres depth
                    texture_depth_2d(),
                    // 4: nearest sampler
                    sampler(SamplerBindingType::NonFiltering),
                    // 5: compositor uniform
                    uniform_buffer::<CompositorUniform>(true),
                ),
            ),
        );

        let render_device = world.resource::<RenderDevice>();
        let nearest_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("pixel_art_compositor nearest sampler"),
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..default()
        });

        Self {
            shader,
            nearest_sampler,
            layout,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
        }
    }
}

// ──────────────────────────────────────────────
//  Specialization
// ──────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompositorKey {
    pub hdr: bool,
}

impl SpecializedRenderPipeline for CompositorPipeline {
    type Key = CompositorKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let format = if key.hdr {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        RenderPipelineDescriptor {
            label: Some("pixel_art_compositor: pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

// ──────────────────────────────────────────────
//  Extract system
// ──────────────────────────────────────────────

pub fn extract_compositor(
    mut commands: Commands,
    compositor_query: Extract<Query<(RenderEntity, &PixelArtCompositor)>>,
    lowres_query: Extract<Query<RenderEntity, With<LowResPixelArtCamera>>>,
) {
    if !DEPTH_TEXTURE_SAMPLING_SUPPORTED {
        info_once!(
            "Disable pixel art compositor on this platform because depth textures aren't supported"
        );
        return;
    }

    for (entity, compositor) in compositor_query.iter() {
        commands
            .get_entity(entity)
            .expect("Compositor entity wasn't synced.")
            .insert(ExtractedCompositor {
                lowres_image: compositor.lowres_image.clone(),
            });
    }

    for entity in lowres_query.iter() {
        commands
            .get_entity(entity)
            .expect("LowRes camera entity wasn't synced.")
            .insert(LowResPixelArtCamera);
    }
}

// ──────────────────────────────────────────────
//  Prepare system
// ──────────────────────────────────────────────

pub fn prepare_compositor_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<CompositorPipeline>>,
    compositor_pipeline: Res<CompositorPipeline>,
    query: Query<(Entity, &ViewTarget), With<ExtractedCompositor>>,
) {
    for (entity, view_target) in &query {
        let hdr = view_target.is_hdr();
        let id = pipelines.specialize(
            &pipeline_cache,
            &compositor_pipeline,
            CompositorKey { hdr },
        );
        commands.entity(entity).insert(CompositorPipelineId(id));
    }
}

// ──────────────────────────────────────────────
//  Render node
// ──────────────────────────────────────────────

pub struct CompositorNode {
    view_query: QueryState<(
        &'static ViewTarget,
        &'static ViewPrepassTextures,
        &'static ExtractedCompositor,
        &'static CompositorPipelineId,
        &'static DynamicUniformIndex<CompositorUniform>,
    )>,
    lowres_query: QueryState<&'static ViewPrepassTextures, With<LowResPixelArtCamera>>,
}

impl FromWorld for CompositorNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            view_query: QueryState::new(world),
            lowres_query: QueryState::new(world),
        }
    }
}

impl Node for CompositorNode {
    fn update(&mut self, world: &mut World) {
        self.view_query.update_archetypes(world);
        self.lowres_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let Ok((view_target, fullres_prepass, extracted, pipeline_id, uniform_index)) =
            self.view_query.get_manual(world, view_entity)
        else {
            return Ok(());
        };

        let compositor_pipeline = world.resource::<CompositorPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            return Ok(());
        };

        // Full-res depth
        let Some(fullres_depth) = &fullres_prepass.depth else {
            return Ok(());
        };

        // Low-res camera prepass textures
        let Some(lowres_prepass) = self.lowres_query.iter_manual(world).next() else {
            return Ok(());
        };
        let Some(lowres_depth) = &lowres_prepass.depth else {
            return Ok(());
        };

        // Low-res color image (the render-to-texture target)
        let Some(lowres_image) = world
            .resource::<RenderAssets<GpuImage>>()
            .get(&extracted.lowres_image)
        else {
            return Ok(());
        };

        // Compositor uniform buffer
        let Some(uniform_binding) = world
            .resource::<ComponentUniforms<CompositorUniform>>()
            .uniforms()
            .binding()
        else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let bind_group = render_context.render_device().create_bind_group(
            "pixel_art_compositor_bind_group",
            &pipeline_cache
                .get_bind_group_layout(&compositor_pipeline.layout),
            &BindGroupEntries::sequential((
                // 0: fullres color (current camera output)
                post_process.source,
                // 1: fullres depth
                &fullres_depth.texture.default_view,
                // 2: lowres color (pixel art render target)
                &lowres_image.texture_view,
                // 3: lowres depth
                &lowres_depth.texture.default_view,
                // 4: nearest sampler
                &compositor_pipeline.nearest_sampler,
                // 5: compositor uniform
                uniform_binding,
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("pixel_art_compositor_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                depth_slice: None,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[uniform_index.index()]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}
