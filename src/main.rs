mod doconeli;
mod graph;
mod mipoga;
mod solution;
mod ui;
mod utils;
mod duaprima;

use crate::solution::Primalization;
use crate::ui::ui;
use crate::utils::{get_bevy_mesh_of_mesh, get_bevy_mesh_of_regions, get_bevy_mesh_of_graph};
use bevy::prelude::*;
use bevy::window::WindowMode;
use bevy::{diagnostic::LogDiagnosticsPlugin, time::common_conditions::on_timer};
use bevy_egui::EguiPlugin;
use doconeli::Doconeli;
use graph::Graph;
use itertools::Itertools;
use rand::Rng;
use smooth_bevy_cameras::{
    controllers::orbit::{OrbitCameraBundle, OrbitCameraController, OrbitCameraPlugin},
    LookTransformPlugin,
};
use smooth_bevy_cameras::{LookTransform, Smoother};
use solution::{MeshResource, PrincipalDirection, LoopScoring};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use strum_macros::EnumIter;
use utils::{get_color, get_random_color, transform_coordinates};

use serde::Deserialize;
use serde::Serialize;

const BACKGROUND_COLOR: bevy::prelude::Color = Color::WHITE;

#[derive(Component)]
pub struct RenderedMesh;

#[derive(Event, Debug)]
pub enum ActionEvent {
    LoadFile(PathBuf),

    RunAlgo,
    AddLoop,
    UndoLoop,
    RemoveLoop,

    PrimalizePlaceCenters,
    PrimalizeConnectCenters,

    InitializeLoops,

    ExportMesh,
    ExportLayout,
    ExportState,

    ResetConfiguration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SaveStateObject {
    mesh: MeshResource,
    config: Configuration,
}

#[derive(Default, Resource, Clone, Debug, Serialize, Deserialize)]
pub struct Configuration {
    pub source: String,
    pub nr_of_faces: usize,
    pub nr_of_edges: usize,
    pub nr_of_vertices: usize,

    pub gamma: f32,

    pub scale: f32,
    pub translation: Vec3,

    pub find_global: bool,

    pub black: bool,

    pub render_type: RenderType,

    pub draw_wireframe: bool,
    pub draw_wireframe_alt: bool,

    pub draw_loops: bool,
    pub draw_centers: bool,
    pub draw_paths: bool,

    pub draw_singularities: bool,    

    pub draw_debug_lines: bool,

    pub loop_scoring_scheme: LoopScoring,

    pub camera_height: f32,
    pub camera_radius: f32,
    pub camera_autorotate: bool,
    pub camera_speed: usize,

    pub choose_direction: PrincipalDirection,
    pub percent_singularities: usize,
    pub choose_component: usize,
    pub draw_next_component: bool,
    pub view_selected_loop: bool,
    
    pub remove_loop: usize,

    pub algorithm_iterations: usize,
    pub algorithm_samples: usize,

    pub color_foreground: [f32; 4],
    pub color_foreground2: [f32; 4],
    pub color_background: [f32; 4],

    pub color_primary1: [f32; 4],
    pub color_primary2: [f32; 4],
    pub color_primary3: [f32; 4],
    pub color_secondary1: [f32; 4],
    pub color_secondary2: [f32; 4],
    pub color_secondary3: [f32; 4],
}

#[derive(Default, Clone, Debug, PartialEq, EnumIter, Serialize, Deserialize)]
pub enum RenderType {
    #[default]
    Original,
    RegionsMesh,
    PatchesMesh,
    PatchesInnerMesh,
    Polycube,
    Nothing,
}

#[derive(PartialEq)]
pub enum ColorType {
    Static(Color),
    Random,
    DirectionPrimary,
    DirectionSecondary,
    Distortion,
}
impl Default for ColorType {
    fn default() -> Self {
        ColorType::Static(Color::BLACK)
    }
}

fn main() {
    App::new()
        .init_resource::<MeshResource>()
        .init_resource::<Configuration>()
        .insert_resource(ClearColor(BACKGROUND_COLOR))
        .insert_resource(AmbientLight {
            brightness: 1.0,
            ..default()
        })
        // Load default plugins
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "pocopi".to_string(),
                mode: WindowMode::BorderlessFullscreen,
                ..Default::default()
            }),
            ..Default::default()
        }))
        // Plugin for diagnostics
        .add_plugins(LogDiagnosticsPlugin::default())
        // Plugin for GUI
        .add_plugins(EguiPlugin)
        // Plugin for smooth camera
        .add_plugins(LookTransformPlugin)
        .add_plugins(OrbitCameraPlugin::default())
        // Pocopi systems
        .add_systems(Startup, setup)
        .add_systems(Update, ui)
        .add_systems(
            Update,
            update_mesh.run_if(on_timer(Duration::from_millis(100))),
        )
        .add_systems(
            Update,
            update_camera.run_if(on_timer(Duration::from_millis(1))),
        )
        .add_systems(Update, handle_events)
        .add_systems(Update, draw_gizmos)
        .add_event::<ActionEvent>()
        .run();
}

/// Set up
fn setup(mut commands: Commands, mut configuration: ResMut<Configuration>) {
    commands
        .spawn(Camera3dBundle::default())
        .insert(OrbitCameraBundle::new(
            OrbitCameraController::default(),
            Vec3::new(0.0, 5.0, 20.0),
            Vec3::new(0., 0., 0.),
            Vec3::Y,
        ));

    setup_configuration(&mut configuration);
}

fn setup_configuration(configuration: &mut ResMut<Configuration>) {

    configuration.algorithm_iterations = 10;
    configuration.algorithm_samples = 100;

    // configuration.camera_autorotate = true;
    // configuration.camera_speed = 20;

    configuration.color_foreground = Color::rgb(0., 0., 0.).as_rgba_f32();
    configuration.color_foreground2 = Color::rgb(0.35, 0.35, 0.35).as_rgba_f32();

    configuration.color_background = Color::WHITE.as_rgba_f32();

    configuration.color_primary1 =
        Color::hsl((0. / 239.) * 360., 190. / 240., 119. / 240.).as_rgba_f32(); // CB dark red
                                                                                // Color::hsl((0. / 239.) * 360., 218./240., 190./240.),  // CB light red
    configuration.color_primary2 =
        Color::hsl((136. / 239.) * 360., 171. / 240., 98. / 240.).as_rgba_f32(); // CB dark blue
                                                                                 // Color::hsl((134. / 239.) * 360., 122./240., 184./240.), // CB light blue
    configuration.color_primary3 =
        Color::hsl((40. / 239.) * 360., 240. / 240., 192. / 240.).as_rgba_f32(); // CB yellow

    configuration.color_secondary1 =
        //Color::hsl((77. / 239.) * 360., 138. / 240., 95. / 240.).as_rgba_f32();
        Color::hsl((61. / 239.) * 360., 0.8, 0.7).as_rgba_f32(); 

    configuration.color_secondary2 =
        //Color::hsl((20. / 239.) * 360., 240. / 240., 120. / 240.).as_rgba_f32();
        Color::hsl((23. / 239.) * 360., 0.9, 0.7).as_rgba_f32();
        
    configuration.color_secondary3 = 
        //Color::hsl((179. / 239.) * 360., 105. / 240., 100. / 240.).as_rgba_f32();
        Color::hsl((187. / 239.) * 360., 0.6, 0.7).as_rgba_f32();

}

fn update_camera(
    mut cameras: Query<(&mut LookTransform, &mut Smoother)>,
    mut configuration: ResMut<Configuration>,
    time: Res<Time>,
) {
    if !configuration.camera_autorotate {
        if let Ok((camera, mut smoother)) = cameras.get_single_mut() {
            smoother.set_lag_weight(0.8);
            configuration.camera_height = camera.eye.y;
            configuration.camera_radius = Vec2::new(camera.target.x, camera.target.z)
                .distance(Vec2::new(camera.eye.x, camera.eye.z));
        }
    }

    if configuration.camera_autorotate {
        let progress =
            (time.elapsed().as_secs_f32() * (configuration.camera_speed as f32)).to_radians();

        if let Ok((mut camera, mut smoother)) = cameras.get_single_mut() {
            smoother.set_lag_weight(0.1);
            camera.eye = camera.target
                + Vec3::new(
                    -configuration.camera_radius * progress.cos(),
                    configuration.camera_height,
                    configuration.camera_radius * progress.sin(),
                );
        }
    }
}

pub fn handle_events(
    mut ev_reader: EventReader<ActionEvent>,
    mut mesh_resmut: ResMut<MeshResource>,
    mut configuration: ResMut<Configuration>,
) {
    for ev in ev_reader.iter() {
        info!("Received event {ev:?}. Handling...");

        match ev {
            ActionEvent::LoadFile(path) => {
                match path.extension().unwrap().to_str() {
                    Some("stl") => match Doconeli::from_stl(&path) {
                        Ok(res) => {
                            *configuration = Configuration::default();
                            configuration.source = String::from(path.to_str().unwrap());

                            setup_configuration(&mut configuration);

                            mesh_resmut.mesh = res;
                            configuration.nr_of_vertices = mesh_resmut.mesh.vertices.len();
                            configuration.nr_of_edges = mesh_resmut.mesh.edges.len() / 2; // dcel -> single edge
                            configuration.nr_of_faces = mesh_resmut.mesh.faces.len();

                            let mesh = get_bevy_mesh_of_mesh(&mesh_resmut.mesh, ColorType::Static(configuration.color_foreground.into()), &configuration);
                            let aabb = mesh.compute_aabb().unwrap();
                            configuration.scale = 10. * (1. / aabb.half_extents.max_element());
                            configuration.translation = (-configuration.scale * aabb.center).into();

                            // mesh_resmut.initialize(&mut configuration);

                            configuration.draw_loops = true;
                            configuration.draw_paths = false;

                            configuration.choose_direction = PrincipalDirection::X;
                            configuration.algorithm_samples = 200;

                        }
                        Err(err) => {
                            error!("Error while parsing STL file {path:?}: {err:?}");
                        }
                    },
                    Some("obj") => {

                        let res = Doconeli::from_obj(&path).unwrap();

                        *configuration = Configuration::default();
                        configuration.source = String::from(path.to_str().unwrap());

                        setup_configuration(&mut configuration);

                        mesh_resmut.mesh = res;
                        configuration.nr_of_vertices = mesh_resmut.mesh.vertices.len();
                        configuration.nr_of_edges = mesh_resmut.mesh.edges.len() / 2; // dcel -> single edge
                        configuration.nr_of_faces = mesh_resmut.mesh.faces.len();

                        let mesh = get_bevy_mesh_of_mesh(&mesh_resmut.mesh, ColorType::Static(configuration.color_foreground.into()), &configuration);
                        let aabb = mesh.compute_aabb().unwrap();
                        configuration.scale = 10. * (1. / aabb.half_extents.max_element());
                        configuration.translation = (-configuration.scale * aabb.center).into();

                        mesh_resmut.initialize(&mut configuration);

                        configuration.draw_loops = true;
                        configuration.draw_paths = false;

                    },
                    Some("lem") => {

                        let lines = std::io::BufRead::lines(std::io::BufReader::new(std::fs::File::open(&path).expect("Error opening file (!)")));

                        for item in lines
                            .into_iter()
                            .map(|x| x.unwrap())
                            .map(|line| {
                                if let Some(rest) = line.strip_prefix("ee ") {
                                    if let Some((pair, path)) = rest.split_once(" : ") {
                                        if let Some((a, b)) = pair.split_once(" ") {
                                            let a = a.parse::<usize>().unwrap();
                                            let b = b.parse::<usize>().unwrap();

                                
                                            let path = path
                                                .split(" ")
                                                .map(|x| x.split(",").map(|p| {
                                                    p.parse::<f32>()
                                                })
                                                .flatten()
                                                .collect_vec())
                                                .collect::<Vec<_>>();

                                            return Some((a, b, path));
                                        }
                                    }
                                }

                                return None;
                                }
                                
                            )
                            {

                            let color = get_random_color();

                            if item.is_none() {
                                continue;
                            }

                            let mut item = item.unwrap().2;

                            item.remove(0);
                            item.remove(item.len()-1);

                            for vertices in item.windows(2) {
                                let pos1 = Vec3::new(vertices[0][0], vertices[0][1], vertices[0][2]);
                                let pos2 = Vec3::new(vertices[1][0], vertices[1][1], vertices[1][2]);
                                mesh_resmut.sol.debug_lines.push((
                                    pos1,
                                    pos2,
                                    color,
                                ));
                                
                            }

                        }
                    },
                    Some("poc") => {
                        let file = File::open(path).unwrap();
                        let reader = BufReader::new(file);

                        // Read the JSON contents of the file as an instance of `User`.
                        let loaded_state: SaveStateObject =
                            serde_json::from_reader(reader).unwrap();

                        *mesh_resmut = loaded_state.mesh;
                        *configuration = loaded_state.config;
                    }

                    _ => panic!("File format not supported."),
                }
            }
            ActionEvent::ResetConfiguration => {
                setup_configuration(&mut configuration);
            }            
            ActionEvent::ExportMesh => {
                let path = format!(
                    "./out/mesh_{}_{:?}.obj",
                    configuration
                        .source
                        .split("\\")
                        .last()
                        .unwrap()
                        .split(".")
                        .next()
                        .unwrap(),
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis());
            
                mesh_resmut.primalization.granulated_mesh.write_to_obj(&PathBuf::from(path));
            }

            ActionEvent::ExportLayout => {
                let path = format!(
                    "./out/layout_{}_{:?}.obj",
                    configuration
                        .source
                        .split("\\")
                        .last()
                        .unwrap()
                        .split(".")
                        .next()
                        .unwrap(),
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("Time went backwards")
                            .as_millis());
                
                mesh_resmut.primalization.patch_graph.write_to_obj(&PathBuf::from(path));
            }
            ActionEvent::ExportState => {
                let path = format!(
                    "./out/state_{}_{:?}.poc",
                    configuration
                        .source
                        .split("\\")
                        .last()
                        .unwrap()
                        .split(".")
                        .next()
                        .unwrap(),
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_millis()
                );

                let state = SaveStateObject {
                    mesh: mesh_resmut.as_mut().clone(),
                    config: configuration.as_mut().clone(),
                };

                fs::write(
                    &PathBuf::from(path),
                    serde_json::to_string(&state).unwrap(),
                );
            }
            ActionEvent::InitializeLoops => {

                mesh_resmut.initialize(&mut configuration);

                configuration.draw_loops = true;
                configuration.draw_paths = false;

            }

            ActionEvent::PrimalizePlaceCenters => {
                mesh_resmut.primalization = Primalization::initialize(&mesh_resmut.mesh, &mesh_resmut.sol);

                let singularities = mesh_resmut.get_top_n_percent_singularities(configuration.percent_singularities);
                mesh_resmut.primalization.place_primals(singularities, &configuration);

            }
            ActionEvent::PrimalizeConnectCenters => {

                mesh_resmut.primalization.connect_primals(&configuration);

                println!("Found valid paths: {:?}/{:?}", mesh_resmut.primalization.edge_to_paths.iter().flatten().count(), mesh_resmut.primalization.edge_to_paths.len()/2);

                configuration.draw_loops = false;
                configuration.draw_paths = true;

            }
            ActionEvent::RunAlgo => {
                for _ in 0..configuration.algorithm_iterations {
                    for i in 0..10 {
                        configuration.choose_component = i;
                        configuration.choose_direction = PrincipalDirection::X;
                        if let Some(new_sol) = mesh_resmut.add_loop(&mut configuration)
                        {
                            mesh_resmut.sol = new_sol;
                            break;
                        }
                    }
                    for i in 0..10 {
                        configuration.choose_component = i;
                        configuration.choose_direction = PrincipalDirection::Y;
                        if let Some(new_sol) = mesh_resmut.add_loop(&mut configuration)                        
                        {
                            mesh_resmut.sol = new_sol;
                            break;
                        }
                    }
                    for i in 0..10 {
                        configuration.choose_component = i;
                        configuration.choose_direction = PrincipalDirection::Z;
                        if let Some(new_sol) = mesh_resmut.add_loop(&mut configuration)
                        {
                            mesh_resmut.sol = new_sol;
                            break;
                        }
                    }
                }

                configuration.draw_loops = true;
                configuration.draw_paths = false;
        
                mesh_resmut.sol.regions = mesh_resmut.get_subsurfaces(&mesh_resmut.sol, &mesh_resmut.sol.intersection_graph, ColorType::Random);
            }
            ActionEvent::AddLoop => {

                for i in 0..4 {
                    if let Some(sol) = mesh_resmut.add_loop(&mut configuration) {
                        mesh_resmut.sol = sol;
                        break;
                    }
                }

                configuration.draw_loops = true;
                configuration.draw_paths = false;

                mesh_resmut.sol.regions = mesh_resmut.get_subsurfaces(&mesh_resmut.sol, &mesh_resmut.sol.intersection_graph, ColorType::Random);                
            },
            ActionEvent::UndoLoop => {

                let last_id = mesh_resmut.sol.paths.len()-1;

                mesh_resmut.remove_path(last_id);

                mesh_resmut.sol.regions = mesh_resmut.get_subsurfaces(&mesh_resmut.sol, &mesh_resmut.sol.intersection_graph, ColorType::Random);  

                configuration.draw_loops = true;
                configuration.draw_paths = false;

            },
            ActionEvent::RemoveLoop => {

                mesh_resmut.remove_path(configuration.remove_loop);
                
                mesh_resmut.sol.regions = mesh_resmut.get_subsurfaces(&mesh_resmut.sol, &mesh_resmut.sol.intersection_graph, ColorType::Random);  

                configuration.draw_loops = true;
                configuration.draw_paths = false;

            },

        }
    }
}

fn check_mesh(mesh_resmut: &Res<MeshResource>) {
    info!("Verifying correctness of mesh.");

    // The edges of a vertex should have that vertex as their root.
    for v in 0..mesh_resmut.mesh.vertices.len() {
        for e in mesh_resmut.mesh.get_edges_of_vertex(v) {
            assert!(
                mesh_resmut.mesh.get_root_of_edge(e) == v,
                "Vertex {v} is indicent to edge {e}. However, edge {e} does not have vertex {v} as root."
            );
        }
    }

    // The edges of a face should have that face as their face.
    for f in 0..mesh_resmut.mesh.faces.len() {
        for e in mesh_resmut.mesh.get_edges_of_face(f) {
            assert!(
                mesh_resmut.mesh.get_face_of_edge(e) == f,
                "Face {f} contains edge {e}. However, edge {e} does have face {f} as face."
            );
        }
    }
}

// Subdivide outlier edges
fn subdivide_outlier_edges(mesh_resmut: &mut ResMut<MeshResource>) {

    mesh_resmut.mesh.split_outliers();
    
}

// Splitting random faces
fn test_split_faces(
    mesh_resmut: &mut ResMut<MeshResource>,
) {
    for _ in 0..100 {
        let face_id = rand::thread_rng().gen_range(0..mesh_resmut.mesh.faces.len());
        mesh_resmut.mesh.split_face(face_id, None);
    }
}

// Splitting random edges
fn test_split_edges(
    mesh_resmut: &mut ResMut<MeshResource>,
) {
    for _ in 0..100 {
        let edge_id = rand::thread_rng().gen_range(0..mesh_resmut.mesh.edges.len());
        println!("{}", edge_id);
        mesh_resmut.mesh.split_edge(edge_id, None);
    }
}

// This function should be called when the mesh (RenderedMesh) is changed, to make sure that modifications are visualized.
fn update_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,

    mesh_resmut: Res<MeshResource>,
    configuration: Res<Configuration>,

    rendered_mesh_query: Query<Entity, With<RenderedMesh>>,
) {
    if !mesh_resmut.is_changed() {
        return;
    }
    info!("Mesh has been changed. Updating the Bevy render.");

    for entity in rendered_mesh_query.iter() {
        commands.entity(entity).despawn();
    }
    info!("Despawning any current meshes.");

    if mesh_resmut.mesh.faces.is_empty() {
        warn!("Mesh is empty (?)");
        return;
    }

    check_mesh(&mesh_resmut);

    let mesh = match configuration.render_type {
        RenderType::Original => get_bevy_mesh_of_mesh(&mesh_resmut.mesh, ColorType::Static(configuration.color_foreground.into()), &configuration),
        // RenderType::Regions => get_bevy_mesh_of_regions(
        //     &mesh_resmut.sol.cached_subsurfaces_regions,
        //     ColorType::DirectionSecondary,
        //     &configuration,
        // ),
        // RenderType::Patches => get_bevy_mesh_of_regions(
        //     &mesh_resmut.sol.cached_subsurfaces_patches,
        //     ColorType::DirectionPrimary,
        //     &configuration,
        // ),
        // RenderType::PatchesWithDistortion => get_bevy_mesh_of_regions(
        //     &mesh_resmut.sol.cached_subsurfaces_patches,
        //     ColorType::Distortion,
        //     &configuration,
        // ),
        RenderType::RegionsMesh => get_bevy_mesh_of_graph(&mesh_resmut.sol.intersection_graph, ColorType::Static(Color::BLACK), &configuration),
        RenderType::PatchesMesh => {
            if configuration.black {
                get_bevy_mesh_of_graph(&mesh_resmut.primalization.patch_graph, ColorType::Static(Color::BLACK), &configuration)
            } else {
                get_bevy_mesh_of_graph(&mesh_resmut.primalization.patch_graph, ColorType::DirectionPrimary, &configuration)
            }
        },
        RenderType::PatchesInnerMesh => {
            if configuration.black {
                get_bevy_mesh_of_regions(&mesh_resmut.primalization.patch_to_surface.clone().into_iter().flatten().collect_vec(), ColorType::Static(Color::BLACK), &configuration)
            } else {
                get_bevy_mesh_of_regions(&mesh_resmut.primalization.patch_to_surface.clone().into_iter().flatten().collect_vec(), ColorType::DirectionPrimary, &configuration)
            }
        },
        RenderType::Polycube => {
            if configuration.black {
                get_bevy_mesh_of_graph(&mesh_resmut.primalization.polycube_graph, ColorType::Static(Color::BLACK), &configuration)
            } else {
                get_bevy_mesh_of_graph(&mesh_resmut.primalization.polycube_graph, ColorType::DirectionPrimary, &configuration)  
            }  
        },
        RenderType::Nothing => return,
        _ => get_bevy_mesh_of_mesh(&mesh_resmut.mesh, ColorType::Static(configuration.color_foreground.into()), &configuration),
    };

    // Spawn new mesh
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(mesh),
            transform: Transform {
                translation: configuration.translation,
                rotation: Quat::from_rotation_z(0f32),
                scale: Vec3::splat(configuration.scale),
            },
            material: materials.add(StandardMaterial {
                perceptual_roughness: 0.9,
                ..default()
            }),
            ..default()
        },
        RenderedMesh,
    ));
}

fn draw_gizmos(
    mut gizmos: Gizmos,
    mesh_resmut: Res<MeshResource>,
    configuration: Res<Configuration>,
) {
    if configuration.draw_debug_lines {
        for &(a, b, color) in &mesh_resmut.sol.debug_lines {
            gizmos.line(
                transform_coordinates(configuration.translation, configuration.scale, a),
                transform_coordinates(configuration.translation, configuration.scale, b),
                color,
            );
        }
    }

    if configuration.draw_loops {
        for loop_id in 0..mesh_resmut.sol.paths.len() {
            let edges = &mesh_resmut.sol.paths[loop_id].edges;
            let dir = mesh_resmut.sol.paths[loop_id].direction;
            let mut color = get_color(dir, false, &configuration);

            if configuration.view_selected_loop && loop_id == configuration.remove_loop % mesh_resmut.sol.paths.len() {
                color = Color::RED;
            };

            for i in 1..edges.len() {
                let u = edges[i - 1];
                let u_normal = mesh_resmut.graphs[dir as usize].nodes[&u].normal;
                let v = edges[i];
                let v_normal = mesh_resmut.graphs[dir as usize].nodes[&v].normal;

                gizmos.line(
                    transform_coordinates(
                        configuration.translation + u_normal * 0.01,
                        configuration.scale,
                        mesh_resmut
                            .sol
                            .get_position_of_path_in_edge(loop_id, &mesh_resmut.mesh, u)
                            .unwrap(),
                    ),
                    transform_coordinates(
                        configuration.translation + v_normal * 0.01,
                        configuration.scale,
                        mesh_resmut
                            .sol
                            .get_position_of_path_in_edge(loop_id, &mesh_resmut.mesh, v)
                            .unwrap(),
                    ),
                    color,
                );
            }
        }
    }

    if configuration.draw_paths {
        for edge_id in 0..mesh_resmut.primalization.patch_graph.edges.len() {
            if let Some(path) = &mesh_resmut.primalization.edge_to_paths[edge_id] {
                let dir = mesh_resmut.primalization.patch_graph.edges[edge_id].direction.unwrap();
                let color = match !configuration.black {
                    true => Color::BLACK,
                    false => get_color(dir, true, &configuration),
                };

                for edge in path.windows(2) {
                    let u = edge[0];
                    let v = edge[1];

                    gizmos.line(
                        transform_coordinates(
                            configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(u) * 0.05,
                            configuration.scale,
                            mesh_resmut
                                .primalization
                                .granulated_mesh.get_position_of_vertex(u),
                        ),
                        transform_coordinates(
                            configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(v) * 0.05,
                            configuration.scale,
                            mesh_resmut.primalization.granulated_mesh.get_position_of_vertex(v)
                        ),
                        color,
                    );
                }
            }
        }
    }

    if configuration.draw_singularities {

        for vertex_id in mesh_resmut.get_top_n_percent_singularities(configuration.percent_singularities) {
            gizmos.line(
                            transform_coordinates(
                                configuration.translation,
                                configuration.scale,
                                mesh_resmut.mesh.get_position_of_vertex(vertex_id),
                            ),
                            transform_coordinates(
                                configuration.translation,
                                configuration.scale,
                                mesh_resmut.mesh.get_position_of_vertex(vertex_id) + mesh_resmut.mesh.get_normal_of_vertex(vertex_id) * 0.05,
                            ),
                            Color::BLACK,
                        );
        }
        

    }

    if configuration.draw_centers {

        for primal in &mesh_resmut.primalization.region_to_primal {
            if let Some(primal) = primal {

                let color = Color::BLACK;

                gizmos.line(
                    transform_coordinates(
                        configuration.translation,
                        configuration.scale,
                        primal.position,
                    ),
                    transform_coordinates(
                        configuration.translation,
                        configuration.scale,
                        primal.position + primal.normal * 0.1,
                    ),
                    color,
                );
                
            }
        }

        for edge_id in 0..mesh_resmut.primalization.patch_graph.edges.len() {
            if let Some(path) = &mesh_resmut.primalization.edge_to_paths[edge_id] {
                let dir = mesh_resmut.primalization.patch_graph.edges[edge_id].direction.unwrap();
                let color = get_color(dir, true, &configuration);

                for edge in path.windows(2) {
                    let u = edge[0];
                    let v = edge[1];

                    gizmos.line(
                        transform_coordinates(
                            configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(u) * 0.01,
                            configuration.scale,
                            mesh_resmut
                                .primalization
                                .granulated_mesh.get_position_of_vertex(u),
                        ),
                        transform_coordinates(
                            configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(v) * 0.01,
                            configuration.scale,
                            mesh_resmut.primalization.granulated_mesh.get_position_of_vertex(v)
                        ),
                        color,
                    );
                }
            }
        }
    }

    if configuration.draw_centers {
        for edge_id in 0..mesh_resmut.primalization.patch_graph.edges.len() {
            if let Some(path) = &mesh_resmut.primalization.edge_to_paths[edge_id] {
                let dir = mesh_resmut.primalization.patch_graph.edges[edge_id].direction.unwrap();
                let color = get_color(dir, true, &configuration);

                for edge in path.windows(2) {
                    let u = edge[0];
                    let v = edge[1];

                    gizmos.line(
                        transform_coordinates(
                            configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(u) * 0.01,
                            configuration.scale,
                            mesh_resmut
                                .primalization
                                .granulated_mesh.get_position_of_vertex(u),
                        ),
                        transform_coordinates(
                            configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(v) * 0.01,
                            configuration.scale,
                            mesh_resmut.primalization.granulated_mesh.get_position_of_vertex(v)
                        ),
                        color,
                    );
                }
            }
        }
    }

    if configuration.draw_wireframe && configuration.render_type == RenderType::Original {
        self::draw_graph_edges(
            &mut gizmos,
            &configuration,
            &mesh_resmut.mesh,
            ColorType::Static(configuration.color_foreground2.into()),
        );
    }

    if configuration.draw_wireframe_alt && configuration.render_type == RenderType::Original {
        self::draw_graph_edges(
            &mut gizmos,
            &configuration,
            &mesh_resmut.primalization.granulated_mesh,
            ColorType::Static(configuration.color_foreground2.into()),
        );
    }

    if configuration.draw_wireframe && configuration.render_type == RenderType::RegionsMesh { 
        let color = match !configuration.black {
            true => ColorType::Static(Color::BLACK),
            false => ColorType::DirectionSecondary,
        };
        self::draw_graph_edges(
            &mut gizmos,
            &configuration,
            &mesh_resmut.sol.intersection_graph,
            color,
        );
    }

    if configuration.draw_wireframe && configuration.render_type == RenderType::PatchesMesh {
        let color = match !configuration.black {
            true => ColorType::Static(Color::BLACK),
            false => ColorType::DirectionPrimary,
        };
        self::draw_graph_edges(
            &mut gizmos,
            &configuration,
            &mesh_resmut.primalization.patch_graph,
            color,
        );
    }

    if configuration.draw_wireframe && configuration.render_type == RenderType::Polycube {
        let color = match !configuration.black {
            true => ColorType::Static(Color::BLACK),
            false => ColorType::DirectionPrimary,
        };
        self::draw_graph_edges(
            &mut gizmos,
            &configuration,
            &mesh_resmut.primalization.polycube_graph,
            color,
        );
    }

    if configuration.draw_next_component {

        // grab region (their vertices)
        let components = mesh_resmut.get_components_between_loops(configuration.choose_direction);
        let target_component =
            mesh_resmut.components_to_singularity_spread(components.clone(), configuration.choose_direction, configuration.percent_singularities)
                [configuration.choose_component % components.len()]
                .0;
        let subset_vertices = components[target_component].clone();

        for vertex_id in subset_vertices {

            gizmos.line(
                transform_coordinates(
                    configuration.translation,
                    configuration.scale,
                    mesh_resmut.mesh.get_position_of_vertex(vertex_id),
                ),
                transform_coordinates(
                    configuration.translation,
                    configuration.scale,
                    mesh_resmut.mesh.get_position_of_vertex(vertex_id) + mesh_resmut.mesh.get_normal_of_vertex(vertex_id) * 0.02,
                ),
                get_color(configuration.choose_direction, false, &configuration),
            );

        }

    }

}

pub fn draw_vertices(
    gizmos: &mut Gizmos,
    configuration: &Res<Configuration>,
    mesh: &Doconeli,
    color: Color,
) {
    for vertex_id in 0..mesh.vertices.len() {
        gizmos.line(
            transform_coordinates(
                configuration.translation,
                configuration.scale,
                mesh.get_position_of_vertex(vertex_id),
            ),
            transform_coordinates(
                configuration.translation,
                configuration.scale,
                mesh.get_position_of_vertex(vertex_id)
                    + mesh.get_normal_of_vertex(vertex_id).normalize() * 0.01,
            ),
            color,
        );
    }
}

pub fn draw_graph_edges(
    gizmos: &mut Gizmos,
    configuration: &Res<Configuration>,
    mesh: &Doconeli,
    color_type: ColorType,
) {
    let mut drawn = HashSet::new();
    for edge_id in 0..mesh.edges.len() {
        let u = mesh.get_root_of_edge(edge_id);
        let v = mesh.get_root_of_edge(mesh.get_next_of_edge(edge_id));
        if drawn.contains(&(u, v)) {
            continue;
        }

        let color = match color_type {
            ColorType::Static(c) => c,
            ColorType::DirectionPrimary => {
                get_color(mesh.edges[edge_id].direction.unwrap(), true, &configuration)
            }
            ColorType::DirectionSecondary => {
                get_color(mesh.edges[edge_id].direction.unwrap(), false, configuration)
            }
            ColorType::Random => get_random_color(),
            _ => Color::PINK,
        };

        gizmos.line(
            transform_coordinates(
                configuration.translation + mesh.get_normal_of_vertex(u).normalize() * 0.01,
                configuration.scale,
                mesh.get_position_of_vertex(u),
            ),
            transform_coordinates(
                configuration.translation + mesh.get_normal_of_vertex(v).normalize() * 0.01,
                configuration.scale,
                mesh.get_position_of_vertex(v),
            ),
            color,
        );
        drawn.insert((u, v));
    }
}
