mod doconeli;
mod graph;
mod mipoga;
mod solution;
mod ui;
mod utils;
mod duaprima;

use crate::duaprima::Duaprima;
use crate::solution::{evaluate, PrimalVertexType, Primalization, Subface, Surface};
use crate::ui::ui;
use crate::utils::{get_bevy_mesh_of_graph, get_bevy_mesh_of_mesh, get_bevy_mesh_of_regions, get_labeling_of_mesh};
use rand::distributions::{Distribution, WeightedIndex};
use rayon::prelude::*;
use rand::prelude::SliceRandom;
use bevy::prelude::*;
use bevy::window::WindowMode;
use bevy::{diagnostic::LogDiagnosticsPlugin, time::common_conditions::on_timer};
use bevy_egui::EguiPlugin;
use doconeli::Doconeli;
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

    InitPrimalize,
    StepPrimalize,

    InitializeLoops,

    ExportMesh,
    ExportLayout,
    ExportLabeling,
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

    pub primal_w_graph: bool,
    pub last_primal_w_graph: Duaprima,

    pub path_weight: f32,

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
    NaiveLabeling,
    Polycube,
    DistortionAlignment,
    DistortionJacobian,
    Nothing,
}

#[derive(PartialEq)]
pub enum ColorType {
    Static(Color),
    Random,
    DirectionPrimary,
    DirectionSecondary,
    DistortionAlignment,
    DistortionJacobian,
    Labeling
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

    configuration.algorithm_iterations = 1;
    configuration.algorithm_samples = 2000;
    configuration.gamma = 15.0;
    configuration.percent_singularities = 50;

    configuration.loop_scoring_scheme = LoopScoring::SingularitySeparationSpread;

    configuration.path_weight = 2.0;

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
            
                let res = mesh_resmut.primalization.granulated_mesh.write_to_obj(&PathBuf::from(path));

                println!("result {:?}", res);

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
            ActionEvent::ExportLabeling => {
                let path = format!(
                    "./out/labeling_{}_{:?}.flag",
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

                let res = get_labeling_of_mesh(&PathBuf::from(path), &mesh_resmut.primalization.granulated_mesh, &mesh_resmut.primalization.patch_to_surface.clone().into_iter().flatten().collect_vec());
                
                println!("result {:?}", res);

                // mesh_resmut.primalization.patch_graph.write_to_obj(&PathBuf::from(path));
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

                // For each crossing, color its face

                mesh_resmut.primalization2 = Primalization::default();

                for intersection_id in 0..mesh_resmut.sol.intersection_graph.vertices.len() {
                    let face = mesh_resmut.sol.intersection_graph.vertices[intersection_id].original_face_id;
                    let paths = mesh_resmut.sol.intersection_graph.vertices[intersection_id].ordering.iter().map(|x| x.0).collect_vec();
                    let directions = paths.iter().map(|x| mesh_resmut.sol.paths[*x].direction).collect_vec();
                    // get the direction that is not in directions
                    let direction = [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z].iter().filter(|x| !directions.contains(x)).next().unwrap();

                    let subface = Subface {
                        face_id: face,
                        bounding_points: mesh_resmut.mesh.get_vertices_of_face(face).iter().map(|&x| (mesh_resmut.mesh.get_position_of_vertex(x), mesh_resmut.mesh.get_normal_of_face(face))).collect_vec(),
                        distortion: None,
                    };

                    let surface = Surface {
                        id: intersection_id,
                        faces: vec![subface],
                        inner_vertices: HashSet::new(),
                        direction: Some(direction.to_owned()),
                        color: None,
                        degree: 0,
                    };

                    mesh_resmut.primalization2.patch_to_surface.push(Some(surface));


                }

                // for each loop edge segment, we find a split such that the two surfaces are balanced (with respect to normal direction of the faces inbetween)
                let mut splits = vec![None; mesh_resmut.sol.intersection_graph.edges.len()];
                
                for edge_id in 0..mesh_resmut.sol.intersection_graph.edges.len() {

                    let (endpoint_1, endpoint_2) = mesh_resmut.sol.intersection_graph.get_endpoints_of_edge(edge_id);

                    // data of endpoint_1
                    let face_1 = mesh_resmut.sol.intersection_graph.vertices[endpoint_1].original_face_id;
                    let paths_1 = mesh_resmut.sol.intersection_graph.vertices[endpoint_1].ordering.iter().map(|x| x.0).collect_vec();
                    let directions_1 = paths_1.iter().map(|x| mesh_resmut.sol.paths[*x].direction).collect_vec();
                    // get the direction that is not in directions
                    let direction_1 = [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z].iter().filter(|x| !directions_1.contains(x)).next().unwrap();
                    // normal of face_1
                    let normal_1 = mesh_resmut.mesh.get_normal_of_face(face_1);
                    // actual direction (based on smaller angle with normal of face_1)
                    let signed_direction_1 = [direction_1.to_vector(), -direction_1.to_vector()].iter().map(|x| (x, normal_1.angle_between(*x))).collect_vec().iter().min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap().0.to_owned();

                    // get the surface of endpoint_1
                    let surface_1 = mesh_resmut.primalization2.patch_to_surface.iter().enumerate().find(|x| x.1.as_ref().unwrap().faces.iter().map(|x| x.face_id).collect_vec().contains(&face_1)).unwrap().0;

                    // data of endpoint_2
                    let face_2 = mesh_resmut.sol.intersection_graph.vertices[endpoint_2].original_face_id;
                    let paths_2 = mesh_resmut.sol.intersection_graph.vertices[endpoint_2].ordering.iter().map(|x| x.0).collect_vec();
                    let directions_2 = paths_2.iter().map(|x| mesh_resmut.sol.paths[*x].direction).collect_vec();
                    // get the direction that is not in directions
                    let direction_2 = [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z].iter().filter(|x| !directions_2.contains(x)).next().unwrap();
                    // normal of face_2
                    let normal_2 = mesh_resmut.mesh.get_normal_of_face(face_2);
                    // actual direction (based on smaller angle with normal of face_2)
                    let signed_direction_2 = [direction_2.to_vector(), -direction_2.to_vector()].iter().map(|x| (x, normal_2.angle_between(*x))).collect_vec().iter().min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap().0.to_owned();
                    
                    // get the surface of endpoint_2
                    let surface_2 = mesh_resmut.primalization2.patch_to_surface.iter().enumerate().find(|x| x.1.as_ref().unwrap().faces.iter().map(|x| x.face_id).collect_vec().contains(&face_2)).unwrap().0;

                    // get all faces inbetween (in order)
                    let mut faces_between = vec![];
                    mesh_resmut.sol.intersection_graph.edges[edge_id].edges_between.as_ref().unwrap().iter().for_each(|&x| {
                        let (a, b) = mesh_resmut.mesh.get_faces_of_edge(x);
                        if !faces_between.contains(&a) && a != face_1 && a != face_2 {
                            faces_between.push(a);
                        }
                        if !faces_between.contains(&b) && b != face_1 && b != face_2 {
                            faces_between.push(b);
                        }
                    });

                    // find split such that the two surfaces are balanced (with respect to normal direction of the faces inbetween)
                    let normals_between = faces_between.iter().map(|&x| mesh_resmut.mesh.get_normal_of_face(x)).collect_vec();
                    // true if angle between normal of face and direction of surface_1 is smaller than direction of surface_2
                    let decision_between = normals_between.iter().map(|x| x.angle_between(signed_direction_1) < x.angle_between(signed_direction_2)).collect_vec();

                    // print the normal angles
                    println!("normals {:?}", normals_between.iter().map(|x| x.angle_between(direction_1.to_vector())).collect_vec());
                    println!("normals {:?}", normals_between.iter().map(|x| x.angle_between(direction_2.to_vector())).collect_vec());
                    println!("decision {:?}", decision_between);

                    // split such that we minimize the number of false in the first half, and minimize the number of true in the second half
                    let mut split = 0;
                    let mut min_score = usize::MAX;
                    for i in 0..faces_between.len() {
                        let score = decision_between.iter().take(i).filter(|x| !**x).count() + decision_between.iter().skip(i).filter(|x| **x).count();
                        if score < min_score {
                            min_score = score;
                            split = i;
                        }
                    }

                    println!("split {:?} and score {:?}", split, min_score);

                    splits[edge_id] = Some(split);

                }

                // for each face in mesh, color each face depending on closest label
                for face in 0..mesh_resmut.mesh.faces.len() {
                    let normal = mesh_resmut.mesh.get_normal_of_face(face);

                    // find the best label (direction), based on smallest angle with normal of face
                    let best_label = [(PrincipalDirection::X, 1.0), (PrincipalDirection::X, -1.0), (PrincipalDirection::Y, 1.0), (PrincipalDirection::Y, -1.0), (PrincipalDirection::Z, 1.0), (PrincipalDirection::Z, -1.0)].iter().map(|x| (x, normal.angle_between(x.1 * x.0.to_vector()))).collect_vec().into_iter().min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap();
                    
                    let label = match best_label.0 {
                        (PrincipalDirection::X, 1.0) => 0,
                        (PrincipalDirection::X, -1.0) => 1,
                        (PrincipalDirection::Y, 1.0) => 2,
                        (PrincipalDirection::Y, -1.0) => 3,
                        (PrincipalDirection::Z, 1.0) => 4,
                        (PrincipalDirection::Z, -1.0) => 5,
                        _ => 999,
                    };

                    mesh_resmut.mesh.faces[face].label = Some(label);
                }

                // for each edge in mesh, color each edge depending on the two adjacent faces
                for edge in 0..mesh_resmut.mesh.edges.len() {
                    let (face_1, face_2) = mesh_resmut.mesh.get_faces_of_edge(edge);
                    let label_1 = mesh_resmut.mesh.faces[face_1].label.unwrap();
                    let label_2 = mesh_resmut.mesh.faces[face_2].label.unwrap();

                    mesh_resmut.mesh.edges[edge].face_labels = Some((label_1, label_2));
                }

                configuration.path_weight = 0.5;

                let direction_choices = [
                    [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z],
                    [PrincipalDirection::X, PrincipalDirection::Z, PrincipalDirection::Y],
                    [PrincipalDirection::Y, PrincipalDirection::X, PrincipalDirection::Z],
                    [PrincipalDirection::Y, PrincipalDirection::Z, PrincipalDirection::X],
                    [PrincipalDirection::Z, PrincipalDirection::X, PrincipalDirection::Y],
                    [PrincipalDirection::Z, PrincipalDirection::Y, PrincipalDirection::X],
                    ];
                let component_choices = [0];
                let algo_choices = [LoopScoring::PathLength, LoopScoring::SingularitySeparationSpread, LoopScoring::SingularitySeparationCount];

                let choices = direction_choices.iter().cartesian_product(component_choices.iter()).cartesian_product(algo_choices.iter()).collect_vec();

                let sols: Vec<_> = choices.par_iter().map(|&((directions, component), &loop_scoring)| {

                    let mut local_mesh_resmut = mesh_resmut.clone();
                    let mut local_configuration = configuration.clone();
                    
                    local_mesh_resmut.initialize(&mut local_configuration);

                    let mut local_score = local_mesh_resmut.evaluation.get_score();

                    for direction in directions {
                        
                        local_configuration.algorithm_samples = 200;
                        local_configuration.choose_direction = *direction;
                        local_configuration.choose_component = *component;
                        local_configuration.loop_scoring_scheme = loop_scoring;
                        if let Some((new_sol, primalization)) = local_mesh_resmut.add_loop(&mut local_configuration) {
                            
                            local_mesh_resmut.sol = new_sol;
                            local_mesh_resmut.primalization = primalization;
                            local_mesh_resmut.evaluation = evaluate(&local_mesh_resmut.primalization).unwrap();
                            
                        } else {
                            return None;
                        }
                    }

                    local_score = local_mesh_resmut.evaluation.get_score();

                    Some((local_mesh_resmut, local_score, directions, component))
                    
                }).flatten().collect();

                for (resmut, score, direction, component) in sols.iter() {
                    println!("!!! Found solution [score: {:?}, direction: {:?}, component: {:?}]", score, direction, component);
                }

                let (best_resmut, best_score, best_direction, best_component) = sols.into_iter().min_by(|a, b| (a.1).partial_cmp(&(b.1)).unwrap()).unwrap();

                *mesh_resmut = best_resmut.clone();
                println!("!!! Found best solution [score: {:?}, direction: {:?}, component: {:?}]", best_score, best_direction, best_component);

                configuration.draw_loops = true;
                configuration.draw_paths = false;

            }

            ActionEvent::PrimalizePlaceCenters => {
                mesh_resmut.primalization = Primalization::initialize(&mesh_resmut.mesh, &mesh_resmut.sol).unwrap();

                let singularities = mesh_resmut.get_top_n_percent_singularities(configuration.percent_singularities);
                mesh_resmut.primalization.place_primals(singularities, &configuration);

            }
            ActionEvent::PrimalizeConnectCenters => {


                mesh_resmut.primalization.connect_primals(&mut configuration);

                println!("Found valid paths: {:?}/{:?}", mesh_resmut.primalization.edge_to_paths.iter().flatten().count(), mesh_resmut.primalization.edge_to_paths.len()/2);

                configuration.draw_loops = false;
                configuration.draw_paths = true;

            }
            ActionEvent::RunAlgo => {

                let number_of_samples = 200;
                let number_of_candidates = 8;
                let number_of_winners = 2;
                let number_of_loops = 10;
                let number_of_removals = 15;
                
                let direction_choices = [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z];
                let component_choices = (0..20).into_iter().collect_vec();
                let component_dist = WeightedIndex::new(&[5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]).unwrap();
                let algo_choices = [LoopScoring::SingularitySeparationCount, LoopScoring::SingularitySeparationSpread, LoopScoring::PathLength, LoopScoring::LoopDistribution];
                let path_weight_choices = 1.0..=1.0;
                let gamma_choices = 2.5..7.5;
                let singularity_choices = 2..=20;

                let converge_threshold = 0.001;
                
                let configuration_clone = configuration.clone();
                let mut winners: Vec<MeshResource> = vec![mesh_resmut.clone(); number_of_winners];

                for iteration in 0..configuration.algorithm_iterations {

                    let best_known_score = winners[0].evaluation.get_score();
                    println!(">> Iteration {:?}/{:?} - best: {}", iteration+1, configuration.algorithm_iterations, best_known_score);

                    let mut sols: Vec<_> = (0..number_of_candidates).into_par_iter().map(|i| {
                            let mut rand = rand::thread_rng();

                            let mut local_mesh_resmut = winners[rand.gen_range(0..number_of_winners)].clone();

                            let mut local_configuration = configuration_clone.clone();

                            let mut local_score = local_mesh_resmut.evaluation.get_score();
                            let mut scores = vec![local_score];

                            for step in 0..number_of_loops {
                                local_configuration.choose_direction = direction_choices[rand.gen_range(0..3)];
                                local_configuration.choose_component = component_choices[component_dist.sample(&mut rand)];
                                local_configuration.loop_scoring_scheme = algo_choices[rand.gen_range(0..algo_choices.len())];
                                local_configuration.path_weight = rand.gen_range(path_weight_choices.clone());
                                local_configuration.gamma = rand.gen_range(gamma_choices.clone());
                                local_configuration.percent_singularities = rand.gen_range(singularity_choices.clone());
                                local_configuration.algorithm_samples = number_of_samples;

                                if let Some((new_sol, primalization)) = local_mesh_resmut.add_loop(&mut local_configuration) {
                                    
                                    local_mesh_resmut.sol = new_sol;
                                    local_mesh_resmut.primalization = primalization;
                                    local_mesh_resmut.evaluation = evaluate(&local_mesh_resmut.primalization).unwrap();
                                    
                                }
                                scores.push(local_mesh_resmut.evaluation.get_score());
                            }

                            local_score = scores.last().unwrap().clone();

                            // println!("!!! Candidate {:?}/{:?} phase 1/2 [{:?}]", i+1, number_of_candidates, local_score);

                            for dir in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z].iter() {
                                let mut paths = local_mesh_resmut.sol.paths.iter().enumerate().filter(|(id, p)| p.direction == *dir).map(|(id, p)| id).collect_vec();
                                if paths.len() == 1 {
                                    continue;
                                }

                                let just_added_paths = paths.split_off(std::cmp::min(paths.len(), std::cmp::max(0, paths.len() - number_of_loops + 1)));
                                
                                paths.shuffle(&mut rand);
                                paths = [paths.into_iter().take(number_of_removals-just_added_paths.len()).collect_vec(), just_added_paths].concat();

                                for &path in paths.iter().take(number_of_removals) {
                                    let mut local_mesh_resmut_copy = local_mesh_resmut.clone();
                                    let score = local_mesh_resmut_copy.evaluation.get_score();
                                    if local_mesh_resmut_copy.remove_path(path, &mut local_configuration) {
                                        local_mesh_resmut_copy.evaluation = evaluate(&local_mesh_resmut_copy.primalization).unwrap();
                                        let new_score = local_mesh_resmut_copy.evaluation.get_score();
                                        // if score increases by at most 1% we remove the loop
                                        let percentual_change = (new_score - score) / score;
                                        if percentual_change < 0. {
                                            local_mesh_resmut = local_mesh_resmut_copy;
                                            scores.push(new_score);
                                            local_score = new_score;
                                        }   
                                    }
                                }
    
                                
                            }

                            println!("> Candidate {:?}(/{:?}): {:?}", i+1, number_of_candidates, scores);

                            Some((local_mesh_resmut, scores.last().unwrap().clone()))
                        
                    }).flatten().collect();

                    sols.push((winners[0].clone(), winners[0].evaluation.get_score()));

                    // get #number_of_winners best solutions
                    // sort the solutions
                    sols.sort_by(|a, b| (a.1).partial_cmp(&(b.1)).unwrap());

                    let top_sols = sols.iter().take(number_of_winners).collect_vec();
                    
                    for (winner_id, (_, score)) in top_sols.iter().enumerate() {
                        println!("> Prospect {}(/{}): {:?}", winner_id+1, number_of_winners, score);
                    }

                    

                    winners = top_sols.into_iter().map(|x| x.0.clone()).collect_vec();

                    let new_best_known_score = winners[0].evaluation.get_score();

                    if (best_known_score - new_best_known_score) < converge_threshold {
                        println!("| converge threshold reached: {}", best_known_score - new_best_known_score);
                        break;
                    }

                }

                *mesh_resmut = winners[0].clone();

                println!("> Final: {}", mesh_resmut.evaluation.get_score()); 

            }
            ActionEvent::AddLoop => {

                for i in 0..4 {
                    if let Some((sol, primalization)) = mesh_resmut.add_loop(&mut configuration) {
                        mesh_resmut.sol = sol;
                        mesh_resmut.primalization = primalization;
                        mesh_resmut.evaluation = evaluate(&mesh_resmut.primalization).unwrap();
                        break;
                    }
                }

                println!("> Final: {}", mesh_resmut.evaluation.get_score());

                configuration.draw_loops = true;
                configuration.draw_paths = false;               
            },
            ActionEvent::UndoLoop => {

                // let last_id = mesh_resmut.sol.paths.len()-1;

                // mesh_resmut.remove_path(last_id);

                // mesh_resmut.sol.regions = mesh_resmut.get_subsurfaces(&mesh_resmut.sol, &mesh_resmut.sol.intersection_graph, ColorType::Random);  

                // configuration.draw_loops = true;
                // configuration.draw_paths = false;

            },
            ActionEvent::RemoveLoop => {

                // mesh_resmut.remove_path(configuration.remove_loop);
                
                // mesh_resmut.sol.regions = mesh_resmut.get_subsurfaces(&mesh_resmut.sol, &mesh_resmut.sol.intersection_graph, ColorType::Random);  

                // configuration.draw_loops = true;
                // configuration.draw_paths = false;

            },
            ActionEvent::InitPrimalize => {

                // for dir in [PrincipalDirection::X, PrincipalDirection::Y, PrincipalDirection::Z].iter() {
                //     let paths = mesh_resmut.sol.paths.iter().enumerate().filter(|(id, p)| p.direction == *dir).map(|(id, p)| id).collect_vec();
                //     if paths.len() == 1 {
                //         continue;
                //     }
                    
                //     for &path in paths.iter() {
                //         let mut mesh_resmut_copy = mesh_resmut.clone();
                //         let score = evaluate(&mesh_resmut_copy.primalization).unwrap();
                //         if mesh_resmut_copy.remove_path(path, &mut configuration) {
                //             let new_score = evaluate(&mesh_resmut_copy.primalization).unwrap();
                //             println!("!!! Score {:?} -> {:?} = {:?}", score, new_score, ((new_score.0+new_score.1) - (score.0+score.1)) / (score.0+score.1));
                //             // if score increases by at most 1% we remove the loop
                //             let percentual_change = ((new_score.0+new_score.1) - (score.0+score.1)) / (score.0+score.1);
                //             if percentual_change < 0. {
                //                 println!("!!! Picked up solution.");
                //                 *mesh_resmut = mesh_resmut_copy;
                //                 return;
                //             }   
                //         }
                //     }
                // }

            },
            ActionEvent::StepPrimalize => {

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
                get_bevy_mesh_of_regions(&mesh_resmut.primalization.patch_to_surface.clone().into_iter().flatten().collect_vec(), &mesh_resmut.primalization.patch_graph, &mesh_resmut.primalization.granulated_mesh, ColorType::Static(Color::BLACK), &configuration, &mesh_resmut.evaluation)
            } else {
                get_bevy_mesh_of_regions(&mesh_resmut.primalization.patch_to_surface.clone().into_iter().flatten().collect_vec(), &mesh_resmut.primalization.patch_graph, &mesh_resmut.primalization.granulated_mesh,ColorType::DirectionPrimary, &configuration, &mesh_resmut.evaluation)
            }
        },
        RenderType::NaiveLabeling => get_bevy_mesh_of_mesh(&mesh_resmut.mesh, ColorType::Labeling, &configuration),
        RenderType::DistortionAlignment => {
            get_bevy_mesh_of_regions(&mesh_resmut.primalization.patch_to_surface.clone().into_iter().flatten().collect_vec(), &mesh_resmut.primalization.patch_graph, &mesh_resmut.primalization.granulated_mesh,ColorType::DistortionAlignment, &configuration, &mesh_resmut.evaluation)
        },
        RenderType::DistortionJacobian => {
            get_bevy_mesh_of_regions(&mesh_resmut.primalization.patch_to_surface.clone().into_iter().flatten().collect_vec(), &mesh_resmut.primalization.patch_graph, &mesh_resmut.primalization.granulated_mesh, ColorType::DistortionJacobian, &configuration, &mesh_resmut.evaluation)
        }
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
                            configuration.color_foreground2.into(),
                        );
        }
        

    }

    // if true {

    //     for primal in &mesh_resmut.primalization.region_to_primal {
    //         if let Some(primal) = primal {

    //             let color = Color::BLACK;

    //             gizmos.line(
    //                 transform_coordinates(
    //                     configuration.translation,
    //                     configuration.scale,
    //                     primal.position,
    //                 ),
    //                 transform_coordinates(
    //                     configuration.translation,
    //                     configuration.scale,
    //                     primal.position + primal.normal * 0.1,
    //                 ),
    //                 color,
    //             );
                
    //         }
    //     }

    //     for edge_id in 0..mesh_resmut.primalization.patch_graph.edges.len() {
    //         if let Some(path) = &mesh_resmut.primalization.edge_to_paths[edge_id] {
    //             let dir = mesh_resmut.primalization.patch_graph.edges[edge_id].direction.unwrap();
    //             let color = get_color(dir, true, &configuration);

    //             for edge in path.windows(2) {
    //                 let u = edge[0];
    //                 let v = edge[1];

    //                 gizmos.line(
    //                     transform_coordinates(
    //                         configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(u) * 0.01,
    //                         configuration.scale,
    //                         mesh_resmut
    //                             .primalization
    //                             .granulated_mesh.get_position_of_vertex(u),
    //                     ),
    //                     transform_coordinates(
    //                         configuration.translation + mesh_resmut.primalization.granulated_mesh.get_normal_of_vertex(v) * 0.01,
    //                         configuration.scale,
    //                         mesh_resmut.primalization.granulated_mesh.get_position_of_vertex(v)
    //                     ),
    //                     color,
    //                 );
    //             }
    //         }
    //     }
    // }

    // if true {
    //     for edge_id in 0..mesh_resmut.primalization2.patch_graph.edges.len() {
    //         if let Some(path) = &mesh_resmut.primalization2.edge_to_paths[edge_id] {
    //             let dir = mesh_resmut.primalization2.patch_graph.edges[edge_id].direction.unwrap();
    //             let color = get_color(dir, true, &configuration);

    //             for edge in path.windows(2) {
    //                 let u = edge[0];
    //                 let v = edge[1];

    //                 gizmos.line(
    //                     transform_coordinates(
    //                         configuration.translation + mesh_resmut.primalization2.granulated_mesh.get_normal_of_vertex(u) * 0.01,
    //                         configuration.scale,
    //                         mesh_resmut
    //                             .primalization2
    //                             .granulated_mesh.get_position_of_vertex(u),
    //                     ),
    //                     transform_coordinates(
    //                         configuration.translation + mesh_resmut.primalization2.granulated_mesh.get_normal_of_vertex(v) * 0.01,
    //                         configuration.scale,
    //                         mesh_resmut.primalization2.granulated_mesh.get_position_of_vertex(v)
    //                     ),
    //                     color,
    //                 );
    //             }
    //         }
    //     }
    // }

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

    if configuration.primal_w_graph && configuration.render_type == RenderType::Original {
        draw_duaprima_edges(&mut gizmos, &configuration, &configuration.last_primal_w_graph);
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


        let mut components = vec![];
                for candidate_direction in [
                    PrincipalDirection::X,
                    PrincipalDirection::Y,
                    PrincipalDirection::Z,
                ] {
                    for component in mesh_resmut.get_components_between_loops(candidate_direction) {
                        components.push((component, candidate_direction));
                    }
                }
                let mut component_to_score = vec![];
                for component_i in 0..components.len() {
                    let mut component_score = 0.;
                    let mut component_area = 0.;
                    for &face_i in &components[component_i].0 {
                        component_score += mesh_resmut.evaluation.face_to_fidelity[&face_i]
                            * mesh_resmut.evaluation.face_to_area[&face_i];
                        component_area += mesh_resmut.evaluation.face_to_area[&face_i];
                    }
                    component_to_score.push((component_score, component_i));
                }

                // sort by score (get index)
                let component_to_score = component_to_score
                    .into_iter()
                    .sorted_by(|a, b| b.0.partial_cmp(&a.0).unwrap())
                    .collect_vec();

                let subset_vertices = 
                    components
                        [component_to_score[configuration.choose_component % components.len()].1]
                        .0
                        .clone();
                let direction = components
                    [component_to_score[configuration.choose_component % components.len()].1]
                    .1;

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
                get_color(direction, false, &configuration),
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

fn draw_duaprima_edges(
    gizmos: &mut Gizmos,
    configuration: &Res<Configuration>,
    duaprima: &Duaprima
) {

    let mut total = 0.0f32;
    let mut max = 0.0f32;
    let mut min = f32::MAX;

    for (node_id, node) in &duaprima.nodes {
        let this_pos = node.position;

        for (neighbor_id, weight) in &duaprima.neighbors[&node_id] {
            let neighbor_pos = duaprima.nodes[&neighbor_id].position;
        
            total += weight;
            max = max.max(*weight);
            min = min.min(*weight);
            
        }
    }

    for (node_id, node) in &duaprima.nodes {
        let this_pos = node.position;

        for (neighbor_id, weight) in &duaprima.neighbors[&node_id] {
            let neighbor_pos = duaprima.nodes[&neighbor_id].position;

            // Color based on weight (and the max)
            // let color = utils::color_map(weight, utils::PARULA.to_vec());

            // map weight to 0 to 1
            let weight = (weight - min) / (max - min);
            let color = utils::color_map(weight, utils::PARULA.to_vec());

            gizmos.line(
                transform_coordinates(
                    configuration.translation,
                    configuration.scale,
                    this_pos,
                ),
                transform_coordinates(
                    configuration.translation,
                    configuration.scale,
                    neighbor_pos,
                ),
                color,
            );
        }
    }
}
