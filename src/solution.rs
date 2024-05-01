use crate::EnumIter;
use crate::{
    doconeli::{Doconeli, Edge, Vertex},
    duaprima::Duaprima,
    graph::Graph,
    mipoga::Mipoga,
    utils::{
        self, average, convert_3d_to_2d, intersection_exact_in_2d, intersection_in_sequence,
        set_intersection, Timer,
    },
    ColorType, Configuration,
};
use nalgebra::Matrix4x2;
use nalgebra::{Matrix2, Matrix2x4};
use serde::Deserialize;
use serde::Serialize;

use bevy::{prelude::*, render::render_resource::encase::rts_array::Length};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::Rng;
use rayon::prelude::*;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    f32::consts::PI,
    mem::swap,
};

#[derive(Default, Clone, Resource, Debug, Serialize, Deserialize)]
pub struct MeshResource {
    pub mesh: Doconeli,

    pub sol: Solution,
    pub primalization: Primalization,
    pub primalization2: Primalization,

    pub graphs: [Mipoga; 3],
}

#[derive(Default, Clone, EnumIter, PartialEq, Eq, Debug, Serialize, Deserialize, Copy)]
pub enum LoopScoring {
    #[default]
    PathLength,
    LoopDistribution,
    SingularitySeparationCount,
    SingularitySeparationSpread,
}

#[derive(Copy, Clone, EnumIter, PartialEq, Eq, Debug, Hash, Default, Serialize, Deserialize)]
pub enum PrincipalDirection {
    #[default]
    X = 0,
    Y = 1,
    Z = 2,
}
impl PrincipalDirection {
    pub fn to_vector(&self) -> Vec3 {
        match self {
            PrincipalDirection::X => Vec3::new(1., 0., 0.),
            PrincipalDirection::Y => Vec3::new(0., 1., 0.),
            PrincipalDirection::Z => Vec3::new(0., 0., 1.),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Path {
    // all half-edges (DCEL) passed with the Mipoga model
    pub edges: Vec<usize>,

    // the direction or labeling associated with the path
    pub direction: PrincipalDirection,

    // token that is used to order all paths going through the same edge
    pub order_token: f32,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Subface {
    pub face_id: usize,
    pub bounding_points: Vec<(Vec3, Vec3)>,
    pub distortion: Option<f32>,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Surface {
    pub id: usize,
    pub faces: Vec<Subface>,
    pub inner_vertices: HashSet<usize>,
    pub direction: Option<PrincipalDirection>,
    pub color: Option<Color>,
    pub degree: usize,
}

// A solution as primalization
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Primalization {
    pub edge_to_paths: Vec<Option<Vec<usize>>>,

    // Make them references?
    pub original_mesh: Doconeli,
    pub dual: Solution,

    pub patch_graph: Doconeli,
    pub polycube_graph: Doconeli,

    pub patch_to_surface: Vec<Option<Surface>>,

    pub region_to_primal: Vec<Option<PrimalVertex>>,

    pub granulated_mesh: Doconeli,
    pub face_to_splitted: Vec<Vec<usize>>,
    pub splitface_to_originalface: HashMap<usize, usize>,
}

impl Primalization {
    // Primalization:
    // 1. Make patch_graph from region_graph (by primalization/dualization)
    // 2. Place centers for each vertex in patch_graph
    // 3. Connect centers with paths for each edge in path_graph

    // 1. Make patch_graph from region_graph (by primalization/dualization)
    pub fn initialize(mesh: &Doconeli, dual: &Solution) -> Option<Primalization> {
        let mut splitface_to_originalface: HashMap<usize, usize> = HashMap::new();
        for face_id in 0..mesh.faces.len() {
            splitface_to_originalface.insert(face_id, face_id);
        }

        let dual_graph = Doconeli::from_dual(&dual.intersection_graph);
        for face_id in 0..dual_graph.faces.len() {
            if dual_graph.get_neighbors_of_face_edgewise(face_id).len() != 4 {
                return None;
            }
        }

        Some(Primalization {
            edge_to_paths: vec![None; dual_graph.edges.len()],
            original_mesh: mesh.clone(),
            dual: dual.clone(),
            patch_graph: dual_graph.clone(),
            polycube_graph: dual_graph.clone(),
            granulated_mesh: mesh.clone(),
            region_to_primal: vec![None; dual.regions.len()],
            face_to_splitted: vec![vec![]; mesh.faces.len()],
            splitface_to_originalface,
            patch_to_surface: vec![None; dual_graph.faces.len()],
        })
    }

    pub fn place_primals(
        &mut self,
        singularities: Vec<usize>,
        configuration: &Configuration,
    ) -> bool {
        // 2. Place centers for each vertex in patch_graph
        let mut region_to_candidates = vec![vec![]; self.dual.regions.len()];

        // For every region, get all candidate corner placements
        for (region_id, region) in self.dual.regions.iter().enumerate() {
            // Inner vertices are candidates
            for &inner_vertex in &region.inner_vertices {
                region_to_candidates[region_id].push(PrimalVertex {
                    vertex_type: PrimalVertexType::Vertex(inner_vertex),
                    position: self.original_mesh.get_position_of_vertex(inner_vertex),
                    normal: self.original_mesh.get_normal_of_vertex(inner_vertex),
                    region_id,
                    weight: 10,
                });
            }

            // Center points of (partial) faces are candidates
            for partialface in &region.faces {
                let center_of_face = average(partialface.bounding_points.iter().map(|x| x.0));
                region_to_candidates[region_id].push(PrimalVertex {
                    vertex_type: PrimalVertexType::PointInFace(partialface.face_id),
                    position: center_of_face,
                    normal: self.original_mesh.get_normal_of_face(partialface.face_id),
                    region_id,
                    weight: 1,
                });
            }
        }

        // Each region has labels inside. Find center for each label. Target should minimize distance to the center of each label
        let mut region_to_labelcenters = vec![[(Vec3::splat(0.), 0); 6]; self.dual.regions.len()];
        for region_id in 0..self.dual.regions.len() {
            let region = &self.dual.regions[region_id];

            for label in [0, 1, 2, 3, 4, 5] {
                let faces = region
                    .faces
                    .iter()
                    .filter(|partialface| {
                        self.original_mesh.faces[partialface.face_id].label.unwrap() == label
                    })
                    .map(|partialface| average(partialface.bounding_points.iter().map(|x| x.0)));

                region_to_labelcenters[region_id][label as usize] =
                    (average(faces.clone()), faces.count());
            }

            // only look at the top 3 labels
            region_to_labelcenters[region_id].sort_by(|a, b| b.1.cmp(&a.1));
        }

        // Each region is part of 3 connected components (corresponding to X, Y, and Z), these components set a target for the region
        let mut region_to_target = vec![Vec3::splat(0.); self.dual.regions.len()];

        for filtered_direction in [
            PrincipalDirection::X,
            PrincipalDirection::Y,
            PrincipalDirection::Z,
        ] {
            // Create a graph where all edges in the filtered direction are removed
            let mut graph = Graph::new(&self.patch_graph);
            for edge_id in 0..self.patch_graph.edges.len() {
                if self.patch_graph.edges[edge_id].direction == Some(filtered_direction) {
                    let (u, v) = self.patch_graph.get_endpoints_of_edge(edge_id);
                    graph.remove_edge(u, v);
                }
            }

            // Get all connected components after removing the edges
            let components = graph.connected_components();

            // Find average coordinate for the regions inside 1 component (for `dir` coordinate component, so 1 value, not a 3-d position)
            // for component in components {
            //     let average_point = average(
            //         component
            //             .iter()
            //             .map(|&region_id| &region_to_candidates[region_id])
            //             .flatten()
            //             .map(|candidate| {
            //                 vec![candidate.position[filtered_direction as usize]; candidate.weight]
            //             })
            //             .flatten(),
            //     );
            //     for region_id in component {
            //         region_to_target[region_id][filtered_direction as usize] = average_point;
            //     }
            // }

            for component in components {
                let average_point = average(component.iter().map(|&region_id| {
                    let all_faces_count = region_to_labelcenters[region_id][0].1
                        + region_to_labelcenters[region_id][1].1
                        + region_to_labelcenters[region_id][2].1;

                    let pos = (region_to_labelcenters[region_id][0].1 as f32
                        / all_faces_count as f32)
                        * region_to_labelcenters[region_id][0].0
                        + (region_to_labelcenters[region_id][1].1 as f32 / all_faces_count as f32)
                            * region_to_labelcenters[region_id][1].0
                        + (region_to_labelcenters[region_id][2].1 as f32 / all_faces_count as f32)
                            * region_to_labelcenters[region_id][2].0;

                    pos[filtered_direction as usize]
                }));
                for region_id in component {
                    region_to_target[region_id][filtered_direction as usize] = average_point;
                }
            }
        }

        // Find the best candidate for each region, based on minimizing distance to the set target per region
        for region_id in 0..self.dual.regions.len() {
            let all_faces_count = region_to_labelcenters[region_id][0].1
                + region_to_labelcenters[region_id][1].1
                + region_to_labelcenters[region_id][2].1;

            self.region_to_primal[region_id] = Some(
                region_to_candidates[region_id]
                    .iter()
                    .map(|candidate| {
                        (
                            candidate,
                            candidate.position.distance(region_to_target[region_id]) * 0.5
                                + 0.5
                                    * ((region_to_labelcenters[region_id][0].1 as f32
                                        / all_faces_count as f32)
                                        * region_to_labelcenters[region_id][0]
                                            .0
                                            .distance(candidate.position)
                                        + (region_to_labelcenters[region_id][1].1 as f32
                                            / all_faces_count as f32)
                                            * region_to_labelcenters[region_id][1]
                                                .0
                                                .distance(candidate.position)
                                        + (region_to_labelcenters[region_id][2].1 as f32
                                            / all_faces_count as f32)
                                            * region_to_labelcenters[region_id][2]
                                                .0
                                                .distance(candidate.position)),
                        )
                    })
                    .min_by(|(_, c1_dist), (_, c2_dist)| c1_dist.total_cmp(&c2_dist))
                    .unwrap()
                    .0
                    .clone(),
            );
        }

        // Set the positions in the patch_graph
        for region_id in 0..self.dual.regions.len() {
            self.patch_graph.vertices[region_id].position =
                self.region_to_primal[region_id].clone().unwrap().position;
            self.patch_graph.vertices[region_id].normal =
                self.region_to_primal[region_id].clone().unwrap().normal;
        }

        // Figure out primals not yet part of the actual mesh (they exist inside a face), these need to be solidified into the mesh
        let mut face_to_regions: HashMap<usize, Vec<usize>> = HashMap::new();
        for region_id in 0..self.dual.regions.len() {
            match self.region_to_primal[region_id]
                .clone()
                .unwrap()
                .vertex_type
            {
                PrimalVertexType::PointInFace(face_id) => {
                    face_to_regions
                        .entry(face_id)
                        .or_insert(vec![])
                        .push(region_id);
                }
                _ => {}
            }
        }

        // Create clone of the input mesh, but granulated such that each new primal actually exists on a vertex in the mesh.
        for (&face_id, regions_in_the_face) in &face_to_regions {
            // Keep track of all faces that are created by splitting this face
            self.face_to_splitted[face_id].push(face_id);

            'outer: for &region_id in regions_in_the_face {
                // A primal is either:
                //     1. on an existing edge, then this edge needs to be split
                //     2. inside a face, then this face needs to be split

                let primal_position = self.region_to_primal[region_id].clone().unwrap().position;

                // Case 1.
                let edges_inside_face = self.face_to_splitted[face_id]
                    .iter()
                    .map(|&face_id| self.granulated_mesh.get_edges_of_face(face_id))
                    .flatten()
                    .collect::<HashSet<_>>();

                for &edge_id in &edges_inside_face {
                    let (u, v) = self.granulated_mesh.get_endpoints_of_edge(edge_id);

                    if utils::point_lies_in_segment(
                        primal_position,
                        (
                            self.granulated_mesh.get_position_of_vertex(u),
                            self.granulated_mesh.get_position_of_vertex(v),
                        ),
                    ) {
                        // split edge
                        let (v0, (f_0, f_1, f_2, f_3)) = self
                            .granulated_mesh
                            .split_edge(edge_id, Some(primal_position));

                        let root_face = self.splitface_to_originalface[&face_id];
                        self.splitface_to_originalface.insert(f_0, root_face);
                        self.splitface_to_originalface.insert(f_1, root_face);
                        self.splitface_to_originalface.insert(f_2, root_face);
                        self.splitface_to_originalface.insert(f_3, root_face);

                        self.face_to_splitted[root_face].push(f_0);
                        self.face_to_splitted[root_face].push(f_1);
                        self.face_to_splitted[root_face].push(f_2);
                        self.face_to_splitted[root_face].push(f_3);

                        self.region_to_primal[region_id] = Some(PrimalVertex {
                            vertex_type: PrimalVertexType::Vertex(v0),
                            position: self.granulated_mesh.get_position_of_vertex(v0),
                            normal: self.granulated_mesh.get_normal_of_vertex(v0),
                            region_id,
                            weight: 2,
                        });

                        continue 'outer;
                    }
                }

                // Case 2.
                // test what face the point lies in:
                // https://blackpawn.com/texts/pointinpoly/default.html
                // A common way to check if a point is in a triangle is to find the vectors connecting the point to each of the triangle's three vertices and sum the angles between those vectors.
                // If the sum of the angles is 2*pi then the point is inside the triangle, otherwise it is not.
                for &inner_face_id in &self.face_to_splitted[face_id] {
                    let triangle_vertices = self
                        .granulated_mesh
                        .get_vertices_of_face(inner_face_id)
                        .into_iter()
                        .map(|v_id| self.granulated_mesh.get_position_of_vertex(v_id))
                        .collect_vec();

                    if utils::point_lies_in_triangle(
                        primal_position,
                        (
                            triangle_vertices[0],
                            triangle_vertices[1],
                            triangle_vertices[2],
                        ),
                    ) {
                        let (v0, (f_0, f_1, f_2)) = self
                            .granulated_mesh
                            .split_face(inner_face_id, Some(primal_position));

                        let root_face = self.splitface_to_originalface[&face_id];
                        self.splitface_to_originalface.insert(f_0, root_face);
                        self.splitface_to_originalface.insert(f_1, root_face);
                        self.splitface_to_originalface.insert(f_2, root_face);

                        self.face_to_splitted[root_face].push(f_0);
                        self.face_to_splitted[root_face].push(f_1);
                        self.face_to_splitted[root_face].push(f_2);

                        let edges_0 = self.granulated_mesh.get_edges_of_face(f_0);
                        // get the edge that does not have the vertex v0
                        let edge_0_id = edges_0
                            .iter()
                            .find(|&edge_id| {
                                let (u, v) = self.granulated_mesh.get_endpoints_of_edge(*edge_id);
                                u != v0 && v != v0
                            })
                            .unwrap()
                            .to_owned();

                        let edges_1 = self.granulated_mesh.get_edges_of_face(f_1);
                        let edge_1_id = edges_1
                            .iter()
                            .find(|&edge_id| {
                                let (u, v) = self.granulated_mesh.get_endpoints_of_edge(*edge_id);
                                u != v0 && v != v0
                            })
                            .unwrap()
                            .to_owned();

                        let edges_2 = self.granulated_mesh.get_edges_of_face(f_2);
                        let edge_2_id = edges_2
                            .iter()
                            .find(|&edge_id| {
                                let (u, v) = self.granulated_mesh.get_endpoints_of_edge(*edge_id);
                                u != v0 && v != v0
                            })
                            .unwrap()
                            .to_owned();

                        // split on edge_0
                        let (v1, (f_3, f_4, f_5, f_6)) =
                            self.granulated_mesh.split_edge(edge_0_id, None);
                        // split on edge_1
                        let (v2, (f_7, f_8, f_9, f_10)) =
                            self.granulated_mesh.split_edge(edge_1_id, None);
                        // split on edge_2
                        let (v3, (f_11, f_12, f_13, f_14)) =
                            self.granulated_mesh.split_edge(edge_2_id, None);

                        let root_face = self.splitface_to_originalface[&face_id];
                        self.splitface_to_originalface.insert(f_3, root_face);
                        self.splitface_to_originalface.insert(f_4, root_face);
                        self.splitface_to_originalface.insert(f_5, root_face);
                        self.splitface_to_originalface.insert(f_6, root_face);
                        self.splitface_to_originalface.insert(f_7, root_face);
                        self.splitface_to_originalface.insert(f_8, root_face);
                        self.splitface_to_originalface.insert(f_9, root_face);
                        self.splitface_to_originalface.insert(f_10, root_face);
                        self.splitface_to_originalface.insert(f_11, root_face);
                        self.splitface_to_originalface.insert(f_12, root_face);
                        self.splitface_to_originalface.insert(f_13, root_face);
                        self.splitface_to_originalface.insert(f_14, root_face);
                        self.face_to_splitted[root_face].push(f_3);
                        self.face_to_splitted[root_face].push(f_4);
                        self.face_to_splitted[root_face].push(f_5);
                        self.face_to_splitted[root_face].push(f_6);
                        self.face_to_splitted[root_face].push(f_7);
                        self.face_to_splitted[root_face].push(f_8);
                        self.face_to_splitted[root_face].push(f_9);
                        self.face_to_splitted[root_face].push(f_10);
                        self.face_to_splitted[root_face].push(f_11);
                        self.face_to_splitted[root_face].push(f_12);
                        self.face_to_splitted[root_face].push(f_13);
                        self.face_to_splitted[root_face].push(f_14);

                        self.region_to_primal[region_id] = Some(PrimalVertex {
                            vertex_type: PrimalVertexType::Vertex(v0),
                            position: self.granulated_mesh.get_position_of_vertex(v0),
                            normal: self.granulated_mesh.get_normal_of_vertex(v0),
                            region_id,
                            weight: 2,
                        });

                        continue 'outer;
                    }
                }

                // if we reach this point, the point is not on an edge or inside a face
                return false;
            }
        }

        self.polycube_graph = MeshResource::polycubify(&self.patch_graph);

        return true;
    }

    pub fn connect_primals(&mut self, configuration: &mut Configuration) -> bool {
        let mut debug_lines = vec![];
        let mut primal_vertex_ids = vec![];
        let mut region_pairs = vec![];

        // 3. Connect centers with paths for each edge in path_graph
        // Figure out all center pairs that require a path between them
        let mut done = HashSet::new();
        for edge_id in 0..self.patch_graph.edges.len() {
            let (region_u, region_v) = self.patch_graph.get_endpoints_of_edge(edge_id);

            if done.contains(&(region_u, region_v)) || done.contains(&(region_v, region_u)) {
                continue;
            }
            done.insert((region_u, region_v));

            if let PrimalVertexType::Vertex(primal_u) =
                self.region_to_primal[region_u].clone().unwrap().vertex_type
            {
                if let PrimalVertexType::Vertex(primal_v) =
                    self.region_to_primal[region_v].clone().unwrap().vertex_type
                {
                    primal_vertex_ids.push(primal_u);
                    primal_vertex_ids.push(primal_v);

                    region_pairs.push((
                        (self
                            .granulated_mesh
                            .get_distance_between_vertices(primal_u, primal_v)),
                        edge_id,
                    ));
                    continue;
                }
            }
            println!("Failed to connect primal vertices [1]");
            return false;
        }

        region_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        'outer: for iterations in 0..10 {
            if iterations == 9 {
                println!("Failed to connect primal vertices [iterations]");
                return false;
            }
            self.edge_to_paths = vec![None; self.patch_graph.edges.len()];

            // Find path for each pair
            for pair_id in 0..region_pairs.len() {
                for face in 0..self.granulated_mesh.faces.len() {
                    let normal = self.granulated_mesh.get_normal_of_face(face);

                    // find the best label (direction), based on smallest angle with normal of face
                    let best_label = [
                        (PrincipalDirection::X, 1.0),
                        (PrincipalDirection::X, -1.0),
                        (PrincipalDirection::Y, 1.0),
                        (PrincipalDirection::Y, -1.0),
                        (PrincipalDirection::Z, 1.0),
                        (PrincipalDirection::Z, -1.0),
                    ]
                    .iter()
                    .map(|x| (x, normal.angle_between(x.1 * x.0.to_vector())))
                    .collect_vec()
                    .into_iter()
                    .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                    .unwrap();

                    let label = match best_label.0 {
                        (PrincipalDirection::X, 1.0) => 0,
                        (PrincipalDirection::X, -1.0) => 1,
                        (PrincipalDirection::Y, 1.0) => 2,
                        (PrincipalDirection::Y, -1.0) => 3,
                        (PrincipalDirection::Z, 1.0) => 4,
                        (PrincipalDirection::Z, -1.0) => 5,
                        _ => 999,
                    };

                    self.granulated_mesh.faces[face].label = Some(label);
                }

                // for each edge in mesh, color each edge depending on the two adjacent faces
                for edge in 0..self.granulated_mesh.edges.len() {
                    let (face_1, face_2) = self.granulated_mesh.get_faces_of_edge(edge);
                    let label_1 = self.granulated_mesh.faces[face_1].label.unwrap();
                    let label_2 = self.granulated_mesh.faces[face_2].label.unwrap();

                    self.granulated_mesh.edges[edge].face_labels = Some((label_1, label_2));
                }

                let mut face_labels = HashMap::new();
                let mut edge_labels = HashMap::new();
                for face_id in 0..self.granulated_mesh.faces.len() {
                    let label = self.granulated_mesh.faces[face_id].label.unwrap();
                    let real_label = match label {
                        0 => 0,
                        1 => 0,
                        2 => 1,
                        3 => 1,
                        4 => 2,
                        5 => 2,
                        _ => 999,
                    };
                    face_labels.insert(face_id, real_label);
                }
                for edge_id in 0..self.granulated_mesh.edges.len() {
                    let labels = self.granulated_mesh.edges[edge_id].face_labels.unwrap();
                    let real_labels: (usize, usize) = [labels.0, labels.1]
                        .iter()
                        .map(|&label| match label {
                            0 => 0,
                            1 => 0,
                            2 => 1,
                            3 => 1,
                            4 => 2,
                            5 => 2,
                            _ => 999,
                        })
                        .collect_tuple()
                        .unwrap();

                    edge_labels.insert(
                        self.granulated_mesh.get_endpoints_of_edge(edge_id),
                        real_labels,
                    );
                }

                let (_, original_edge_id) = region_pairs[pair_id];

                let (region_u, region_v) = self.patch_graph.get_endpoints_of_edge(original_edge_id);

                let mut remove_vertices = HashSet::new();
                let mut remove_edges = HashSet::new();

                // remove all vertices and edges covered by prev edges..
                // vertices in the paths
                for path in self.edge_to_paths.iter().flatten() {
                    for vertex_id in path.iter() {
                        remove_vertices.insert(vertex_id.clone());
                    }
                }

                // edges in the paths (not allowed to cross them through face to face edges)
                for path in self.edge_to_paths.iter().flatten() {
                    for edge in path.windows(2) {
                        let edge_between = self
                            .granulated_mesh
                            .get_edge_between_vertex_and_vertex(edge[0], edge[1])
                            .unwrap();
                        let (face_a, face_b) = self.granulated_mesh.get_faces_of_edge(edge_between);
                        remove_edges.insert((
                            self.granulated_mesh.vertices.len() + face_a,
                            self.granulated_mesh.vertices.len() + face_b,
                        ));
                    }
                }

                // Block all primal vertices, as youre never allowed to trace paths over primals
                for vertex_id in &primal_vertex_ids {
                    remove_vertices.insert(vertex_id.clone());
                    remove_vertices.insert(vertex_id.clone());
                }

                // Block all vertices that are not in the current region
                // Basically just keep everything in the current regions
                // let mut faces_to_keep = HashSet::new();
                // for face_id in self.dual.regions[region_u]
                //     .faces
                //     .iter()
                //     .chain(self.dual.regions[region_v].faces.iter())
                //     .map(|subface| subface.face_id)
                // {
                //     faces_to_keep.insert(face_id);

                //     self.face_to_splitted[face_id]
                //         .iter()
                //         .for_each(|&splitface_id| {
                //             faces_to_keep.insert(splitface_id);
                //         });
                // }

                // let mut vertices_to_keep = HashSet::new();
                // for &face_id in &faces_to_keep {
                //     for vertex_id in self.granulated_mesh.get_vertices_of_face(face_id) {
                //         vertices_to_keep.insert(vertex_id);
                //     }
                // }

                // for face_id in 0..self.granulated_mesh.faces.len() {
                //     if !faces_to_keep.contains(&face_id) {
                //         remove_vertices.insert(self.granulated_mesh.vertices.len() + face_id);

                //         for vertex_id in self.granulated_mesh.get_vertices_of_face(face_id) {
                //             // if !vertices_to_keep.contains(&vertex_id) {
                //             remove_vertices.insert(vertex_id);
                //             // }
                //         }
                //     }
                // }

                // Actually find the path
                if let PrimalVertexType::Vertex(primal_u) =
                    self.region_to_primal[region_u].clone().unwrap().vertex_type
                {
                    if let PrimalVertexType::Vertex(primal_v) =
                        self.region_to_primal[region_v].clone().unwrap().vertex_type
                    {
                        remove_vertices.remove(&primal_u);
                        remove_vertices.remove(&primal_v);

                        let mut ggg: Duaprima = Duaprima::from_mesh_with_mask(
                            &self.granulated_mesh,
                            remove_vertices,
                            remove_edges,
                        );

                        // find target_labels (look at the two regions that need to be connected)
                        // we have the two regions
                        // these two regions MUST be adjacent faces in the intersection graph
                        assert!(
                            self.dual
                                .intersection_graph
                                .get_neighbors_of_face_edgewise(region_u)
                                .iter()
                                .find(|&&x| x == region_v)
                                .unwrap()
                                == &region_v
                        );

                        let edge_between_regions = self
                            .dual
                            .intersection_graph
                            .get_edge_of_face_and_face(region_u, region_v)
                            .unwrap();

                        let (intersection_a, intersection_b) = self
                            .dual
                            .intersection_graph
                            .get_endpoints_of_edge(edge_between_regions);

                        let intersection_a_labels = self.dual.intersection_graph.vertices
                            [intersection_a]
                            .ordering
                            .iter()
                            .map(|(label, _)| label.to_owned())
                            .collect::<HashSet<_>>();

                        let all_directions = [0, 1, 2];

                        // direction of an intersection is the third label that is missing
                        let intersection_a_direction = intersection_a_labels
                            .symmetric_difference(&all_directions.iter().cloned().collect())
                            .next()
                            .unwrap()
                            .to_owned();

                        let intersection_b_labels = self.dual.intersection_graph.vertices
                            [intersection_b]
                            .ordering
                            .iter()
                            .map(|(label, _)| label.to_owned())
                            .collect::<HashSet<_>>();

                        let intersection_b_direction = intersection_b_labels
                            .symmetric_difference(&all_directions.iter().cloned().collect())
                            .next()
                            .unwrap()
                            .to_owned();

                        let target_labels = (intersection_a_direction, intersection_b_direction);

                        ggg.precompute_label_weights(
                            &configuration,
                            target_labels,
                            face_labels.clone(),
                            edge_labels.clone(),
                        );

                        configuration.last_primal_w_graph = ggg.clone();

                        if let Some(path) = ggg.shortest_path(primal_u, primal_v) {
                            let mut realized_path = vec![];

                            let mut last_face: Option<(usize, usize, usize)> = None;
                            for vertex in &path {
                                match ggg.nodes[&vertex].node_type {
                                    crate::duaprima::NodeType::Face(face_id) => {
                                        let (vertex_id, faces) =
                                            self.granulated_mesh.split_face(face_id, None);

                                        let root_face = self.splitface_to_originalface[&face_id];

                                        self.splitface_to_originalface.insert(faces.0, root_face);
                                        self.splitface_to_originalface.insert(faces.1, root_face);
                                        self.splitface_to_originalface.insert(faces.2, root_face);

                                        self.face_to_splitted[root_face].push(faces.0);
                                        self.face_to_splitted[root_face].push(faces.1);
                                        self.face_to_splitted[root_face].push(faces.2);

                                        if let Some(prev_faces) = last_face {
                                            for face in [faces.0, faces.1, faces.2] {
                                                for prev_face in
                                                    [prev_faces.0, prev_faces.1, prev_faces.2]
                                                {
                                                    if let Some(edge_id) = self
                                                        .granulated_mesh
                                                        .get_edge_of_face_and_face(face, prev_face)
                                                    {
                                                        let (mid_id, faces) = self
                                                            .granulated_mesh
                                                            .split_edge(edge_id, None);
                                                        realized_path.push(mid_id);

                                                        let root_prevface = self
                                                            .splitface_to_originalface[&prev_face];

                                                        self.splitface_to_originalface
                                                            .insert(faces.0, root_face);
                                                        self.splitface_to_originalface
                                                            .insert(faces.2, root_face);
                                                        self.splitface_to_originalface
                                                            .insert(faces.1, root_prevface);
                                                        self.splitface_to_originalface
                                                            .insert(faces.3, root_prevface);

                                                        self.face_to_splitted[root_face]
                                                            .push(faces.0);
                                                        self.face_to_splitted[root_prevface]
                                                            .push(faces.1);
                                                        self.face_to_splitted[root_face]
                                                            .push(faces.2);
                                                        self.face_to_splitted[root_prevface]
                                                            .push(faces.3);
                                                    }
                                                }
                                            }
                                        }
                                        realized_path.push(vertex_id);

                                        last_face = Some(faces);
                                    }
                                    crate::duaprima::NodeType::Vertex(vertex_id) => {
                                        realized_path.push(vertex_id);
                                        last_face = None;
                                    }
                                    crate::duaprima::NodeType::Phantom => {}
                                }
                            }

                            self.edge_to_paths[original_edge_id] = Some(realized_path.clone());
                            self.edge_to_paths
                                [self.patch_graph.get_twin_of_edge(original_edge_id)] =
                                Some(realized_path.clone());
                        } else {
                            debug_lines.push((
                                self.region_to_primal[region_u].clone().unwrap().position,
                                self.region_to_primal[region_u].clone().unwrap().position
                                    + self.region_to_primal[region_u].clone().unwrap().normal * 0.2,
                                Color::GREEN,
                            ));
                            debug_lines.push((
                                self.region_to_primal[region_v].clone().unwrap().position,
                                self.region_to_primal[region_v].clone().unwrap().position
                                    + self.region_to_primal[region_v].clone().unwrap().normal * 0.2,
                                Color::GREEN,
                            ));

                            for (node_id, neighbors) in &ggg.neighbors {
                                for (neighbor_id, weight) in neighbors {
                                    debug_lines.push((
                                        ggg.nodes[&node_id].position,
                                        ggg.nodes[&neighbor_id].position,
                                        Color::RED,
                                    ));
                                }
                            }

                            // IF IT FAILS, MOVE EDGE TO FRONT OF PRIORITY
                            let item = region_pairs.remove(pair_id);
                            region_pairs.insert(0, item);
                            continue 'outer;
                        }
                    }
                }
            }

            // Success!
            break;
        }

        let mut edge_to_faces = vec![HashSet::new(); self.patch_graph.edges.len()];

        for edge_id in 0..self.patch_graph.edges.len() {
            let mut path = self.edge_to_paths[edge_id].clone().unwrap();
            path.remove(path.len() - 1);
            path.remove(0);
            let faces = path
                .into_iter()
                .map(|vertex_id| self.granulated_mesh.get_faces_of_vertex(vertex_id))
                .flatten()
                .collect::<HashSet<_>>();
            edge_to_faces[edge_id] = faces;
        }

        // find patches
        // and compute a score for each patch
        for patch_id in 0..self.patch_graph.faces.len() {
            let boundary_edges = self.patch_graph.get_edges_of_face(patch_id);

            let illegal_faces = (0..self.patch_graph.edges.len())
                .filter(|edge_id| {
                    !boundary_edges.contains(edge_id)
                        && !boundary_edges.contains(&self.patch_graph.get_twin_of_edge(*edge_id))
                })
                .map(|edge_id| edge_to_faces[edge_id].clone())
                .flatten()
                .collect::<HashSet<_>>();

            let boundary_vertices = boundary_edges
                .clone()
                .into_iter()
                .map(|edge_id| self.edge_to_paths[edge_id].clone())
                .flatten()
                .flatten()
                .collect::<HashSet<_>>();

            let graph = Graph::new(&self.granulated_mesh);
            let components = MeshResource::get_subsurfaces3(&graph, &boundary_vertices);

            // get all vertices in the subsurfaces that ONLY share faces with the paths of this boundary.
            let inner_vertices = components
                .into_iter()
                .filter(|component| {
                    component
                        .iter()
                        .map(|&vertex_id| self.granulated_mesh.get_faces_of_vertex(vertex_id))
                        .flatten()
                        .all(|face_id| !illegal_faces.contains(&face_id))
                })
                .flatten()
                .collect::<HashSet<_>>();

            let all_vertices = inner_vertices
                .clone()
                .into_iter()
                .chain(boundary_vertices.clone().into_iter())
                .collect::<HashSet<_>>();

            let all_faces = (0..self.granulated_mesh.faces.len())
                .filter(|&face_id| {
                    self.granulated_mesh
                        .get_vertices_of_face(face_id)
                        .iter()
                        .all(|&vertex_id| all_vertices.contains(&vertex_id))
                })
                .collect_vec();
            let subfaces = all_faces
                .iter()
                .map(|&face_id| {
                    let bounding_points = self
                        .granulated_mesh
                        .get_vertices_of_face(face_id)
                        .iter()
                        .map(|&vertex_id| {
                            (
                                self.granulated_mesh.get_position_of_vertex(vertex_id),
                                self.granulated_mesh.get_normal_of_vertex(vertex_id),
                            )
                        })
                        .collect_vec();
                    let distortion = None;
                    Subface {
                        face_id,
                        bounding_points,
                        distortion,
                    }
                })
                .collect_vec();

            let dir = [
                PrincipalDirection::X,
                PrincipalDirection::Y,
                PrincipalDirection::Z,
            ]
            .into_iter()
            .filter(|&dir| {
                !boundary_edges
                    .iter()
                    .map(|&edge_id| self.patch_graph.edges[edge_id].direction)
                    .contains(&Some(dir))
            })
            .next();

            let surface = Surface {
                id: patch_id,
                faces: subfaces,
                inner_vertices: all_vertices,
                direction: dir,
                color: None,
                degree: 0,
            };

            self.patch_to_surface[patch_id] = Some(surface);
        }

        true
    }
}

pub fn evaluate(primalization: &Primalization) -> Option<f32> {
    let mut worst_patch_scores = vec![];
    let mut avg_patch_scores = vec![];

    for patch_id in 0..primalization.patch_graph.faces.len() {
        let surface = primalization.patch_to_surface[patch_id].clone().unwrap();
        let dir = surface.direction;

        // compute avg normal of the surface
        let avg_normal = compute_average_normal(&surface, &primalization.granulated_mesh);

        // direction is negative or positive based on angle with average normal
        let positive = dir.unwrap().to_vector().dot(avg_normal).signum();
        let direction = dir.unwrap().to_vector() * positive;

        let mut areas = vec![];
        let mut flatness_devs = vec![];
        let mut alignment_devs = vec![];
        for subface in &surface.faces {
            let flatness_dev =
                compute_deviation(subface.face_id, &primalization.granulated_mesh, avg_normal);
            flatness_devs.push(flatness_dev);

            let alignment_dev =
                compute_deviation(subface.face_id, &primalization.granulated_mesh, direction);
            alignment_devs.push(alignment_dev);

            let area = primalization
                .granulated_mesh
                .get_area_of_face(subface.face_id);
            areas.push(area);
        }

        let max_flatness_dev = flatness_devs
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_flatness_dev = flatness_devs.iter().cloned().fold(f32::INFINITY, f32::min);
        let avg_flatness_dev = flatness_devs.iter().sum::<f32>() / flatness_devs.len() as f32;
        let avg_flatness_dev_scaled = flatness_devs
            .iter()
            .zip(areas.iter())
            .map(|(dev, area)| dev * area)
            .sum::<f32>()
            / areas.iter().sum::<f32>();

        let max_alignment_dev = alignment_devs
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_alignment_dev = alignment_devs.iter().cloned().fold(f32::INFINITY, f32::min);
        let avg_alignment_dev = alignment_devs.iter().sum::<f32>() / alignment_devs.len() as f32;
        let avg_alignment_dev_scaled = alignment_devs
            .iter()
            .zip(areas.iter())
            .map(|(dev, area)| dev * area)
            .sum::<f32>()
            / areas.iter().sum::<f32>();

        // compute Jacobian of the patch
        let corners = primalization.patch_graph.get_vertices_of_face(patch_id);
        let corner_positions = corners
            .iter()
            .map(|&vertex_id| primalization.patch_graph.vertices[vertex_id].position)
            .collect_vec();
        assert!(corners.len() == 4);

        let quad = match (surface.direction, positive < 0.) {
            (Some(PrincipalDirection::X), false) => {
                let mut m = Matrix4x2::zeros();
                m[(0, 0)] = corner_positions[3].y;
                m[(0, 1)] = corner_positions[3].z;
                m[(1, 0)] = corner_positions[2].y;
                m[(1, 1)] = corner_positions[2].z;
                m[(2, 0)] = corner_positions[1].y;
                m[(2, 1)] = corner_positions[1].z;
                m[(3, 0)] = corner_positions[0].y;
                m[(3, 1)] = corner_positions[0].z;
                m
            }
            (Some(PrincipalDirection::Y), true) => {
                let mut m = Matrix4x2::zeros();
                m[(0, 0)] = corner_positions[3].x;
                m[(0, 1)] = corner_positions[3].z;
                m[(1, 0)] = corner_positions[2].x;
                m[(1, 1)] = corner_positions[2].z;
                m[(2, 0)] = corner_positions[1].x;
                m[(2, 1)] = corner_positions[1].z;
                m[(3, 0)] = corner_positions[0].x;
                m[(3, 1)] = corner_positions[0].z;
                m
            }
            (Some(PrincipalDirection::Z), false) => {
                let mut m = Matrix4x2::zeros();
                m[(0, 0)] = corner_positions[3].x;
                m[(0, 1)] = corner_positions[3].y;
                m[(1, 0)] = corner_positions[2].x;
                m[(1, 1)] = corner_positions[2].y;
                m[(2, 0)] = corner_positions[1].x;
                m[(2, 1)] = corner_positions[1].y;
                m[(3, 0)] = corner_positions[0].x;
                m[(3, 1)] = corner_positions[0].y;
                m
            }
            (Some(PrincipalDirection::X), true) => {
                let mut m = Matrix4x2::zeros();
                m[(0, 0)] = corner_positions[0].y;
                m[(0, 1)] = corner_positions[0].z;
                m[(1, 0)] = corner_positions[1].y;
                m[(1, 1)] = corner_positions[1].z;
                m[(2, 0)] = corner_positions[2].y;
                m[(2, 1)] = corner_positions[2].z;
                m[(3, 0)] = corner_positions[3].y;
                m[(3, 1)] = corner_positions[3].z;
                m
            }
            (Some(PrincipalDirection::Y), false) => {
                let mut m = Matrix4x2::zeros();
                m[(0, 0)] = corner_positions[0].x;
                m[(0, 1)] = corner_positions[0].z;
                m[(1, 0)] = corner_positions[1].x;
                m[(1, 1)] = corner_positions[1].z;
                m[(2, 0)] = corner_positions[2].x;
                m[(2, 1)] = corner_positions[2].z;
                m[(3, 0)] = corner_positions[3].x;
                m[(3, 1)] = corner_positions[3].z;
                m
            }
            (Some(PrincipalDirection::Z), true) => {
                let mut m = Matrix4x2::zeros();
                m[(0, 0)] = corner_positions[0].x;
                m[(0, 1)] = corner_positions[0].y;
                m[(1, 0)] = corner_positions[1].x;
                m[(1, 1)] = corner_positions[1].y;
                m[(2, 0)] = corner_positions[2].x;
                m[(2, 1)] = corner_positions[2].y;
                m[(3, 0)] = corner_positions[3].x;
                m[(3, 1)] = corner_positions[3].y;
                m
            }
            _ => Matrix4x2::from_vec(
                corner_positions
                    .iter()
                    .map(|&pos| vec![0., 0.])
                    .flatten()
                    .collect(),
            ),
        };

        for (i, n, z) in [(0, -1., -1.), (1, 1., -1.), (2, 1., 1.), (3, -1., 1.)] {
            // get distance (length) between p1 and p2
            let length1 = Vec2::from([quad.row(i)[0], quad.row(i)[1]]).distance(Vec2::from([
                quad.row((i - 1) % 4)[0],
                quad.row((i - 1) % 4)[1],
            ]));

            // get distance (length) between p2 and p3
            let length2 = Vec2::from([quad.row(i)[0], quad.row(i)[1]]).distance(Vec2::from([
                quad.row((i + 1) % 4)[0],
                quad.row((i + 1) % 4)[1],
            ]));

            let jacobian = jacobian(n, z, &quad);
            let det_jacobian = det_jacobian(n, z, &quad);
            let scaled_det_jacobian = det_jacobian / ((length1 / 2.) * (length2 / 2.));
        }

        worst_patch_scores.push(max_alignment_dev);
        avg_patch_scores.push(avg_alignment_dev_scaled);
    }

    let score = worst_patch_scores
        .into_iter()
        .max_by(|a, b| a.total_cmp(b))
        .unwrap()
        * 0.2
        + utils::average(avg_patch_scores.into_iter());

    if score.is_nan() {
        println!("Failed to connect primal vertices [score]");
        return None;
    }

    return Some(score);
}

// Derivatives of shape functions with respect to xi and eta
pub fn dphi(n: f32, z: f32) -> Matrix4x2<f32> {
    let mut m = Matrix4x2::zeros();

    m[(0, 0)] = -0.25 * (1. - n);
    m[(0, 1)] = -0.25 * (1. - z);
    m[(1, 0)] = 0.25 * (1. - n);
    m[(1, 1)] = -0.25 * (1. + z);
    m[(2, 0)] = 0.25 * (1. + n);
    m[(2, 1)] = 0.25 * (1. + z);
    m[(3, 0)] = -0.25 * (1. + n);
    m[(3, 1)] = 0.25 * (1. - z);

    m
}

pub fn jacobian(n: f32, z: f32, quad: &Matrix4x2<f32>) -> Matrix2<f32> {
    dphi(n, z).transpose() * quad
}

pub fn det_jacobian(n: f32, z: f32, quad: &Matrix4x2<f32>) -> f32 {
    jacobian(n, z, quad).determinant()
}

// Given a surface, compute the average normal of the surface (weighted by the area of the faces)
pub fn compute_average_normal(surface: &Surface, mesh: &Doconeli) -> Vec3 {
    let mut normal = Vec3::splat(0.);
    let mut total_area = 0.;

    for subface in &surface.faces {
        let area = mesh.get_area_of_face(subface.face_id);
        total_area += area;
        normal += mesh.get_normal_of_face(subface.face_id) * area;
    }

    normal / total_area
}

pub fn compute_deviation(face_id: usize, mesh: &Doconeli, vector: Vec3) -> f32 {
    mesh.get_normal_of_face(face_id).angle_between(vector) / std::f32::consts::PI
}

// Given a surface, and a vector, compute the average deviation of the normals of the faces to the vector
pub fn compute_average_deviation(surface: &Surface, mesh: &Doconeli, vector: Vec3) -> f32 {
    let mut deviation = 0.;
    let mut total_area = 0.;

    for subface in &surface.faces {
        let area = mesh.get_area_of_face(subface.face_id);
        total_area += area;
        deviation += compute_deviation(subface.face_id, mesh, vector) * area;
    }

    deviation / total_area
}

// A solution is a collection of paths.
// Furthermore, a DCEL on the intersection graph is available.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Solution {
    pub paths: Vec<Path>,

    pub intersection_graph: Doconeli,

    pub regions: Vec<Surface>,

    pub debug_lines: Vec<(Vec3, Vec3, Color)>,

    // f32  =  1.0 -> incoming path
    //      = -1.0 -> outgoing path
    pub paths_passing_per_edge: Vec<Vec<(usize, f32)>>,
}

impl Solution {
    pub fn pre_process(&mut self) {
        for edge_id in 0..self.paths_passing_per_edge.len() {
            self.paths_passing_per_edge[edge_id]
                .sort_by_key(|&(path_id, io)| OrderedFloat(io * self.paths[path_id].order_token));
        }
    }

    pub fn add_path(
        &mut self,
        path: &Vec<usize>,
        principal_direction: PrincipalDirection,
        order_token: f32,
        is_loop: bool,
    ) -> usize {
        let path_id = self.paths.len();

        for (i, &edge) in path.iter().enumerate() {
            // Skip last one // duplicate...
            if is_loop && i == path.len() - 1 {
                continue;
            };
            assert!(!self.paths_passing_per_edge[edge].contains(&(path_id, 1.0)));
            assert!(!self.paths_passing_per_edge[edge].contains(&(path_id, -1.0)));
            if i % 2 == 0 {
                self.paths_passing_per_edge[edge].push((path_id, 1.0));
            } else {
                self.paths_passing_per_edge[edge].push((path_id, -1.0));
            }
        }

        self.paths.push(Path {
            edges: path.clone(),
            direction: principal_direction,
            order_token,
        });

        self.pre_process();

        return path_id;
    }

    pub fn get_order_of_path_in_edge(&self, path_id: usize, edge_id: usize) -> Option<usize> {
        self.paths_passing_per_edge[edge_id]
            .iter()
            .position(|&(p, _)| p == path_id)
    }

    pub fn get_offset_of_path_in_edge(&self, path_id: usize, edge_id: usize) -> Option<f32> {
        if let Some(i) = self.get_order_of_path_in_edge(path_id, edge_id) {
            let n = self.paths_passing_per_edge[edge_id].len();
            let offset = (i as f32 + 1.) / (n as f32 + 1.);
            return Some(offset);
        } else {
            return None;
        }
    }

    pub fn get_position_of_path_in_edge(
        &self,
        path_id: usize,
        mesh: &Doconeli,
        edge_id: usize,
    ) -> Option<Vec3> {
        if let Some(offset) = self.get_offset_of_path_in_edge(path_id, edge_id) {
            return Some(mesh.get_midpoint_of_edge_with_offset(edge_id, offset));
        } else {
            return None;
        }
    }

    pub fn get_normal_of_path_in_edge(
        &self,
        path_id: usize,
        mesh: &Doconeli,
        edge_id: usize,
    ) -> Option<Vec3> {
        if let Some(offset) = self.get_offset_of_path_in_edge(path_id, edge_id) {
            return Some(mesh.get_normal_of_edge_with_offset(edge_id, offset));
        } else {
            return None;
        }
    }

    pub fn get_sequence_around_face(&self, mesh: &Doconeli, face_id: usize) -> Vec<(usize, usize)> {
        mesh.get_edges_of_face(face_id)
            .iter()
            .flat_map(|edge_id| {
                self.paths_passing_per_edge[*edge_id]
                    .iter()
                    .map(|&(path_id, _)| (path_id, *edge_id))
            })
            .collect()
    }

    pub fn intersection_in_face(
        &self,
        mesh: &Doconeli,
        path_a: usize,
        path_b: usize,
        face_id: usize,
    ) -> bool {
        let sequence = self
            .get_sequence_around_face(mesh, face_id)
            .iter()
            .map(|x| x.0)
            .collect();
        intersection_in_sequence(path_a, path_b, &sequence)
    }

    pub fn intersection_in_face_exact(
        &self,
        mesh: &Doconeli,
        segment_a: (Vec3, Vec3),
        segment_b: (Vec3, Vec3),
        face_id: usize,
    ) -> Option<Vec3> {
        let edges = mesh.get_edges_of_face(face_id);
        assert!(edges.len() == 3);

        let root = mesh.get_position_of_vertex(mesh.get_root_of_edge(edges[0]));

        // convert to 2d for easier intersection
        let e0 = mesh.get_vector_of_edge(edges[0]);
        let e2 = mesh.get_vector_of_edge(edges[2]);

        let p1_2d = convert_3d_to_2d(segment_a.0 - root, e0);
        let p2_2d = convert_3d_to_2d(segment_a.1 - root, e0);
        let p3_2d = convert_3d_to_2d(segment_b.0 - root, e0);
        let p4_2d = convert_3d_to_2d(segment_b.1 - root, e0);

        // get intersection and convert back to 3d
        if let Some(intersection_2d) = intersection_exact_in_2d(p1_2d, p2_2d, p3_2d, p4_2d) {
            let x_axis_in_3d = e0;
            let altitude = (e2.length() * e0.angle_between(-e2).cos()) / e0.length();
            let y_axis_in_3d = -e0 * altitude - e2;

            let intersection_position = root
                + x_axis_in_3d.normalize() * intersection_2d.x
                + y_axis_in_3d.normalize() * intersection_2d.y;

            return Some(intersection_position);
        }

        None
    }

    pub fn get_common_faces(
        &self,
        mesh: &Doconeli,
        loop_i_id: usize,
        loop_j_id: usize,
    ) -> HashSet<usize> {
        // Get the common passed edges
        let common_passed_edges =
            set_intersection(&self.paths[loop_i_id].edges, &self.paths[loop_j_id].edges);

        // Get the common passed faces
        let common_passed_faces: HashSet<usize> = common_passed_edges
            .iter()
            .map(|&edge_id| mesh.get_face_of_edge(edge_id))
            .collect();

        common_passed_faces
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PrimalVertexType {
    Vertex(usize),
    PointInFace(usize),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrimalVertex {
    pub vertex_type: PrimalVertexType,
    pub position: Vec3,
    pub normal: Vec3,
    pub region_id: usize,
    pub weight: usize,
}

impl MeshResource {
    pub fn get_singularities(&self) -> Vec<(usize, f32)> {
        (0..self.mesh.vertices.len())
            .map(|vertex_id| {
                (
                    vertex_id,
                    (self.mesh.get_curvature_of_vertex(vertex_id) - 2. * PI).abs(),
                )
            })
            .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
            .collect_vec()
    }

    pub fn remove_path(&mut self, path_id: usize, configuration: &mut Configuration) -> bool {
        let mut new_sol = self.sol.clone();

        let singularities =
            self.get_top_n_percent_singularities(configuration.percent_singularities);

        let normalized_path_id = path_id % new_sol.paths.len();

        for edge_id in 0..new_sol.paths_passing_per_edge.len() {
            new_sol.paths_passing_per_edge[edge_id]
                .retain(|&(path_id, _)| path_id != normalized_path_id);

            for i in 0..new_sol.paths_passing_per_edge[edge_id].len() {
                let passing_path_id = new_sol.paths_passing_per_edge[edge_id][i].0;
                if passing_path_id > normalized_path_id {
                    new_sol.paths_passing_per_edge[edge_id][i].0 = passing_path_id - 1;
                }
            }
        }

        new_sol.paths.remove(normalized_path_id);

        new_sol.pre_process();

        self.sol.regions =
            self.get_subsurfaces(&self.sol, &self.sol.intersection_graph, ColorType::Random);

        if let Some((ok_sol, ok_primal)) = self.verify_sol(&new_sol, configuration) {
            self.sol = ok_sol;
            self.primalization = ok_primal;
            return true;
        } else {
            return false;
        }
    }

    pub fn get_components_between_loops(
        &self,
        principal_direction: PrincipalDirection,
    ) -> Vec<Vec<usize>> {
        let mut segmentation = HashSet::new();
        for path in &self.sol.paths {
            if path.direction == principal_direction {
                segmentation.extend(path.edges.iter());
            }
        }

        let graph = Graph::new(&self.mesh);

        MeshResource::get_subsurfaces2(&self.mesh, &graph, &segmentation)
            .into_iter()
            .collect_vec()
    }

    pub fn get_top_n_percent_singularities(&self, n: usize) -> Vec<usize> {
        self.get_singularities()
            .into_iter()
            .take(n * self.mesh.vertices.len() / 100)
            .map(|(id, _)| id)
            .collect_vec()
    }

    pub fn components_to_singularity_count(
        &self,
        components: Vec<Vec<usize>>,
        principal_direction: PrincipalDirection,
        n: usize,
    ) -> Vec<(usize, usize)> {
        let top_curvature_vertices = self.get_top_n_percent_singularities(n);

        components
            .iter()
            .enumerate()
            .map(|(component_id, vertices)| {
                (
                    component_id,
                    vertices
                        .iter()
                        .filter(|vertex_id| top_curvature_vertices.contains(vertex_id))
                        .count(),
                )
            })
            .sorted_by(|a, b| b.1.cmp(&a.1))
            .collect_vec()
    }

    pub fn components_to_singularity_spread(
        &self,
        components: Vec<Vec<usize>>,
        principal_direction: PrincipalDirection,
        n: usize,
    ) -> Vec<(usize, f32)> {
        let top_curvature_vertices = self.get_top_n_percent_singularities(n);

        components
            .iter()
            .enumerate()
            .map(|(component_id, vertices)| {
                let coords = vertices
                    .iter()
                    .filter(|vertex_id| top_curvature_vertices.contains(vertex_id))
                    .map(|&vertex_id| {
                        self.mesh.get_position_of_vertex(vertex_id)[principal_direction as usize]
                    });

                if let Some(max) = coords.clone().into_iter().reduce(f32::max) {
                    if let Some(min) = coords.clone().into_iter().reduce(f32::min) {
                        return (component_id, max - min);
                    }
                }

                (component_id, 0.)
            })
            .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
            .collect_vec()
    }

    pub fn initialize(&mut self, configuration: &mut Configuration) {
        self.graphs[0] = Mipoga::midpoint_edge_from_mesh(&self.mesh, 0.5);
        self.graphs[1] = Mipoga::midpoint_edge_from_mesh(&self.mesh, 0.5);
        self.graphs[2] = Mipoga::midpoint_edge_from_mesh(&self.mesh, 0.5);

        self.sol = Solution {
            paths: vec![],
            paths_passing_per_edge: vec![vec![]; self.mesh.edges.len()],
            debug_lines: vec![],
            intersection_graph: Doconeli::empty(),
            ..default()
        };
    }

    pub fn polycubify(connectivity_graph: &Doconeli) -> Doconeli {
        let mut polycube_graph = connectivity_graph.clone();

        // Each corner is part of 3 connected components (corresponding to X, Y, and Z), these components set a coordinate for the corner, we simply choose the average coord.
        let mut corner_to_target = vec![Vec3::splat(0.); connectivity_graph.vertices.len()];

        for filtered_direction in [
            PrincipalDirection::X,
            PrincipalDirection::Y,
            PrincipalDirection::Z,
        ] {
            // Create a graph where all edges in the filtered direction are removed
            let mut graph = Graph::new(&connectivity_graph);
            for edge_id in 0..connectivity_graph.edges.len() {
                if connectivity_graph.edges[edge_id].direction == Some(filtered_direction) {
                    let (u, v) = connectivity_graph.get_endpoints_of_edge(edge_id);
                    graph.remove_edge(u, v);
                }
            }

            // Get all connected components after removing the edges
            let components = graph.connected_components();

            // Find average coordinate for the corners inside 1 component (for `dir` coordinate component, so 1 value, not a 3-d position)
            for component in components {
                let average_point = average(component.iter().map(|&corner_id| {
                    connectivity_graph.get_position_of_vertex(corner_id)
                        [filtered_direction as usize]
                }));
                for corner_id in component {
                    corner_to_target[corner_id][filtered_direction as usize] = average_point;
                }
            }
        }

        // Find the best candidate for each region, based on minimizing distance to the set target per region
        for corner_id in 0..connectivity_graph.vertices.len() {
            polycube_graph.vertices[corner_id].position = corner_to_target[corner_id];
        }

        polycube_graph
    }

    // Given a segmentation s.t.
    //      segmentation is DCEL with vertices inside faces of the mesh
    //      each edge in segmentation is a "super-edge" over the mesh: has a defined path of edges passed in the mesh
    pub fn get_subsurfaces(
        &self,
        sol: &Solution,
        segmentation: &Doconeli,
        color_type: ColorType,
    ) -> Vec<Surface> {
        let dual_graph = Graph::dual(&self.mesh);

        let mut subsurfaces: Vec<Surface> = vec![];

        // Go through all faces of the segmentation (each face corresponds to a region for which we want to find the subsurface)
        for region_id in 0..segmentation.faces.len() {
            let superedges = segmentation.get_edges_of_face(region_id);

            // All faces passed by the superedges in the mesh
            let faces_passed_by_superedges: HashSet<usize> = superedges
                .iter()
                .map(|e| segmentation.edges[*e].edges_between.clone().unwrap())
                .flatten()
                .map(|e| self.mesh.get_face_of_edge(e))
                .chain(superedges.iter().map(|&superedge| {
                    segmentation.vertices[segmentation.get_root_of_edge(superedge)].original_face_id
                }))
                .collect();

            let mut dual_graph_copy = dual_graph.clone();
            dual_graph_copy.remove_nodes(&faces_passed_by_superedges);

            let components = dual_graph_copy.connected_components();

            let edges_of_loops: HashSet<usize> = sol
                .paths
                .iter()
                .map(|path| path.edges.clone())
                .flatten()
                .collect();

            let all_faces: HashSet<usize> = components
                .iter()
                .filter(|&component| {
                    !component.iter().any(|&face_id| {
                        self.mesh
                            .get_edges_of_face(face_id)
                            .iter()
                            .any(|edge_id| edges_of_loops.contains(edge_id))
                    })
                })
                .flatten()
                .copied()
                .chain(faces_passed_by_superedges)
                .collect();

            // Initialize a mapping from each passed face, to all "bounding" points inside this face
            // these points will be used to form partial triangles when building the subsurface
            let mut face_to_points = HashMap::new();
            for &face_id in &all_faces {
                face_to_points.insert(face_id, vec![]);
            }

            let inner_vertices: HashSet<usize> = all_faces
                .iter()
                .map(|&face_id| self.mesh.get_vertices_of_face(face_id))
                .flatten()
                .collect::<HashSet<usize>>()
                .into_iter()
                .filter(|&vertex_id| {
                    self.mesh
                        .get_faces_of_vertex(vertex_id)
                        .iter()
                        .all(|face_id| all_faces.contains(face_id))
                })
                .collect();

            for &inner_vertex in &inner_vertices {
                for face_id in self.mesh.get_faces_of_vertex(inner_vertex) {
                    face_to_points.entry(face_id).and_modify(|v| {
                        v.push((
                            self.mesh.get_position_of_vertex(inner_vertex),
                            self.mesh.get_normal_of_vertex(inner_vertex),
                        ))
                    });
                }
            }

            for &superedge_id in &superedges {
                let edges_inside_superedge = segmentation.edges[superedge_id]
                    .edges_between
                    .clone()
                    .unwrap();

                let (vertex_start, vertex_end) = segmentation.get_endpoints_of_edge(superedge_id);

                let (face_start, face_end) = (
                    segmentation.vertices[vertex_start].original_face_id,
                    segmentation.vertices[vertex_end].original_face_id,
                );

                face_to_points.entry(face_start).and_modify(|v| {
                    v.push((
                        segmentation.get_position_of_vertex(vertex_start),
                        self.mesh.get_normal_of_face(face_start),
                    ));
                });

                face_to_points.entry(face_end).and_modify(|v| {
                    v.push((
                        segmentation.get_position_of_vertex(vertex_end),
                        self.mesh.get_normal_of_face(face_end),
                    ));
                });

                assert!((face_start == face_end) == edges_inside_superedge.is_empty());

                for edge_id in edges_inside_superedge {
                    let position = sol
                        .get_position_of_path_in_edge(
                            segmentation.edges[superedge_id].part_of_path.unwrap(),
                            &self.mesh,
                            edge_id,
                        )
                        .unwrap();

                    let normal = sol
                        .get_normal_of_path_in_edge(
                            segmentation.edges[superedge_id].part_of_path.unwrap(),
                            &self.mesh,
                            edge_id,
                        )
                        .unwrap();

                    face_to_points
                        .entry(self.mesh.get_face_of_edge(edge_id))
                        .and_modify(|v| {
                            v.push((position, normal));
                        });
                }
            }

            let subfaces = face_to_points
                .iter()
                .map(|(face_id, bounding_points)| Subface {
                    face_id: face_id.clone(),
                    bounding_points: bounding_points.clone(),
                    distortion: None,
                })
                .collect();

            let direction = [
                PrincipalDirection::X,
                PrincipalDirection::Y,
                PrincipalDirection::Z,
            ]
            .into_iter()
            .filter(|&dir| {
                !superedges
                    .iter()
                    .map(|&edge_id| segmentation.edges[edge_id].direction)
                    .contains(&Some(dir))
            })
            .next();

            // let color = match color_type {
            //     ColorType::Static(c) => c,
            //     ColorType::DirectionPrimary => get_color(direction.unwrap(), true, &configuration),
            //     ColorType::DirectionSecondary => get_color(direction.unwrap(), false, &configuration),
            //     ColorType::Random => get_random_color(),
            // };

            subsurfaces.push(Surface {
                id: region_id,
                faces: subfaces,
                direction,
                color: match superedges.len() {
                    3 => Some(Color::rgb(0.25, 0.25, 0.25)),
                    4 => Some(Color::rgb(0.2, 0.2, 0.2)),
                    5 => Some(Color::rgb(0.15, 0.15, 0.15)),
                    6 => Some(Color::rgb(0.1, 0.1, 0.1)),
                    _ => Some(Color::BLACK),
                },
                inner_vertices,
                degree: superedges.len(),
            });
        }

        subsurfaces
    }

    // Given a segmentation (list of edges to remove)
    // return connected components of vertices
    pub fn get_subsurfaces2(
        mesh: &Doconeli,
        graph: &Graph,
        segmentation: &HashSet<usize>, // edges to remove
    ) -> HashSet<Vec<usize>> {
        let mut graph = graph.clone();

        for &edge_id in segmentation {
            let (node_i, node_j) = mesh.get_endpoints_of_edge(edge_id);
            graph.remove_edge(node_i, node_j);
        }

        let components = graph.connected_components();

        return components;
    }

    // Given a segmentation (list of vertices to remove)
    // return connected components of vertices
    pub fn get_subsurfaces3(
        graph: &Graph,
        segmentation: &HashSet<usize>, // vertices to remove
    ) -> HashSet<Vec<usize>> {
        let mut graph = graph.clone();

        graph.remove_nodes(segmentation);

        let components = graph.connected_components();

        return components;
    }

    pub fn add_loop(&self, configuration: &mut Configuration) -> Option<(Solution, Primalization)> {
        let principal_direction = configuration.choose_direction;

        let mut timer = Timer::new();

        // Dual graph of mesh, with directed weighted edges. Edge weights see paper.
        let mut graph = self.graphs[principal_direction as usize].clone();
        graph
            .precompute_angular_loopy_weights(principal_direction.to_vector(), configuration.gamma);

        let mut vertex_graph = Graph::new(&self.mesh);

        //      Two options depending on configuration.find_global=true/false
        //          - add anywhere on the mesh
        //          - add only in certain region / between previous loops... decide starting points etc...
        let mut all_edges = (0..self.mesh.edges.len()).collect();
        let mut all_vertices = (0..self.mesh.vertices.len()).collect();
        // only look in the region with lots of curvature
        if !configuration.find_global {
            // grab region (their vertices)
            let components = self.get_components_between_loops(principal_direction);
            let target_component = self.components_to_singularity_spread(
                components.clone(),
                principal_direction,
                configuration.percent_singularities,
            )[configuration.choose_component % components.len()]
            .0;
            let subset_vertices = components[target_component].clone();

            // vertices to remove
            let vertices_to_remove = (0..self.mesh.vertices.len())
                .into_iter()
                .filter(|vertex_id| !subset_vertices.contains(vertex_id));

            // get edges part of this region (all incident to vertices)
            let subset_edges = subset_vertices
                .clone()
                .into_iter()
                .map(|vertex_id| self.mesh.get_edges_of_vertex(vertex_id))
                .flatten()
                .map(|edge_id| [edge_id, self.mesh.get_twin_of_edge(edge_id)])
                .flatten()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect_vec();

            // edges to remove
            let edges_to_remove = (0..self.mesh.edges.len())
                .into_iter()
                .filter(|edge_id| !subset_edges.contains(edge_id));

            graph.remove_nodes(&HashSet::from_iter(edges_to_remove));
            vertex_graph.remove_nodes(&HashSet::from_iter(vertices_to_remove));

            all_vertices = subset_vertices.clone();
            all_edges = subset_edges.clone();
        }

        // Find all candidate loops (based on configuration settings)
        let mut candidate_loops: BinaryHeap<(OrderedFloat<f32>, (usize, Vec<usize>))> = (0
            ..configuration.algorithm_samples)
            .into_par_iter()
            .map(|candidate_id| {
                // Sample a random edge as starting point
                let edge = all_edges[rand::thread_rng().gen_range(0..all_edges.len())];
                // Find shortest path to itself (for a loop)
                if let Some(path) = graph.shortest_path(edge, edge) {
                    Some((candidate_id, path))
                } else {
                    None
                }
            })
            .flatten()
            .filter(|(candidate_id, path)| {
                let faces: HashSet<usize> = path
                    .iter()
                    .map(|&edge_id| self.mesh.get_face_of_edge(edge_id))
                    .collect();

                for face_id in faces {
                    if path[1..]
                        .iter()
                        .filter(|&&edge_id| self.mesh.get_face_of_edge(edge_id) == face_id)
                        .count()
                        != 2
                    {
                        return false;
                    }
                }
                true
            })
            .map(|(candidate_id, path)| {
                // odd length paths.. they take 3 edges inside 1 triangle.. must not be used
                if path.len() % 2 != 1 {
                    return None;
                }

                let score = match configuration.loop_scoring_scheme {
                    LoopScoring::PathLength => {
                        // Path length
                        let length = path
                            .windows(2)
                            .map(|e| {
                                self.graphs[principal_direction as usize]
                                    .get_position_of_vertex(e[0])
                                    .distance(
                                        self.graphs[principal_direction as usize]
                                            .get_position_of_vertex(e[1]),
                                    )
                            })
                            .sum::<f32>();

                        length
                    }
                    LoopScoring::LoopDistribution => {
                        // Distance to other loops
                        let avg_coord = average(path.iter().map(|&x| {
                            self.graphs[principal_direction as usize].get_position_of_vertex(x)
                        }));

                        let distances = self
                            .sol
                            .paths
                            .iter()
                            .filter(|&other_path| other_path.direction == principal_direction)
                            .map(|other_path| {
                                avg_coord.distance(average(other_path.edges.iter().map(|&x| {
                                    self.graphs[principal_direction as usize]
                                        .get_position_of_vertex(x)
                                })))
                            });

                        let min_distance = distances.clone().into_iter().reduce(f32::min).unwrap();

                        min_distance
                    }
                    LoopScoring::SingularitySeparationCount => {
                        // Separating singularities by seperating singularities evenly
                        let components = MeshResource::get_subsurfaces2(
                            &self.mesh,
                            &vertex_graph,
                            &path.clone().into_iter().collect::<HashSet<_>>(),
                        )
                        .into_iter()
                        .map(|c| {
                            c.into_iter()
                                .filter(|vertex_id| all_vertices.contains(vertex_id))
                                .collect_vec()
                        })
                        .collect_vec();

                        let components_with_singularities = self.components_to_singularity_count(
                            components,
                            principal_direction,
                            100,
                        );

                        if components_with_singularities.len() != 2 {
                            return None;
                        }

                        let score = std::cmp::min(
                            components_with_singularities[0].1,
                            components_with_singularities[1].1,
                        );

                        score as f32
                    }

                    LoopScoring::SingularitySeparationSpread => {
                        // Separating singularities by minimizing the spread
                        let components = MeshResource::get_subsurfaces2(
                            &self.mesh,
                            &vertex_graph,
                            &path.clone().into_iter().collect::<HashSet<_>>(),
                        )
                        .into_iter()
                        .map(|c| {
                            c.into_iter()
                                .filter(|vertex_id| all_vertices.contains(vertex_id))
                                .collect_vec()
                        })
                        .collect_vec();

                        let components_with_singularities: Vec<(usize, f32)> = self
                            .components_to_singularity_spread(
                                components,
                                principal_direction,
                                configuration.percent_singularities,
                            );

                        if components_with_singularities.len() != 2 {
                            return None;
                        }

                        let score = -f32::max(
                            components_with_singularities[0].1,
                            components_with_singularities[1].1,
                        );

                        score
                    }
                };

                Some((OrderedFloat(score as f32), (candidate_id, path)))
            })
            .flatten()
            .collect();

        timer.report(&format!(
            "Sampled {} {principal_direction:?}-loops. Found {} valid loops.",
            configuration.algorithm_samples,
            candidate_loops.len(),
        ));

        let mut new_sol = self.sol.clone();
        // Go through candidate loops in the ordering of best score
        'outer: while let Some((score, (candidate_id, candidate_loop))) = candidate_loops.pop() {
            timer.reset();
            timer.report(&format!(
                "Verifying candidate loop {candidate_id} with score {score}"
            ));

            // Start a new solution, with the candidate path added.
            new_sol = self.sol.clone();

            let this_loop_id = new_sol.add_path(
                &candidate_loop,
                principal_direction,
                rand::thread_rng().gen_range(0.0..1.0),
                true,
            );

            // Validate intersection patterns
            // Get the intersection patterns
            let intersection_patterns =
                new_sol
                    .paths
                    .par_iter()
                    .enumerate()
                    .map(|(other_loop_id, other_loop)| {
                        let number_of_intersections = new_sol
                            .get_common_faces(&self.mesh, this_loop_id, other_loop_id)
                            .into_iter()
                            .filter(|&face_id| {
                                new_sol.intersection_in_face(
                                    &self.mesh,
                                    this_loop_id,
                                    other_loop_id,
                                    face_id,
                                )
                            })
                            .count();

                        (other_loop.direction, number_of_intersections)
                    });

            // ACTUALY USE CONFIG FOR RULES...
            if !intersection_patterns
                .clone()
                .all(|(dir, nrx)| nrx == 0 || (nrx == 2 && principal_direction != dir))
            {
                timer.report(&format!("❌  Detected invalid intersections"));
                continue 'outer;
            }

            if new_sol.paths.len() >= 3 {
                let intersects_x = intersection_patterns
                    .clone()
                    .any(|(dir, nrx)| nrx > 0 && dir == PrincipalDirection::X)
                    || principal_direction == PrincipalDirection::X;
                let intersects_y = intersection_patterns
                    .clone()
                    .any(|(dir, nrx)| nrx > 0 && dir == PrincipalDirection::Y)
                    || principal_direction == PrincipalDirection::Y;
                let intersects_z = intersection_patterns
                    .clone()
                    .any(|(dir, nrx)| nrx > 0 && dir == PrincipalDirection::Z)
                    || principal_direction == PrincipalDirection::Z;
                if !(intersects_x && intersects_y && intersects_z) {
                    timer.report(&format!("❌  Detected invalid intersections (must intersect atleast one loop of other two labels (X: {intersects_x}, Y: {intersects_y}, Z: {intersects_z})"));
                    continue 'outer;
                }
            }

            timer.report(&format!("🆗  Loop passed intersections checks"));
            timer.reset();

            let mut prim = Primalization::initialize(&self.mesh.clone(), &new_sol).unwrap();
            let singularities =
                self.get_top_n_percent_singularities(configuration.percent_singularities);

            // ACTUALY USE CONFIG FOR RULES...
            if new_sol.paths.len() >= 3 {
                let mut intersections: Vec<Vertex> = vec![];
                let mut loop_to_intersections = HashMap::new();

                let mut intersection_connections = vec![];

                for loop_id in 0..(new_sol.paths.len()) {
                    loop_to_intersections.insert(loop_id, vec![]);
                }

                for loop_i_id in 0..(new_sol.paths.len() - 1) {
                    for loop_j_id in (loop_i_id + 1)..new_sol.paths.len() {
                        let common_faces =
                            new_sol.get_common_faces(&self.mesh, loop_i_id, loop_j_id);

                        for common_passed_face in common_faces {
                            // find intersection between loop_i and loop_j in common_passed_face ->

                            // first find the endpoints of both loops inside this face

                            let common_passed_face_edges =
                                self.mesh.get_edges_of_face(common_passed_face);

                            let segment_i = common_passed_face_edges
                                .iter()
                                .filter(|edge_id| new_sol.paths[loop_i_id].edges.contains(edge_id))
                                .map(|&edge_id| {
                                    new_sol
                                        .get_position_of_path_in_edge(
                                            loop_i_id, &self.mesh, edge_id,
                                        )
                                        .unwrap()
                                })
                                .collect_tuple()
                                .unwrap();

                            let segment_j = common_passed_face_edges
                                .iter()
                                .filter(|edge_id| new_sol.paths[loop_j_id].edges.contains(edge_id))
                                .map(|&edge_id| {
                                    new_sol
                                        .get_position_of_path_in_edge(
                                            loop_j_id, &self.mesh, edge_id,
                                        )
                                        .unwrap()
                                })
                                .collect_tuple()
                                .unwrap();

                            if let Some(intersection_position) = new_sol.intersection_in_face_exact(
                                &self.mesh,
                                segment_i,
                                segment_j,
                                common_passed_face,
                            ) {
                                if intersections
                                    .iter()
                                    .find(|&v| {
                                        v.position.distance(intersection_position) <= 0.000001
                                    })
                                    .is_some()
                                {
                                    continue 'outer;
                                }

                                loop_to_intersections
                                    .entry(loop_i_id)
                                    .or_default()
                                    .push(intersections.len());

                                loop_to_intersections
                                    .entry(loop_j_id)
                                    .or_default()
                                    .push(intersections.len());

                                let seq: Vec<(usize, usize)> = new_sol
                                    .get_sequence_around_face(&self.mesh, common_passed_face)
                                    .iter()
                                    .filter(|x| x.0 == loop_i_id || x.0 == loop_j_id)
                                    .copied()
                                    .collect();

                                intersections.push(Vertex {
                                    some_edge: None,
                                    position: intersection_position,
                                    normal: self.mesh.get_normal_of_face(common_passed_face),
                                    original_face_id: common_passed_face,
                                    ordering: seq,
                                    ..default()
                                });
                            }
                        }
                    }
                }

                let mut vertexpair_to_edge = HashMap::new();

                for loop_id in 0..(new_sol.paths.len()) {
                    let loop_intersections = loop_to_intersections.entry(loop_id).or_default();
                    let passed_edges = new_sol.paths[loop_id].edges.clone();

                    loop_intersections.sort_by(|&intersection_a, &intersection_b| {
                        let intersection_a_face = intersections[intersection_a].original_face_id;
                        let intersection_b_face = intersections[intersection_b].original_face_id;
                        let mut intersection_a_edges: Vec<usize> = self
                            .mesh
                            .get_edges_of_face(intersection_a_face)
                            .iter()
                            .filter(|e| passed_edges.contains(e))
                            .copied()
                            .collect();

                        assert!(intersection_a_edges.len() == 2);

                        let mut intersection_b_edges: Vec<usize> = self
                            .mesh
                            .get_edges_of_face(intersection_b_face)
                            .iter()
                            .filter(|e| passed_edges.contains(e))
                            .copied()
                            .collect();

                        assert!(intersection_b_edges.len() == 2);

                        intersection_a_edges.sort_by(|a, b| {
                            let pos_a = passed_edges.iter().position(|x| x == a).unwrap();
                            let pos_b = passed_edges.iter().position(|x| x == b).unwrap();
                            (&pos_a).cmp(&pos_b)
                        });

                        intersection_b_edges.sort_by(|a, b| {
                            let pos_a = passed_edges.iter().position(|x| x == a).unwrap();
                            let pos_b = passed_edges.iter().position(|x| x == b).unwrap();
                            (&pos_a).cmp(&pos_b)
                        });

                        if intersection_a_face != intersection_b_face {
                            let pos_intersection_a = passed_edges
                                .iter()
                                .position(|&x| x == intersection_a_edges[0])
                                .unwrap();
                            let pos_intersection_b = passed_edges
                                .iter()
                                .position(|&x| x == intersection_b_edges[0])
                                .unwrap();

                            (&pos_intersection_a).cmp(&pos_intersection_b)
                        } else {
                            assert!(intersection_a_edges[0] == intersection_b_edges[0]);

                            let vector_root_edge =
                                self.mesh.get_vector_of_edge(intersection_a_edges[0]);
                            let pos_root_edge = self.mesh.get_position_of_vertex(
                                self.mesh.get_root_of_edge(intersection_a_edges[0]),
                            );

                            let pos_intersection_a = intersections[intersection_a].position;
                            let pos_intersection_b = intersections[intersection_b].position;

                            let vector_intersection_a = pos_intersection_a - pos_root_edge;
                            let vector_intersection_b = pos_intersection_b - pos_root_edge;

                            let angle_intersection_a =
                                vector_root_edge.angle_between(vector_intersection_a);
                            let angle_intersection_b =
                                vector_root_edge.angle_between(vector_intersection_b);

                            (&angle_intersection_a)
                                .partial_cmp(&angle_intersection_b)
                                .unwrap()
                        }
                    });

                    loop_intersections.push(loop_intersections[0]);

                    for consecutive_intersections in loop_intersections.windows(2) {
                        let mut loop_edges = new_sol.paths[loop_id].edges[1..].to_vec();

                        let intersection_a = consecutive_intersections[0];
                        let intersection_a_face = intersections[intersection_a].original_face_id;
                        let intersection_a_edges = self.mesh.get_edges_of_face(intersection_a_face);

                        let intersection_b = consecutive_intersections[1];
                        let intersection_b_face = intersections[intersection_b].original_face_id;
                        let intersection_b_edges = self.mesh.get_edges_of_face(intersection_b_face);

                        let edges_bet = if intersection_a_face == intersection_b_face {
                            vec![]
                        } else if self
                            .mesh
                            .get_neighbors_of_face_edgewise(intersection_a_face)
                            .contains(&intersection_b_face)
                        {
                            let the_edge: Vec<usize> = intersection_a_edges
                                .iter()
                                .copied()
                                .filter(|&edge_id| {
                                    intersection_b_edges
                                        .contains(&self.mesh.get_twin_of_edge(edge_id))
                                })
                                .collect();

                            assert!(the_edge.length() == 1);

                            vec![the_edge[0], self.mesh.get_twin_of_edge(the_edge[0])]
                        } else {
                            let start_edge_positions: Vec<usize> = loop_edges
                                .iter()
                                .positions(|edge_id| intersection_a_edges.contains(edge_id))
                                .collect();

                            assert!(start_edge_positions.len() == 2);

                            loop_edges.remove(start_edge_positions[1]);

                            let end_edge_positions: Vec<usize> = loop_edges
                                .iter()
                                .positions(|edge_id| intersection_b_edges.contains(edge_id))
                                .collect();

                            assert!(end_edge_positions.len() == 2);

                            loop_edges.remove(end_edge_positions[1]);

                            let start_edge_position: Vec<usize> = loop_edges
                                .iter()
                                .positions(|edge_id| intersection_a_edges.contains(edge_id))
                                .collect();

                            assert!(start_edge_position.len() == 1);

                            let end_edge_position: Vec<usize> = loop_edges
                                .iter()
                                .positions(|edge_id| intersection_b_edges.contains(edge_id))
                                .collect();

                            assert!(end_edge_position.len() == 1);

                            let mut edges_between_intersections =
                                if start_edge_position[0] <= end_edge_position[0] {
                                    loop_edges[(start_edge_position[0] + 1)..end_edge_position[0]]
                                        .to_vec()
                                } else {
                                    [
                                        &loop_edges[(start_edge_position[0] + 1)..],
                                        &loop_edges[..end_edge_position[0]],
                                    ]
                                    .concat()
                                };

                            let first_edge = self
                                .mesh
                                .get_twin_of_edge(*edges_between_intersections.first().unwrap());
                            let last_edge = self
                                .mesh
                                .get_twin_of_edge(*edges_between_intersections.last().unwrap());

                            edges_between_intersections.insert(0, first_edge);
                            edges_between_intersections.push(last_edge);

                            edges_between_intersections
                        };

                        let mut rev_edges_between_intersections = edges_bet.clone();
                        rev_edges_between_intersections.reverse();

                        // println!("{} {}", start_edge_position, end_edge_position);
                        // println!("edges between: {:?}", edges_between_intersections);
                        // println!("all edges: {:?}", new_sol.paths[loop_id].edges);

                        let this_connection = intersection_connections.len();
                        let twin_connection = intersection_connections.len() + 1;

                        intersection_connections.push(Edge {
                            root: intersection_a,
                            face: None,
                            next: None,
                            twin: Some(twin_connection),
                            label: Some(new_sol.paths[loop_id].direction as usize),
                            direction: Some(new_sol.paths[loop_id].direction),
                            part_of_path: Some(loop_id),
                            edges_between: Some(edges_bet),
                            ..default()
                        });

                        vertexpair_to_edge.insert(
                            (
                                intersection_a,
                                intersection_b,
                                new_sol.paths[loop_id].direction as usize,
                            ),
                            this_connection,
                        );
                        vertexpair_to_edge.insert(
                            (
                                intersection_b,
                                intersection_a,
                                new_sol.paths[loop_id].direction as usize,
                            ),
                            twin_connection,
                        );

                        intersection_connections.push(Edge {
                            root: intersection_b,
                            face: None,
                            next: None,
                            twin: Some(this_connection),
                            label: Some(new_sol.paths[loop_id].direction as usize),
                            direction: Some(new_sol.paths[loop_id].direction),
                            part_of_path: Some(loop_id),
                            edges_between: Some(rev_edges_between_intersections),
                            ..default()
                        });
                    }
                }

                for intersection_connection in 0..intersection_connections.len() {
                    let edge = &intersection_connections[intersection_connection];

                    let loop_id = edge.part_of_path.unwrap();

                    let intersection_ordering_in_loop =
                        loop_to_intersections.get(&loop_id).unwrap();

                    let u = edge.root;
                    let v = intersection_connections[edge.twin.unwrap()].root;

                    let pair = intersection_ordering_in_loop
                        .windows(2)
                        .filter(|x| (x[0] == u && x[1] == v) || (x[0] == v && x[1] == u))
                        .next()
                        .unwrap();

                    let forwards =
                        pair.iter().position(|&x| x == u) < pair.iter().position(|&x| x == v);

                    let seq = &intersections[v].ordering;

                    let this_edges: Vec<usize> =
                        seq.iter().filter(|x| x.0 == loop_id).map(|x| x.1).collect();

                    let other_loop = seq.iter().filter(|x| x.0 != loop_id).next().unwrap().0;
                    let other_edges: Vec<usize> = seq
                        .iter()
                        .filter(|x| x.0 == other_loop)
                        .map(|x| x.1)
                        .collect();

                    let mut this_first = new_sol.paths[loop_id]
                        .edges
                        .iter()
                        .find_or_first(|x| this_edges.contains(x))
                        .unwrap()
                        .clone();
                    let other_first = new_sol.paths[other_loop]
                        .edges
                        .iter()
                        .find_or_first(|x| other_edges.contains(x))
                        .unwrap()
                        .clone();

                    let mut this_second = this_edges
                        .iter()
                        .filter(|&&x| x != this_first)
                        .next()
                        .unwrap()
                        .clone();
                    let other_second = other_edges
                        .iter()
                        .filter(|&&x| x != other_first)
                        .next()
                        .unwrap()
                        .clone();

                    if !forwards {
                        swap(&mut this_first, &mut this_second);
                    }

                    let next_step = seq[(seq
                        .iter()
                        .position(|x| x.0 == loop_id && x.1 == this_first)
                        .unwrap()
                        + 1)
                        % 4]
                    .1;

                    let other_forwards = next_step == other_second;

                    let mut intersection_ordering_in_other =
                        loop_to_intersections.get(&other_loop).unwrap().clone();

                    intersection_ordering_in_other.remove(intersection_ordering_in_other.len() - 1);

                    let cur_pos = intersection_ordering_in_other
                        .iter()
                        .position(|&x| x == v)
                        .unwrap();

                    let next_other_step = if other_forwards {
                        (cur_pos + intersection_ordering_in_other.len() + 1)
                            % intersection_ordering_in_other.len()
                    } else {
                        (cur_pos + intersection_ordering_in_other.len() - 1)
                            % intersection_ordering_in_other.len()
                    };

                    let next_pos = intersection_ordering_in_other[next_other_step];

                    let next_edge_id = vertexpair_to_edge
                        .get(&(v, next_pos, new_sol.paths[other_loop].direction as usize))
                        .copied()
                        .unwrap();

                    intersection_connections[intersection_connection].next = Some(next_edge_id);
                }

                new_sol.intersection_graph =
                    Doconeli::from_vertices_and_edges(intersections, intersection_connections);

                for face_id in 0..new_sol.intersection_graph.faces.len() {
                    let labels = new_sol
                        .intersection_graph
                        .get_edges_of_face(face_id)
                        .into_iter()
                        .map(|edge_id| new_sol.intersection_graph.edges[edge_id].direction);

                    if labels.clone().any(|l| l.is_none()) {
                        timer.report(&format!(
                            "[❌] Detected incorrect region (some labels undefined)"
                        ));
                        continue 'outer;
                    }

                    let boundary_size = labels.clone().count();
                    if boundary_size > 6 || boundary_size < 3 {
                        timer.report(&format!(
                            "❌  Detected incorrect region (boundary has size: {boundary_size})"
                        ));
                        continue 'outer;
                    }

                    let count_x = labels
                        .clone()
                        .filter(|&l| l == Some(PrincipalDirection::X))
                        .count();
                    if count_x > 2 {
                        timer.report(&format!("❌  Detected incorrect region {face_id} (has too many Z in boundary: {count_x})"));
                        continue 'outer;
                    }

                    let count_y = labels
                        .clone()
                        .filter(|&l| l == Some(PrincipalDirection::Y))
                        .count();
                    if count_y > 2 {
                        timer.report(&format!("❌  Detected incorrect region {face_id} (has too many Y in boundary: {count_y})"));
                        continue 'outer;
                    }

                    let count_z = labels
                        .clone()
                        .filter(|&l| l == Some(PrincipalDirection::Z))
                        .count();
                    if count_z > 2 {
                        timer.report(&format!("❌  Detected incorrect region {face_id} (has too many Z in boundary: {count_z})"));
                        continue 'outer;
                    }
                }

                timer.report(&format!("🆗  Loop passed regions checks"));

                new_sol.regions =
                    self.get_subsurfaces(&new_sol, &new_sol.intersection_graph, ColorType::Random);

                if let Some(new_prim) = Primalization::initialize(&self.mesh.clone(), &new_sol) {
                    prim = new_prim;
                } else {
                    timer.report(&format!("❌  Detected impossible primalization 1111"));
                    continue 'outer;
                }

                if !prim.place_primals(singularities, configuration) {
                    timer.report(&format!(
                        "❌  Detected impossible primal placement {:?}",
                        new_sol.paths.len()
                    ));
                    continue 'outer;
                }

                timer.report(&format!("🆗  Primals placed"));

                if !prim.connect_primals(configuration) {
                    timer.report(&format!(
                        "❌  Detected impossible primal connections {:?}",
                        new_sol.paths.len()
                    ));
                    continue 'outer;
                }

                timer.report(&format!("🆗  Primals connected"));
            }

            return Some((new_sol, prim));
        }

        return None;
    }

    pub fn verify_sol(
        &self,
        mut sol: &Solution,
        configuration: &mut Configuration,
    ) -> Option<(Solution, Primalization)> {
        let mut new_sol = sol.clone();

        let mut timer = Timer::new();

        for this_loop_id in 0..new_sol.paths.len() {
            let principal_direction = new_sol.paths[this_loop_id].direction;

            // Validate intersection patterns
            // Get the intersection patterns
            let intersection_patterns =
                new_sol
                    .paths
                    .par_iter()
                    .enumerate()
                    .map(|(other_loop_id, other_loop)| {
                        let number_of_intersections = new_sol
                            .get_common_faces(&self.mesh, this_loop_id, other_loop_id)
                            .into_iter()
                            .filter(|&face_id| {
                                new_sol.intersection_in_face(
                                    &self.mesh,
                                    this_loop_id,
                                    other_loop_id,
                                    face_id,
                                )
                            })
                            .count();

                        (other_loop.direction, number_of_intersections)
                    });

            // ACTUALY USE CONFIG FOR RULES...
            if !intersection_patterns
                .clone()
                .all(|(dir, nrx)| nrx == 0 || (nrx == 2 && principal_direction != dir))
            {
                timer.report(&format!("❌  Detected invalid intersections"));
                return None;
            }

            if new_sol.paths.len() >= 3 {
                let intersects_x = intersection_patterns
                    .clone()
                    .any(|(dir, nrx)| nrx > 0 && dir == PrincipalDirection::X)
                    || principal_direction == PrincipalDirection::X;
                let intersects_y = intersection_patterns
                    .clone()
                    .any(|(dir, nrx)| nrx > 0 && dir == PrincipalDirection::Y)
                    || principal_direction == PrincipalDirection::Y;
                let intersects_z = intersection_patterns
                    .clone()
                    .any(|(dir, nrx)| nrx > 0 && dir == PrincipalDirection::Z)
                    || principal_direction == PrincipalDirection::Z;
                if !(intersects_x && intersects_y && intersects_z) {
                    timer.report(&format!("❌  Detected invalid intersections (must intersect atleast one loop of other two labels (X: {intersects_x}, Y: {intersects_y}, Z: {intersects_z})"));
                    return None;
                }
            }
        }

        timer.report(&format!("🆗  Loops passed intersections checks"));
        timer.reset();

        // ACTUALY USE CONFIG FOR RULES...

        let mut intersections: Vec<Vertex> = vec![];
        let mut loop_to_intersections = HashMap::new();

        let mut intersection_connections = vec![];

        for loop_id in 0..(new_sol.paths.len()) {
            loop_to_intersections.insert(loop_id, vec![]);
        }

        for loop_i_id in 0..(new_sol.paths.len() - 1) {
            for loop_j_id in (loop_i_id + 1)..new_sol.paths.len() {
                let common_faces = new_sol.get_common_faces(&self.mesh, loop_i_id, loop_j_id);

                for common_passed_face in common_faces {
                    // find intersection between loop_i and loop_j in common_passed_face ->

                    // first find the endpoints of both loops inside this face

                    let common_passed_face_edges = self.mesh.get_edges_of_face(common_passed_face);

                    let segment_i = common_passed_face_edges
                        .iter()
                        .filter(|edge_id| new_sol.paths[loop_i_id].edges.contains(edge_id))
                        .map(|&edge_id| {
                            new_sol
                                .get_position_of_path_in_edge(loop_i_id, &self.mesh, edge_id)
                                .unwrap()
                        })
                        .collect_tuple()
                        .unwrap();

                    let segment_j = common_passed_face_edges
                        .iter()
                        .filter(|edge_id| new_sol.paths[loop_j_id].edges.contains(edge_id))
                        .map(|&edge_id| {
                            new_sol
                                .get_position_of_path_in_edge(loop_j_id, &self.mesh, edge_id)
                                .unwrap()
                        })
                        .collect_tuple()
                        .unwrap();

                    if let Some(intersection_position) = new_sol.intersection_in_face_exact(
                        &self.mesh,
                        segment_i,
                        segment_j,
                        common_passed_face,
                    ) {
                        if intersections
                            .iter()
                            .find(|&v| v.position.distance(intersection_position) <= 0.00001)
                            .is_some()
                        {
                            return None;
                        }

                        loop_to_intersections
                            .entry(loop_i_id)
                            .or_default()
                            .push(intersections.len());

                        loop_to_intersections
                            .entry(loop_j_id)
                            .or_default()
                            .push(intersections.len());

                        let seq: Vec<(usize, usize)> = new_sol
                            .get_sequence_around_face(&self.mesh, common_passed_face)
                            .iter()
                            .filter(|x| x.0 == loop_i_id || x.0 == loop_j_id)
                            .copied()
                            .collect();

                        intersections.push(Vertex {
                            some_edge: None,
                            position: intersection_position,
                            normal: self.mesh.get_normal_of_face(common_passed_face),
                            original_face_id: common_passed_face,
                            ordering: seq,
                            ..default()
                        });
                    }
                }
            }
        }

        if intersections.len() < 6 {
            return None;
        }

        let mut vertexpair_to_edge = HashMap::new();

        for loop_id in 0..(new_sol.paths.len()) {
            let loop_intersections = loop_to_intersections.entry(loop_id).or_default();
            let passed_edges = new_sol.paths[loop_id].edges.clone();

            loop_intersections.sort_by(|&intersection_a, &intersection_b| {
                let intersection_a_face = intersections[intersection_a].original_face_id;
                let intersection_b_face = intersections[intersection_b].original_face_id;
                let mut intersection_a_edges: Vec<usize> = self
                    .mesh
                    .get_edges_of_face(intersection_a_face)
                    .iter()
                    .filter(|e| passed_edges.contains(e))
                    .copied()
                    .collect();

                assert!(intersection_a_edges.len() == 2);

                let mut intersection_b_edges: Vec<usize> = self
                    .mesh
                    .get_edges_of_face(intersection_b_face)
                    .iter()
                    .filter(|e| passed_edges.contains(e))
                    .copied()
                    .collect();

                assert!(intersection_b_edges.len() == 2);

                intersection_a_edges.sort_by(|a, b| {
                    let pos_a = passed_edges.iter().position(|x| x == a).unwrap();
                    let pos_b = passed_edges.iter().position(|x| x == b).unwrap();
                    (&pos_a).cmp(&pos_b)
                });

                intersection_b_edges.sort_by(|a, b| {
                    let pos_a = passed_edges.iter().position(|x| x == a).unwrap();
                    let pos_b = passed_edges.iter().position(|x| x == b).unwrap();
                    (&pos_a).cmp(&pos_b)
                });

                if intersection_a_face != intersection_b_face {
                    let pos_intersection_a = passed_edges
                        .iter()
                        .position(|&x| x == intersection_a_edges[0])
                        .unwrap();
                    let pos_intersection_b = passed_edges
                        .iter()
                        .position(|&x| x == intersection_b_edges[0])
                        .unwrap();

                    (&pos_intersection_a).cmp(&pos_intersection_b)
                } else {
                    assert!(intersection_a_edges[0] == intersection_b_edges[0]);

                    let vector_root_edge = self.mesh.get_vector_of_edge(intersection_a_edges[0]);
                    let pos_root_edge = self.mesh.get_position_of_vertex(
                        self.mesh.get_root_of_edge(intersection_a_edges[0]),
                    );

                    let pos_intersection_a = intersections[intersection_a].position;
                    let pos_intersection_b = intersections[intersection_b].position;

                    let vector_intersection_a = pos_intersection_a - pos_root_edge;
                    let vector_intersection_b = pos_intersection_b - pos_root_edge;

                    let angle_intersection_a =
                        vector_root_edge.angle_between(vector_intersection_a);
                    let angle_intersection_b =
                        vector_root_edge.angle_between(vector_intersection_b);

                    (&angle_intersection_a)
                        .partial_cmp(&angle_intersection_b)
                        .unwrap()
                }
            });

            if loop_intersections.is_empty() {
                return None;
            }

            loop_intersections.push(loop_intersections[0]);

            for consecutive_intersections in loop_intersections.windows(2) {
                let mut loop_edges = new_sol.paths[loop_id].edges[1..].to_vec();

                let intersection_a = consecutive_intersections[0];
                let intersection_a_face = intersections[intersection_a].original_face_id;
                let intersection_a_edges = self.mesh.get_edges_of_face(intersection_a_face);

                let intersection_b = consecutive_intersections[1];
                let intersection_b_face = intersections[intersection_b].original_face_id;
                let intersection_b_edges = self.mesh.get_edges_of_face(intersection_b_face);

                let edges_bet = if intersection_a_face == intersection_b_face {
                    vec![]
                } else if self
                    .mesh
                    .get_neighbors_of_face_edgewise(intersection_a_face)
                    .contains(&intersection_b_face)
                {
                    let the_edge: Vec<usize> = intersection_a_edges
                        .iter()
                        .copied()
                        .filter(|&edge_id| {
                            intersection_b_edges.contains(&self.mesh.get_twin_of_edge(edge_id))
                        })
                        .collect();

                    assert!(the_edge.length() == 1);

                    vec![the_edge[0], self.mesh.get_twin_of_edge(the_edge[0])]
                } else {
                    let start_edge_positions: Vec<usize> = loop_edges
                        .iter()
                        .positions(|edge_id| intersection_a_edges.contains(edge_id))
                        .collect();

                    assert!(start_edge_positions.len() == 2);

                    loop_edges.remove(start_edge_positions[1]);

                    let end_edge_positions: Vec<usize> = loop_edges
                        .iter()
                        .positions(|edge_id| intersection_b_edges.contains(edge_id))
                        .collect();

                    assert!(end_edge_positions.len() == 2);

                    loop_edges.remove(end_edge_positions[1]);

                    let start_edge_position: Vec<usize> = loop_edges
                        .iter()
                        .positions(|edge_id| intersection_a_edges.contains(edge_id))
                        .collect();

                    assert!(start_edge_position.len() == 1);

                    let end_edge_position: Vec<usize> = loop_edges
                        .iter()
                        .positions(|edge_id| intersection_b_edges.contains(edge_id))
                        .collect();

                    assert!(end_edge_position.len() == 1);

                    let mut edges_between_intersections =
                        if start_edge_position[0] <= end_edge_position[0] {
                            loop_edges[(start_edge_position[0] + 1)..end_edge_position[0]].to_vec()
                        } else {
                            [
                                &loop_edges[(start_edge_position[0] + 1)..],
                                &loop_edges[..end_edge_position[0]],
                            ]
                            .concat()
                        };

                    let first_edge = self
                        .mesh
                        .get_twin_of_edge(*edges_between_intersections.first().unwrap());
                    let last_edge = self
                        .mesh
                        .get_twin_of_edge(*edges_between_intersections.last().unwrap());

                    edges_between_intersections.insert(0, first_edge);
                    edges_between_intersections.push(last_edge);

                    edges_between_intersections
                };

                let mut rev_edges_between_intersections = edges_bet.clone();
                rev_edges_between_intersections.reverse();

                // println!("{} {}", start_edge_position, end_edge_position);
                // println!("edges between: {:?}", edges_between_intersections);
                // println!("all edges: {:?}", new_sol.paths[loop_id].edges);

                let this_connection = intersection_connections.len();
                let twin_connection = intersection_connections.len() + 1;

                intersection_connections.push(Edge {
                    root: intersection_a,
                    face: None,
                    next: None,
                    twin: Some(twin_connection),
                    label: Some(new_sol.paths[loop_id].direction as usize),
                    direction: Some(new_sol.paths[loop_id].direction),
                    part_of_path: Some(loop_id),
                    edges_between: Some(edges_bet),
                    ..default()
                });

                vertexpair_to_edge.insert(
                    (
                        intersection_a,
                        intersection_b,
                        new_sol.paths[loop_id].direction as usize,
                    ),
                    this_connection,
                );
                vertexpair_to_edge.insert(
                    (
                        intersection_b,
                        intersection_a,
                        new_sol.paths[loop_id].direction as usize,
                    ),
                    twin_connection,
                );

                intersection_connections.push(Edge {
                    root: intersection_b,
                    face: None,
                    next: None,
                    twin: Some(this_connection),
                    label: Some(new_sol.paths[loop_id].direction as usize),
                    direction: Some(new_sol.paths[loop_id].direction),
                    part_of_path: Some(loop_id),
                    edges_between: Some(rev_edges_between_intersections),
                    ..default()
                });
            }
        }

        for intersection_connection in 0..intersection_connections.len() {
            let edge = &intersection_connections[intersection_connection];

            let loop_id = edge.part_of_path.unwrap();

            let intersection_ordering_in_loop = loop_to_intersections.get(&loop_id).unwrap();

            let u = edge.root;
            let v = intersection_connections[edge.twin.unwrap()].root;

            let pair = intersection_ordering_in_loop
                .windows(2)
                .filter(|x| (x[0] == u && x[1] == v) || (x[0] == v && x[1] == u))
                .next()
                .unwrap();

            let forwards = pair.iter().position(|&x| x == u) < pair.iter().position(|&x| x == v);

            let seq = &intersections[v].ordering;

            let this_edges: Vec<usize> =
                seq.iter().filter(|x| x.0 == loop_id).map(|x| x.1).collect();

            let other_loop = seq.iter().filter(|x| x.0 != loop_id).next().unwrap().0;
            let other_edges: Vec<usize> = seq
                .iter()
                .filter(|x| x.0 == other_loop)
                .map(|x| x.1)
                .collect();

            let mut this_first = new_sol.paths[loop_id]
                .edges
                .iter()
                .find_or_first(|x| this_edges.contains(x))
                .unwrap()
                .clone();
            let other_first = new_sol.paths[other_loop]
                .edges
                .iter()
                .find_or_first(|x| other_edges.contains(x))
                .unwrap()
                .clone();

            let mut this_second = this_edges
                .iter()
                .filter(|&&x| x != this_first)
                .next()
                .unwrap()
                .clone();
            let other_second = other_edges
                .iter()
                .filter(|&&x| x != other_first)
                .next()
                .unwrap()
                .clone();

            if !forwards {
                swap(&mut this_first, &mut this_second);
            }

            let next_step = seq[(seq
                .iter()
                .position(|x| x.0 == loop_id && x.1 == this_first)
                .unwrap()
                + 1)
                % 4]
            .1;

            let other_forwards = next_step == other_second;

            let mut intersection_ordering_in_other =
                loop_to_intersections.get(&other_loop).unwrap().clone();

            intersection_ordering_in_other.remove(intersection_ordering_in_other.len() - 1);

            let cur_pos = intersection_ordering_in_other
                .iter()
                .position(|&x| x == v)
                .unwrap();

            let next_other_step = if other_forwards {
                (cur_pos + intersection_ordering_in_other.len() + 1)
                    % intersection_ordering_in_other.len()
            } else {
                (cur_pos + intersection_ordering_in_other.len() - 1)
                    % intersection_ordering_in_other.len()
            };

            let next_pos = intersection_ordering_in_other[next_other_step];

            let next_edge_id = vertexpair_to_edge
                .get(&(v, next_pos, new_sol.paths[other_loop].direction as usize))
                .copied()
                .unwrap();

            intersection_connections[intersection_connection].next = Some(next_edge_id);
        }

        new_sol.intersection_graph =
            Doconeli::from_vertices_and_edges(intersections, intersection_connections);

        for face_id in 0..new_sol.intersection_graph.faces.len() {
            let labels = new_sol
                .intersection_graph
                .get_edges_of_face(face_id)
                .into_iter()
                .map(|edge_id| new_sol.intersection_graph.edges[edge_id].direction);

            if labels.clone().any(|l| l.is_none()) {
                timer.report(&format!(
                    "[❌] Detected incorrect region (some labels undefined)"
                ));
                return None;
            }

            let boundary_size = labels.clone().count();
            if boundary_size > 6 || boundary_size < 3 {
                timer.report(&format!(
                    "❌  Detected incorrect region (boundary has size: {boundary_size})"
                ));
                return None;
            }

            let count_x = labels
                .clone()
                .filter(|&l| l == Some(PrincipalDirection::X))
                .count();
            if count_x > 2 {
                timer.report(&format!("❌  Detected incorrect region {face_id} (has too many Z in boundary: {count_x})"));
                return None;
            }

            let count_y = labels
                .clone()
                .filter(|&l| l == Some(PrincipalDirection::Y))
                .count();
            if count_y > 2 {
                timer.report(&format!("❌  Detected incorrect region {face_id} (has too many Y in boundary: {count_y})"));
                return None;
            }

            let count_z = labels
                .clone()
                .filter(|&l| l == Some(PrincipalDirection::Z))
                .count();
            if count_z > 2 {
                timer.report(&format!("❌  Detected incorrect region {face_id} (has too many Z in boundary: {count_z})"));
                return None;
            }
        }

        timer.report(&format!("🆗  Loop passed regions checks"));

        new_sol.regions =
            self.get_subsurfaces(&new_sol, &new_sol.intersection_graph, ColorType::Random);

        let prim_res = Primalization::initialize(&self.mesh.clone(), &new_sol);

        if prim_res.is_none() {
            timer.report(&format!("❌  Detected impossible primalization 1111"));
            return None;
        }

        let mut prim = prim_res.unwrap();

        let singularities =
            self.get_top_n_percent_singularities(configuration.percent_singularities);

        if let Some(new_prim) = Primalization::initialize(&self.mesh.clone(), &new_sol) {
            prim = new_prim;
        } else {
            timer.report(&format!("❌  Detected impossible primalization 1111"));
            return None;
        }

        if !prim.place_primals(singularities, configuration) {
            timer.report(&format!(
                "❌  Detected impossible primal placement {:?}",
                new_sol.paths.len()
            ));
            return None;
        }

        timer.report(&format!("🆗  Primals placed"));

        if !prim.connect_primals(configuration) {
            timer.report(&format!(
                "❌  Detected impossible primal connections {:?}",
                new_sol.paths.len()
            ));
            return None;
        }

        timer.report(&format!("🆗  Primals connected"));

        return Some((new_sol, prim));
    }
}
