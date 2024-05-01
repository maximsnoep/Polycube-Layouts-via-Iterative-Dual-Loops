use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    f32::consts::PI,
    ops::Sub,
};

use bevy::prelude::*;

use ordered_float::OrderedFloat;

use crate::{doconeli::Doconeli, Configuration};

use serde::Deserialize;
use serde::Serialize;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub enum NodeType {
    Vertex(usize),
    Face(usize),
    #[default]
    Phantom,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub position: Vec3,
    pub normal: Vec3,
    pub node_type: NodeType,
}

// The dual-primal graph (duaprima), is a graph G' of a planar (DCEL) graph G, such that G' is a copy of G, with the addition of a vertices for each face of G, and edges connecting the face-vertices to adjacent edges and vertices.
// Important to keep track of the face-vertices, as they do not exist in the original graph, and may need to be realized depending on the purpose.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Duaprima {
    pub nodes: HashMap<usize, Node>,
    pub neighbors: HashMap<usize, Vec<(usize, f32)>>,
}

impl Duaprima {
    pub fn from_mesh(mesh: &Doconeli) -> Duaprima {
        let mut nodes = HashMap::new();
        let mut neighbors = HashMap::new();

        for vertex_id in 0..mesh.vertices.len() {
            let node = Node {
                position: mesh.get_position_of_vertex(vertex_id),
                normal: mesh.get_normal_of_vertex(vertex_id),
                node_type: NodeType::Vertex(vertex_id),
            };

            let mut node_neighbors = vec![];
            for neighbor_id in mesh.get_neighbors_of_vertex(vertex_id) {
                if neighbor_id == vertex_id {
                    continue;
                }
                node_neighbors.push((neighbor_id, 1.0));
            }
            for neighbor_id in mesh.get_faces_of_vertex(vertex_id) {
                node_neighbors.push((mesh.vertices.len() + neighbor_id, 1.0));
            }

            nodes.insert(vertex_id, node);
            neighbors.insert(vertex_id, node_neighbors);
        }

        for face_id in 0..mesh.faces.len() {
            let node = Node {
                position: mesh.get_centroid_of_face(face_id),
                normal: mesh.get_normal_of_face(face_id),
                node_type: NodeType::Face(face_id),
            };

            let mut node_neighbors = vec![];
            for neighbor_id in mesh.get_vertices_of_face(face_id) {
                node_neighbors.push((neighbor_id, 1.0));
            }
            for neighbor_id in mesh.get_neighbors_of_face_edgewise(face_id) {
                if neighbor_id == face_id {
                    continue;
                }
                node_neighbors.push((mesh.vertices.len() + neighbor_id, 1.0));
            }

            nodes.insert(mesh.vertices.len() + face_id, node);
            neighbors.insert(mesh.vertices.len() + face_id, node_neighbors);
        }

        Duaprima { nodes, neighbors }
    }

    pub fn from_mesh_with_mask(
        mesh: &Doconeli,
        vertex_mask: HashSet<usize>,
        edge_mask: HashSet<(usize, usize)>,
    ) -> Duaprima {
        let mut nodes = HashMap::new();
        let mut neighbors = HashMap::new();

        for vertex_id in 0..mesh.vertices.len() {
            if vertex_mask.contains(&vertex_id) {
                continue;
            }

            let node = Node {
                position: mesh.get_position_of_vertex(vertex_id),
                normal: mesh.get_normal_of_vertex(vertex_id),
                node_type: NodeType::Vertex(vertex_id),
            };

            let mut node_neighbors = vec![];
            for neighbor_id in mesh.get_neighbors_of_vertex(vertex_id) {
                if vertex_mask.contains(&neighbor_id) {
                    continue;
                }
                if edge_mask.contains(&(vertex_id, neighbor_id))
                    || edge_mask.contains(&(neighbor_id, vertex_id))
                {
                    continue;
                }

                if neighbor_id == vertex_id {
                    continue;
                }

                node_neighbors.push((neighbor_id, 1.0));
            }
            for neighbor_id in mesh.get_faces_of_vertex(vertex_id) {
                if vertex_mask.contains(&(mesh.vertices.len() + neighbor_id)) {
                    continue;
                }
                if edge_mask.contains(&(vertex_id, mesh.vertices.len() + neighbor_id))
                    || edge_mask.contains(&(mesh.vertices.len() + neighbor_id, vertex_id))
                {
                    continue;
                }

                node_neighbors.push((mesh.vertices.len() + neighbor_id, 1.0));
            }

            nodes.insert(vertex_id, node);
            neighbors.insert(vertex_id, node_neighbors);
        }

        for face_id in 0..mesh.faces.len() {
            if vertex_mask.contains(&(mesh.vertices.len() + face_id)) {
                continue;
            }

            let node = Node {
                position: mesh.get_centroid_of_face(face_id),
                normal: mesh.get_normal_of_face(face_id),
                node_type: NodeType::Face(face_id),
            };

            let mut node_neighbors = vec![];

            for neighbor_id in mesh.get_vertices_of_face(face_id) {
                if vertex_mask.contains(&neighbor_id) {
                    continue;
                }
                if edge_mask.contains(&(mesh.vertices.len() + face_id, neighbor_id))
                    || edge_mask.contains(&(neighbor_id, mesh.vertices.len() + face_id))
                {
                    continue;
                }

                node_neighbors.push((neighbor_id, 1.0));
            }

            for neighbor_id in mesh.get_neighbors_of_face_edgewise(face_id) {
                if vertex_mask.contains(&(mesh.vertices.len() + neighbor_id)) {
                    continue;
                }
                if edge_mask.contains(&(
                    mesh.vertices.len() + face_id,
                    mesh.vertices.len() + neighbor_id,
                )) || edge_mask.contains(&(
                    mesh.vertices.len() + neighbor_id,
                    mesh.vertices.len() + face_id,
                )) {
                    continue;
                }

                if neighbor_id == face_id {
                    continue;
                }

                node_neighbors.push((mesh.vertices.len() + neighbor_id, 1.0));
            }

            nodes.insert(mesh.vertices.len() + face_id, node);
            neighbors.insert(mesh.vertices.len() + face_id, node_neighbors);
        }

        Duaprima { nodes, neighbors }
    }

    pub fn set_weights_based_on_distance(&mut self) {
        for node in self.nodes.keys() {
            for &(neighbor, mut weight) in &self.neighbors[node] {
                let distance = self.nodes[node]
                    .position
                    .distance(self.nodes[&neighbor].position);
                weight = distance;
            }
        }
    }

    pub fn set_weight(&mut self, node_a: usize, node_b: usize, weight: f32) {
        self.neighbors.entry(node_a).and_modify(|node_neighbors| {
            let pos = node_neighbors
                .iter()
                .position(|&(neighbor, _)| neighbor == node_b)
                .unwrap();
            node_neighbors[pos].1 = weight
        });
        self.neighbors.entry(node_b).and_modify(|node_neighbors| {
            let pos = node_neighbors
                .iter()
                .position(|&(neighbor, _)| neighbor == node_a)
                .unwrap();
            node_neighbors[pos].1 = weight
        });
    }

    pub fn add_vertex_with_neighbors(
        &mut self,
        node_id: usize,
        node: Node,
        neighbors: Vec<(usize, f32)>,
    ) {
        let this_node = self.nodes.len();
        self.nodes.insert(node_id, node);

        for &(neighbor_id, weight) in &neighbors {
            self.neighbors
                .entry(neighbor_id)
                .and_modify(|node_neighbors| node_neighbors.push((this_node, weight)));
        }
        self.neighbors.insert(node_id, neighbors);
    }

    pub fn remove_nodes(&mut self, the_removed: &HashSet<usize>) {
        for node in the_removed {
            self.nodes.remove(&node);
            self.neighbors.remove(&node);
        }
        for cur_node in &self.nodes {
            self.neighbors
                .entry(*cur_node.0)
                .and_modify(|node_neighbors| {
                    node_neighbors.retain(|(x, _)| !the_removed.contains(x));
                });
        }
    }

    pub fn remove_edges(&mut self, the_removed: &HashSet<(usize, usize)>) {
        for edge in the_removed {
            self.remove_edge(edge.0, edge.1);
        }
    }

    pub fn remove_edge(&mut self, node_a: usize, node_b: usize) {
        self.neighbors.entry(node_a).and_modify(|node_neighbors| {
            node_neighbors.retain(|(x, _)| *x != node_b);
        });
        self.neighbors.entry(node_b).and_modify(|node_neighbors| {
            node_neighbors.retain(|(x, _)| *x != node_a);
        });
    }

    pub fn get_position_of_vertex(&self, vertex_id: usize) -> Vec3 {
        self.nodes.get(&vertex_id).unwrap().position
    }

    pub fn euclidean_distance_between_vertices(&self, vertex1_id: usize, vertex2_id: usize) -> f32 {
        self.get_position_of_vertex(vertex1_id)
            .distance(self.get_position_of_vertex(vertex2_id))
    }

    pub fn precompute_euclidean_weights(&mut self) {
        self.nodes.keys().for_each(|v_i| {
            (0..self.neighbors[v_i].len()).for_each(|j| {
                let v_j = self.neighbors[v_i][j].0;

                self.neighbors.entry(*v_i).and_modify(|node_neighbors| {
                    let mut mult = 1.;
                    if let NodeType::Face(_) = self.nodes[v_i].node_type {
                        mult *= 2.;
                    }
                    if let NodeType::Face(_) = self.nodes[&v_j].node_type {
                        mult *= 2.;
                    }

                    node_neighbors[j].1 =
                        self.nodes[v_i].position.distance(self.nodes[&v_j].position) * mult
                });
            })
        });
    }

    pub fn precompute_label_weights(
        &mut self,
        configuration: &Configuration,
        target_labels: (usize, usize),
        face_labels: HashMap<usize, usize>,
        edge_labels: HashMap<(usize, usize), (usize, usize)>,
    ) {
        let cutting_path = target_labels.0 != target_labels.1;

        self.nodes.keys().for_each(|v_i| {
            (0..self.neighbors[v_i].len()).for_each(|j| {
                let v_j = self.neighbors[v_i][j].0;

                self.neighbors.entry(*v_i).and_modify(|node_neighbors| {
                    let mut mult = 3.;

                    // if not a cutting_path, then we want any edges that have target_labels.0 in the labels
                    if !cutting_path {
                        match (
                            self.nodes[v_i].node_type.clone(),
                            self.nodes[&v_j].node_type.clone(),
                        ) {
                            (NodeType::Vertex(i), NodeType::Vertex(j)) => {
                                let edge_label = edge_labels.get(&(i, j));
                                if edge_label.is_some() {
                                    if edge_label == Some(&(target_labels.0, target_labels.1))
                                        || edge_label == Some(&(target_labels.1, target_labels.0))
                                    {
                                        mult = 0.25;
                                    }
                                }
                                let edge_label = edge_labels.get(&(j, i));
                                if edge_label.is_some() {
                                    if edge_label == Some(&(target_labels.0, target_labels.1))
                                        || edge_label == Some(&(target_labels.1, target_labels.0))
                                    {
                                        mult = 0.25;
                                    }
                                }
                            }
                            (NodeType::Vertex(i), NodeType::Face(j)) => {
                                if face_labels.get(&j) == Some(&target_labels.0) {
                                    mult = 0.5;
                                }
                            }
                            (NodeType::Face(i), NodeType::Vertex(j)) => {
                                if face_labels.get(&i) == Some(&target_labels.0) {
                                    mult = 0.5;
                                }
                            }
                            (NodeType::Face(i), NodeType::Face(j)) => {
                                if face_labels.get(&i) == Some(&target_labels.0)
                                    && face_labels.get(&j) == Some(&target_labels.0)
                                {
                                    mult = 0.5;
                                }
                            }
                            _ => {}
                        }
                    } else {
                        // this is a cutting path, so we want the edge to have target_labels.0 and target_labels.1, preferably we do not want to cross faces!
                        match (
                            self.nodes[v_i].node_type.clone(),
                            self.nodes[&v_j].node_type.clone(),
                        ) {
                            (NodeType::Vertex(i), NodeType::Vertex(j)) => {
                                mult = 2.0;
                                let edge_label = edge_labels.get(&(i, j));
                                if edge_label.is_some() {
                                    if edge_label == Some(&(target_labels.0, target_labels.1))
                                        || edge_label == Some(&(target_labels.1, target_labels.0))
                                    {
                                        mult = 2.0_f32.powf(configuration.path_weight);
                                    }
                                }
                                let edge_label = edge_labels.get(&(j, i));
                                if edge_label.is_some() {
                                    if edge_label == Some(&(target_labels.0, target_labels.1))
                                        || edge_label == Some(&(target_labels.1, target_labels.0))
                                    {
                                        mult = 2.0_f32.powf(configuration.path_weight);
                                    }
                                }
                            }

                            (NodeType::Vertex(i), NodeType::Face(j)) => {
                                mult = 4.;
                            }
                            (NodeType::Face(i), NodeType::Vertex(j)) => {
                                mult = 4.;
                            }
                            (NodeType::Face(i), NodeType::Face(j)) => {
                                mult = 3.;
                            }
                            _ => {}
                        }
                    }

                    node_neighbors[j].1 =
                        self.nodes[v_i].position.distance(self.nodes[&v_j].position) * mult
                });
            })
        });
    }

    pub fn precompute_angular_loopy_weights(&mut self, principal_direction: Vec3) {
        self.nodes.keys().for_each(|v_i| {
            let mut filter_neighbors = vec![];
            (0..self.neighbors[v_i].len()).for_each(|j| {
                let v_j = self.neighbors[v_i][j].0;
                let v_i_normal = self.nodes[v_i].normal;
                let direction_i_j = self.nodes[&v_j].position.sub(self.nodes[v_i].position);
                let cross = direction_i_j.cross(v_i_normal);
                let angle = (principal_direction.angle_between(cross) / PI) * 180.;

                self.neighbors
                    .entry(*v_i)
                    .and_modify(|node_neighbors| node_neighbors[j].1 = (angle).powi(5));

                if angle > 90. {
                    filter_neighbors.push(v_j);
                }
            });

            self.neighbors.entry(*v_i).and_modify(|node_neighbors| {
                node_neighbors.retain(|(i, _)| !filter_neighbors.contains(i))
            });
        });
    }

    pub fn shortest_path(&self, v_a: usize, v_b: usize) -> Option<Vec<usize>> {
        let mut distance = HashMap::with_capacity(self.nodes.len());
        let mut pred = HashMap::new();

        for &i in self.nodes.keys() {
            if i != v_a {
                distance.insert(i, f32::MAX);
            } else {
                distance.insert(v_a, f32::MAX - 42.);
            }
        }

        let mut unvisited = BinaryHeap::new();
        unvisited.push((OrderedFloat(0.), v_a));

        let mut visited = HashSet::new();

        while let Some((cost, v_i)) = unvisited.pop() {
            let cur_distance = {
                if v_i == v_a {
                    0.
                } else {
                    distance[&v_i]
                }
            };

            if !visited.insert(v_i) || cost > OrderedFloat(cur_distance) {
                continue;
            };

            for &(v_j, distance_i_j) in &self.neighbors[&v_i] {
                let new_distance = cur_distance + distance_i_j;

                if !distance.contains_key(&v_j) {
                    println!(
                        "v_j {} not found for v_i: {} and neigh: {:?}",
                        v_j, v_i, self.neighbors[&v_i]
                    );
                }

                if new_distance <= distance[&v_j] {
                    distance.insert(v_j, new_distance);
                    pred.insert(v_j, v_i);
                    unvisited.push((OrderedFloat(-new_distance), v_j));
                }

                if v_j == v_b {
                    let mut path = vec![v_b];
                    while *path.last().unwrap() != v_a || path.len() == 1 {
                        let maybe_previous = pred.get(path.last().unwrap());

                        if maybe_previous.is_some() {
                            path.push(*maybe_previous.unwrap())
                        } else {
                            break;
                        }
                    }
                    path.reverse();

                    return Some(path);
                }
            }
        }

        None
    }
}
