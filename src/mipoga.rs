use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    f32::consts::PI,
    ops::Sub,
};

use bevy::prelude::*;

use ordered_float::OrderedFloat;

use crate::doconeli::Doconeli;

use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub position: Vec3,
    pub normal: Vec3,
}

// The midpoint edge graph (Mipoga), is a graph G' of a planar (DCEL) graph G, such that G' has a node for each half-edge in G, and nodes of G' are connected iff the half-edges share a face, or if the half-edges are twins.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Mipoga {
    pub nodes: HashMap<usize, Node>,
    pub neighbors: HashMap<usize, Vec<(usize, f32)>>,
}

impl Mipoga {
    pub fn midpoint_edge_from_mesh(mesh: &Doconeli, center_weight: f32) -> Mipoga {
        let mut nodes = HashMap::new();
        let mut neighbors = HashMap::new();

        for edge_id in 0..mesh.edges.len() {
            let node = Node {
                position: (1.0 - center_weight) * mesh.get_midpoint_of_edge(edge_id)
                    + center_weight * mesh.get_centroid_of_face(mesh.get_face_of_edge(edge_id)),
                normal: mesh.get_midpoint_normal_of_edge(edge_id),
            };

            // create edge for each edge
            let mut node_neighbors = vec![];
            for neighbor_id in mesh.get_edges_of_face(mesh.get_face_of_edge(edge_id)) {
                if neighbor_id == edge_id {
                    continue;
                }
                node_neighbors.push((neighbor_id, 1.0));
            }

            node_neighbors.push((mesh.get_twin_of_edge(edge_id), 1.0));

            nodes.insert(edge_id, node);
            neighbors.insert(edge_id, node_neighbors);
        }

        Mipoga { nodes, neighbors }
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
                    node_neighbors[j].1 =
                        self.nodes[v_i].position.distance(self.nodes[&v_j].position)
                });
            })
        });
    }

    pub fn precompute_angular_loopy_weights(&mut self, principal_direction: Vec3, gamma: f32) {
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
                    .and_modify(|node_neighbors| node_neighbors[j].1 = (angle).powf(gamma));

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
        if !self.nodes.contains_key(&v_a) || !self.nodes.contains_key(&v_b) {
            return None;
        }

        let mut distance = HashMap::new();
        distance.insert(v_a, f32::MAX - 42.);

        for &i in self.nodes.keys() {
            if i != v_a {
                distance.insert(i, f32::MAX);
            };
        }

        let mut pred = HashMap::new();

        let mut unvisited = BinaryHeap::new();
        unvisited.push((OrderedFloat(0.), v_a));

        let mut visited = HashSet::new();

        while let Some((_, v_i)) = unvisited.pop() {
            if !visited.insert(v_i) {
                continue;
            };

            let cur_distance = {
                if v_i == v_a {
                    0.
                } else {
                    *distance.get(&v_i).unwrap()
                }
            };

            for &(v_j, distance_i_j) in &self.neighbors[&v_i] {
                let new_distance = cur_distance + distance_i_j;

                if new_distance <= *distance.get(&v_j).unwrap() {
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
