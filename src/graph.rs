use crate::doconeli::Doconeli;
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
pub struct Node {}

// The midpoint edge graph (Mipoga), is a graph G' of a planar (DCEL) graph G, such that G' has a node for each half-edge in G, and nodes of G' are connected iff the half-edges share a face, or if the half-edges are twins.
#[derive(Default, Clone)]
pub struct Graph {
    pub nodes: HashMap<usize, Node>,
    pub neighbors: HashMap<usize, Vec<usize>>,
}

impl Graph {
    pub fn empty() -> Graph {
        let nodes = HashMap::new();
        let neighbors = HashMap::new();

        Graph { nodes, neighbors }
    }

    pub fn add_node(&mut self, node_id: usize, node: Node) {
        self.nodes.insert(node_id, node);
        self.neighbors.insert(node_id, vec![]);
    }

    pub fn add_edge(&mut self, node_i: usize, node_j: usize) {
        self.neighbors.entry(node_i).and_modify(|node_neighbors| {
            node_neighbors.push(node_j);
        });
        self.neighbors.entry(node_j).and_modify(|node_neighbors| {
            node_neighbors.push(node_i);
        });
    }

    pub fn new(mesh: &Doconeli) -> Graph {
        let mut nodes = HashMap::new();
        let mut neighbors = HashMap::new();

        for vertex_id in 0..mesh.vertices.len() {
            let node = Node {};

            // create edge to all neighbors
            let mut node_neighbors = vec![];
            for neighbor_id in mesh.get_neighbors_of_vertex(vertex_id) {
                node_neighbors.push(neighbor_id);
            }

            nodes.insert(vertex_id, node);
            neighbors.insert(vertex_id, node_neighbors);
        }

        Graph { nodes, neighbors }
    }

    pub fn dual(mesh: &Doconeli) -> Graph {
        let mut nodes = HashMap::new();
        let mut neighbors = HashMap::new();

        for face_id in 0..mesh.faces.len() {
            let node = Node {};

            // create edge to all neighbors
            let mut node_neighbors = vec![];
            for neighbor_id in mesh.get_neighbors_of_face_vertexwise(face_id) {
                node_neighbors.push(neighbor_id);
            }

            nodes.insert(face_id, node);
            neighbors.insert(face_id, node_neighbors);
        }

        Graph { nodes, neighbors }
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
                    node_neighbors.retain(|x| !the_removed.contains(x));
                });
        }
    }

    pub fn remove_edge(&mut self, node_i: usize, node_j: usize) {
        self.neighbors.entry(node_i).and_modify(|node_neighbors| {
            node_neighbors.retain(|&x| x != node_j);
        });
        self.neighbors.entry(node_j).and_modify(|node_neighbors| {
            node_neighbors.retain(|&x| x != node_i);
        });
    }

    pub fn connected_components(&self) -> HashSet<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = HashSet::new();

        for (node_id, _) in &self.nodes {
            if visited.contains(node_id) {
                continue;
            }
            visited.insert(*node_id);
            let mut component = vec![*node_id];
            let mut neighbors_to_visit = self.neighbors.get(node_id).unwrap().clone();

            while !neighbors_to_visit.is_empty() {
                let neighbor_id = neighbors_to_visit.pop().unwrap();
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);
                component.push(neighbor_id);
                neighbors_to_visit.extend(self.neighbors.get(&neighbor_id).clone().unwrap());
            }

            components.insert(component);
        }

        return components;
    }

    pub fn two_coloring(&self) -> Option<(HashSet<usize>, HashSet<usize>)> {
        let mut odd = HashSet::new();
        let mut even = HashSet::new();

        for component in self.connected_components() {
            odd.insert(component[0]);
        }

        while odd.iter().chain(even.iter()).count() < self.nodes.len() {
            for odd_id in &odd {
                for &neighbor_id in self.neighbors.get(odd_id).unwrap() {
                    even.insert(neighbor_id);
                }
            }
            for even_id in &even {
                for &neighbor_id in self.neighbors.get(even_id).unwrap() {
                    odd.insert(neighbor_id);
                }
            }
        }

        if odd.is_disjoint(&even) {
            Some((odd, even))
        } else {
            None
        }
    }
}
