use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::OpenOptions,
    ops::Div,
    path::PathBuf,
};

use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use std::io::Write;

use simple_error::bail;

use bevy::prelude::*;

use rayon::prelude::*;
use stl_io::{IndexedMesh, IndexedTriangle, Vector};

use crate::{
    solution::{PrincipalDirection, Surface},
    utils::average,
};

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Vertex {
    pub some_edge: Option<usize>,

    // Auxiliary data
    pub position: Vec3,
    pub normal: Vec3,
    pub original_face_id: usize,
    pub ordering: Vec<(usize, usize)>,
    pub respective_surface: Option<Surface>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Face {
    pub some_edge: Option<usize>,

    // Auxiliary data
    pub color: Color,
    pub normal: Vec3,
    pub dual_position: Option<Vec3>,
    pub dual_normal: Option<Vec3>,
    pub original_face: Option<usize>,
    pub respective_surface: Option<Surface>,
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub root: usize,
    pub face: Option<usize>,
    pub next: Option<usize>,
    pub twin: Option<usize>,

    // Auxiliary data
    pub label: Option<usize>,
    pub part_of_path: Option<usize>,
    pub edges_between: Option<Vec<usize>>,
    pub edges_between_endpoints: Option<(usize, usize)>,
    pub direction: Option<PrincipalDirection>,
}

// The doubly connected edge list (DCEL or Doconeli), also known as half-edge data structure,
// is a data structure to represent an embedding of a planar graph in the plane, and polytopes in 3D.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Doconeli {
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,
    pub vertices: Vec<Vertex>,
}

impl Doconeli {
    // Initialize an empty DCEL with `n` vertices and `m` faces.
    pub fn new(n: usize, m: usize) -> Doconeli {
        Doconeli {
            edges: vec![],
            faces: vec![Face::default(); m],
            vertices: vec![Vertex::default(); n],
        }
    }

    // Initialize an empty DCEL.
    pub fn empty() -> Doconeli {
        Doconeli {
            edges: vec![],
            faces: vec![],
            vertices: vec![],
        }
    }

    // Read an STL file at `path`, and construct a DCEL.
    // Fails if the input mesh is NOT manifold or NOT watertight.
    pub fn from_stl(path: &PathBuf) -> Result<Doconeli, Box<dyn Error>> {
        info!("Creating a Doconeli (or DCEL: doubly connected edge list) by parsing STL file {path:?}.");
        let stl = stl_io::read_stl(&mut OpenOptions::new().read(true).open(path)?)?;

        Self::from_indexedmesh(&stl)
    }

    // Read an STL file at `path`, and construct a DCEL.
    // Fails if the input mesh is NOT manifold or NOT watertight.
    pub fn from_obj(path: &PathBuf) -> Result<Doconeli, Box<dyn Error>> {
        info!("Creating a Doconeli (or DCEL: doubly connected edge list) by parsing STL file {path:?}.");

        let lines = std::io::BufRead::lines(std::io::BufReader::new(
            std::fs::File::open(&path).expect("Error opening file (!)"),
        ));

        let mut vertices = vec![];
        let mut faces = vec![];

        for line in lines.into_iter().map(|x| x.unwrap()) {
            if let Some(rest) = line.strip_prefix("v ") {
                let r: Option<(f32, f32, f32)> = rest
                    .split(" ")
                    .map(|x| x.parse::<f32>().unwrap())
                    .collect_tuple();

                vertices.push(Vector::new(r.unwrap().into()));
            } else if let Some(rest) = line.strip_prefix("f ") {
                let r: Option<(usize, usize, usize)> = rest
                    .split(" ")
                    .map(|x| x.parse::<usize>().unwrap() - 1)
                    .collect_tuple();

                faces.push(IndexedTriangle {
                    normal: Vector::new([0., 0., 0.]),
                    vertices: r.unwrap().into(),
                });
            }
        }

        let obj = IndexedMesh { vertices, faces };

        Self::from_indexedmesh(&obj)
    }

    pub fn from_indexedmesh(indexedmesh: &IndexedMesh) -> Result<Doconeli, Box<dyn Error>> {
        let mut mesh = Doconeli::new(indexedmesh.vertices.len(), indexedmesh.faces.len());

        info!("Gather information of {} faces.", indexedmesh.faces.len());
        for (face_id, face) in indexedmesh.faces.iter().enumerate() {
            let e0_id = mesh.edges.len();
            mesh.faces[face_id] = Face {
                some_edge: Some(e0_id),
                color: Color::BLACK,
                normal: Vec3::new(face.normal[0], face.normal[1], face.normal[2]),
                ..default()
            };

            // Add three half-edges (e0, e1, e2) for the triangle face.
            // Add the root vertices (v0, v1, v2) if they do not exist yet.
            //
            //            v0
            //             *
            //            ^ \
            //           /   \ e0
            //       e2 /     \
            //         /       v
            //     v2 * < - - - * v1
            //             e1
            for (i, &v_id) in face.vertices.iter().enumerate() {
                mesh.edges.push(Edge {
                    root: v_id,
                    face: Some(face_id),
                    next: Some(e0_id + ((i + 1) % face.vertices.len())),
                    twin: None,
                    ..default()
                });

                if mesh.vertices[v_id].some_edge.is_none() {
                    mesh.vertices[v_id] = Vertex {
                        some_edge: Some(e0_id + i),
                        position: Vec3::new(
                            indexedmesh.vertices[v_id][0],
                            indexedmesh.vertices[v_id][1],
                            indexedmesh.vertices[v_id][2],
                        ),
                        ..default()
                    };
                }
            }
        }

        info!(
            "Found {} faces, {} edges, {} vertices.",
            mesh.faces.len(),
            mesh.edges.len(),
            mesh.vertices.len()
        );

        info!("For each edge (a,b) find its twin (b,a).");
        let mut vertex_pair_to_edge_map = HashMap::new();
        for edge_id in 0..mesh.edges.len() {
            let v_a = mesh.get_root_of_edge(edge_id);
            let v_b = mesh.get_root_of_edge(mesh.get_next_of_edge(edge_id));
            if let Some(&dupl_id) = vertex_pair_to_edge_map.get(&(v_a, v_b)) {
                bail!("NOT MANIFOLD: Duplicate vertex pair ({v_a}, {v_b}) for both edge {dupl_id} and edge {edge_id}.")
            }
            vertex_pair_to_edge_map.insert((v_a, v_b), edge_id);
        }

        for (&(v_a, v_b), &e_ab) in &vertex_pair_to_edge_map {
            if let Some(&e_ba) = vertex_pair_to_edge_map.get(&(v_b, v_a)) {
                match mesh.edges[e_ab].twin {
                    Some(cur) => bail!("NOT MANIFOLD: Edge {e_ab} has two twins ({cur}, {e_ba})."),
                    None => mesh.edges[e_ab].twin = Some(e_ba),
                };
            } else {
                bail!("NOT WATERTIGHT: Edge {e_ab} has no twin.");
            }
        }

        info!("Setting normals of vertices...");
        for vertex_id in 0..mesh.vertices.len() {
            let normals = mesh
                .get_faces_of_vertex(vertex_id)
                .into_iter()
                .map(|face_id| mesh.faces[face_id].normal);
            mesh.vertices[vertex_id].normal = average(normals.clone());
        }

        Ok(mesh)
    }

    pub fn from_vertices_and_edges(vs: Vec<Vertex>, es: Vec<Edge>) -> Doconeli {
        let mut graph = Doconeli {
            edges: es,
            faces: vec![],
            vertices: vs,
        };

        for edge_id in 0..graph.edges.len() {
            if graph.edges[edge_id].face.is_none() {
                let this_face = graph.faces.len();
                graph.faces.push(Face {
                    some_edge: Some(edge_id),
                    ..default()
                });
                for edge_in_face in graph.get_edges_of_edge(edge_id) {
                    graph.edges[edge_in_face].face = Some(this_face);
                }
            }
        }

        graph
    }

    // TODO: Should be Doconeli that has nodes defined for each face ;)
    // can be implented when we have doconeli with dynamic element types
    pub fn from_dual(dual: &Doconeli) -> Doconeli {
        let mut vertices: Vec<Vertex> = dual
            .faces
            .iter()
            .map(|face| Vertex {
                some_edge: None,
                ..default()
            })
            .collect();

        let mut edges: Vec<Edge> = vec![];

        let mut pair_to_edge: HashMap<(usize, usize), usize> = HashMap::new();

        for face_id in 0..dual.faces.len() {
            let neighbors_of_face_id = dual.get_neighbors_of_face_edgewise(face_id);

            for &neighbor_id in &neighbors_of_face_id {
                if pair_to_edge.contains_key(&(face_id, neighbor_id)) {
                    continue;
                }

                let direction = dual.edges[dual
                    .get_edge_of_face_and_face(face_id, neighbor_id)
                    .unwrap()]
                .direction;

                // Face will be filled by `from_vertices_and_edges`
                // Next will be defined in the next part
                let this_edge = edges.len();
                let edge = Edge {
                    root: face_id,
                    face: None,
                    next: None,
                    twin: Some(this_edge + 1),
                    direction,
                    ..default()
                };
                let twin = Edge {
                    root: neighbor_id,
                    face: None,
                    next: None,
                    twin: Some(this_edge),
                    direction,
                    ..default()
                };

                edges.push(edge);
                edges.push(twin);

                pair_to_edge.insert((face_id, neighbor_id), this_edge);
                pair_to_edge.insert((neighbor_id, face_id), this_edge + 1);
            }
        }

        for edge_id in 0..edges.len() {
            let root = edges[edge_id].root;
            let dest = edges[edges[edge_id].twin.unwrap()].root;

            let neighbors_of_dest = dual.get_neighbors_of_face_edgewise(dest);

            let next_dest_pos = (neighbors_of_dest.iter().position(|&id| id == root).unwrap()
                + neighbors_of_dest.len()
                - 1)
                % neighbors_of_dest.len();

            let next_dest = neighbors_of_dest[next_dest_pos];

            let next_edge = pair_to_edge.get(&(dest, next_dest)).unwrap().clone();

            edges[edge_id].next = Some(next_edge);
            vertices[root].some_edge = Some(edge_id);
        }

        println!("oki dokii");

        Self::from_vertices_and_edges(vertices, edges)
    }

    pub fn get_position_of_vertex(&self, vertex_id: usize) -> Vec3 {
        self.vertices[vertex_id].position
    }

    pub fn get_normal_of_vertex(&self, vertex_id: usize) -> Vec3 {
        self.vertices[vertex_id].normal
    }

    pub fn get_normal_of_face(&self, face_id: usize) -> Vec3 {
        self.faces[face_id].normal
    }

    pub fn get_next_of_edge(&self, edge_id: usize) -> usize {
        self.edges[edge_id]
            .next
            .expect("Edge is not properly initialized")
    }

    pub fn get_twin_of_edge(&self, edge_id: usize) -> usize {
        self.edges[edge_id]
            .twin
            .expect("Edge is not properly initialized")
    }

    pub fn get_face_of_edge(&self, edge_id: usize) -> usize {
        self.edges[edge_id]
            .face
            .expect("Edge is not properly initialized")
    }

    pub fn get_faces_of_edge(&self, edge_id: usize) -> (usize, usize) {
        (
            self.get_face_of_edge(edge_id),
            self.get_face_of_edge(self.get_twin_of_edge(edge_id)),
        )
    }

    pub fn get_normal_of_edge(&self, edge_id: usize) -> Vec3 {
        let (face1, face2) = self.get_faces_of_edge(edge_id);
        (self.get_normal_of_face(face1) + self.get_normal_of_face(face2)) / 2.
    }

    pub fn get_root_of_edge(&self, edge_id: usize) -> usize {
        self.edges[edge_id].root
    }

    pub fn get_endpoints_of_edge(&self, edge_id: usize) -> (usize, usize) {
        (
            self.get_root_of_edge(edge_id),
            self.get_root_of_edge(self.get_twin_of_edge(edge_id)),
        )
    }

    pub fn get_distance_between_vertices(&self, v_a: usize, v_b: usize) -> f32 {
        self.get_position_of_vertex(v_a)
            .distance(self.get_position_of_vertex(v_b))
    }

    pub fn get_length_of_edge(&self, edge_id: usize) -> f32 {
        let (v_a, v_b) = self.get_endpoints_of_edge(edge_id);
        self.get_distance_between_vertices(v_a, v_b)
    }

    pub fn get_vector_of_edge(&self, edge_id: usize) -> Vec3 {
        self.get_position_of_vertex(self.get_root_of_edge(self.get_next_of_edge(edge_id)))
            - self.get_position_of_vertex(self.get_root_of_edge(edge_id))
    }

    pub fn get_edge_of_face(&self, face_id: usize) -> usize {
        self.faces[face_id]
            .some_edge
            .expect("Face is not properly initialized")
    }

    pub fn get_edge_of_vertex(&self, vertex_id: usize) -> usize {
        self.vertices[vertex_id]
            .some_edge
            .expect("Vertex is not properly initialized")
    }

    // To get all edges following an edge, traverse the "next" of an edge, until you get back to the starting edge.
    pub fn get_edges_of_edge(&self, edge_id: usize) -> Vec<usize> {
        let mut edges = vec![edge_id];
        loop {
            let next = self.get_next_of_edge(*edges.last().unwrap());
            if edges.contains(&next) {
                return edges;
            }
            edges.push(next);
        }
    }

    // To get all edges of a face, traverse the first edge of the face, until you get back to the starting edge.
    pub fn get_edges_of_face(&self, face_id: usize) -> Vec<usize> {
        self.get_edges_of_edge(self.get_edge_of_face(face_id))
    }

    // To get all vertices of a face, get the roots of all edges of the face.
    pub fn get_vertices_of_face(&self, face_id: usize) -> Vec<usize> {
        self.get_edges_of_face(face_id)
            .par_iter()
            .map(|&edge_id| self.get_root_of_edge(edge_id))
            .collect()
    }

    // To get all neighbors of a face, get the faces of the twins of its edges.
    pub fn get_neighbors_of_face_edgewise(&self, face_id: usize) -> Vec<usize> {
        self.get_edges_of_face(face_id)
            .par_iter()
            .map(|&edge_id| self.get_face_of_edge(self.get_twin_of_edge(edge_id)))
            .filter(|&neighbor_id| neighbor_id != face_id)
            .collect()
    }

    // To get all neighbors of a face, get the faces of its vertices.
    pub fn get_neighbors_of_face_vertexwise(&self, face_id: usize) -> HashSet<usize> {
        self.get_vertices_of_face(face_id)
            .par_iter()
            .map(|&vertex_id| self.get_faces_of_vertex(vertex_id))
            .flatten()
            .filter(|&neighbor_id| neighbor_id != face_id)
            .collect()
    }

    // To get the edge between two faces, compare the edges of one face, with the twins of the edges of the other face.
    pub fn get_edge_of_face_and_face(&self, face_a: usize, face_b: usize) -> Option<usize> {
        let edges_a = self.get_edges_of_face(face_a);
        self.get_edges_of_face(face_b)
            .par_iter()
            .map(|&edge_id| self.get_twin_of_edge(edge_id))
            .find_any(|edge_id| edges_a.contains(edge_id))
    }

    // Get curvature of a vertex (summed interior angles of its incident edges)
    pub fn get_curvature_of_vertex(&self, vertex_id: usize) -> f32 {
        let mut curvature = 0.;
        for outgoing_edge_id in self.get_edges_of_vertex(vertex_id) {
            let incoming_edge_id = self.get_twin_of_edge(outgoing_edge_id);
            let next_edge_id = self.get_next_of_edge(incoming_edge_id);
            let v1 = self.get_vector_of_edge(outgoing_edge_id);
            let v2 = self.get_vector_of_edge(next_edge_id);
            let angle = v1.angle_between(v2);
            curvature += angle;
        }
        curvature
    }

    // To get all edges of a vertex, traverse the first edge of the vertex (twin, next, twin, next, etc.) until you get back to the starting edge.
    pub fn get_edges_of_vertex(&self, vertex_id: usize) -> Vec<usize> {
        let mut edges = vec![self.get_edge_of_vertex(vertex_id)];
        loop {
            let twin = self.get_twin_of_edge(edges[edges.len() - 1]);
            let next = self.get_next_of_edge(twin);
            if edges.contains(&next) {
                return edges;
            }
            edges.push(next);
        }
    }

    // Get edge between two vertices
    pub fn get_edge_between_vertex_and_vertex(&self, v_a: usize, v_b: usize) -> Option<usize> {
        let edges_a = self.get_edges_of_vertex(v_a);
        let edges_b = self.get_edges_of_vertex(v_b);
        let edge_between = edges_b
            .into_iter()
            .find(|edge_id| edges_a.contains(&self.get_twin_of_edge(*edge_id)));

        edge_between
    }

    // To get all neighbors of a vertex, get the roots of the twins of its edges.
    pub fn get_neighbors_of_vertex(&self, vertex_id: usize) -> Vec<usize> {
        self.get_edges_of_vertex(vertex_id)
            .par_iter()
            .map(|&edge_id| self.get_root_of_edge(self.get_twin_of_edge(edge_id)))
            .collect()
    }

    // To get all faces of a vertex, get the faces of its edges.
    pub fn get_faces_of_vertex(&self, vertex_id: usize) -> Vec<usize> {
        self.get_edges_of_vertex(vertex_id)
            .par_iter()
            .map(|&edge_id| self.get_face_of_edge(edge_id))
            .collect()
    }

    // Be careful with concave faces, the centroid might lay outside the face.
    pub fn get_centroid_of_face(&self, face_id: usize) -> Vec3 {
        self.get_edges_of_face(face_id)
            .iter()
            .map(|&edge_id| self.vertices[self.get_root_of_edge(edge_id)].position)
            .sum::<Vec3>()
            .div(self.get_edges_of_face(face_id).len() as f32)
    }

    pub fn get_midpoint_of_edge(&self, edge_id: usize) -> Vec3 {
        self.get_midpoint_of_edge_with_offset(edge_id, 0.5)
    }

    pub fn get_midpoint_of_edge_with_offset(&self, edge_id: usize, offset: f32) -> Vec3 {
        self.get_position_of_vertex(self.get_root_of_edge(edge_id))
            + self.get_vector_of_edge(edge_id) * offset
    }

    pub fn get_normal_of_edge_with_offset(&self, edge_id: usize, offset: f32) -> Vec3 {
        let (start, end) = self.get_endpoints_of_edge(edge_id);

        self.get_normal_of_vertex(start) * (1. - offset) + self.get_normal_of_vertex(end) * offset
    }

    pub fn get_midpoint_normal_of_edge(&self, edge_id: usize) -> Vec3 {
        let u = self.get_root_of_edge(edge_id);
        let v = self.get_root_of_edge(self.get_twin_of_edge(edge_id));

        (self.get_normal_of_vertex(u) + self.get_normal_of_vertex(v)) / 2.
    }

    pub fn edges_share_face(&self, edge_a: usize, edge_b: usize) -> bool {
        self.get_face_of_edge(edge_a) == self.get_face_of_edge(edge_b)
    }

    pub fn split_face(
        &mut self,
        face_id: usize,
        split_point_maybe: Option<Vec3>,
    ) -> (usize, (usize, usize, usize)) {
        let edges = self.get_edges_of_face(face_id);

        let split_point = match split_point_maybe {
            Some(point) => point,
            None => self.get_centroid_of_face(face_id),
        };

        // Original face
        let e_01 = edges[0];
        let v_0 = self.edges[e_01].root;

        let e_12 = edges[1];
        let v_1 = self.edges[e_12].root;

        let e_20 = edges[2];
        let v_2 = self.edges[e_20].root;

        // Two new faces (original face stays the same)
        let f_0 = face_id;

        let f_1 = self.faces.len();
        self.faces.push(Face {
            some_edge: Some(e_12),
            color: self.faces[face_id].color,
            normal: self.faces[face_id].normal,
            dual_position: self.faces[face_id].dual_position,
            dual_normal: self.faces[face_id].dual_normal,
            original_face: self.faces[face_id].original_face,
            respective_surface: None,
        });

        let f_2 = self.faces.len();
        self.faces.push(Face {
            some_edge: Some(e_20),
            color: self.faces[face_id].color,
            normal: self.faces[face_id].normal,
            dual_position: self.faces[face_id].dual_position,
            dual_normal: self.faces[face_id].dual_normal,
            original_face: self.faces[face_id].original_face,
            respective_surface: None,
        });

        // Six new edges (with next six available ids)
        let e_x0 = self.edges.len();
        self.edges.push(Edge::default());
        let e_x1 = self.edges.len();
        self.edges.push(Edge::default());
        let e_x2 = self.edges.len();
        self.edges.push(Edge::default());

        let e_0x = self.edges.len();
        self.edges.push(Edge::default());
        let e_1x = self.edges.len();
        self.edges.push(Edge::default());
        let e_2x = self.edges.len();
        self.edges.push(Edge::default());

        // One new vertex (with next available id)
        let v_x = self.vertices.len();
        self.vertices.push(Vertex {
            some_edge: Some(e_x0),
            position: split_point,
            normal: self.faces[face_id].normal,
            ..default()
        });

        // Set the edges correctly
        self.edges[e_x0] = Edge {
            root: v_x,
            face: Some(f_0),
            next: Some(e_01),
            twin: Some(e_0x),
            ..default()
        };
        self.edges[e_x1] = Edge {
            root: v_x,
            face: Some(f_1),
            next: Some(e_12),
            twin: Some(e_1x),
            ..default()
        };
        self.edges[e_x2] = Edge {
            root: v_x,
            face: Some(f_2),
            next: Some(e_20),
            twin: Some(e_2x),
            ..default()
        };

        self.edges[e_0x] = Edge {
            root: v_0,
            face: Some(f_2),
            next: Some(e_x2),
            twin: Some(e_x0),
            ..default()
        };
        self.edges[e_1x] = Edge {
            root: v_1,
            face: Some(f_0),
            next: Some(e_x0),
            twin: Some(e_x1),
            ..default()
        };
        self.edges[e_2x] = Edge {
            root: v_2,
            face: Some(f_1),
            next: Some(e_x1),
            twin: Some(e_x2),
            ..default()
        };

        self.edges[e_01].face = Some(f_0);
        self.edges[e_01].next = Some(e_1x);

        self.edges[e_12].face = Some(f_1);
        self.edges[e_12].next = Some(e_2x);

        self.edges[e_20].face = Some(f_2);
        self.edges[e_20].next = Some(e_0x);

        return (v_x, (f_0, f_1, f_2));
    }

    pub fn split_edge(
        &mut self,
        edge_id: usize,
        split_point_maybe: Option<Vec3>,
    ) -> (usize, (usize, usize, usize, usize)) {
        // First face
        let e_ab = edge_id;
        let e_b0 = self.get_next_of_edge(e_ab);
        let e_0a = self.get_next_of_edge(e_b0);

        assert!(self.get_next_of_edge(e_0a) == e_ab);

        let v_a = self.get_root_of_edge(e_ab);
        let v_b = self.get_root_of_edge(e_b0);
        let v_0 = self.get_root_of_edge(e_0a);

        let split_point = match split_point_maybe {
            Some(point) => point,
            None => (self.get_position_of_vertex(v_a) + self.get_position_of_vertex(v_b)) / 2.,
        };

        // Second face
        let e_ba = self.get_twin_of_edge(edge_id);
        let e_a1 = self.get_next_of_edge(e_ba);
        let e_1b = self.get_next_of_edge(e_a1);
        println!("e_ba: {}, e_a1: {}, e_1b: {}", e_ba, e_a1, e_1b);
        println!("next of e_1b: {}", self.get_next_of_edge(e_1b));
        assert!(self.get_next_of_edge(e_1b) == e_ba);

        assert!(self.get_root_of_edge(e_ba) == v_b);
        assert!(self.get_root_of_edge(e_a1) == v_a);
        let v_1 = self.get_root_of_edge(e_1b);

        // Four new faces (re-use original id for first 2)
        let f_0 = self.get_face_of_edge(e_ab);
        self.faces[f_0].some_edge = Some(e_0a);

        let f_1 = self.get_face_of_edge(e_ba);
        self.faces[f_1].some_edge = Some(e_a1);

        let f_2 = self.faces.len();
        self.faces.push(Face {
            some_edge: Some(e_b0),
            color: self.faces[f_0].color,
            normal: self.faces[f_0].normal,
            dual_position: self.faces[f_0].dual_position,
            dual_normal: self.faces[f_0].dual_normal,
            original_face: self.faces[f_0].original_face,
            respective_surface: None,
        });

        let f_3 = self.faces.len();
        self.faces.push(Face {
            some_edge: Some(e_1b),
            color: self.faces[f_1].color,
            normal: self.faces[f_1].normal,
            dual_position: self.faces[f_1].dual_position,
            dual_normal: self.faces[f_1].dual_normal,
            original_face: self.faces[f_1].original_face,
            respective_surface: None,
        });

        // Six new edges (with next six available ids)

        // f_0
        let e_ax = e_ab;
        let e_x0 = self.edges.len();
        self.edges.push(Edge::default());

        // f_1
        let e_xa = e_ba;
        let e_1x = self.edges.len();
        self.edges.push(Edge::default());

        // f_2
        let e_xb = self.edges.len();
        self.edges.push(Edge::default());
        let e_0x = self.edges.len();
        self.edges.push(Edge::default());

        // f_3
        let e_bx = self.edges.len();
        self.edges.push(Edge::default());
        let e_x1 = self.edges.len();
        self.edges.push(Edge::default());

        // One new vertex (with next available id)
        let v_x = self.vertices.len();
        self.vertices.push(Vertex {
            some_edge: Some(e_xa),
            position: split_point,
            normal: self.faces[f_0].normal,
            ..default()
        });

        self.vertices[v_b].some_edge = Some(e_b0);
        self.vertices[v_a].some_edge = Some(e_a1);

        // Set the edges correctly
        self.edges[e_ax] = Edge {
            root: v_a,
            face: Some(f_0),
            next: Some(e_x0),
            twin: Some(e_xa),
            ..default()
        };
        self.edges[e_xa] = Edge {
            root: v_x,
            face: Some(f_1),
            next: Some(e_a1),
            twin: Some(e_ax),
            ..default()
        };

        self.edges[e_bx] = Edge {
            root: v_b,
            face: Some(f_3),
            next: Some(e_x1),
            twin: Some(e_xb),
            ..default()
        };
        self.edges[e_xb] = Edge {
            root: v_x,
            face: Some(f_2),
            next: Some(e_b0),
            twin: Some(e_bx),
            ..default()
        };

        self.edges[e_0x] = Edge {
            root: v_0,
            face: Some(f_2),
            next: Some(e_xb),
            twin: Some(e_x0),
            ..default()
        };
        self.edges[e_x0] = Edge {
            root: v_x,
            face: Some(f_0),
            next: Some(e_0a),
            twin: Some(e_0x),
            ..default()
        };

        self.edges[e_1x] = Edge {
            root: v_1,
            face: Some(f_1),
            next: Some(e_xa),
            twin: Some(e_x1),
            ..default()
        };
        self.edges[e_x1] = Edge {
            root: v_x,
            face: Some(f_3),
            next: Some(e_1b),
            twin: Some(e_1x),
            ..default()
        };

        self.edges[e_a1].face = Some(f_1);
        self.edges[e_a1].next = Some(e_1x);

        self.edges[e_1b].face = Some(f_3);
        self.edges[e_1b].next = Some(e_bx);

        self.edges[e_b0].face = Some(f_2);
        self.edges[e_b0].next = Some(e_0x);

        self.edges[e_0a].face = Some(f_0);
        self.edges[e_0a].next = Some(e_ax);

        return (v_x, (f_0, f_1, f_2, f_3));
    }

    pub fn split_long_edges(&mut self) {
        let long_edges = (0..self.edges.len())
            .filter(|&edge_id| {
                let length = self.get_length_of_edge(edge_id);
                let total_length = self
                    .get_edges_of_face(self.get_face_of_edge(edge_id))
                    .iter()
                    .map(|&e| self.get_length_of_edge(e))
                    .sum::<f32>();
                length > 0.45 * total_length
            })
            .collect_vec();

        info!(
            "Splitting {} edges with length larger than 40% of triangle",
            long_edges.len(),
        );

        for edge_id in long_edges {
            println!("Splitting edge {edge_id}", edge_id = edge_id);
            self.split_edge(edge_id, None);
        }
    }

    pub fn split_outliers(&mut self) {
        let edge_lengths = (0..self.edges.len()).map(|edge_id| {
            let v_a = self.get_root_of_edge(edge_id);
            let v_b = self.get_root_of_edge(self.get_twin_of_edge(edge_id));

            self.vertices[v_a]
                .position
                .distance(self.vertices[v_b].position)
        });

        let mean = average(edge_lengths.clone());
        let variance = average(edge_lengths.clone().map(|length| (length - mean).powf(2.0)));
        let sd = variance.sqrt();
        let threshold = mean + 1.0 * sd;

        let outliers: Vec<usize> = edge_lengths
            .enumerate()
            .filter(|&(_, length)| length > threshold)
            .map(|(i, _)| i)
            .collect();

        info!(
            "Splitting {} edges with length larger than 3 standard deviations from the mean {threshold:.3}",
            outliers.len(),
        );

        for edge_id in outliers {
            println!("Splitting edge {edge_id}", edge_id = edge_id);
            self.split_edge(edge_id, None);
        }
    }

    pub fn write_to_obj(&self, path: &PathBuf) -> Result<(), Box<dyn Error>> {
        let mut file = std::fs::File::create(path)?;

        for vertex_id in 0..self.vertices.len() {
            writeln!(
                file,
                "v {x:.6} {y:.6} {z:.6}",
                x = self.vertices[vertex_id].position.x,
                y = self.vertices[vertex_id].position.y,
                z = self.vertices[vertex_id].position.z
            )?;
        }

        for face_id in 0..self.faces.len() {
            let vertices = self.get_vertices_of_face(face_id);
            write!(file, "f")?;
            for vertex_id in vertices {
                write!(file, " {}", vertex_id + 1)?;
            }
        }

        Ok(())
    }
}
