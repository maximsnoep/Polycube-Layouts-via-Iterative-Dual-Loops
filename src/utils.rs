use crate::{
    doconeli::Doconeli,
    solution::{PrincipalDirection, Surface},
    ColorType, Configuration,
};
use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    utils::Instant,
};
use itertools::Itertools;
use rand::Rng;
use std::{f32::consts::PI, ops::Add};

pub fn average<'a, T>(list: impl Iterator<Item = T>) -> T
where
    T: Add<Output = T> + std::default::Default + std::ops::Div<f32, Output = T>,
{
    let (sum, count) = list.fold((T::default(), 0.), |(sum, count), elem| {
        (sum + elem, count + 1.)
    });
    sum / (count as f32)
}

pub fn set_intersection<T: std::cmp::PartialEq + Clone>(
    collection_a: &Vec<T>,
    collection_b: &Vec<T>,
) -> Vec<T> {
    let mut intesection = collection_b.clone();
    intesection.retain(|edge_id| collection_a.contains(edge_id));
    return intesection;
}

pub fn transform_coordinates(translation: Vec3, scale: f32, position: Vec3) -> Vec3 {
    (position * scale) + translation
}

pub fn intersection_in_sequence(elem_a: usize, elem_b: usize, sequence: &Vec<usize>) -> bool {
    let mut sequence_copy = sequence.clone();
    sequence_copy.retain(|&elem| elem == elem_a || elem == elem_b);
    debug_assert!(sequence_copy.len() == 4, "{:?}", sequence_copy);
    sequence_copy.dedup();
    sequence_copy.len() >= 4
}

// Report times
#[derive(Clone)]
pub struct Timer {
    pub start: Instant,
}

impl Timer {
    pub fn new() -> Timer {
        Timer {
            start: Instant::now(),
        }
    }

    pub fn reset(&mut self) {
        self.start = Instant::now();
    }

    pub fn report(&self, note: &str) {
        info!("{:>12?}  >  {note}", self.start.elapsed());
    }
}

// X,      Y,       Z
// red,    blue,    yellow
// green,  orange,  purple
pub fn get_color(dir: PrincipalDirection, primary: bool, configuration: &Configuration) -> Color {
    match (dir, primary) {
        (PrincipalDirection::X, true) => configuration.color_primary1.into(),
        (PrincipalDirection::Y, true) => configuration.color_primary2.into(),
        (PrincipalDirection::Z, true) => configuration.color_primary3.into(),
        (PrincipalDirection::X, false) => configuration.color_secondary1.into(),
        (PrincipalDirection::Y, false) => configuration.color_secondary2.into(),
        (PrincipalDirection::Z, false) => configuration.color_secondary3.into(),
    }
}

pub fn get_random_color() -> Color {
    let hue = rand::thread_rng().gen_range(0.0..360.0);
    let sat = rand::thread_rng().gen_range(0.6..0.8);
    let lit = rand::thread_rng().gen_range(0.6..0.8);
    Color::hsl(hue, sat, lit)
}

// Construct a mesh object that can be rendered using the Bevy framework.
pub fn get_bevy_mesh_of_regions(
    regions: &Vec<Surface>,
    color_type: ColorType,
    configuration: &Configuration,
) -> Mesh {
    let mut mesh_triangle_list = Mesh::new(PrimitiveTopology::TriangleList);
    let mut vertex_positions = vec![];
    let mut vertex_normals = vec![];
    let mut vertex_colors = vec![];

    for surface in regions {
        let color = match color_type {
            ColorType::DirectionPrimary => {
                get_color(surface.direction.unwrap(), true, configuration)
            }
            ColorType::DirectionSecondary => surface.color.unwrap().into(),
            ColorType::Random => get_random_color(),
            ColorType::Static(color) => color,
            _ => surface.color.unwrap().into(),
        };

        let mut color_f32 = color.as_rgba_f32();

        for subface in &surface.faces {
            if color_type == ColorType::Distortion {
                println!("{}", subface.distortion.unwrap());
                color_f32 = Color::hsl(0., 0.5, 1. - 0.5 * (subface.distortion.unwrap() / 90.))
                    .as_rgba_f32();
            }

            for &p1 in &subface.bounding_points {
                for &p2 in &subface.bounding_points {
                    for &p3 in &subface.bounding_points {
                        vertex_positions.push(p1.0);
                        vertex_positions.push(p2.0);
                        vertex_positions.push(p3.0);
                        vertex_normals.push(p1.1);
                        vertex_normals.push(p2.1);
                        vertex_normals.push(p3.1);
                        vertex_colors.push(color_f32);
                        vertex_colors.push(color_f32);
                        vertex_colors.push(color_f32);
                    }
                }
            }
        }
    }

    let length = vertex_positions.len();
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertex_positions);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertex_normals);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);
    mesh_triangle_list.set_indices(Some(Indices::U32((0..length as u32).collect())));

    mesh_triangle_list
}

// Construct a mesh object that can be rendered using the Bevy framework.
pub fn get_bevy_mesh_of_mesh(
    mesh: &Doconeli,
    color_type: ColorType,
    configuration: &Configuration,
) -> Mesh {
    let mut mesh_triangle_list = Mesh::new(PrimitiveTopology::TriangleList);
    let mut vertex_positions = Vec::with_capacity(mesh.faces.len() * 3);
    let mut vertex_normals = Vec::with_capacity(mesh.faces.len() * 3);
    let mut vertex_colors = Vec::with_capacity(mesh.faces.len() * 3);

    for face_id in 0..mesh.faces.len() {
        let color = match color_type {
            ColorType::Static(c) => c,
            _ => mesh.faces[face_id].color,
        };

        for vertex_id in mesh.get_vertices_of_face(face_id) {
            vertex_positions.push(mesh.get_position_of_vertex(vertex_id));
            vertex_normals.push(mesh.vertices[vertex_id].normal);
            vertex_colors.push(color.as_rgba_f32());
        }
    }

    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertex_positions);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertex_normals);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);
    mesh_triangle_list.set_indices(Some(Indices::U32(
        (0..mesh.faces.len() as u32 * 3).collect(),
    )));

    mesh_triangle_list
}

// Construct a mesh object that can be rendered using the Bevy framework.
pub fn get_bevy_mesh_of_graph(
    graph: &Doconeli,
    color_type: ColorType,
    configuration: &Configuration,
) -> Mesh {
    let mut mesh_triangle_list = Mesh::new(PrimitiveTopology::TriangleList);
    let mut vertex_positions = vec![];
    let mut vertex_normals = vec![];
    let mut vertex_colors = vec![];

    for face_id in 0..graph.faces.len() {
        let vertices = graph.get_vertices_of_face(face_id);
        let edges = graph.get_edges_of_face(face_id);

        let dir = [
            PrincipalDirection::X,
            PrincipalDirection::Y,
            PrincipalDirection::Z,
        ]
        .into_iter()
        .filter(|&dir| {
            !edges
                .iter()
                .map(|&edge_id| graph.edges[edge_id].direction)
                .contains(&Some(dir))
        })
        .next();

        let color = match color_type {
            ColorType::Static(color) => color,
            ColorType::Random => get_random_color(),
            ColorType::DirectionPrimary => get_color(dir.unwrap(), true, configuration),
            ColorType::DirectionSecondary => get_color(dir.unwrap(), false, configuration),
            ColorType::Distortion => todo!(),
        };

        for &vertex1 in &vertices {
            for &vertex2 in &vertices {
                for &vertex3 in &vertices {
                    vertex_positions.push(graph.get_position_of_vertex(vertex1));
                    vertex_normals.push(graph.vertices[vertex1].normal);
                    vertex_colors.push(color.as_rgba_f32());
                    vertex_positions.push(graph.get_position_of_vertex(vertex2));
                    vertex_normals.push(graph.vertices[vertex2].normal);
                    vertex_colors.push(color.as_rgba_f32());
                    vertex_positions.push(graph.get_position_of_vertex(vertex3));
                    vertex_normals.push(graph.vertices[vertex3].normal);
                    vertex_colors.push(color.as_rgba_f32());
                }
            }
        }
    }

    mesh_triangle_list.set_indices(Some(Indices::U32(
        (0..vertex_positions.len() as u32).collect(),
    )));
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_POSITION, vertex_positions);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vertex_normals);
    mesh_triangle_list.insert_attribute(Mesh::ATTRIBUTE_COLOR, vertex_colors);

    mesh_triangle_list
}

pub fn intersection_exact_in_2d(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Option<Vec2> {
    // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    let t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x))
        / ((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x));

    let u = ((p1.x - p3.x) * (p1.y - p2.y) - (p1.y - p3.y) * (p1.x - p2.x))
        / ((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x));

    if t >= 0. && t <= 1. && u >= 0. && u <= 1. {
        let intersection_x = p1.x + t * (p2.x - p1.x);
        let intersection_y = p1.y + t * (p2.y - p1.y);

        return Some(Vec2::new(intersection_x, intersection_y));
    }

    None
}

pub fn convert_3d_to_2d(point: Vec3, reference: Vec3) -> Vec2 {
    let alpha = point.angle_between(reference);
    Vec2::new(point.length() * alpha.cos(), point.length() * alpha.sin())
}

pub fn point_lies_in_triangle(point: Vec3, triangle: (Vec3, Vec3, Vec3)) -> bool {
    let vectors = vec![triangle.0 - point, triangle.1 - point, triangle.2 - point];

    let sum_of_angles = vectors[0].angle_between(vectors[1])
        + vectors[1].angle_between(vectors[2])
        + vectors[2].angle_between(vectors[0]);

    return (sum_of_angles - 2. * PI).abs() < 0.00001;
}

pub fn point_lies_in_segment(point: Vec3, segment: (Vec3, Vec3)) -> bool {
    let segment_length = segment.0.distance(segment.1);

    let segment_length_through_point = point.distance(segment.0) + point.distance(segment.1);

    return (segment_length - segment_length_through_point).abs() < 0.00001;
}

pub fn score_normal_based_on_orientation(normal: Vec3, orientation: PrincipalDirection) -> i32 {
    let orientation_vec = orientation.to_vector();
    let orientation_vec_neg = -orientation_vec;

    std::cmp::min(
        normal.angle_between(orientation_vec).to_degrees() as i32,
        normal.angle_between(orientation_vec_neg).to_degrees() as i32,
    )
}
