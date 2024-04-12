use crate::{
    doconeli::Doconeli,
    solution::{
        compute_average_normal, compute_deviation, det_jacobian, jacobian, PrincipalDirection,
        Surface,
    },
    ColorType, Configuration,
};
use bevy::{
    prelude::*,
    render::{mesh::Indices, render_resource::PrimitiveTopology},
    utils::Instant,
};
use itertools::Itertools;
use nalgebra::Matrix4x2;
use rand::Rng;
use std::error::Error;
use std::io::Write;
use std::{f32::consts::PI, ops::Add, path::PathBuf};

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

// Parula colormap
// 53, 42, 134
pub const PARULA_1: Color = Color::rgb(35. / 255., 42. / 255., 134. / 255.);
// 53, 60, 172
pub const PARULA_2: Color = Color::rgb(53. / 255., 60. / 255., 172. / 255.);
// 31, 82, 211
pub const PARULA_3: Color = Color::rgb(31. / 255., 82. / 255., 211. / 255.);
// 4, 108, 224
pub const PARULA_4: Color = Color::rgb(4. / 255., 108. / 255., 224. / 255.);
// 16, 120, 218
pub const PARULA_5: Color = Color::rgb(16. / 255., 120. / 255., 218. / 255.);
// 20, 132, 211
pub const PARULA_6: Color = Color::rgb(20. / 255., 132. / 255., 211. / 255.);
// 8, 152, 209
pub const PARULA_7: Color = Color::rgb(8. / 255., 152. / 255., 209. / 255.);
// 37, 180, 169
pub const PARULA_8: Color = Color::rgb(37. / 255., 180. / 255., 169. / 255.);
// 9, 171, 189
pub const PARULA_9: Color = Color::rgb(9. / 255., 171. / 255., 189. / 255.);
// 37, 180, 169
pub const PARULA_10: Color = Color::rgb(37. / 255., 180. / 255., 169. / 255.);
// 65, 186, 151
pub const PARULA_11: Color = Color::rgb(65. / 255., 186. / 255., 151. / 255.);
// 112, 190, 128
pub const PARULA_12: Color = Color::rgb(112. / 255., 190. / 255., 128. / 255.);
// 145, 190, 114
pub const PARULA_13: Color = Color::rgb(145. / 255., 190. / 255., 114. / 255.);
// 174, 189, 103
pub const PARULA_14: Color = Color::rgb(174. / 255., 189. / 255., 103. / 255.);
// 208, 186, 89
pub const PARULA_15: Color = Color::rgb(208. / 255., 186. / 255., 89. / 255.);
// 233, 185, 78
pub const PARULA_16: Color = Color::rgb(233. / 255., 185. / 255., 78. / 255.);
// 253, 190, 61
pub const PARULA_17: Color = Color::rgb(253. / 255., 190. / 255., 61. / 255.);
// 249, 210, 41
pub const PARULA_18: Color = Color::rgb(249. / 255., 210. / 255., 41. / 255.);
// 244, 228, 28
pub const PARULA_19: Color = Color::rgb(244. / 255., 228. / 255., 28. / 255.);

pub const PARULA: [Color; 19] = [
    PARULA_1, PARULA_2, PARULA_3, PARULA_4, PARULA_5, PARULA_6, PARULA_7, PARULA_8, PARULA_9,
    PARULA_10, PARULA_11, PARULA_12, PARULA_13, PARULA_14, PARULA_15, PARULA_16, PARULA_17,
    PARULA_18, PARULA_19,
];

// CB dark red: rgb(226, 26, 27) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_X: Color = Color::rgb(0.8862745098, 0.10196078431, 0.10588235294);
// CB dark blue: rgb(30, 119, 179) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Y: Color = Color::rgb(0.11764705882, 0.46666666666, 0.70196078431);
// CB yellow: rgb(255, 215, 0) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Z: Color = Color::rgb(1., 1., 0.6);

// CB light red: rgb(250, 153, 153) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_X_LIGHT: Color = Color::rgb(0.98039215686, 0.6, 0.6);
// CB light blue: rgb(166, 205, 226) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Y_LIGHT: Color = Color::rgb(0.65098039215, 0.80392156862, 0.8862745098);
// white: rgb(255, 255, 255) ranged from 0.0 to 1.0
pub const COLOR_PRIMARY_Z_LIGHT: Color = Color::rgb(1.0, 1.0, 0.8);

pub fn color_map(value: f32, colors: Vec<Color>) -> Color {
    let index = (value * (colors.len() - 1) as f32).round() as usize;
    colors[index]
}

// Construct a mesh object that can be rendered using the Bevy framework.
pub fn get_bevy_mesh_of_regions(
    regions: &Vec<Surface>,
    patch_graph: &Doconeli,
    granulated_mesh: &Doconeli,
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
            _ => Color::PINK,
        };

        let mut areas = vec![];
        let mut alignment_devs = vec![];
        let mut flatness_devs = vec![];

        if color_type == ColorType::DistortionAlignment {
            // direction is negative or positive based on angle with average normal
            let avg_normal = compute_average_normal(&surface, &granulated_mesh);
            let positive = surface
                .direction
                .unwrap()
                .to_vector()
                .dot(avg_normal)
                .signum();
            let direction = surface.direction.unwrap().to_vector() * positive;

            for subface in &surface.faces {
                let alignment_dev = compute_deviation(subface.face_id, &granulated_mesh, direction);
                alignment_devs.push(alignment_dev);

                let area = granulated_mesh.get_area_of_face(subface.face_id);
                areas.push(area);
            }

            let max_alignment_dev = alignment_devs
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_alignment_dev = alignment_devs.iter().cloned().fold(f32::INFINITY, f32::min);
            let avg_alignment_dev =
                alignment_devs.iter().sum::<f32>() / alignment_devs.len() as f32;
            let avg_alignment_dev_scaled = alignment_devs
                .iter()
                .zip(areas.iter())
                .map(|(dev, area)| dev * area)
                .sum::<f32>()
                / areas.iter().sum::<f32>();
        }

        if color_type == ColorType::DistortionFlatness {
            // compute avg normal of the surface
            let avg_normal = compute_average_normal(&surface, &granulated_mesh);
            for subface in &surface.faces {
                let flatness_dev = compute_deviation(subface.face_id, &granulated_mesh, avg_normal);
                flatness_devs.push(flatness_dev);

                let area = granulated_mesh.get_area_of_face(subface.face_id);
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
        }

        let mut det_j = 0.;
        if color_type == ColorType::DistortionJacobian {
            let avg_normal = compute_average_normal(&surface, &granulated_mesh);

            let positive = surface
                .direction
                .unwrap()
                .to_vector()
                .dot(avg_normal)
                .signum();

            let corners = patch_graph.get_vertices_of_face(surface.id);
            let corner_positions = corners
                .iter()
                .map(|&vertex_id| patch_graph.vertices[vertex_id].position)
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

            let mut min = f32::MAX;

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

                if scaled_det_jacobian < min {
                    min = scaled_det_jacobian;
                }
            }

            det_j = min;
        }

        let mut color_f32 = color.as_rgba_f32();

        for (i, subface) in surface.faces.iter().enumerate() {
            if color_type == ColorType::DistortionAlignment {
                let alignment_dev = alignment_devs[i];

                color_f32 = color_map(alignment_dev, PARULA.to_vec()).as_rgba_f32();
            }
            if color_type == ColorType::DistortionFlatness {
                let flatness_dev = flatness_devs[i];

                color_f32 = color_map(flatness_dev, PARULA.to_vec()).as_rgba_f32();
            }
            if color_type == ColorType::DistortionJacobian {
                let det_j = det_j;

                color_f32 = color_map(1.0 - det_j, PARULA.to_vec()).as_rgba_f32();

                if det_j <= 0. {
                    color_f32 = Color::RED.as_rgba_f32();
                }
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

pub fn get_labeling_of_mesh(
    path: &PathBuf,
    granulated_mesh: &Doconeli,
    regions: &Vec<Surface>,
) -> Result<(), Box<dyn Error>> {
    let mut labels = vec![-1; granulated_mesh.faces.len()];

    for surface in regions {
        // get avg normal of surface
        let avg_normal = average(
            surface
                .faces
                .iter()
                .map(|subface| granulated_mesh.faces[subface.face_id].normal),
        );
        // positive or negative based on angle with the direction
        let positive = surface.direction.unwrap().to_vector().dot(avg_normal) > 0.;
        // set label for all faces in the surface
        let label = match surface.direction.unwrap() {
            PrincipalDirection::X => {
                if positive {
                    0
                } else {
                    1
                }
            }
            PrincipalDirection::Y => {
                if positive {
                    2
                } else {
                    3
                }
            }
            PrincipalDirection::Z => {
                if positive {
                    4
                } else {
                    5
                }
            }
        };

        for subface in surface.faces.iter() {
            labels[subface.face_id] = label;
        }
    }

    let mut file = std::fs::File::create(path)?;

    for label in labels {
        writeln!(file, "{label:?}");
    }

    Ok(())
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
            ColorType::Labeling => match mesh.faces[face_id].label {
                Some(0) => COLOR_PRIMARY_X,
                Some(1) => COLOR_PRIMARY_X_LIGHT,
                Some(2) => COLOR_PRIMARY_Y,
                Some(3) => COLOR_PRIMARY_Y_LIGHT,
                Some(4) => COLOR_PRIMARY_Z,
                Some(5) => COLOR_PRIMARY_Z_LIGHT,
                _ => Color::WHITE,
            },
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
            _ => todo!(),
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
