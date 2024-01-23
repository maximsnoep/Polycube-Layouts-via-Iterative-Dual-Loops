# Polycube Layouts via Iterative Dual Loops

Polycube layouts for 3D models effectively support a wide variety of methods such as hex-mesh construction, seamless texture mapping, spline fitting, and multi-block grid generation. Our study of polycube layouts and their corresponding polycubes is motivated by automated conformal mesh generation for aerospace modelling. In this setting, quality and correctness guarantees are of the utmost importance. 
However, currently the fully automatic construction of valid polycube layouts still poses significant challenges: state-of-the-art methods are generally not guaranteed to return a proper solution, even after post-processing, or they use a prohibitively large number of voxels that add detail indiscriminately.

In this paper we present a robust, flexible, and efficient method to generate polycube layouts fully automatically. Our approach is based on a dual representation for polycube layouts and builds the final layout and its polycube by iteratively adding dual loops. Our construction is robust by design: at any iterative step we maintain a valid polycube layout. We offer the flexibility of manual intervention if the user so desires: while our method is able to compute a complete polycube layout without user intervention, the user can interrupt after each iteration and target further refinement on both the local and the global level. Last but not least, our method is efficient and can be implemented using comparatively simple algorithmic building blocks. Our implementation is publicly available and we present its output for numerous benchmark models.

## Prerequisites
Before you can run the project, make sure you have the following prerequisites installed on your system:
 - Rust and Cargo: This project is written in Rust, so you'll need Rust and Cargo installed. You can download and install Rust from https://www.rust-lang.org/tools/install.

## Getting Started
To get started with the project, follow these steps:

1. Clone the repository
2. Navigate to the project directory
3. Build and rn the project using Cargo: `cargo run --release`
4. Explore the project: load a `.stl` file of a  genus 0 manifold triangle surface meshes of a three-dimensional shape (if you do not have any, download samples from https://anonymous.4open.science/r/QD9tt), and use the user interface to compute a polycube layout
