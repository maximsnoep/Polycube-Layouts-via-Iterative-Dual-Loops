# Polycube Layouts via Iterative Dual Loops

Polycube layouts for 3D models effectively support a wide variety of applications such as hexahedral mesh construction, seamless texture mapping, spline fitting, and multi-block grid generation. However, the automated construction of valid polycube layouts 
suffers from robustness issues: the state-of-the-art deformation-based methods are  not guaranteed to find a valid solution. In this paper we present a novel approach which is guaranteed to return a valid polycube layout for 3D models of genus 0. Our algorithm is based on a dual representation of polycubes; we construct polycube layouts by iteratively adding or removing dual loops. The iterative nature of our algorithm facilitates a seamless trade-off between quality and complexity of the solution. Our method is efficient and can be implemented using comparatively simple algorithmic building blocks. We experimentally compare the results of our algorithm against state-of-the-art methods. Our fully automated method always produces provably valid polycube layouts whose quality - assessed via the quality of derived hexahedral meshes - is on par with state-of-the-art deformation methods.

## Prerequisites
Before you can run the project, make sure you have the following prerequisites installed on your system:
 - Rust and Cargo: This project is written in Rust, so you will need Rust and Cargo (the Rust package manager) installed. You can download and install both from [https://www.rust-lang.org/tools/install](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## Getting Started
To get started with the project, follow these steps:

1. Clone the repository
2. Build and run the project using Cargo: `cargo run --release` (in the root of the project directory)
3. Explore the project through the user interface
    - load a `.stl` file of a genus 0 manifold triangle surface mesh of a three-dimensional shape, and compute a dual loop structure and polycube layout using the `run` button (see the console output for current status)
    - load a `.poc` save file of previously computed dual loop strucutres and polycube layouts, for example [`/out/bone-003_128.poc`](/out/bone-003_128.poc)
    - export current polycube layout as a `.obj` mesh with accompanying `.flag` labeling, which can be used in [robustPolycube](https://github.com/fprotais/robustPolycube) for hexahedral meshing
