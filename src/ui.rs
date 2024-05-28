use crate::{
    solution::{self, LoopScoring, PrincipalDirection},
    ActionEvent, Configuration, RenderType,
};
use bevy::{prelude::*, render::Render};
use bevy_egui::{
    egui::{self, Color32, RichText, Ui},
    EguiContexts,
};
use solution::MeshResource;
use strum::IntoEnumIterator;

pub fn ui(
    mut egui_ctx: EguiContexts,
    mut mesh_resmut: ResMut<MeshResource>,
    mut ev_writer: EventWriter<ActionEvent>,
    mut configuration: ResMut<Configuration>,
) {
    egui::SidePanel::left("ui_side_panel_left").show(egui_ctx.ctx_mut(), |ui| {
        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui_software_info(ui);

        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui_file_info(ui, &mut ev_writer, &configuration);

        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui_algorithm_info(ui, &mut ev_writer, &mut configuration, &mut mesh_resmut);
    });
}

fn ui_software_info(ui: &mut Ui) {
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("    DualLoops    ").strong().code().heading());
        ui.add_space(10.);
        ui.label(RichText::new("Polycube Layouts via Iterative Dual Loops"));
    });
    ui.add_space(10.);
    ui.vertical_centered_justified(|ui| {
        ui.label(RichText::new("Rotate"));
        ui.label(RichText::new("CTRL + drag").italics());
    });
    ui.vertical_centered_justified(|ui| {
        ui.label(RichText::new("Pan"));
        ui.label(RichText::new("RMB + drag").italics());
    });
    ui.vertical_centered_justified(|ui| {
        ui.label(RichText::new("Zoom"));
        ui.label(RichText::new("mouse wheel").italics());
    });
}

fn ui_file_info(
    ui: &mut Ui,
    ev_writer: &mut EventWriter<ActionEvent>,
    configuration: &ResMut<Configuration>,
) {
    if ui.button("Load file (supported: .stl)").clicked() {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("triangulated geometry", &["stl", "poc"])
            .pick_file()
        {
            ev_writer.send(ActionEvent::LoadFile(path));
        }
    }

    ui.add_space(10.);

    if !configuration.source.is_empty() {
        ui.label(RichText::new(format!(
            "\tLoaded: {}\n\t(v: {}, e: {}, f: {})",
            configuration.source,
            configuration.nr_of_vertices,
            configuration.nr_of_edges,
            configuration.nr_of_faces
        )));
    } else {
        ui.label(RichText::new("\tNo file yet loaded.").color(Color32::LIGHT_RED));
    }

    ui.add_space(10.);

    ui.horizontal(|ui| {
        ui.label("Export ");
        if ui.button("mesh and labeling").clicked() {
            ev_writer.send(ActionEvent::ExportMesh);
            ev_writer.send(ActionEvent::ExportLabeling);
        };
        if ui.button("save state").clicked() {
            ev_writer.send(ActionEvent::ExportState);
        };
    });
}

fn ui_algorithm_info(
    ui: &mut Ui,
    ev_writer: &mut EventWriter<ActionEvent>,
    configuration: &mut ResMut<Configuration>,
    mesh_resmut: &mut ResMut<MeshResource>,
) {
    ui.add_space(10.);

    ui.horizontal(|ui| {
        ui.label("DualLoops");
        ui.add_space(10.);
        if ui.button("run").clicked() {
            ev_writer.send(ActionEvent::RunAlgo);
        };
        ui.add_space(10.);
        ui.checkbox(&mut configuration.reinit, "reinit");
        ui.add_space(10.);
        ui.add(
            egui::Slider::new(&mut configuration.algorithm_iterations, 1..=10)
                .text("max iterations"),
        );
    });

    if !configuration.debug {
        ui.add_space(5.);

        ui.horizontal(|ui| {
            ui.label("Visualize");
            if ui
                .radio(
                    configuration.render_type == RenderType::Original
                        && configuration.draw_wireframe,
                    format!("Input mesh"),
                )
                .clicked()
            {
                configuration.render_type = RenderType::Original;
                configuration.draw_loops = false;
                configuration.draw_wireframe = true;
                configuration.draw_paths = false;
                mesh_resmut.as_mut();
            }
            if ui
                .radio(
                    configuration.render_type == RenderType::Original && configuration.draw_loops,
                    format!("Loop structure"),
                )
                .clicked()
            {
                configuration.render_type = RenderType::Original;
                configuration.draw_loops = true;
                configuration.draw_wireframe = false;
                configuration.draw_paths = false;
                mesh_resmut.as_mut();
            }
            if ui
                .radio(
                    configuration.render_type == RenderType::PatchesInnerMesh,
                    format!("Patch layout"),
                )
                .clicked()
            {
                configuration.render_type = RenderType::PatchesInnerMesh;
                configuration.draw_loops = false;
                configuration.draw_wireframe = false;
                configuration.draw_paths = true;
                mesh_resmut.as_mut();
            }
        });
    }

    ui.add_space(10.);
    ui.add_space(10.);

    ui.checkbox(&mut configuration.debug, "toggle debug mode");

    if configuration.debug {
        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui.label("Evolutionary algo parameters");
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut configuration.n, 1..=16).text("N"));
            ui.add_space(10.);
            ui.add(egui::Slider::new(&mut configuration.n_prime, 1..=16).text("N'"));

            if configuration.n_prime < configuration.n {
                configuration.n_prime = configuration.n;
            }
        });

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut configuration.n_l, 1..=20).text("n_l"));
            ui.add_space(10.);
            ui.add(egui::Slider::new(&mut configuration.n_r, 1..=20).text("n_r"));
            ui.add_space(10.);
            ui.add(egui::Slider::new(&mut configuration.algorithm_samples, 1..=2000).text("n_c"));
        });

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut configuration.slack_min, 1.0..=20.).text("slack_min"));
            ui.add_space(10.);
            ui.add(egui::Slider::new(&mut configuration.slack_max, 1.0..=20.).text("slack_max"));
            if configuration.slack_min > configuration.slack_max {
                configuration.slack_max = configuration.slack_min;
            }
        });

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut configuration.rho_min, 0.0..=1.).text("rho_min"));
            ui.add_space(10.);
            ui.add(egui::Slider::new(&mut configuration.rho_max, 0.0..=1.).text("rho_max"));
            if configuration.rho_min > configuration.rho_max {
                configuration.rho_max = configuration.rho_min;
            }
        });

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut configuration.tau, 0.0..=1.0).text("tau"));
        });

        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui.label("Add valid loop");

        for loop_scoring in LoopScoring::iter() {
            if ui
                .radio(
                    configuration.loop_scoring_scheme == loop_scoring,
                    format!("{loop_scoring:?}"),
                )
                .clicked()
            {
                configuration.loop_scoring_scheme = loop_scoring;
            }
        }

        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut configuration.algorithm_samples, 1..=2000).text("n_c"));
            ui.add_space(10.);
            ui.add(egui::Slider::new(&mut configuration.gamma, 1.0..=20.0).text("slack"));
            ui.add_space(10.);
            ui.add(
                egui::Slider::new(&mut configuration.percent_singularities, 0.0..=1.0)
                    .text("rho (x100)"),
            );
        });

        ui.horizontal(|ui| {
            ui.horizontal(|ui| {
                if ui.button("<<").clicked() {
                    configuration.choose_component -= 1;
                };
                ui.label(format!(
                    "  selected zone {}  ",
                    configuration.choose_component
                ));
                if ui.button(">>").clicked() {
                    configuration.choose_component += 1;
                    mesh_resmut.as_mut();
                };
            });

            if ui.button("add loop to zone (if valid)").clicked() {
                ev_writer.send(ActionEvent::AddLoop);
            };
        });

        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui.label("Remove valid loop");

        ui.horizontal(|ui| {
            if ui.button("<<").clicked() {
                configuration.remove_loop -= 1;
            };
            ui.label(format!("  selected loop {}  ", configuration.remove_loop));
            if ui.button(">>").clicked() {
                configuration.remove_loop += 1;
            };

            ui.checkbox(&mut configuration.view_selected_loop, "highlight loop");

            if ui.button("remove loop (if valid)").clicked() {
                ev_writer.send(ActionEvent::RemoveLoop);
            };
        });

        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui.label("Visualize");
        for render_type in RenderType::iter() {
            if ui
                .radio(
                    configuration.render_type == render_type,
                    format!("{render_type:?}"),
                )
                .clicked()
            {
                configuration.render_type = render_type;
                mesh_resmut.as_mut();
            }
        }

        ui.horizontal(|ui| {
            ui.checkbox(&mut configuration.draw_wireframe, "mesh M");
            ui.add_space(10.);
            ui.checkbox(&mut configuration.draw_loops, "loops");
            ui.add_space(10.);
            ui.checkbox(&mut configuration.draw_paths, "paths");
            ui.add_space(10.);
            ui.checkbox(&mut configuration.draw_wireframe_alt, "mesh M'");
            ui.add_space(10.);
            ui.checkbox(&mut configuration.draw_singularities, "critical vertices");
            ui.add_space(10.);
            if ui
                .checkbox(&mut configuration.black, "toggle black")
                .clicked()
            {
                mesh_resmut.as_mut();
            }
            ui.add_space(10.);
            ui.checkbox(&mut configuration.draw_debug_lines, "debug lines");
        });

        ui.add_space(5.);

        ui.add_space(5.);
        ui.horizontal(|ui| {
            ui.label("Camera ");
            ui.checkbox(&mut configuration.camera_autorotate, "auto-rotate");
            ui.add(egui::Slider::new(&mut configuration.camera_speed, 1..=100).text("speed"));
        });
    }
}
