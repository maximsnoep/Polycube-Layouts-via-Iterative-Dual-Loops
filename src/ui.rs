use crate::{
    solution::{self, LoopScoring, PrincipalDirection},
    ActionEvent, Configuration, RenderType,
};
use bevy::prelude::*;
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

        ui_algorithm_info(ui, &mut ev_writer, &mut configuration);

        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);

        ui_rendering_info(ui, &mut ev_writer, &mut configuration, &mut mesh_resmut);

        ui.add_space(10.);
        ui.separator();
        ui.add_space(10.);
    });
}

fn ui_software_info(ui: &mut Ui) {
    ui.vertical_centered(|ui| {
        ui.label(RichText::new("    pola dulo    ").strong().code().heading());
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
        if ui.button("mesh").clicked() {
            ev_writer.send(ActionEvent::ExportMesh);
        };
        if ui.button("layout").clicked() {
            ev_writer.send(ActionEvent::ExportLayout);
        };
        if ui.button("labeling").clicked() {
            ev_writer.send(ActionEvent::ExportLabeling);
        };
        if ui.button("state").clicked() {
            ev_writer.send(ActionEvent::ExportState);
        };
    });
}

fn ui_algorithm_info(
    ui: &mut Ui,
    ev_writer: &mut EventWriter<ActionEvent>,
    configuration: &mut ResMut<Configuration>,
) {
    ui.add_space(10.);

    ui.label("Algorithm");

    if ui.button("re-init").clicked() {
        ev_writer.send(ActionEvent::InitializeLoops);
    };

    ui.add(egui::Slider::new(&mut configuration.algorithm_iterations, 1..=5).text("iterations"));
    // ui.add(egui::Slider::new(&mut configuration.algorithm_samples, 1..=2000).text("samples"));
    // ui.add(egui::Slider::new(&mut configuration.gamma, 1.0..=20.0).text("gamma"));

    // ui.add(
    //     egui::Slider::new(&mut configuration.percent_singularities, 1..=100).text("%singularities"),
    // );

    // for loop_scoring in LoopScoring::iter() {
    //     if ui
    //         .radio(
    //             configuration.loop_scoring_scheme == loop_scoring,
    //             format!("{loop_scoring:?}"),
    //         )
    //         .clicked()
    //     {
    //         configuration.loop_scoring_scheme = loop_scoring;
    //     }
    // }

    // ui.add_space(5.);

    // ui.horizontal(|ui| {
    //     if ui.button("<<").clicked() {
    //         configuration.choose_component -= 1;
    //     };
    //     ui.label(format!("  component {}  ", configuration.choose_component));
    //     if ui.button(">>").clicked() {
    //         configuration.choose_component += 1;
    //     };
    // });

    // ui.checkbox(
    //     &mut configuration.draw_next_component,
    //     "view selected component",
    // );

    // ui.checkbox(&mut configuration.find_global, "find global");

    // ui.add_space(5.);

    // ui.horizontal(|ui| {
    //     if ui.button("<<").clicked() {
    //         configuration.remove_loop -= 1;
    //     };
    //     ui.label(format!("  loop {}  ", configuration.remove_loop));
    //     if ui.button(">>").clicked() {
    //         configuration.remove_loop += 1;
    //     };
    // });

    // ui.checkbox(&mut configuration.view_selected_loop, "view selected loop");

    // if ui.button("remove").clicked() {
    //     ev_writer.send(ActionEvent::RemoveLoop);
    // };

    // ui.add_space(10.);

    ui.horizontal(|ui| {
        // ui.label("Dual structure");
        // for principal_direction in PrincipalDirection::iter() {
        //     if ui
        //         .radio(
        //             configuration.choose_direction == principal_direction,
        //             format!("{principal_direction:?}"),
        //         )
        //         .clicked()
        //     {
        //         configuration.choose_direction = principal_direction;
        //         configuration.choose_component = 0;
        //     }
        // }
        // if ui.button("add").clicked() {
        //     ev_writer.send(ActionEvent::AddLoop);
        // };
        // if ui.button("undo").clicked() {
        //     ev_writer.send(ActionEvent::UndoLoop);
        // };
        if ui.button("automatic").clicked() {
            ev_writer.send(ActionEvent::RunAlgo);
        };
    });

    // ui.add_space(10.);

    // ui.add(egui::Slider::new(&mut configuration.path_weight, 0.0..=1.0).text("path weight"));

    // ui.horizontal(|ui| {
    //     ui.label("Primalization embedding");
    //     if ui.button("vertices").clicked() {
    //         ev_writer.send(ActionEvent::PrimalizePlaceCenters);
    //     };
    //     if ui.button("edges").clicked() {
    //         ev_writer.send(ActionEvent::PrimalizeConnectCenters);
    //     };
    // });

    // ui.horizontal(|ui| {
    //     ui.label("Primalization embedding (NEW)");
    //     if ui.button("init").clicked() {
    //         ev_writer.send(ActionEvent::InitPrimalize);
    //     };
    //     if ui.button("step").clicked() {
    //         ev_writer.send(ActionEvent::StepPrimalize);
    //     };
    // });

    ui.add_space(10.);
}

fn ui_rendering_info(
    ui: &mut Ui,
    ev_writer: &mut EventWriter<ActionEvent>,
    configuration: &mut ResMut<Configuration>,
    mesh_resmut: &mut ResMut<MeshResource>,
) {
    ui.add_space(5.);
    ui.add_space(10.);

    ui.label("Rendered object");
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

    ui.add_space(10.);
    ui.label("Visualize");
    ui.horizontal(|ui| {
        ui.checkbox(&mut configuration.draw_wireframe, "input mesh");
        ui.checkbox(&mut configuration.draw_wireframe_alt, "granulated mesh");
        // ui.checkbox(&mut configuration.primal_w_graph, "primal w graph");
    });

    ui.horizontal(|ui| {
        ui.checkbox(&mut configuration.draw_loops, "loops");
        ui.checkbox(&mut configuration.draw_paths, "paths");
    });

    ui.checkbox(&mut configuration.draw_singularities, "singularities");

    if ui
        .checkbox(&mut configuration.black, "toggle black")
        .clicked()
    {
        mesh_resmut.as_mut();
    }

    // ui.add_space(10.);
    // ui.label("Debug");
    // ui.checkbox(&mut configuration.draw_debug_lines, "Debug lines");

    // ui.add_space(10.);
    // ui.label("Colors");
    // ui.horizontal(|ui| {
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_background);
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_foreground);
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_foreground2);
    //     ui.label("Main");
    // });

    // ui.horizontal(|ui| {
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_primary1);
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_primary2);
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_primary3);
    //     ui.label("Primary");
    // });

    // ui.horizontal(|ui| {
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_secondary1);
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_secondary2);
    //     ui.color_edit_button_rgba_unmultiplied(&mut configuration.color_secondary3);
    //     ui.label("Secondary");
    // });

    // ui.add_space(5.);

    // ui.horizontal(|ui| {
    //     if ui.button("apply colors").clicked() {
    //         mesh_resmut.as_mut();
    //     };
    //     if ui.button("reset colors").clicked() {
    //         ev_writer.send(ActionEvent::ResetConfiguration);
    //     };
    // });

    // ui.add_space(10.);
    // ui.label("Camera controls");

    // ui.horizontal(|ui| {
    //     ui.checkbox(&mut configuration.camera_autorotate, "auto-rotate");
    //     ui.add(egui::Slider::new(&mut configuration.camera_speed, 1..=100).text("speed"));
    // });

    // ui.add_space(10.);

    // ui.add_space(5.);
}
