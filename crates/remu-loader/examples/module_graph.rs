// SPDX-License-Identifier: GPL-2.0-or-later

use std::path::PathBuf;
use std::process::ExitCode;

use remu_loader::resolve_module_dependency_graph;

fn main() -> ExitCode {
    let Some(entry) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: cargo run -p remu-loader --example module_graph -- <eboot.bin>");
        return ExitCode::FAILURE;
    };
    let graph = match resolve_module_dependency_graph(&entry) {
        Ok(graph) => graph,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::FAILURE;
        }
    };

    println!("app_root={}", graph.app_root.display());
    println!("entry={}", graph.entry.path.display());
    for (index, root) in graph.search_roots.iter().enumerate() {
        println!("search_root[{index}]={}", root.display());
    }
    for (index, module) in graph.dependency_order.iter().enumerate() {
        println!("dependency[{index}]={}", module.path.display());
    }
    for unresolved in &graph.unresolved_dependencies {
        println!(
            "unresolved={} requested_by={}",
            unresolved.name,
            unresolved.requested_by.display()
        );
    }
    println!("bundled_dependencies={}", graph.dependency_order.len());
    println!(
        "unresolved_dependencies={}",
        graph.unresolved_dependencies.len()
    );
    ExitCode::SUCCESS
}
