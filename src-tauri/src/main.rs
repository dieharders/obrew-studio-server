// @TODO This is not preventing terminal window from opening on release builds.
// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(
  all(not(debug_assertions), target_os = "windows"),
  windows_subsystem = "windows"
)]
// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
// #![windows_subsystem = "windows"]

use std::process::Command;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;
use command_group::CommandGroup;
use tauri::api::process::Command as TCommand;
use tauri::WindowEvent;
use tauri::Manager;
// use std::path::PathBuf;

// #[tauri::command]
// fn open_file_dialog(window: Window) {
//     let file_dialog_result: Option<PathBuf> = window.dialog().open_directory();
//     if let Some(path) = file_dialog_result {
//         window.emit("selected-directory", Some(path.display().to_string()));
//     }
// }

fn start_backend(receiver: Receiver<i32>) {
  // `new_sidecar()` expects just the filename, NOT the whole path
  let t = TCommand::new_sidecar("main")
    .expect("[Error] Failed to create `main.exe` binary command");
  let mut group = Command::from(t).group_spawn().expect("[Error] spawning api server process.");
  thread::spawn(move || {
    loop{
      let s = receiver.recv();
      if s.unwrap()==-1 {
        group.kill().expect("[Error] killing api server process.");
      }
    }
  });
}

fn main() {
  // Startup the python binary (api service)
  let (tx,rx) = sync_channel(1);
  start_backend(rx);

  tauri::Builder::default()
    .setup(|app| {
      #[cfg(debug_assertions)] // only include this code on debug builds
      {
        let window = app.get_window("main").unwrap();
        window.open_devtools();
        window.close_devtools();
      }
      Ok(())
    })
    // Tell the child process to shutdown when app exits
    .on_window_event(move |event| match event.event() {
      WindowEvent::Destroyed => {
        tx.send(-1).expect("[Error] sending msg.");
        println!("[Event] App closed, shutting down API...");
      }
      _ => {}
    })
    .run(tauri::generate_context!())
    .expect("[Error] while running tauri application");
}
